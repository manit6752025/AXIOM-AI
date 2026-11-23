from __future__ import annotations

from fastapi.testclient import TestClient

from axiom.api.server import create_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_WORKSPACES_HOME", str(tmp_path / ".axiom-test-home"))
    monkeypatch.setenv("AXIOM_WORKSPACES_DIR", str(tmp_path / ".axiom-managed-workspaces"))
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app())


def test_knowledge_api_creates_base_ingests_document_and_keeps_text_out_of_metadata(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    source = documents / "guide.md"
    source_text = "# Secret title\nPrivate document body."
    source.write_text(source_text, encoding="utf-8")

    with _client(tmp_path, monkeypatch) as client:
        created = client.post("/api/workspace/knowledge-bases", json={
            "name": "Engineering", "description": "Local team notes",
        })
        assert created.status_code == 201, created.text
        base = created.json()["knowledge_base"]
        assert base["name"] == "Engineering"
        assert base["document_count"] == 0

        ingested = client.post(
            f"/api/workspace/knowledge-bases/{base['id']}/documents",
            json={"path": "documents/guide.md"},
        )
        assert ingested.status_code == 201, ingested.text
        payload = ingested.json()
        assert payload["source_text_returned"] is False
        assert payload["source_text_persisted"] is True
        assert payload["document"]["content_sha256"]
        assert source_text not in ingested.text

        listing = client.get(f"/api/workspace/knowledge-bases/{base['id']}/documents")
        assert listing.status_code == 200, listing.text
        assert listing.json()["total"] == 1
        assert source_text not in listing.text

        bases = client.get("/api/workspace/knowledge-bases")
        assert bases.status_code == 200
        assert bases.json()["knowledge_bases"][0]["document_count"] == 1

        removed = client.delete(
            f"/api/workspace/knowledge-bases/{base['id']}/documents/{payload['document']['id']}"
        )
        assert removed.status_code == 200
        assert removed.json()["source_file_modified"] is False
        assert source.read_text(encoding="utf-8") == source_text


def test_knowledge_api_rejects_unsafe_and_unsupported_document_paths(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "source.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (documents / "binary.txt").write_bytes(b"\x00\x01")
    (tmp_path / "outside.md").write_text("outside", encoding="utf-8")
    (documents / "external.md").symlink_to(tmp_path / "outside.md")

    with _client(tmp_path, monkeypatch) as client:
        base = client.post("/api/workspace/knowledge-bases", json={"name": "Safe"}).json()["knowledge_base"]
        traversal = client.post(f"/api/workspace/knowledge-bases/{base['id']}/documents", json={"path": "../outside.md"})
        unsupported = client.post(f"/api/workspace/knowledge-bases/{base['id']}/documents", json={"path": "outside.md"})
        symlink = client.post(f"/api/workspace/knowledge-bases/{base['id']}/documents", json={"path": "documents/external.md"})
        binary = client.post(f"/api/workspace/knowledge-bases/{base['id']}/documents", json={"path": "documents/binary.txt"})
        extra = client.post(f"/api/workspace/knowledge-bases/{base['id']}/documents", json={
            "path": "documents/source.csv", "absolute_path": "/etc/passwd",
        })
        assert traversal.status_code in {400, 422}
        assert unsupported.status_code == 422
        assert symlink.status_code in {400, 422}
        assert binary.status_code == 422
        assert extra.status_code == 422


def test_knowledge_api_is_workspace_scoped_and_removes_bases(tmp_path, monkeypatch):
    doc = tmp_path / "notes.txt"
    doc.write_text("isolated content", encoding="utf-8")
    with _client(tmp_path, monkeypatch) as client:
        first = client.post("/api/workspace/knowledge-bases", json={"name": "First"}).json()["knowledge_base"]
        second = client.post("/api/workspace/knowledge-bases", json={"name": "Second"}).json()["knowledge_base"]
        assert first["id"] != second["id"]
        ingested = client.post(f"/api/workspace/knowledge-bases/{first['id']}/documents", json={"path": "notes.txt"})
        assert ingested.status_code == 201
        removed = client.delete(f"/api/workspace/knowledge-bases/{first['id']}")
        assert removed.status_code == 200
        remaining = client.get("/api/workspace/knowledge-bases").json()
        assert [item["id"] for item in remaining["knowledge_bases"]] == [second["id"]]
        assert doc.read_text(encoding="utf-8") == "isolated content"



def test_knowledge_api_search_returns_cited_ranked_chunks_and_applies_metadata_filters(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    nested = documents / "engineering"
    nested.mkdir(parents=True)
    first_text = (
        "AXIOM local inference runtime keeps sensitive model execution on the workstation. "
        "This guide explains the local inference service."
    )
    second_text = (
        "A completely unrelated cooking note describes sourdough starter, flour, hydration and baking."
    )
    first = nested / "runtime.md"
    second = documents / "bread.txt"
    first.write_text(first_text, encoding="utf-8")
    second.write_text(second_text, encoding="utf-8")

    with _client(tmp_path, monkeypatch) as client:
        base = client.post("/api/workspace/knowledge-bases", json={"name": "Search"}).json()["knowledge_base"]
        ingested_first = client.post(
            f"/api/workspace/knowledge-bases/{base['id']}/documents", json={"path": "documents/engineering/runtime.md"},
        )
        ingested_second = client.post(
            f"/api/workspace/knowledge-bases/{base['id']}/documents", json={"path": "documents/bread.txt"},
        )
        assert ingested_first.status_code == 201
        assert ingested_second.status_code == 201

        response = client.post(f"/api/workspace/knowledge-bases/{base['id']}/search", json={
            "query": "AXIOM local inference runtime",
            "limit": 5,
            "formats": ["md"],
            "source_prefixes": ["documents/engineering"],
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["engine"] == "axiom-lexical-knowledge-search"
        assert payload["query"]["candidate_documents"] == 1
        assert payload["query"]["vector_enabled"] is False
        assert payload["results"]
        citation = payload["results"][0]["citation"]
        assert citation["source_path"] == "documents/engineering/runtime.md"
        assert citation["content_sha256"] == ingested_first.json()["document"]["content_sha256"]
        assert citation["start_char"] < citation["end_char"]
        assert citation["chunk_index"] == 1
        assert "AXIOM" in payload["results"][0]["snippet"]
        assert second_text not in response.text
        assert payload["privacy"]["source_documents_modified"] is False


def test_knowledge_api_search_rejects_invalid_limits_and_unsafe_filters(tmp_path, monkeypatch):
    (tmp_path / "notes.md").write_text("local AXIOM notes", encoding="utf-8")
    with _client(tmp_path, monkeypatch) as client:
        base = client.post("/api/workspace/knowledge-bases", json={"name": "Search"}).json()["knowledge_base"]
        client.post(f"/api/workspace/knowledge-bases/{base['id']}/documents", json={"path": "notes.md"})
        bad_limit = client.post(f"/api/workspace/knowledge-bases/{base['id']}/search", json={
            "query": "AXIOM", "limit": 1000,
        })
        bad_prefix = client.post(f"/api/workspace/knowledge-bases/{base['id']}/search", json={
            "query": "AXIOM", "source_prefixes": ["../outside"],
        })
        blank = client.post(f"/api/workspace/knowledge-bases/{base['id']}/search", json={
            "query": "   ",
        })
        assert bad_limit.status_code == 422
        assert bad_prefix.status_code == 422
        assert blank.status_code == 422


def test_knowledge_api_hybrid_search_invokes_local_embeddings_only_when_model_path_is_supplied(tmp_path, monkeypatch):
    (tmp_path / "notes.md").write_text("AXIOM uses local inference", encoding="utf-8")
    with _client(tmp_path, monkeypatch) as client:
        base = client.post("/api/workspace/knowledge-bases", json={"name": "Hybrid"}).json()["knowledge_base"]
        client.post(f"/api/workspace/knowledge-bases/{base['id']}/documents", json={"path": "notes.md"})
        captured = {}
        def fake_search(store, base_id, query, **kwargs):
            captured.update(base_id=base_id, query=query, **kwargs)
            return {"schema_version": 1, "engine": "axiom-hybrid-knowledge-search", "results": []}
        monkeypatch.setattr(knowledge_api, "search_knowledge_base", fake_search)
        response = client.post(f"/api/workspace/knowledge-bases/{base['id']}/search", json={
            "query": "AXIOM local inference", "embedding_model_path": str(tmp_path / "models" / "encoder"),
            "vector_weight": 0.7, "device": "cpu",
        })
        assert response.status_code == 200, response.text
        assert captured["embedding_model_path"] == str((tmp_path / "models" / "encoder").resolve())
        assert captured["vector_weight"] == 0.7
        assert captured["device"] == "cpu"
