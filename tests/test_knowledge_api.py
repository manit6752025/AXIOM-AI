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
