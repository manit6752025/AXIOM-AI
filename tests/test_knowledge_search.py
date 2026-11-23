from __future__ import annotations

from pathlib import Path

import pytest

from axiom.core.knowledge import KnowledgeBaseStore, KnowledgeStoreCapacityError
from axiom.core.knowledge_search import (
    KnowledgeSearchError,
    search_knowledge_base,
)


def _seed_store(tmp_path: Path):
    (tmp_path / "docs" / "engineering").mkdir(parents=True)
    (tmp_path / "docs" / "engineering" / "runtime.md").write_text(
        "AXIOM local inference runtime uses private models. "
        "The local runtime serves inference requests without cloud access.",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "bread.txt").write_text(
        "Sourdough starter hydration, flour and baking temperatures are unrelated.",
        encoding="utf-8",
    )
    store = KnowledgeBaseStore(tmp_path)
    base = store.create_base("Searchable")
    store.ingest_document(base["id"], "docs/engineering/runtime.md")
    store.ingest_document(base["id"], "docs/bread.txt")
    return store, base


def test_knowledge_search_chunks_ranks_and_returns_verifiable_citations(tmp_path):
    store, base = _seed_store(tmp_path)
    result = search_knowledge_base(store, base["id"], "AXIOM local inference runtime")

    assert result["engine"] == "axiom-lexical-knowledge-search"
    assert result["query"]["vector_enabled"] is False
    assert result["results"]
    best = result["results"][0]
    assert best["citation"]["source_path"] == "docs/engineering/runtime.md"
    assert best["citation"]["start_char"] == 0
    assert best["citation"]["end_char"] <= 1200
    assert best["citation"]["content_sha256"]
    assert "AXIOM" in best["snippet"]
    assert best["lexical_score"] > 0
    assert best["vector_score"] is None
    assert result["privacy"]["full_source_documents_returned"] is False


def test_knowledge_search_metadata_filters_limit_candidate_documents(tmp_path):
    store, base = _seed_store(tmp_path)
    result = search_knowledge_base(
        store, base["id"], "AXIOM local",
        formats=["md"], source_prefixes=["docs/engineering"],
    )
    assert result["query"]["candidate_documents"] == 1
    assert all(item["citation"]["source_path"].startswith("docs/engineering/") for item in result["results"])

    selected_document = store.list_documents(base["id"])["documents"][1]
    by_id = search_knowledge_base(
        store, base["id"], "flour",
        document_ids=[selected_document["id"]],
    )
    assert by_id["query"]["candidate_documents"] == 1
    assert by_id["results"][0]["citation"]["document_id"] == selected_document["id"]


def test_knowledge_search_breaks_equal_score_ties_by_stable_source_path(tmp_path):
    (tmp_path / "docs" / "ties").mkdir(parents=True)
    (tmp_path / "docs" / "ties" / "zeta.md").write_text(
        "Deterministic retrieval ranking signal.", encoding="utf-8",
    )
    (tmp_path / "docs" / "ties" / "alpha.md").write_text(
        "Deterministic retrieval ranking signal.", encoding="utf-8",
    )
    store = KnowledgeBaseStore(tmp_path)
    base = store.create_base("Stable ties")
    # Deliberately ingest in reverse lexical order; ranking tie-breaks must not
    # depend on insertion/list order when scores and chunk indices are equal.
    store.ingest_document(base["id"], "docs/ties/zeta.md")
    store.ingest_document(base["id"], "docs/ties/alpha.md")

    first = search_knowledge_base(
        store, base["id"], "deterministic retrieval ranking",
        source_prefixes=["docs/ties"], limit=2,
    )
    second = search_knowledge_base(
        store, base["id"], "deterministic retrieval ranking",
        source_prefixes=["docs/ties"], limit=2,
    )

    paths = [item["citation"]["source_path"] for item in first["results"]]
    assert paths == ["docs/ties/alpha.md", "docs/ties/zeta.md"]
    assert paths == [item["citation"]["source_path"] for item in second["results"]]
    assert first["results"][0]["score"] == first["results"][1]["score"]


def test_knowledge_search_fuses_local_embedding_scores_when_model_is_supplied(tmp_path, monkeypatch):
    store, base = _seed_store(tmp_path)
    calls = []

    def fake_embed(model_path, texts, **kwargs):
        calls.append((model_path, list(texts), kwargs))
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "AXIOM" in text or text == "AXIOM" else [0.0, 1.0])
        return {"embeddings": vectors}

    monkeypatch.setattr("axiom.core.knowledge_search.embed_local_texts", fake_embed)
    result = search_knowledge_base(
        store, base["id"], "AXIOM",
        embedding_model_path="/trusted/encoder", device="cpu",
    )
    assert result["engine"] == "axiom-hybrid-knowledge-search"
    assert result["query"]["vector_enabled"] is True
    assert calls[0][1] == ["AXIOM"]
    assert calls[0][2]["device"] == "cpu"
    assert len(result["results"]) >= 1
    assert result["results"][0]["vector_score"] is not None


def test_knowledge_search_hybrid_ranking_recovers_semantic_only_matches(tmp_path, monkeypatch):
    store, base = _seed_store(tmp_path)
    (tmp_path / "docs" / "mobility.md").write_text(
        "A vehicle drivetrain transfers torque from the engine to the wheels.",
        encoding="utf-8",
    )
    store.ingest_document(base["id"], "docs/mobility.md")

    def fake_embed(model_path, texts, **kwargs):
        vectors = []
        for text in texts:
            normalized = text.casefold()
            if "automotive" in normalized or "vehicle" in normalized:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return {"embeddings": vectors}

    monkeypatch.setattr("axiom.core.knowledge_search.embed_local_texts", fake_embed)
    result = search_knowledge_base(
        store,
        base["id"],
        "automotive",
        embedding_model_path="/trusted/encoder",
        vector_weight=1.0,
        device="cpu",
    )

    assert result["engine"] == "axiom-hybrid-knowledge-search"
    assert result["query"]["vector_enabled"] is True
    assert result["query"]["vector_weight"] == 1.0
    assert result["results"]
    best = result["results"][0]
    assert best["citation"]["source_path"] == "docs/mobility.md"
    assert best["lexical_score"] == 0.0
    assert best["vector_score"] == 1.0
    assert best["score"] == 1.0
    assert best["citation"]["content_sha256"]


@pytest.mark.parametrize("kwargs", [
    {"limit": 0},
    {"max_documents": 65},
    {"chunk_size_chars": 64},
    {"chunk_overlap_chars": 1200},
    {"vector_weight": 1.5},
    {"device": "remote"},
    {"device": []},
])
def test_knowledge_search_rejects_unbounded_parameters(tmp_path, kwargs):
    store, base = _seed_store(tmp_path)
    with pytest.raises(KnowledgeSearchError):
        search_knowledge_base(store, base["id"], "AXIOM", **kwargs)


def test_knowledge_search_refuses_to_truncate_when_chunk_budget_is_exceeded(tmp_path):
    (tmp_path / "large.md").write_text("AXIOM retrieval text. " * 5000, encoding="utf-8")
    store = KnowledgeBaseStore(tmp_path)
    base = store.create_base("Chunk bound")
    store.ingest_document(base["id"], "large.md")
    with pytest.raises(KnowledgeStoreCapacityError, match="chunk limit"):
        search_knowledge_base(
            store, base["id"], "AXIOM",
            chunk_size_chars=256, chunk_overlap_chars=0,
        )
