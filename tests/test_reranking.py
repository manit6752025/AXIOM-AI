from __future__ import annotations

import pytest

from axiom.core.reranking import RerankerError, _ranked_results, validate_rerank_inputs


def test_reranker_ranks_descending_and_preserves_original_order_for_ties():
    ranked = _ranked_results([0.4, 0.8, 0.8], ["first", "second", "third"])
    assert [item["document_index"] for item in ranked] == [2, 3, 1]
    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert [item["score"] for item in ranked] == [0.8, 0.8, 0.4]
    assert all("first" != item.get("document") for item in ranked)


def test_reranker_rejects_mismatched_and_nonfinite_scores():
    with pytest.raises(RerankerError, match="different number"):
        _ranked_results([0.1], ["one", "two"])
    with pytest.raises(RerankerError, match="invalid score"):
        _ranked_results([float("nan")], ["one"])


def test_rerank_input_validation_enforces_candidate_query_and_device_bounds():
    validate_rerank_inputs("query", ["one", "two"], device="cpu")
    with pytest.raises(ValueError, match="Query"):
        validate_rerank_inputs(" ", ["one"])
    with pytest.raises(ValueError, match="documents"):
        validate_rerank_inputs("query", [])
    with pytest.raises(ValueError, match="per-document"):
        validate_rerank_inputs("query", ["x" * 8193])
    with pytest.raises(ValueError, match="Combined"):
        validate_rerank_inputs("q" * 8192, ["x" * 8192] * 32)
    with pytest.raises(ValueError, match="Batch size"):
        validate_rerank_inputs("query", ["one"], batch_size=17)
    with pytest.raises(ValueError, match="Device"):
        validate_rerank_inputs("query", ["one"], device="remote")
    with pytest.raises(ValueError, match="Device"):
        validate_rerank_inputs("query", ["one"], device=[])


def test_reranker_executes_a_real_local_safetensors_checkpoint_on_cpu(tiny_local_transformer_checkpoints):
    from axiom.core.reranking import rerank_local_documents, unload_local_reranker

    checkpoint = tiny_local_transformer_checkpoints["reranker"]
    documents = [
        "AXIOM uses a local model for semantic document retrieval.",
        "Sourdough bread uses flour, water and a starter culture.",
    ]
    try:
        report = rerank_local_documents(
            str(checkpoint),
            "How does local model retrieval work?",
            documents,
            batch_size=2,
            max_length=32,
            device="cpu",
        )
        assert report["engine"] == "axiom-local-cross-encoder"
        assert report["model"]["name"] == checkpoint.name
        assert report["model"]["score_mode"] == "positive_class_softmax"
        assert report["device"] == "cpu"
        assert report["request"]["document_count"] == 2
        assert [item["document_index"] for item in report["results"]] in ([1, 2], [2, 1])
        assert [item["rank"] for item in report["results"]] == [1, 2]
        assert all(0.0 <= item["score"] <= 1.0 for item in report["results"])
        assert report["privacy"]["query_returned"] is False
        assert report["privacy"]["documents_returned"] is False
        assert report["privacy"]["network_used"] is False
    finally:
        unload_local_reranker()
