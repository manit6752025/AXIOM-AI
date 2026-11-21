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
