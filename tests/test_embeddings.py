from __future__ import annotations

import pytest

from axiom.core.embeddings import split_text_into_chunks, validate_embedding_inputs


def test_embedding_chunks_preserve_offsets_and_overlap():
    text = ("AXIOM local embeddings preserve source offsets. " * 20).strip()
    chunks = split_text_into_chunks([text], chunk_size_chars=300, chunk_overlap_chars=40)
    assert len(chunks) > 1
    assert all(item["text"] == text[item["start_char"]:item["end_char"]] for item in chunks)
    assert all(item["characters"] == item["end_char"] - item["start_char"] for item in chunks)
    assert all(item["chunk_index"] == index for index, item in enumerate(chunks, 1))
    assert all(left["start_char"] < right["start_char"] < left["end_char"] for left, right in zip(chunks, chunks[1:]))


def test_embedding_chunks_reject_pathological_counts_and_overlap():
    with pytest.raises(ValueError, match="overlap"):
        split_text_into_chunks(["text"], chunk_size_chars=256, chunk_overlap_chars=256)
    with pytest.raises(ValueError, match="chunk request limit"):
        split_text_into_chunks(["x" * 65_536], chunk_size_chars=256, chunk_overlap_chars=128)


def test_embedding_input_validation_enforces_bounds():
    validate_embedding_inputs(["one", "two"], batch_size=2, max_length=128, device="cpu")
    with pytest.raises(ValueError, match="non-whitespace"):
        validate_embedding_inputs(["   "])
    with pytest.raises(ValueError, match="per-input"):
        validate_embedding_inputs(["x" * 65_537])
    with pytest.raises(ValueError, match="Combined"):
        validate_embedding_inputs(["x" * 65_536] * 5)
    with pytest.raises(ValueError, match="Batch size"):
        validate_embedding_inputs(["x"], batch_size=17)
    with pytest.raises(ValueError, match="Device"):
        validate_embedding_inputs(["x"], device="cloud")
    with pytest.raises(ValueError, match="Device"):
        validate_embedding_inputs(["x"], device=[])
    with pytest.raises(ValueError, match="normalization"):
        validate_embedding_inputs(["x"], normalize="yes")
