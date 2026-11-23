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


def test_embeddings_execute_a_real_local_safetensors_checkpoint_on_cpu(tiny_local_transformer_checkpoints):
    from axiom.core.embeddings import embed_local_texts, unload_local_embeddings

    checkpoint = tiny_local_transformer_checkpoints["embedding"]
    try:
        report = embed_local_texts(
            str(checkpoint),
            ["AXIOM local runtime embeddings preserve semantic document vectors."],
            batch_size=1,
            max_length=32,
            device="cpu",
            normalize=True,
        )
        assert report["engine"] == "axiom-local-embeddings"
        assert report["model"]["name"] == checkpoint.name
        assert report["model"]["dimension"] == 16
        assert report["device"] == "cpu"
        assert report["request"]["input_count"] == 1
        assert len(report["embeddings"]) == 1
        vector = report["embeddings"][0]
        assert len(vector) == 16
        assert all(isinstance(value, float) and math.isfinite(value) for value in vector)
        assert sum(value * value for value in vector) == pytest.approx(1.0, abs=1e-5)
        assert report["privacy"]["source_text_persisted"] is False
        assert report["privacy"]["network_used"] is False
    finally:
        unload_local_embeddings()
