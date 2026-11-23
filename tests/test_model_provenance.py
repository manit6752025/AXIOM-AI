from __future__ import annotations

import hashlib

import pytest

from axiom.models import provenance
from axiom.models.provenance import ModelProvenanceError, inspect_model_provenance


def test_model_provenance_hashes_weights_and_license_evidence_deterministically(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weight = b"private model tensor bytes"
    (model / "model.safetensors").write_bytes(weight)
    (model / "LICENSE").write_text("SPDX-License-Identifier: Apache-2.0\n", encoding="utf-8")
    (model / "README.md").write_text("Model card text must not be returned.", encoding="utf-8")

    metadata = {
        "source": "example/model",
        "source_url": "https://huggingface.co/example/model",
        "source_revision": "immutable-revision",
        "source_attribution": "Example publisher",
        "license": "Apache-2.0",
        "license_review": "unreviewed",
        "license_review_notes": "Needs legal review.",
    }
    report = inspect_model_provenance(model, registry_metadata=metadata)

    weight_sha = hashlib.sha256(weight).hexdigest()
    expected_tree = hashlib.sha256()
    expected_tree.update(b"model.safetensors")
    expected_tree.update(b"\0")
    expected_tree.update(str(len(weight)).encode("ascii"))
    expected_tree.update(b"\0")
    expected_tree.update(bytes.fromhex(weight_sha))

    assert report["schema_version"] == 1
    assert report["format_evidence"]["detected_formats"] == ["safetensors"]
    assert report["format_evidence"]["weight_file_count"] == 1
    assert report["format_evidence"]["pickle_capable_weight_files_present"] is False
    assert report["hash_evidence"]["status"] == "complete"
    assert report["hash_evidence"]["files"] == [{
        "path": "model.safetensors",
        "format": "safetensors",
        "size_bytes": len(weight),
        "sha256": weight_sha,
    }]
    assert report["hash_evidence"]["checkpoint_sha256"] == expected_tree.hexdigest()
    assert report["license_evidence"]["status"] == "local_files_found"
    assert report["license_evidence"]["files"][0]["path"] == "LICENSE"
    assert report["license_evidence"]["declared_license"] == "Apache-2.0"
    assert report["license_evidence"]["registry_review_state"] == "unreviewed"
    assert report["source_evidence"]["status"] == "registry_claim_not_verified"
    assert report["source_evidence"]["network_used"] is False
    assert report["security_evidence"]["model_weights_loaded"] is False
    assert report["security_evidence"]["remote_code_executed"] is False
    assert "private model tensor bytes" not in str(report)
    assert "Model card text must not be returned" not in str(report)


def test_model_provenance_skips_symlinks_and_flags_pickle_capable_formats(tmp_path):
    model = tmp_path / "model"
    outside = tmp_path / "outside"
    model.mkdir()
    outside.mkdir()
    (model / "weights.bin").write_bytes(b"pickle-capable format")
    (outside / "outside.safetensors").write_bytes(b"must not be scanned")
    try:
        (model / "linked.safetensors").symlink_to(outside / "outside.safetensors")
        (model / "linked-directory").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    report = inspect_model_provenance(model)
    assert report["format_evidence"]["detected_formats"] == ["pytorch_pickle_or_binary"]
    assert report["format_evidence"]["pickle_capable_weight_files_present"] is True
    assert report["security_evidence"]["symlink_files_skipped"] == 1
    assert report["security_evidence"]["symlink_directories_skipped"] == 1
    assert all("linked" not in item["path"] for item in report["format_evidence"]["files"])


def test_model_provenance_skips_hashing_above_declared_byte_limit(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.gguf").write_bytes(b"x" * 100)
    monkeypatch.setattr(provenance, "MAX_PROVENANCE_HASH_BYTES", 10)

    report = inspect_model_provenance(model)
    assert report["format_evidence"]["weight_size_bytes"] == 100
    assert report["hash_evidence"]["status"] == "skipped_size_or_file_count_limit"
    assert report["hash_evidence"]["checkpoint_sha256"] is None
    assert report["hash_evidence"]["files"] == []


def test_model_provenance_skips_license_hashing_over_aggregate_bounds(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for index in range(3):
        (model / f"LICENSE-{index}.txt").write_bytes(b"license evidence")
    monkeypatch.setattr(provenance, "MAX_LICENSE_EVIDENCE_FILES", 2)

    report = inspect_model_provenance(model)
    evidence = report["license_evidence"]
    assert evidence["status"] == "skipped_count_or_total_size_limit"
    assert evidence["candidate_file_count"] == 3
    assert evidence["files"] == []


def test_model_provenance_skips_license_hashing_over_total_byte_bound(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "LICENSE").write_bytes(b"x" * 32)
    (model / "NOTICE").write_bytes(b"y" * 32)
    monkeypatch.setattr(provenance, "MAX_TOTAL_LICENSE_EVIDENCE_BYTES", 32)

    report = inspect_model_provenance(model)
    evidence = report["license_evidence"]
    assert evidence["status"] == "skipped_count_or_total_size_limit"
    assert evidence["candidate_total_bytes"] == 64
    assert evidence["files"] == []


def test_model_provenance_returns_explicit_no_weight_status(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    report = inspect_model_provenance(model)
    assert report["format_evidence"]["weight_file_count"] == 0
    assert report["hash_evidence"]["status"] == "no_weight_files"
    assert report["hash_evidence"]["checkpoint_sha256"] is None


def test_model_provenance_rejects_symlinked_ancestor_components(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_model = real_parent / "model"
    real_model.mkdir(parents=True)
    (real_model / "model.gguf").write_bytes(b"weights")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ModelProvenanceError, match="symbolic links"):
        inspect_model_provenance(linked_parent / "model")


def test_model_provenance_rejects_files_and_symlinked_roots(tmp_path):
    not_directory = tmp_path / "not-a-directory"
    not_directory.write_text("not a model folder", encoding="utf-8")
    with pytest.raises(ModelProvenanceError, match="directory"):
        inspect_model_provenance(not_directory)

    real = tmp_path / "real-model"
    real.mkdir()
    linked = tmp_path / "linked-model"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ModelProvenanceError, match="symbolic link"):
        inspect_model_provenance(linked)

def test_model_provenance_api_enforces_workspace_paths_and_labels_registry_claims(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    model = tmp_path / "models" / "demo"
    model.mkdir(parents=True)
    weights = b"local weights"
    (model / "model.safetensors").write_bytes(weights)
    (model / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        registered = client.post("/api/workspace/models", json={
            "name": "Demo local model",
            "source": "models/demo",
            "format": "safetensors",
            "license": "Apache-2.0",
            "license_review": "unreviewed",
            "source_attribution": "Example publisher",
            "source_url": "https://huggingface.co/example/demo",
            "source_revision": "deadbeef",
        })
        assert registered.status_code == 201, registered.text
        model_id = registered.json()["model"]["id"]

        response = client.post("/api/workspace/models/provenance", json={
            "path": "demo",
            "model_id": model_id,
        })
        assert response.status_code == 200, response.text
        report = response.json()["provenance"]
        assert report["model_directory"] == "models/demo"
        assert report["hash_evidence"]["checkpoint_sha256"]
        assert report["hash_evidence"]["files"][0]["sha256"] == hashlib.sha256(weights).hexdigest()
        assert report["registry_metadata"]["status"] == "supplied"
        assert report["source_evidence"]["status"] == "registry_claim_not_verified"
        assert report["source_evidence"]["network_used"] is False
        assert report["license_evidence"]["declared_license"] == "Apache-2.0"
        assert report["license_evidence"]["registry_review_state"] == "unreviewed"

        other_model = tmp_path / "models" / "other"
        other_model.mkdir()
        mismatch = client.post("/api/workspace/models/provenance", json={
            "path": "other",
            "model_id": model_id,
        })
        assert mismatch.status_code == 409

        unsafe = client.post("/api/workspace/models/provenance", json={"path": "../outside"})
        assert unsafe.status_code == 422
        missing = client.post("/api/workspace/models/provenance", json={
            "path": "demo",
            "model_id": "0" * 32,
        })
        assert missing.status_code == 404

        linked = tmp_path / "models" / "linked"
        try:
            linked.symlink_to(model, target_is_directory=True)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        linked_response = client.post("/api/workspace/models/provenance", json={"path": "linked"})
        assert linked_response.status_code == 422
        for invalid_path in ("../demo", "demo/../demo", "demo//model", "demo\\\\model", "/models/demo"):
            invalid_response = client.post("/api/workspace/models/provenance", json={"path": invalid_path})
            assert invalid_response.status_code == 422


def test_model_provenance_cli_rejects_registry_claims_for_a_different_model_path(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from axiom.cli.main import app

    model = tmp_path / "models" / "demo"
    model.mkdir(parents=True)
    (model / "model.gguf").write_bytes(b"gguf test bytes")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "axiom.cli.main.ModelRegistry",
        lambda: SimpleNamespace(get=lambda _model_id: SimpleNamespace(source="models/other")),
    )

    result = CliRunner().invoke(app, [
        "--no-banner", "model", "provenance", str(model), "--model-id", "a" * 32, "--json",
    ])
    assert result.exit_code != 0
    assert "does not refer to the selected local model directory" in result.output


def test_model_provenance_cli_emits_machine_readable_hash_evidence(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from axiom.cli.main import app

    model = tmp_path / "demo"
    model.mkdir()
    (model / "model.gguf").write_bytes(b"gguf test bytes")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, [
        "--no-banner", "model", "provenance", str(model), "--json",
    ])
    assert result.exit_code == 0, result.stdout
    report = json.loads(result.stdout)["provenance"]
    assert report["format_evidence"]["detected_formats"] == ["gguf"]
    assert report["hash_evidence"]["status"] == "complete"
    assert report["hash_evidence"]["checkpoint_sha256"]
    assert report["security_evidence"]["model_weights_loaded"] is False


def test_model_provenance_hashes_a_valid_safetensors_checkpoint_artifact(tmp_path):
    """Exercise provenance hashing against a real safetensors binary container."""
    import json
    import struct

    model = tmp_path / "tiny-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    checkpoint = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
    (model / "model.safetensors").write_bytes(checkpoint)

    report = inspect_model_provenance(model)

    assert report["format_evidence"]["detected_formats"] == ["safetensors"]
    assert report["hash_evidence"]["status"] == "complete"
    assert report["hash_evidence"]["files"][0]["sha256"] == hashlib.sha256(checkpoint).hexdigest()
    assert report["hash_evidence"]["files"][0]["size_bytes"] == len(checkpoint)
    assert report["security_evidence"]["model_weights_loaded"] is False


def test_model_provenance_enforces_real_weight_file_count_boundary(tmp_path):
    model = tmp_path / "large-model"
    model.mkdir()
    for index in range(provenance.MAX_PROVENANCE_WEIGHT_FILES + 1):
        (model / f"shard-{index:04}.safetensors").write_bytes(bytes([index % 256]))

    report = inspect_model_provenance(model)

    assert report["format_evidence"]["weight_file_count"] == provenance.MAX_PROVENANCE_WEIGHT_FILES + 1
    assert report["hash_evidence"]["status"] == "skipped_size_or_file_count_limit"
    assert report["hash_evidence"]["hashed_file_count"] == 0
    assert report["hash_evidence"]["files"] == []
    assert report["hash_evidence"]["checkpoint_sha256"] is None


def test_model_provenance_enforces_real_license_file_count_boundary(tmp_path):
    model = tmp_path / "license-heavy-model"
    model.mkdir()
    for index in range(provenance.MAX_LICENSE_EVIDENCE_FILES + 1):
        (model / f"LICENSE-{index:03}.txt").write_bytes(b"license evidence")

    report = inspect_model_provenance(model)
    evidence = report["license_evidence"]

    assert evidence["candidate_file_count"] == provenance.MAX_LICENSE_EVIDENCE_FILES + 1
    assert evidence["status"] == "skipped_count_or_total_size_limit"
    assert evidence["files"] == []


def test_model_provenance_enforces_real_directory_enumeration_boundary(tmp_path):
    model = tmp_path / "directory-heavy-model"
    model.mkdir()
    for index in range(provenance.MAX_PROVENANCE_DIRECTORIES + 1):
        (model / f"directory-{index:04}").mkdir()

    with pytest.raises(ModelProvenanceError, match="too many directories"):
        inspect_model_provenance(model)

def test_model_provenance_enforces_exact_aggregate_license_byte_boundary(tmp_path):
    model = tmp_path / "license-byte-boundary"
    model.mkdir()
    per_file_bytes = provenance.MAX_TOTAL_LICENSE_EVIDENCE_BYTES // provenance.MAX_LICENSE_EVIDENCE_FILES
    assert per_file_bytes * provenance.MAX_LICENSE_EVIDENCE_FILES == provenance.MAX_TOTAL_LICENSE_EVIDENCE_BYTES
    for index in range(provenance.MAX_LICENSE_EVIDENCE_FILES):
        (model / f"LICENSE-{index:03}.txt").write_bytes(b"x" * per_file_bytes)

    at_limit = inspect_model_provenance(model)["license_evidence"]
    assert at_limit["candidate_file_count"] == provenance.MAX_LICENSE_EVIDENCE_FILES
    assert at_limit["candidate_total_bytes"] == provenance.MAX_TOTAL_LICENSE_EVIDENCE_BYTES
    assert at_limit["status"] == "local_files_found"
    assert len(at_limit["files"]) == provenance.MAX_LICENSE_EVIDENCE_FILES
    assert all(item["hash_status"] == "complete" for item in at_limit["files"])

    (model / f"LICENSE-{provenance.MAX_LICENSE_EVIDENCE_FILES - 1:03}.txt").write_bytes(
        b"x" * (per_file_bytes + 1)
    )
    over_limit = inspect_model_provenance(model)["license_evidence"]
    assert over_limit["candidate_total_bytes"] == provenance.MAX_TOTAL_LICENSE_EVIDENCE_BYTES + 1
    assert over_limit["status"] == "skipped_count_or_total_size_limit"
    assert over_limit["files"] == []


def test_model_provenance_enforces_total_regular_file_inventory_boundary(tmp_path):
    model = tmp_path / "file-heavy-model"
    model.mkdir()
    for index in range(provenance.MAX_PROVENANCE_FILES + 1):
        (model / f"metadata-{index:05}.txt").touch()

    with pytest.raises(ModelProvenanceError, match="too many files"):
        inspect_model_provenance(model)

def test_model_provenance_hashes_all_weight_files_at_file_count_limit(tmp_path):
    model = tmp_path / "weight-file-boundary"
    model.mkdir()
    for index in range(provenance.MAX_PROVENANCE_WEIGHT_FILES):
        (model / f"shard-{index:04}.safetensors").write_bytes(bytes([index % 256]))

    report = inspect_model_provenance(model)

    assert report["format_evidence"]["weight_file_count"] == provenance.MAX_PROVENANCE_WEIGHT_FILES
    assert report["hash_evidence"]["status"] == "complete"
    assert report["hash_evidence"]["hashed_file_count"] == provenance.MAX_PROVENANCE_WEIGHT_FILES
    assert len(report["hash_evidence"]["files"]) == provenance.MAX_PROVENANCE_WEIGHT_FILES
    assert report["hash_evidence"]["checkpoint_sha256"]


def test_model_provenance_hashes_license_evidence_at_exact_aggregate_byte_limit(tmp_path):
    model = tmp_path / "license-byte-boundary"
    model.mkdir()
    per_file_bytes = provenance.MAX_TOTAL_LICENSE_EVIDENCE_BYTES // provenance.MAX_LICENSE_EVIDENCE_FILES
    assert per_file_bytes * provenance.MAX_LICENSE_EVIDENCE_FILES == provenance.MAX_TOTAL_LICENSE_EVIDENCE_BYTES
    for index in range(provenance.MAX_LICENSE_EVIDENCE_FILES):
        (model / f"LICENSE-{index:03}.txt").write_bytes(b"x" * per_file_bytes)

    at_limit = inspect_model_provenance(model)["license_evidence"]
    assert at_limit["candidate_file_count"] == provenance.MAX_LICENSE_EVIDENCE_FILES
    assert at_limit["candidate_total_bytes"] == provenance.MAX_TOTAL_LICENSE_EVIDENCE_BYTES
    assert at_limit["status"] == "local_files_found"
    assert len(at_limit["files"]) == provenance.MAX_LICENSE_EVIDENCE_FILES
    assert all(item["hash_status"] == "complete" for item in at_limit["files"])

    (model / f"LICENSE-{provenance.MAX_LICENSE_EVIDENCE_FILES - 1:03}.txt").write_bytes(
        b"x" * (per_file_bytes + 1)
    )
    over_limit = inspect_model_provenance(model)["license_evidence"]
    assert over_limit["candidate_total_bytes"] == provenance.MAX_TOTAL_LICENSE_EVIDENCE_BYTES + 1
    assert over_limit["status"] == "skipped_count_or_total_size_limit"
    assert over_limit["files"] == []



def test_model_provenance_rejects_sparse_checkpoint_over_production_hash_limit_without_reading(
    tmp_path,
    monkeypatch,
):
    """A sparse >16 GiB weight must be rejected on metadata before a digest read."""
    model = tmp_path / "oversized-sparse-checkpoint"
    model.mkdir()
    sparse_weight = model / "large-model.safetensors"
    with sparse_weight.open("wb") as stream:
        stream.seek(provenance.MAX_PROVENANCE_HASH_BYTES)
        stream.write(bytes([0]))

    assert sparse_weight.stat().st_size == provenance.MAX_PROVENANCE_HASH_BYTES + 1
    monkeypatch.setattr(
        provenance,
        "_hash_file",
        lambda *_args, **_kwargs: pytest.fail("over-limit checkpoint must not be opened for hashing"),
    )

    report = inspect_model_provenance(model)
    assert report["format_evidence"]["weight_file_count"] == 1
    assert report["format_evidence"]["weight_size_bytes"] == provenance.MAX_PROVENANCE_HASH_BYTES + 1
    assert report["hash_evidence"]["status"] == "skipped_size_or_file_count_limit"
    assert report["hash_evidence"]["checkpoint_sha256"] is None
    assert report["hash_evidence"]["files"] == []


def test_model_provenance_hashes_at_exact_configured_weight_byte_limit(tmp_path, monkeypatch):
    model = tmp_path / "exact-weight-byte-limit"
    model.mkdir()
    weight = b"0123456789abcdef"
    path = model / "model.safetensors"
    path.write_bytes(weight)
    monkeypatch.setattr(provenance, "MAX_PROVENANCE_HASH_BYTES", len(weight))

    at_limit = inspect_model_provenance(model)
    assert at_limit["hash_evidence"]["status"] == "complete"
    assert at_limit["hash_evidence"]["checkpoint_sha256"]
    assert at_limit["hash_evidence"]["files"] == [{
        "path": "model.safetensors",
        "format": "safetensors",
        "size_bytes": len(weight),
        "sha256": hashlib.sha256(weight).hexdigest(),
    }]

    path.write_bytes(weight + b"x")
    above_limit = inspect_model_provenance(model)
    assert above_limit["hash_evidence"]["status"] == "skipped_size_or_file_count_limit"
    assert above_limit["hash_evidence"]["checkpoint_sha256"] is None
    assert above_limit["hash_evidence"]["files"] == []


def test_model_provenance_hashes_a_valid_safetensors_artifact_at_the_full_16_gib_limit(tmp_path):
    """Hash a valid sparse safetensors container through the production 16 GiB limit."""
    import json
    import struct

    model = tmp_path / "full-production-weight-budget"
    model.mkdir()
    sparse_weight = model / "full-limit.safetensors"
    total_size = provenance.MAX_PROVENANCE_HASH_BYTES
    payload_size = total_size

    # Build a valid safetensors header, accounting for its own padded size
    # before creating a sparse tensor payload of exactly the remaining bytes.
    for _ in range(4):
        header = json.dumps(
            {
                "weight": {
                    "dtype": "U8",
                    "shape": [payload_size],
                    "data_offsets": [0, payload_size],
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        padded = header + b" " * (-len(header) % 8)
        next_payload_size = total_size - 8 - len(padded)
        if next_payload_size == payload_size:
            break
        payload_size = next_payload_size

    header = json.dumps(
        {
            "weight": {
                "dtype": "U8",
                "shape": [payload_size],
                "data_offsets": [0, payload_size],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    padded = header + b" " * (-len(header) % 8)
    assert 8 + len(padded) + payload_size == total_size

    with sparse_weight.open("wb") as stream:
        stream.write(struct.pack("<Q", len(padded)))
        stream.write(padded)
        stream.truncate(total_size)

    with sparse_weight.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        parsed = json.loads(stream.read(header_length))
    assert parsed["weight"]["dtype"] == "U8"
    assert parsed["weight"]["shape"] == [payload_size]
    assert parsed["weight"]["data_offsets"] == [0, payload_size]

    # Do not call safe_open on this 16 GiB sparse fixture: its public API
    # memory-maps the complete tensor and exceeds ordinary CI runner limits.
    # The header is decoded above, and the provenance scanner must stream-hash
    # the entire file without loading model weights into memory.

    assert sparse_weight.stat().st_size == total_size
    report = inspect_model_provenance(model)
    evidence = report["hash_evidence"]
    assert report["format_evidence"]["detected_formats"] == ["safetensors"]
    assert report["format_evidence"]["weight_file_count"] == 1
    assert report["format_evidence"]["weight_size_bytes"] == total_size
    assert evidence["status"] == "complete"
    assert evidence["hashed_file_count"] == 1
    assert evidence["files"][0]["size_bytes"] == total_size
    assert len(evidence["files"][0]["sha256"]) == 64
    # The checkpoint digest is a deterministic tree digest over path, size,
    # and the independently computed per-file digest (not the raw file digest).
    tree_digest = hashlib.sha256()
    tree_digest.update(b"full-limit.safetensors")
    tree_digest.update(bytes((0,)))
    tree_digest.update(str(total_size).encode("ascii"))
    tree_digest.update(bytes((0,)))
    tree_digest.update(bytes.fromhex(evidence["files"][0]["sha256"]))
    assert evidence["checkpoint_sha256"] == tree_digest.hexdigest()
