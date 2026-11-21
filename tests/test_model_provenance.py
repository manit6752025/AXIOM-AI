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


def test_model_provenance_returns_explicit_no_weight_status(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    report = inspect_model_provenance(model)
    assert report["format_evidence"]["weight_file_count"] == 0
    assert report["hash_evidence"]["status"] == "no_weight_files"
    assert report["hash_evidence"]["checkpoint_sha256"] is None


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
