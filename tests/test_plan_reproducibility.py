from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from axiom.core.plan_reproducibility import (
    PLANNER_ALGORITHM_VERSION,
    diff_plan_records,
    reproducible_plan_document,
)
from axiom.core.experiments import ExperimentStore
from axiom.cli.main import app


def _record(
    record_id: str,
    *,
    parameters: float = 7,
    method: str = "qlora",
    vram: float = 16,
    learning_rate: float = 0.0002,
    notes: str | None = None,
    seed: int = 42,
    torch_version: str | None = "2.4.0",
) -> dict:
    return {
        "schema_version": 1,
        "id": record_id,
        "name": f"{record_id} plan",
        "created_at": "2026-11-18T10:00:00+00:00",
        "notes": notes,
        "tags": ["baseline"],
        "model": "models/qwen",
        "dataset": "data/train.jsonl",
        "parameter_billions": parameters,
        "requested_method": method,
        "planner_algorithm_version": PLANNER_ALGORITHM_VERSION,
        "axiom_version": "0.3.0-beta.8",
        "seed": seed,
        "software_manifest": {
            "schema_version": 1,
            "captured": True,
            "axiom_version": "0.3.0-beta.8",
            "python_version": "3.12.3",
            "python_implementation": "CPython",
            "platform": {"system": "Linux", "machine": "x86_64"},
            "packages": {
                "torch": torch_version,
                "transformers": "5.0.0",
                "peft": "0.20.0",
                "accelerate": "1.1.0",
                "numpy": "2.1.0",
                "safetensors": "0.4.5",
            },
        },
        "status": "planned",
        "hardware": {
            "os": "TestOS",
            "architecture": "x86_64",
            "ram_gb": 32,
            "gpu_name": "Test GPU",
            "vram_gb": vram,
            "cuda_available": True,
        },
        "plan": {
            "method": "qlora",
            "precision": "4-bit",
            "lora_rank": 16,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "sequence_length": 4096,
            "learning_rate": learning_rate,
            "estimated_vram_gb": 7.84,
            "fits_hardware": True,
            "reason": "Configuration is estimated to fit available GPU memory.",
        },
    }


def test_reproducible_plan_fingerprint_is_stable_and_ignores_record_labels():
    left = _record("left", notes="first note")
    right = _record("right", notes="updated note")
    right["name"] = "A different display name"
    right["created_at"] = "2026-11-19T20:00:00+11:00"
    right["tags"] = ["release"]

    first = reproducible_plan_document(left)
    second = reproducible_plan_document(right)

    assert first["fingerprint"] == second["fingerprint"]
    assert first["schema_version"] == 1
    assert first["planner"]["algorithm_version"] == PLANNER_ALGORITHM_VERSION
    assert first["execution"]["started"] is False
    assert first["reproducibility"]["seed"] == 42
    assert first["reproducibility"]["seed_status"] == "recorded_only"
    assert first["reproducibility"]["seed_applied"] is False
    assert first["reproducibility"]["software"]["captured"] is True
    assert first["reproducibility"]["software"]["python_version"] == "3.12.3"
    assert first["reproducibility"]["software"]["packages"]["torch"] == "2.4.0"
    assert first["format"] == "application/vnd.axiom.training-plan+json;version=1"
    assert json.dumps(first, sort_keys=True, allow_nan=False)


def test_reproducible_plan_legacy_record_does_not_invent_provenance():
    legacy = _record("legacy")
    legacy.pop("planner_algorithm_version")
    legacy.pop("axiom_version")
    legacy.pop("seed")
    legacy.pop("software_manifest")

    snapshot = reproducible_plan_document(legacy)

    assert snapshot["planner"]["algorithm_version"] is None
    assert snapshot["planner"]["axiom_version"] is None
    assert snapshot["reproducibility"]["seed"] is None
    assert snapshot["reproducibility"]["seed_status"] == "not_recorded"
    assert snapshot["reproducibility"]["seed_applied"] is False
    assert snapshot["reproducibility"]["software"]["captured"] is False
    assert snapshot["reproducibility"]["software"]["packages"] == {}
    assert snapshot["planner"]["snapshot_kind"] == "saved_estimate"


def test_plan_diff_reports_changed_inputs_hardware_estimates_and_reproducibility():
    left = _record("left")
    right = _record("right", parameters=13, vram=24, learning_rate=0.0001, seed=1234, torch_version="2.5.0")

    result = diff_plan_records(left, right)

    assert result["identical"] is False
    assert result["left"]["fingerprint"] != result["right"]["fingerprint"]
    changes = {item["path"]: item for item in result["changes"]}
    assert changes["inputs.parameter_billions"]["before"] == 7
    assert changes["inputs.parameter_billions"]["after"] == 13
    assert changes["hardware.vram_gb"]["category"] == "hardware"
    assert changes["plan.learning_rate"]["category"] == "estimate"
    assert changes["reproducibility.seed"]["category"] == "provenance"
    assert changes["reproducibility.seed"]["after"] == 1234
    assert changes["reproducibility.software.packages.torch"]["category"] == "provenance"
    assert result["scope_note"]


def test_plan_diff_is_identical_for_same_saved_plan_even_with_different_record_ids():
    result = diff_plan_records(_record("first"), _record("second"))

    assert result["identical"] is True
    assert result["changes"] == []


def test_plan_printer_and_diff_cli_emit_canonical_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ExperimentStore()
    left = _record("plan-left")
    right = _record("plan-right", parameters=13)
    store.add(left)
    store.add(right)
    runner = CliRunner()

    printed = runner.invoke(app, ["experiment", "print", "plan-left"])
    assert printed.exit_code == 0, printed.output
    printed_payload = json.loads(printed.stdout)
    assert printed_payload["fingerprint"]["value"] == reproducible_plan_document(left)["fingerprint"]["value"]
    assert printed_payload["execution"]["started"] is False
    assert printed_payload["reproducibility"]["seed"] == 42
    assert printed_payload["reproducibility"]["seed_applied"] is False
    assert printed_payload["reproducibility"]["software"]["packages"]["torch"] == "2.4.0"

    compared = runner.invoke(app, ["experiment", "diff", "plan-left", "plan-right", "--json"])
    assert compared.exit_code == 0, compared.output
    compared_payload = json.loads(compared.stdout)
    assert compared_payload["identical"] is False
    assert any(item["path"] == "inputs.parameter_billions" for item in compared_payload["changes"])


def test_plan_cli_rejects_missing_or_legacy_incomplete_snapshots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ExperimentStore().add({"id": "metadata-only", "name": "No plan"})
    runner = CliRunner()

    result = runner.invoke(app, ["experiment", "print", "metadata-only"])

    assert result.exit_code != 0
    assert "no complete training-plan snapshot" in result.output.lower()


def test_reproducible_plan_and_diff_api_routes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    monkeypatch.chdir(tmp_path)
    store = ExperimentStore()
    store.add(_record("api-left"))
    store.add(_record("api-right", parameters=13))

    with TestClient(create_app()) as client:
        printed = client.get("/api/workspace/experiments/api-left/reproducible")
        assert printed.status_code == 200, printed.text
        snapshot = printed.json()["snapshot"]
        assert snapshot["fingerprint"]["algorithm"] == "sha256"
        assert snapshot["planner"]["algorithm_version"] == PLANNER_ALGORITHM_VERSION
        assert snapshot["reproducibility"]["seed"] == 42
        assert snapshot["reproducibility"]["seed_applied"] is False
        assert snapshot["reproducibility"]["software"]["packages"]["torch"] == "2.4.0"

        diff = client.get("/api/workspace/experiments/api-left/diff/api-right")
        assert diff.status_code == 200, diff.text
        result = diff.json()["diff"]
        assert result["identical"] is False
        assert any(item["path"] == "inputs.parameter_billions" for item in result["changes"])

        missing = client.get("/api/workspace/experiments/missing/reproducible")
        assert missing.status_code == 404


def test_experiment_create_captures_seed_and_software_versions(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        created = client.post("/api/workspace/experiments", json={
            "name": "Seeded plan",
            "parameter_billions": 7,
            "method": "qlora",
            "seed": 987654,
        })
        assert created.status_code == 201, created.text
        record = created.json()["experiment"]
        assert record["seed"] == 987654
        assert record["software_manifest"]["captured"] is True
        assert record["software_manifest"]["axiom_version"]
        assert record["software_manifest"]["python_version"]
        assert "torch" in record["software_manifest"]["packages"]

        exported = client.get(f"/api/workspace/experiments/{record['id']}/reproducible")
        assert exported.status_code == 200, exported.text
        manifest = exported.json()["snapshot"]["reproducibility"]
        assert manifest["seed"] == 987654
        assert manifest["seed_status"] == "recorded_only"
        assert manifest["seed_applied"] is False

        invalid = client.post("/api/workspace/experiments", json={
            "name": "Boolean seed",
            "parameter_billions": 7,
            "method": "qlora",
            "seed": True,
        })
        assert invalid.status_code == 422


@pytest.mark.parametrize("seed", [True, -1, 4_294_967_296, "42"])
def test_reproducible_plan_rejects_invalid_seed_values(seed):
    record = _record("invalid-seed")
    record["seed"] = seed

    with pytest.raises(ValueError, match="reproducibility seed"):
        reproducible_plan_document(record)


def test_reproducible_plan_rejects_invalid_software_package_versions():
    record = _record("invalid-version")
    record["software_manifest"]["packages"]["torch"] = {"version": "2.4.0"}

    with pytest.raises(ValueError, match="software package version"):
        reproducible_plan_document(record)


def test_plan_diff_rejects_nonfinite_snapshot_values():
    record = _record("invalid")
    record["plan"]["learning_rate"] = float("nan")

    with pytest.raises(ValueError, match="canonical JSON"):
        reproducible_plan_document(record)
