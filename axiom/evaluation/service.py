"""Shared persistence and baseline rules for offline evaluation runs."""

from __future__ import annotations

import math
import platform
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from axiom.evaluation.jsonl import (
    BaselineCorpusMismatchError,
    compare_to_baseline,
    create_output_manifest,
)
from axiom.evaluation.store import EvaluationStore
from axiom.version import __version__

_PROVENANCE_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_PROVENANCE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVENANCE_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,159}$")


def validate_prediction_provenance(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate bounded metadata saved from an AXIOM prediction job."""
    if value is None:
        return None
    expected_keys = {
        "schema_version",
        "kind",
        "prediction_job_id",
        "dataset_path",
        "output_path",
        "model_path",
        "adapter_path",
        "prompt_field",
        "prediction_field",
        "reference_field",
        "generation",
        "rows_completed",
        "prompts_truncated",
        "artifact_sha256",
        "reproducibility_note",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("Evaluation provenance is malformed.")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["kind"] != "axiom_prediction_job"
        or not isinstance(value["prediction_job_id"], str)
        or not _PROVENANCE_JOB_ID.fullmatch(value["prediction_job_id"])
        or not isinstance(value["artifact_sha256"], str)
        or not _PROVENANCE_SHA256.fullmatch(value["artifact_sha256"])
    ):
        raise ValueError("Evaluation provenance identifiers are invalid.")

    def workspace_path(raw: Any, *, required_prefix: str | None = None) -> str:
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw) > 4096
            or raw.startswith("/")
            or "\\" in raw
            or any(ord(character) < 32 or ord(character) == 127 for character in raw)
        ):
            raise ValueError("Evaluation provenance paths must be workspace-relative.")
        parts = raw.split("/")
        if (
            any(part in {"", ".", ".."} for part in parts)
            or (len(parts[0]) >= 2 and parts[0][1] == ":")
            or (required_prefix is not None and not raw.startswith(required_prefix))
        ):
            raise ValueError("Evaluation provenance paths must be safe workspace-relative paths.")
        return raw

    dataset_path = workspace_path(value["dataset_path"], required_prefix="data/")
    output_path = workspace_path(value["output_path"], required_prefix="data/")
    if dataset_path == output_path:
        raise ValueError("Evaluation provenance input and output paths must be different.")
    model_path = workspace_path(value["model_path"])
    adapter_path = (
        workspace_path(value["adapter_path"]) if value["adapter_path"] is not None else None
    )
    fields: dict[str, str] = {}
    for field_name in ("prompt_field", "prediction_field", "reference_field"):
        field = value[field_name]
        if not isinstance(field, str) or not _PROVENANCE_FIELD.fullmatch(field):
            raise ValueError("Evaluation provenance field names are invalid.")
        fields[field_name] = field
    if len(set(fields.values())) != 3:
        raise ValueError("Evaluation provenance field names must be different.")

    generation = value["generation"]
    generation_keys = {
        "max_new_tokens",
        "temperature",
        "seed",
        "device_preference",
        "precision_preference",
    }
    if not isinstance(generation, dict) or set(generation) != generation_keys:
        raise ValueError("Evaluation provenance generation settings are malformed.")
    max_new_tokens = generation["max_new_tokens"]
    temperature = generation["temperature"]
    seed = generation["seed"]
    device_preference = generation["device_preference"]
    precision_preference = generation["precision_preference"]
    try:
        normalized_temperature = float(temperature)
    except (OverflowError, ValueError, TypeError):
        normalized_temperature = float("nan")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 64
        or isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(normalized_temperature)
        or not 0 <= normalized_temperature <= 2
        or (
            seed is not None
            and (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= 2**63 - 1
            )
        )
        or not isinstance(device_preference, str)
        or device_preference not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}
        or not isinstance(precision_preference, str)
        or precision_preference not in {"auto", "fp32", "fp16", "bf16"}
    ):
        raise ValueError("Evaluation provenance generation settings are invalid.")
    prompts_truncated = value["prompts_truncated"]
    if (
        isinstance(prompts_truncated, bool)
        or not isinstance(prompts_truncated, int)
        or not 0 <= prompts_truncated <= 100
    ):
        raise ValueError("Evaluation provenance truncation count is invalid.")
    rows_completed = value["rows_completed"]
    if (
        isinstance(rows_completed, bool)
        or not isinstance(rows_completed, int)
        or not 0 <= rows_completed <= 100
        or prompts_truncated > rows_completed
    ):
        raise ValueError("Evaluation provenance row count is invalid.")
    reproducibility_note = value["reproducibility_note"]
    if reproducibility_note != (
        "Paths and settings only; model files, runtime versions, and hardware may change."
    ):
        raise ValueError("Evaluation provenance reproducibility note is invalid.")

    return {
        "schema_version": 1,
        "kind": "axiom_prediction_job",
        "prediction_job_id": value["prediction_job_id"],
        "dataset_path": dataset_path,
        "output_path": output_path,
        "model_path": model_path,
        "adapter_path": adapter_path,
        **fields,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": normalized_temperature,
            "seed": seed,
            "device_preference": device_preference,
            "precision_preference": precision_preference,
        },
        "rows_completed": rows_completed,
        "prompts_truncated": prompts_truncated,
        "artifact_sha256": value["artifact_sha256"],
        "reproducibility_note": reproducibility_note,
    }


class EvaluationBaselineNotFoundError(ValueError):
    """A requested baseline run does not exist."""


class EvaluationBaselinePermissionError(ValueError):
    """A requested comparison target is not marked as a baseline."""


class EvaluationHistoryReadError(RuntimeError):
    """Saved evaluation history could not be read safely."""


class EvaluationPersistenceError(RuntimeError):
    """An aggregate evaluation record could not be persisted safely."""


def persist_evaluation_run(
    report: dict[str, Any],
    *,
    name: str | None = None,
    model_label: str | None = None,
    baseline: bool = False,
    baseline_run_id: str | None = None,
    regression_metric: str = "token_f1_micro",
    regression_tolerance: float = 0.05,
    python_version: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare against a baseline, then persist aggregate metrics and optional provenance."""
    if not isinstance(report, dict):
        raise ValueError("Evaluation report is malformed.")
    manifest = report.get("dataset_manifest", {})
    if not isinstance(manifest, dict):
        raise ValueError("Evaluation report manifest is malformed.")
    complete = not report.get("truncated") and manifest.get("full_content_hashed") is True
    comparison_corpus_available = manifest.get("comparison_corpus_fully_hashed") is True
    if baseline and not complete:
        raise ValueError("A partial evaluation cannot be marked as a baseline.")
    if baseline and not comparison_corpus_available:
        raise ValueError("A run without a complete comparison fingerprint cannot be marked as a baseline.")
    validated_provenance = validate_prediction_provenance(provenance)
    if validated_provenance is not None:
        report_fields = report.get("fields", {})
        manifest_sha256 = manifest.get("sha256")
        if (
            not isinstance(report_fields, dict)
            or report_fields.get("prediction") != validated_provenance["prediction_field"]
            or report_fields.get("reference") != validated_provenance["reference_field"]
            or report.get("path") != validated_provenance["output_path"][5:]
            or manifest_sha256 != validated_provenance["artifact_sha256"]
            or manifest.get("full_content_hashed") is not True
        ):
            raise ValueError("Evaluation provenance does not match the evaluated prediction artifact.")

    store = EvaluationStore()
    comparison = None
    if baseline_run_id and not complete:
        comparison = {"status": "unavailable", "reason": "current_evaluation_partial"}
    elif baseline_run_id and not comparison_corpus_available:
        comparison = {"status": "unavailable", "reason": "current_comparison_unavailable"}
    elif baseline_run_id:
        try:
            baseline_record = store.get(baseline_run_id)
        except (OSError, ValueError):
            raise EvaluationHistoryReadError("Evaluation history could not be read safely.") from None
        if baseline_record is None:
            raise EvaluationBaselineNotFoundError("Baseline evaluation run was not found.")
        if baseline_record.get("baseline") is not True:
            raise EvaluationBaselinePermissionError(
                "Only a run explicitly marked as a baseline can be compared."
            )
        baseline_report = baseline_record.get("report", {})
        if not isinstance(baseline_report, dict):
            raise ValueError("Baseline evaluation record is malformed.")
        baseline_manifest = baseline_report.get("dataset_manifest", {})
        if not isinstance(baseline_manifest, dict):
            raise ValueError("Baseline evaluation record is malformed.")
        if baseline_report.get("truncated") or baseline_manifest.get("full_content_hashed") is not True:
            comparison = {"status": "unavailable", "reason": "baseline_run_partial"}
        elif baseline_manifest.get("comparison_corpus_fully_hashed") is not True:
            comparison = {"status": "unavailable", "reason": "baseline_comparison_unavailable"}
        else:
            try:
                comparison = compare_to_baseline(
                    report,
                    baseline_record,
                    baseline_run_id=baseline_run_id,
                    metric=regression_metric,
                    regression_tolerance=regression_tolerance,
                )
            except BaselineCorpusMismatchError as exc:
                comparison = {"status": "unavailable", "reason": exc.reason}

    now = datetime.now(UTC).isoformat()
    software = {"axiom": __version__}
    if python_version is not None:
        software["python"] = python_version or platform.python_version()
    record = {
        "schema_version": EvaluationStore.SCHEMA_VERSION,
        "id": uuid.uuid4().hex,
        "name": name or f"{report.get('file_name', 'dataset')} offline evaluation",
        "created_at": now,
        "model_label": model_label,
        "baseline": baseline,
        "comparison": comparison,
        "status": "complete" if complete else "partial",
        "execution_status": "precomputed_outputs_only",
        "software": software,
        "output_manifest": create_output_manifest(report),
        "report": report,
        "scope": {
            "model_inference_run": False,
            "network_accessed": False,
            "raw_values_returned": False,
            "raw_values_persisted": False,
            "metrics_are_lexical": True,
            "model_label_is_user_supplied": True,
        },
    }
    if validated_provenance is not None:
        record["provenance"] = validated_provenance
    try:
        store.add(record)
    except (OSError, TimeoutError, ValueError):
        raise EvaluationPersistenceError(
            "Offline evaluation metrics could not be saved safely."
        ) from None
    return record
