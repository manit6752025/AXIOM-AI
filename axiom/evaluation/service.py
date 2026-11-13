"""Shared persistence and baseline rules for offline evaluation runs."""

from __future__ import annotations

import platform
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
) -> dict[str, Any]:
    """Compare against an eligible baseline, then save an aggregate-only run."""
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
    try:
        store.add(record)
    except (OSError, TimeoutError, ValueError):
        raise EvaluationPersistenceError(
            "Offline evaluation metrics could not be saved safely."
        ) from None
    return record
