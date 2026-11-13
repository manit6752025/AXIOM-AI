"""Bounded, privacy-preserving metrics for reference/prediction JSONL files."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, BinaryIO

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_LINE_BYTES = 1024 * 1024
MAX_EVALUATION_BYTES = 256 * 1024 * 1024
METRIC_SCHEMA_VERSION = 1


class BaselineCorpusMismatchError(ValueError):
    """A report cannot be gated against a baseline from a different corpus."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def evaluate_jsonl_stream(
    stream: BinaryIO,
    *,
    path: str,
    file_name: str,
    file_size_bytes: int,
    prediction_field: str,
    reference_field: str,
    split_field: str | None = None,
    split_value: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
) -> dict[str, Any]:
    _validate_options(
        file_size_bytes=file_size_bytes,
        prediction_field=prediction_field,
        reference_field=reference_field,
        split_field=split_field,
        split_value=split_value,
        max_bytes=max_bytes,
        max_rows=max_rows,
        max_line_bytes=max_line_bytes,
    )

    stream.seek(0)
    digest = hashlib.sha256()
    comparison_corpus_digest = hashlib.sha256()
    bytes_scanned = 0
    lines_scanned = 0
    invalid_json_rows = 0
    non_object_rows = 0
    overlong_rows = 0
    missing_value_rows = 0
    non_string_value_rows = 0
    filtered_split_rows = 0
    valid_samples = 0
    comparison_rows = 0
    comparison_corpus_hashable = True
    exact_matches = 0
    normalized_exact_matches = 0
    prediction_tokens = 0
    reference_tokens = 0
    overlapping_tokens = 0
    truncation_reasons: set[str] = set()

    def digest_comparison_row(row_index: int, status: str, sample: Any = None) -> None:
        nonlocal comparison_rows, comparison_corpus_hashable
        if not comparison_corpus_hashable:
            return
        try:
            payload = json.dumps(
                [row_index, status, sample],
                ensure_ascii=True,
                allow_nan=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except RecursionError:
            comparison_corpus_hashable = False
            return
        comparison_corpus_digest.update(len(payload).to_bytes(8, "big"))
        comparison_corpus_digest.update(payload)
        comparison_rows += 1

    while lines_scanned < max_rows and bytes_scanned < max_bytes:
        read_limit = min(max_line_bytes + 1, max_bytes - bytes_scanned)
        raw_line = stream.readline(read_limit)
        if not raw_line:
            break

        bytes_scanned += len(raw_line)
        digest.update(raw_line)
        lines_scanned += 1
        overlong = len(raw_line) > max_line_bytes
        if overlong:
            digest_comparison_row(lines_scanned, "overlong")
            overlong_rows += 1
            truncation_reasons.add("max_line_bytes")
            if not raw_line.endswith(b"\n"):
                while bytes_scanned < max_bytes:
                    discarded = stream.readline(min(64 * 1024, max_bytes - bytes_scanned))
                    if not discarded:
                        break
                    bytes_scanned += len(discarded)
                    digest.update(discarded)
                    if discarded.endswith(b"\n"):
                        break
            continue

        if bytes_scanned == max_bytes and file_size_bytes > bytes_scanned and not raw_line.endswith(b"\n"):
            truncation_reasons.add("max_bytes")
            break

        if not raw_line.strip():
            digest_comparison_row(lines_scanned, "blank", hashlib.sha256(raw_line).hexdigest())
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            digest_comparison_row(lines_scanned, "invalid_json", hashlib.sha256(raw_line).hexdigest())
            invalid_json_rows += 1
            continue
        if not isinstance(row, dict):
            digest_comparison_row(lines_scanned, "non_object", hashlib.sha256(raw_line).hexdigest())
            non_object_rows += 1
            continue

        # Predictions differ between candidate runs. Hash every other field so
        # prompt/context changes cannot masquerade as an identical corpus.
        comparison_sample = {key: value for key, value in row.items() if key != prediction_field}

        if split_field is not None and row.get(split_field) != split_value:
            digest_comparison_row(lines_scanned, "filtered", comparison_sample)
            filtered_split_rows += 1
            continue

        prediction = row.get(prediction_field)
        reference = row.get(reference_field)
        if isinstance(reference, str):
            status = "selected"
        elif reference is None:
            status = "missing_reference"
        else:
            status = "non_string_reference"
        digest_comparison_row(lines_scanned, status, comparison_sample)
        if prediction is None or reference is None:
            missing_value_rows += 1
            continue
        if not isinstance(prediction, str) or not isinstance(reference, str):
            non_string_value_rows += 1
            continue

        normalized_prediction = " ".join(prediction.split()).casefold()
        normalized_reference = " ".join(reference.split()).casefold()
        prediction_word_counts = Counter(normalized_prediction.split())
        reference_word_counts = Counter(normalized_reference.split())
        prediction_count = sum(prediction_word_counts.values())
        reference_count = sum(reference_word_counts.values())
        overlap = sum((prediction_word_counts & reference_word_counts).values())

        valid_samples += 1
        exact_matches += prediction == reference
        normalized_exact_matches += normalized_prediction == normalized_reference
        prediction_tokens += prediction_count
        reference_tokens += reference_count
        overlapping_tokens += overlap

    if bytes_scanned >= max_bytes and file_size_bytes > bytes_scanned:
        truncation_reasons.add("max_bytes")
    if lines_scanned >= max_rows and file_size_bytes > bytes_scanned:
        truncation_reasons.add("max_rows")

    if valid_samples == 0:
        raise ValueError("No rows with string prediction and reference values were found within the scan limits.")

    precision = overlapping_tokens / prediction_tokens if prediction_tokens else float(reference_tokens == 0)
    recall = overlapping_tokens / reference_tokens if reference_tokens else float(prediction_tokens == 0)
    token_f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    complete_hash = bytes_scanned == file_size_bytes
    complete_comparison_corpus = (
        complete_hash and not truncation_reasons and comparison_corpus_hashable
    )
    exact_interval = _wilson_interval(exact_matches, valid_samples)
    normalized_interval = _wilson_interval(normalized_exact_matches, valid_samples)
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "evaluation_type": "offline_text_metrics",
        "path": path,
        "file_name": file_name,
        "dataset_manifest": {
            "sha256": digest.hexdigest() if complete_hash else None,
            "partial_sha256": digest.hexdigest() if not complete_hash else None,
            "bytes_hashed": bytes_scanned,
            "file_size_bytes": file_size_bytes,
            "full_content_hashed": complete_hash,
            "comparison_corpus_sha256": (
                comparison_corpus_digest.hexdigest() if complete_comparison_corpus else None
            ),
            "comparison_corpus_fully_hashed": complete_comparison_corpus,
            "comparison_corpus_rows_hashed": comparison_rows,
        },
        "fields": {
            "prediction": prediction_field,
            "reference": reference_field,
            "split": split_field,
            "selected_split": split_value,
        },
        "counts": {
            "lines_scanned": lines_scanned,
            "valid_samples": valid_samples,
            "invalid_json_rows": invalid_json_rows,
            "non_object_rows": non_object_rows,
            "overlong_rows": overlong_rows,
            "missing_value_rows": missing_value_rows,
            "non_string_value_rows": non_string_value_rows,
            "filtered_split_rows": filtered_split_rows,
        },
        "metrics": {
            "exact_match": exact_matches / valid_samples,
            "normalized_exact_match": normalized_exact_matches / valid_samples,
            "token_precision_micro": precision,
            "token_recall_micro": recall,
            "token_f1_micro": token_f1,
        },
        "confidence_intervals": {
            "confidence_level": 0.95,
            "method": "Wilson score interval",
            "exact_match": exact_interval,
            "normalized_exact_match": normalized_interval,
        },
        "truncated": bool(truncation_reasons),
        "truncation_reasons": sorted(truncation_reasons),
        "limits": {
            "max_bytes": max_bytes,
            "max_rows": max_rows,
            "max_line_bytes": max_line_bytes,
        },
        "privacy": {
            "raw_values_returned": False,
            "raw_values_persisted": False,
            "content_fingerprints_persisted_locally": True,
        },
    }


def create_output_manifest(report: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "format": "application/vnd.axiom.evaluation-report+json;version=1",
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "bytes": len(serialized),
    }


def _validate_options(
    *,
    file_size_bytes: int,
    prediction_field: str,
    reference_field: str,
    split_field: str | None,
    split_value: str | None,
    max_bytes: int,
    max_rows: int,
    max_line_bytes: int,
) -> None:
    if type(file_size_bytes) is not int or file_size_bytes < 0:
        raise ValueError("Dataset file size must be a non-negative integer.")
    if any(
        not isinstance(field, str) or not field.strip() or len(field) > 160
        for field in (prediction_field, reference_field)
    ):
        raise ValueError("Prediction and reference field names must contain 1 to 160 characters.")
    if prediction_field == reference_field:
        raise ValueError("Prediction and reference fields must be different.")
    if (split_field is None) != (split_value is None):
        raise ValueError("Split field and selected split value must be supplied together.")
    if split_field is not None and (not split_field.strip() or len(split_field) > 160 or len(split_value or "") > 160):
        raise ValueError("Split field and selected split value must contain at most 160 characters.")
    if any(type(limit) is not int or limit < 1 for limit in (max_bytes, max_rows, max_line_bytes)):
        raise ValueError("Evaluation scan limits must be positive integers.")
    if max_bytes > MAX_EVALUATION_BYTES:
        raise ValueError(f"Evaluation byte limit cannot exceed {MAX_EVALUATION_BYTES}.")


def _wilson_interval(successes: int, sample_count: int) -> dict[str, float]:
    z = 1.959963984540054
    proportion = successes / sample_count
    denominator = 1 + z * z / sample_count
    center = (proportion + z * z / (2 * sample_count)) / denominator
    margin = z * (
        (proportion * (1 - proportion) / sample_count + z * z / (4 * sample_count * sample_count)) ** 0.5
    ) / denominator
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def compare_to_baseline(
    report: dict[str, Any],
    baseline_record: dict[str, Any],
    *,
    baseline_run_id: str,
    metric: str,
    regression_tolerance: float,
) -> dict[str, Any]:
    if not isinstance(report, dict) or not isinstance(baseline_record, dict):
        raise ValueError("Baseline report is malformed.")
    if metric not in {"exact_match", "normalized_exact_match", "token_f1_micro"}:
        raise ValueError("Unsupported regression metric.")
    try:
        tolerance = float(regression_tolerance)
    except (OverflowError, TypeError, ValueError):
        tolerance = float("nan")
    if (
        isinstance(regression_tolerance, bool)
        or not isinstance(regression_tolerance, (int, float))
        or not math.isfinite(tolerance)
        or not 0 <= tolerance <= 1
    ):
        raise ValueError("Regression tolerance must be between 0 and 1.")

    baseline_report = baseline_record.get("report", {})
    if not isinstance(baseline_report, dict):
        raise ValueError("Baseline report is malformed.")
    baseline_manifest = baseline_report.get("dataset_manifest", {})
    current_manifest = report.get("dataset_manifest", {})
    baseline_fields = baseline_report.get("fields", {})
    current_fields = report.get("fields", {})
    baseline_counts = baseline_report.get("counts", {})
    current_counts = report.get("counts", {})
    baseline_metrics = baseline_report.get("metrics", {})
    current_metrics = report.get("metrics", {})
    if any(
        not isinstance(value, dict)
        for value in (
            baseline_manifest,
            current_manifest,
            baseline_fields,
            current_fields,
            baseline_counts,
            current_counts,
            baseline_metrics,
            current_metrics,
        )
    ):
        raise ValueError("Baseline report is malformed.")
    if (
        baseline_fields.get("reference") != current_fields.get("reference")
        or baseline_fields.get("split") != current_fields.get("split")
        or baseline_fields.get("selected_split") != current_fields.get("selected_split")
    ):
        raise BaselineCorpusMismatchError("evaluation_configuration_mismatch")
    baseline_comparison_digest = baseline_manifest.get("comparison_corpus_sha256")
    current_comparison_digest = current_manifest.get("comparison_corpus_sha256")
    same_comparison_corpus = (
        baseline_manifest.get("comparison_corpus_fully_hashed") is True
        and current_manifest.get("comparison_corpus_fully_hashed") is True
        and isinstance(baseline_comparison_digest, str)
        and len(baseline_comparison_digest) == 64
        and all(character in "0123456789abcdef" for character in baseline_comparison_digest)
        and baseline_comparison_digest == current_comparison_digest
    )
    baseline_input_digest = baseline_manifest.get("sha256")
    current_input_digest = current_manifest.get("sha256")
    same_full_input_snapshot = (
        baseline_manifest.get("full_content_hashed") is True
        and current_manifest.get("full_content_hashed") is True
        and isinstance(baseline_input_digest, str)
        and len(baseline_input_digest) == 64
        and all(character in "0123456789abcdef" for character in baseline_input_digest)
        and baseline_input_digest == current_input_digest
    )
    if (
        not (same_comparison_corpus or same_full_input_snapshot)
        or baseline_report.get("truncated")
        or report.get("truncated")
        or baseline_counts.get("invalid_json_rows") != current_counts.get("invalid_json_rows")
        or baseline_counts.get("non_object_rows") != current_counts.get("non_object_rows")
        or baseline_counts.get("filtered_split_rows") != current_counts.get("filtered_split_rows")
        or baseline_counts.get("missing_value_rows")
        or current_counts.get("missing_value_rows")
        or baseline_counts.get("non_string_value_rows")
        or current_counts.get("non_string_value_rows")
        or baseline_counts.get("valid_samples") != current_counts.get("valid_samples")
    ):
        raise BaselineCorpusMismatchError("sample_corpus_mismatch")

    current_score = current_metrics.get(metric)
    baseline_score = baseline_metrics.get(metric)
    if any(
        isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        for score in (current_score, baseline_score)
    ):
        raise ValueError("Baseline does not contain the selected metric.")
    delta = current_score - baseline_score
    return {
        "baseline_run_id": baseline_run_id,
        "metric": metric,
        "baseline_score": baseline_score,
        "current_score": current_score,
        "delta": delta,
        "regression_tolerance": tolerance,
        "status": "regression" if delta < -tolerance else "passed",
    }
