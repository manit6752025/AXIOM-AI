"""Bounded, privacy-preserving metrics for reference/prediction JSONL files."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, BinaryIO, Callable

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_LINE_BYTES = 1024 * 1024
MAX_EVALUATION_BYTES = 256 * 1024 * 1024
MAX_EVALUATION_ROWS = 500_000
MAX_EVALUATION_LINE_BYTES = 8 * 1024 * 1024
METRIC_SCHEMA_VERSION = 1


def text_pair_counts(prediction: str, reference: str) -> tuple[bool, bool, int, int, int]:
    """Return the row counts shared by offline evaluation and paired job comparison."""
    normalized_prediction = " ".join(prediction.split()).casefold()
    normalized_reference = " ".join(reference.split()).casefold()
    prediction_words = Counter(normalized_prediction.split())
    reference_words = Counter(normalized_reference.split())
    prediction_tokens = sum(prediction_words.values())
    reference_tokens = sum(reference_words.values())
    overlapping_tokens = sum((prediction_words & reference_words).values())
    return (
        prediction == reference,
        normalized_prediction == normalized_reference,
        prediction_tokens,
        reference_tokens,
        overlapping_tokens,
    )


def aggregate_text_metrics(
    *,
    valid_samples: int,
    exact_matches: int,
    normalized_exact_matches: int,
    prediction_tokens: int,
    reference_tokens: int,
    overlapping_tokens: int,
) -> dict[str, float]:
    """Aggregate the same lexical metrics returned by the offline evaluator."""
    if valid_samples <= 0:
        raise ValueError("At least one valid text pair is required for metric aggregation.")
    precision = (
        overlapping_tokens / prediction_tokens
        if prediction_tokens
        else float(reference_tokens == 0)
    )
    recall = (
        overlapping_tokens / reference_tokens
        if reference_tokens
        else float(prediction_tokens == 0)
    )
    token_f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "exact_match": exact_matches / valid_samples,
        "normalized_exact_match": normalized_exact_matches / valid_samples,
        "token_precision_micro": precision,
        "token_recall_micro": recall,
        "token_f1_micro": token_f1,
    }


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
    on_valid_pair: Callable[[int, str, str], None] | None = None,
) -> dict[str, Any]:
    """Score a bounded stream and optionally report valid selected pairs in-process."""
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

        row_counts = text_pair_counts(prediction, reference)
        if on_valid_pair is not None:
            on_valid_pair(lines_scanned, prediction, reference)
        valid_samples += 1
        exact_matches += row_counts[0]
        normalized_exact_matches += row_counts[1]
        prediction_tokens += row_counts[2]
        reference_tokens += row_counts[3]
        overlapping_tokens += row_counts[4]

    if bytes_scanned >= max_bytes and file_size_bytes > bytes_scanned:
        truncation_reasons.add("max_bytes")
    if lines_scanned >= max_rows and file_size_bytes > bytes_scanned:
        truncation_reasons.add("max_rows")

    if valid_samples == 0:
        raise ValueError("No rows with string prediction and reference values were found within the scan limits.")

    metrics = aggregate_text_metrics(
        valid_samples=valid_samples,
        exact_matches=exact_matches,
        normalized_exact_matches=normalized_exact_matches,
        prediction_tokens=prediction_tokens,
        reference_tokens=reference_tokens,
        overlapping_tokens=overlapping_tokens,
    )
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
        "metrics": metrics,
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


def compare_evaluation_examples_streams(
    baseline_stream: BinaryIO,
    candidate_stream: BinaryIO,
    *,
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    category: str,
    limit: int = 3,
    include_text: bool = False,
    max_excerpt_chars: int = 2_000,
    sample_strategy: str = "first",
    sample_seed: str | None = None,
) -> dict[str, Any]:
    """Verify saved evaluation snapshots and inspect exact-match or token-F1 examples."""
    categories = {"fixed", "regressed", "token_f1_improved", "token_f1_worsened"}
    if not isinstance(category, str) or category not in categories:
        raise ValueError("Choose a fixed/regressed exact-match or token-F1 transition.")
    if not isinstance(sample_strategy, str) or sample_strategy not in {"first", "uniform_hash"}:
        raise ValueError("Choose the first or uniform_hash example-sampling strategy.")
    if category.startswith("token_f1_") and sample_strategy != "first":
        raise ValueError("uniform_hash sampling is available only for exact-match transitions.")
    if sample_strategy == "uniform_hash" and (
        not isinstance(sample_seed, str) or not sample_seed or len(sample_seed) > 256
    ):
        raise ValueError("A bounded comparison seed is required for uniform_hash sampling.")
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError("Evaluation example limit must be between 1 and 3.")
    if type(include_text) is not bool:
        raise ValueError("include_text must be a boolean.")
    if type(max_excerpt_chars) is not int or not 1 <= max_excerpt_chars <= 2_000:
        raise ValueError("Evaluation example text limit must be between 1 and 2000 characters.")
    try:
        baseline_stream.seek(0)
        candidate_stream.seek(0)
    except (OSError, ValueError) as exc:
        raise ValueError("Evaluation source streams could not be positioned safely.") from exc

    def snapshot(report: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(report, dict)
            or report.get("evaluation_type") != "offline_text_metrics"
            or report.get("truncated") is not False
        ):
            raise ValueError("Only complete evaluation reports can be compared.")
        manifest = report.get("dataset_manifest")
        fields = report.get("fields")
        counts = report.get("counts")
        limits = report.get("limits")
        if any(not isinstance(value, dict) for value in (manifest, fields, counts, limits)):
            raise ValueError("Saved evaluation report is malformed.")
        content_digest = manifest.get("sha256")
        corpus_digest = manifest.get("comparison_corpus_sha256")
        file_size = manifest.get("file_size_bytes")
        corpus_rows = manifest.get("comparison_corpus_rows_hashed")
        if (
            manifest.get("full_content_hashed") is not True
            or manifest.get("comparison_corpus_fully_hashed") is not True
            or not isinstance(content_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", content_digest)
            or not isinstance(corpus_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", corpus_digest)
            or type(file_size) is not int
            or not 1 <= file_size <= MAX_EVALUATION_BYTES
            or type(manifest.get("bytes_hashed")) is not int
            or manifest.get("bytes_hashed") != file_size
            or type(corpus_rows) is not int
            or corpus_rows < 1
        ):
            raise ValueError("Saved evaluation report has no bounded complete snapshot.")
        prediction_field = fields.get("prediction")
        reference_field = fields.get("reference")
        split_field = fields.get("split")
        split_value = fields.get("selected_split")
        if (
            not isinstance(prediction_field, str)
            or not prediction_field.strip()
            or len(prediction_field) > 160
            or not isinstance(reference_field, str)
            or not reference_field.strip()
            or len(reference_field) > 160
            or prediction_field == reference_field
            or (split_field is None) != (split_value is None)
            or (split_field is not None and (
                not isinstance(split_field, str)
                or not split_field.strip()
                or len(split_field) > 160
                or not isinstance(split_value, str)
                or len(split_value) > 160
            ))
        ):
            raise ValueError("Saved evaluation fields are invalid.")
        max_bytes = limits.get("max_bytes")
        max_rows = limits.get("max_rows")
        max_line_bytes = limits.get("max_line_bytes")
        if any(type(value) is not int or value < 1 for value in (max_bytes, max_rows, max_line_bytes)):
            raise ValueError("Saved evaluation limits are malformed.")
        if (
            max_bytes > MAX_EVALUATION_BYTES
            or max_rows > MAX_EVALUATION_ROWS
            or max_line_bytes > MAX_EVALUATION_LINE_BYTES
            or file_size > max_bytes
        ):
            raise ValueError("Saved evaluation exceeds supported comparison bounds.")
        required_counts = (
            "lines_scanned",
            "valid_samples",
            "invalid_json_rows",
            "non_object_rows",
            "overlong_rows",
            "missing_value_rows",
            "non_string_value_rows",
            "filtered_split_rows",
        )
        if any(type(counts.get(key)) is not int or counts[key] < 0 for key in required_counts):
            raise ValueError("Saved evaluation counts are malformed.")
        if corpus_rows != counts["lines_scanned"]:
            raise ValueError("Saved comparison fingerprint row count is invalid.")
        return {
            "content_digest": content_digest,
            "corpus_digest": corpus_digest,
            "file_size": file_size,
            "corpus_rows": corpus_rows,
            "prediction_field": prediction_field,
            "reference_field": reference_field,
            "split_field": split_field,
            "split_value": split_value,
            "counts": counts,
            "limits": {
                "max_bytes": min(max_bytes, MAX_EVALUATION_BYTES),
                "max_rows": min(max_rows, MAX_EVALUATION_ROWS),
                "max_line_bytes": min(max_line_bytes, MAX_EVALUATION_LINE_BYTES),
            },
        }

    baseline = snapshot(baseline_report)
    candidate = snapshot(candidate_report)
    if (
        baseline["reference_field"] != candidate["reference_field"]
        or baseline["split_field"] != candidate["split_field"]
        or baseline["split_value"] != candidate["split_value"]
    ):
        raise ValueError("Saved evaluation settings do not match.")
    same_full_snapshot = baseline["content_digest"] == candidate["content_digest"]
    same_comparison_corpus = baseline["corpus_digest"] == candidate["corpus_digest"]
    if not same_full_snapshot and (
        baseline["prediction_field"] != candidate["prediction_field"]
        or not same_comparison_corpus
    ):
        raise ValueError("Saved evaluation reports do not describe matching rows.")
    if (
        baseline["counts"]["valid_samples"] < 1
        or baseline["counts"]["valid_samples"] != candidate["counts"]["valid_samples"]
        or baseline["counts"]["missing_value_rows"] != 0
        or candidate["counts"]["missing_value_rows"] != 0
        or baseline["counts"]["non_string_value_rows"] != 0
        or candidate["counts"]["non_string_value_rows"] != 0
    ):
        raise ValueError("Saved evaluations do not contain the same valid row set.")

    states = (baseline, candidate)
    streams = (baseline_stream, candidate_stream)
    bytes_scanned = [0, 0]
    rows_scanned = [0, 0]
    full_digests = [hashlib.sha256(), hashlib.sha256()]
    actual_counts = [
        {
            "lines_scanned": 0,
            "valid_samples": 0,
            "invalid_json_rows": 0,
            "non_object_rows": 0,
            "overlong_rows": 0,
            "missing_value_rows": 0,
            "non_string_value_rows": 0,
            "filtered_split_rows": 0,
        }
        for _ in range(2)
    ]
    transitions = {"fixed": 0, "regressed": 0}
    token_f1_changes = {"improved": 0, "worsened": 0}
    examples: list[dict[str, Any]] = []
    ranked_examples: list[tuple[float, int, dict[str, Any]]] = []
    hash_examples: list[tuple[bytes, int, dict[str, Any]]] = []
    hash_priority_seed = b""
    if sample_strategy == "uniform_hash":
        assert sample_seed is not None
        hash_priority_seed = (
            b"axiom:evaluation-example-sample:v1\x00"
            + sample_seed.encode("utf-8")
            + b"\x00"
            + category.encode("ascii")
            + b"\x00"
        )

    def read_row(raw: bytes, state: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        if not raw.strip():
            return "blank", None
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            return "invalid_json", None
        if not isinstance(row, dict):
            return "non_object", None
        if state["split_field"] is not None and row.get(state["split_field"]) != state["split_value"]:
            return "filtered", row
        return "selected", row

    def row_token_f1(pair_counts: tuple[bool, bool, int, int, int]) -> float:
        token_total = pair_counts[2] + pair_counts[3]
        return 2 * pair_counts[4] / token_total if token_total else 1.0

    def add_text_excerpts(
        example: dict[str, Any],
        baseline_prediction: str,
        candidate_prediction: str,
        reference: str,
    ) -> None:
        if not include_text:
            return
        example.update({
            "baseline_prediction_excerpt": baseline_prediction[:max_excerpt_chars],
            "candidate_prediction_excerpt": candidate_prediction[:max_excerpt_chars],
            "reference_excerpt": reference[:max_excerpt_chars],
            "text_truncated": {
                "baseline_prediction": len(baseline_prediction) > max_excerpt_chars,
                "candidate_prediction": len(candidate_prediction) > max_excerpt_chars,
                "reference": len(reference) > max_excerpt_chars,
            },
        })

    for line_number in range(1, min(baseline["limits"]["max_rows"], candidate["limits"]["max_rows"]) + 1):
        raw_lines: list[bytes] = []
        for index, stream in enumerate(streams):
            remaining = states[index]["limits"]["max_bytes"] - bytes_scanned[index]
            read_limit = min(states[index]["limits"]["max_line_bytes"] + 1, remaining)
            raw_lines.append(stream.readline(read_limit) if remaining > 0 else b"")
        if not raw_lines[0] and not raw_lines[1]:
            break
        if not raw_lines[0] or not raw_lines[1]:
            raise ValueError("Evaluation files contain different row counts.")

        parsed: list[tuple[str, dict[str, Any] | None]] = []
        for index, raw in enumerate(raw_lines):
            if len(raw) > states[index]["limits"]["max_line_bytes"]:
                raise ValueError("Evaluation file contains an oversized JSONL line.")
            bytes_scanned[index] += len(raw)
            rows_scanned[index] += 1
            if bytes_scanned[index] > states[index]["limits"]["max_bytes"]:
                raise ValueError("Evaluation file exceeds the bounded scan size.")
            full_digests[index].update(raw)
            status, row = read_row(raw, states[index])
            parsed.append((status, row))
            counts = actual_counts[index]
            counts["lines_scanned"] += 1
            if status == "invalid_json":
                counts["invalid_json_rows"] += 1
            elif status == "non_object":
                counts["non_object_rows"] += 1
            elif status == "filtered":
                counts["filtered_split_rows"] += 1
            elif status == "selected" and row is not None:
                prediction = row.get(states[index]["prediction_field"])
                reference = row.get(states[index]["reference_field"])
                if prediction is None or reference is None:
                    counts["missing_value_rows"] += 1
                elif not isinstance(prediction, str) or not isinstance(reference, str):
                    counts["non_string_value_rows"] += 1
                else:
                    counts["valid_samples"] += 1

        baseline_status, baseline_row = parsed[0]
        candidate_status, candidate_row = parsed[1]
        if baseline_status in {"blank", "invalid_json", "non_object"}:
            continue
        if baseline_status == "filtered" or candidate_status == "filtered":
            if baseline_status != candidate_status:
                raise ValueError("Evaluation files select different rows.")
            continue
        if baseline_row is None or candidate_row is None:
            continue
        if baseline["prediction_field"] == candidate["prediction_field"]:
            baseline_corpus_row = {
                key: value for key, value in baseline_row.items()
                if key != baseline["prediction_field"]
            }
            candidate_corpus_row = {
                key: value for key, value in candidate_row.items()
                if key != candidate["prediction_field"]
            }
            try:
                baseline_corpus = json.dumps(
                    baseline_corpus_row,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=True,
                    separators=(",", ":"),
                )
                candidate_corpus = json.dumps(
                    candidate_corpus_row,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError, RecursionError):
                raise ValueError("Evaluation rows cannot be compared safely.") from None
            if baseline_corpus != candidate_corpus:
                raise ValueError("Evaluation files do not contain the same non-prediction rows.")
        baseline_prediction = baseline_row.get(baseline["prediction_field"])
        candidate_prediction = candidate_row.get(candidate["prediction_field"])
        baseline_reference = baseline_row.get(baseline["reference_field"])
        candidate_reference = candidate_row.get(candidate["reference_field"])
        if not all(
            isinstance(value, str)
            for value in (baseline_prediction, candidate_prediction, baseline_reference, candidate_reference)
        ):
            raise ValueError("A selected evaluation row no longer has string fields.")
        if baseline_reference != candidate_reference:
            raise ValueError("Evaluation files do not contain the same reference values.")
        baseline_pair = text_pair_counts(baseline_prediction, baseline_reference)
        candidate_pair = text_pair_counts(candidate_prediction, candidate_reference)
        baseline_correct = baseline_pair[0]
        candidate_correct = candidate_pair[0]
        baseline_token_f1 = row_token_f1(baseline_pair)
        candidate_token_f1 = row_token_f1(candidate_pair)
        token_f1_delta = candidate_token_f1 - baseline_token_f1
        if not baseline_correct and candidate_correct:
            transitions["fixed"] += 1
        elif baseline_correct and not candidate_correct:
            transitions["regressed"] += 1
        if token_f1_delta > 0:
            token_f1_changes["improved"] += 1
        elif token_f1_delta < 0:
            token_f1_changes["worsened"] += 1

        if category in {"fixed", "regressed"}:
            row_matches_category = (
                (category == "fixed" and not baseline_correct and candidate_correct)
                or (category == "regressed" and baseline_correct and not candidate_correct)
            )
            if not row_matches_category:
                continue
            if sample_strategy == "first":
                if len(examples) >= limit:
                    continue
                example: dict[str, Any] = {"row_number": line_number}
                add_text_excerpts(
                    example, baseline_prediction, candidate_prediction, baseline_reference
                )
                examples.append(example)
            else:
                priority = hashlib.sha256(
                    hash_priority_seed + line_number.to_bytes(8, byteorder="big")
                ).digest()
                priority_key = (priority, line_number)
                if len(hash_examples) >= limit and priority_key >= hash_examples[-1][:2]:
                    continue
                example = {"row_number": line_number}
                add_text_excerpts(
                    example, baseline_prediction, candidate_prediction, baseline_reference
                )
                hash_examples.append((priority, line_number, example))
                hash_examples.sort(key=lambda item: (item[0], item[1]))
                if len(hash_examples) > limit:
                    hash_examples.pop()
            continue
        row_matches_category = (
            (category == "token_f1_improved" and token_f1_delta > 0)
            or (category == "token_f1_worsened" and token_f1_delta < 0)
        )
        if not row_matches_category:
            continue
        priority = -token_f1_delta if category == "token_f1_improved" else token_f1_delta
        priority_key = (priority, line_number)
        if len(ranked_examples) >= limit and priority_key >= ranked_examples[-1][:2]:
            continue
        example = {
            "row_number": line_number,
            "baseline_token_f1_per_row": baseline_token_f1,
            "current_token_f1_per_row": candidate_token_f1,
            "token_f1_per_row_delta": token_f1_delta,
        }
        add_text_excerpts(example, baseline_prediction, candidate_prediction, baseline_reference)
        ranked_examples.append((priority, line_number, example))
        ranked_examples.sort(key=lambda item: (item[0], item[1]))
        if len(ranked_examples) > limit:
            ranked_examples.pop()

    for index, state in enumerate(states):
        if (
            bytes_scanned[index] != state["file_size"]
            or rows_scanned[index] != state["counts"].get("lines_scanned")
            or full_digests[index].hexdigest() != state["content_digest"]
            or actual_counts[index] != {
                key: state["counts"].get(key) for key in actual_counts[index]
            }
        ):
            raise ValueError("Evaluation source no longer matches its saved complete snapshot.")

    if category in {"fixed", "regressed"} and sample_strategy == "uniform_hash":
        sample_strategy = "uniform_hash"
        examples = [item[2] for item in sorted(hash_examples, key=lambda item: item[1])]
    elif category in {"fixed", "regressed"}:
        sample_strategy = "first_in_file_order_for_raw_exact_match_transition"
    elif category == "token_f1_improved":
        sample_strategy = "largest_per_row_token_f1_delta"
        examples = [item[2] for item in ranked_examples]
    else:
        sample_strategy = "smallest_per_row_token_f1_delta"
        examples = [item[2] for item in ranked_examples]
    return {
        "rows_compared": actual_counts[0]["valid_samples"],
        "transition_counts": transitions,
        "token_f1_change_counts": token_f1_changes,
        "examples": examples,
        "sample_limit": limit,
        "sample_strategy": sample_strategy,
        "scope": {
            "complete_source_snapshots_verified": True,
            "prediction_reference_text_returned": include_text and bool(examples),
            "prompt_text_returned": False,
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
