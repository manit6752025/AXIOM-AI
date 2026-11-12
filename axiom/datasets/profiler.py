"""Bounded JSONL profiling. Only structural metadata is returned."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Sequence
from axiom.datasets.pii import MAX_PII_TEXT_BYTES, MAX_PII_VALUES_PER_ROW, scan_row_for_pii

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_LINE_BYTES = 1024 * 1024
MAX_TRACKED_FIELDS = 500

SENSITIVE = (
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "access_key", "authorization", "email", "phone", "mobile",
    "address", "ssn", "social_security", "credit_card", "card_number",
    "date_of_birth", "birth_date", "cookie",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _normalise_field_name(name: str) -> str:
    return name.casefold().replace("-", "_").replace(" ", "_")


def _sensitive(
    name: str,
    rules: Sequence[str] = (),
    ignored_fields: Sequence[str] = (),
) -> bool:
    normalised = _normalise_field_name(name)
    if normalised in {_normalise_field_name(item) for item in ignored_fields}:
        return False
    return any(pattern in normalised for pattern in (*SENSITIVE, *(rule.casefold() for rule in rules)))


def _validate_limits(max_bytes: int, max_rows: int, max_line_bytes: int) -> None:
    if any(
        type(value) is not int or value < 1
        for value in (max_bytes, max_rows, max_line_bytes)
    ):
        raise ValueError("Profiling limits must be positive integers.")


def _validate_profile_options(
    duplicate_key_fields: Sequence[str],
    duplicate_rate_threshold: float | None,
    text_fields: Sequence[str],
    sensitive_field_rules: Sequence[str],
    ignored_sensitive_fields: Sequence[str],
) -> None:
    for label, fields in (
        ("Duplicate-key", duplicate_key_fields),
        ("Text", text_fields),
        ("Sensitive-field rule", sensitive_field_rules),
        ("Sensitive-field exclusion", ignored_sensitive_fields),
    ):
        if len(fields) > 100 or any(not isinstance(name, str) or not name.strip() or len(name) > 160 for name in fields):
            raise ValueError(f"{label} fields must contain at most 100 non-empty names of 160 characters or fewer.")
        if len(set(name.strip() for name in fields)) != len(fields):
            raise ValueError(f"{label} field names must be unique.")
    if duplicate_rate_threshold is not None and (
        isinstance(duplicate_rate_threshold, bool)
        or not isinstance(duplicate_rate_threshold, (int, float))
        or not 0 <= duplicate_rate_threshold <= 1
    ):
        raise ValueError("Duplicate-rate threshold must be between 0 and 1.")


def profile_jsonl(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    duplicate_key_fields: Sequence[str] = (),
    duplicate_rate_threshold: float | None = None,
    text_fields: Sequence[str] = (),
    sensitive_field_rules: Sequence[str] = (),
    ignored_sensitive_fields: Sequence[str] = (),
    scan_pii: bool = False,
) -> dict[str, Any]:
    """Profile a local file path. Intended for trusted local CLI callers."""
    _validate_limits(max_bytes, max_rows, max_line_bytes)
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Dataset path must be a regular file.")

    with source.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        return profile_jsonl_stream(
            stream,
            file_name=source.name,
            file_size_bytes=size,
            max_bytes=max_bytes,
            max_rows=max_rows,
            max_line_bytes=max_line_bytes,
            duplicate_key_fields=duplicate_key_fields,
            duplicate_rate_threshold=duplicate_rate_threshold,
            text_fields=text_fields,
            sensitive_field_rules=sensitive_field_rules,
            ignored_sensitive_fields=ignored_sensitive_fields,
            scan_pii=scan_pii,
        )


def profile_jsonl_stream(
    stream: BinaryIO,
    *,
    file_name: str,
    file_size_bytes: int,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    duplicate_key_fields: Sequence[str] = (),
    duplicate_rate_threshold: float | None = None,
    text_fields: Sequence[str] = (),
    sensitive_field_rules: Sequence[str] = (),
    ignored_sensitive_fields: Sequence[str] = (),
    scan_pii: bool = False,
) -> dict[str, Any]:
    """Profile an already-open file without resolving or reopening a client path."""
    duplicate_key_fields = tuple(name.strip() for name in duplicate_key_fields)
    text_fields = tuple(name.strip() for name in text_fields)
    sensitive_field_rules = tuple(rule.strip() for rule in sensitive_field_rules)
    ignored_sensitive_fields = tuple(name.strip() for name in ignored_sensitive_fields)
    _validate_limits(max_bytes, max_rows, max_line_bytes)
    _validate_profile_options(
        duplicate_key_fields,
        duplicate_rate_threshold,
        text_fields,
        sensitive_field_rules,
        ignored_sensitive_fields,
    )
    if type(scan_pii) is not bool:
        raise ValueError("PII scanning opt-in must be a boolean.")
    if type(file_size_bytes) is not int or file_size_bytes < 0:
        raise ValueError("File size must be a non-negative integer.")

    size = file_size_bytes
    read_bytes = lines = valid = invalid = empty = 0
    duplicates = overlong = omitted = 0
    encoding_errors = malformed_json = non_object_json = 0
    key_duplicates = key_rows = 0
    truncated = False
    scan_budget_exhausted = False
    truncation_reasons: set[str] = set()
    seen: set[str] = set()
    seen_keys: set[str] = set()
    counts: Counter[str] = Counter()
    nulls: Counter[str] = Counter()
    types: dict[str, Counter[str]] = {}
    newline_counts: Counter[str] = Counter()
    token_lengths = {name: Counter() for name in text_fields}
    pii_matches: Counter[str] = Counter()
    pii_fields: dict[str, set[str]] = {}
    pii_values_scanned = pii_long_values_skipped = pii_rows_truncated = 0

    def count_newlines(raw_line: bytes) -> None:
        crlf = raw_line.count(b"\r\n")
        newline_counts["CRLF"] += crlf
        newline_counts["LF"] += raw_line.count(b"\n") - crlf
        newline_counts["CR"] += raw_line.count(b"\r") - crlf

    while lines < max_rows:
        remaining = max_bytes - read_bytes
        if remaining <= 0:
            truncated = True
            truncation_reasons.add("max_bytes")
            break

        raw = stream.readline(min(max_line_bytes + 1, remaining + 1))
        if not raw:
            break
        if len(raw) > remaining:
            read_bytes += remaining
            truncated = True
            truncation_reasons.add("max_bytes")
            break

        read_bytes += len(raw)
        lines += 1
        count_newlines(raw)

        if len(raw) > max_line_bytes:
            invalid += 1
            overlong += 1
            truncated = True
            truncation_reasons.add("max_line_bytes")
            while raw and not raw.endswith(b"\n"):
                remaining = max_bytes - read_bytes
                if remaining <= 0:
                    truncated = True
                    scan_budget_exhausted = True
                    truncation_reasons.add("max_bytes")
                    break
                raw = stream.readline(min(65536, remaining + 1))
                if len(raw) > remaining:
                    read_bytes += remaining
                    truncated = True
                    scan_budget_exhausted = True
                    truncation_reasons.add("max_bytes")
                    break
                read_bytes += len(raw)
                count_newlines(raw)
            if scan_budget_exhausted:
                break
            continue

        raw = raw.strip()
        if not raw:
            empty += 1
            continue

        try:
            row = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except UnicodeDecodeError:
            invalid += 1
            encoding_errors += 1
            continue
        except (json.JSONDecodeError, ValueError):
            invalid += 1
            malformed_json += 1
            continue
        if not isinstance(row, dict):
            invalid += 1
            non_object_json += 1
            continue

        valid += 1
        if scan_pii:
            pii_summary = scan_row_for_pii(row)
            pii_values_scanned += pii_summary["values_scanned"]
            pii_long_values_skipped += pii_summary["long_values_skipped"]
            pii_rows_truncated += int(pii_summary["row_scan_truncated"])
            pii_matches.update(pii_summary["matches_by_type"])
            for kind, names in pii_summary["fields_by_type"].items():
                pii_fields.setdefault(kind, set()).update(names)
        canonical = json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            duplicates += 1
        else:
            seen.add(fingerprint)

        if duplicate_key_fields and all(name in row for name in duplicate_key_fields):
            key_rows += 1
            key_identity = json.dumps(
                [row[name] for name in duplicate_key_fields],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            key_fingerprint = hashlib.sha256(key_identity.encode("utf-8")).hexdigest()
            if key_fingerprint in seen_keys:
                key_duplicates += 1
            else:
                seen_keys.add(key_fingerprint)

        for name, histogram in token_lengths.items():
            value = row.get(name)
            if isinstance(value, str):
                token_count = len(value.split())
                bucket = (
                    "0" if token_count == 0 else "1" if token_count == 1 else
                    "2-4" if token_count <= 4 else "5-9" if token_count <= 9 else
                    "10-19" if token_count <= 19 else "20-49" if token_count <= 49 else "50+"
                )
                histogram[bucket] += 1

        for key, value in row.items():
            name = str(key)
            if name not in counts and len(counts) >= MAX_TRACKED_FIELDS:
                omitted += 1
                continue
            counts[name] += 1
            if value is None:
                nulls[name] += 1
            types.setdefault(name, Counter())[_type_name(value)] += 1

    if lines >= max_rows and read_bytes < size:
        truncated = True
        truncation_reasons.add("max_rows")
    if read_bytes >= max_bytes and read_bytes < size:
        truncated = True
        truncation_reasons.add("max_bytes")

    fields = []
    truncated_field_names = 0
    for name in sorted(counts):
        occurrences = counts[name]
        shown = name if len(name) <= 160 else name[:159] + "…"
        truncated_field_names += int(len(name) > 160)
        fields.append({
            "name": shown,
            "occurrences": occurrences,
            "null_count": nulls[name],
            "null_rate": round(nulls[name] / occurrences, 4) if occurrences else 0.0,
            "types": dict(sorted(types[name].items())),
            "possibly_sensitive": _sensitive(name, sensitive_field_rules, ignored_sensitive_fields),
        })

    sensitive = [field["name"] for field in fields if field["possibly_sensitive"]]
    warnings = []
    if invalid:
        warnings.append("Invalid or unsupported rows were detected.")
    if duplicates:
        warnings.append("Duplicate valid rows were detected.")
    if sensitive:
        warnings.append("Potentially sensitive field names were detected; review locally.")
    if truncated:
        warnings.append("Profile is partial because a scanning limit was reached.")
        warnings.append("Limit causes: " + ", ".join(sorted(truncation_reasons)) + ".")
    if omitted:
        warnings.append("Field tracking limit reached; some fields were omitted.")

    duplicate_key_rate = key_duplicates / key_rows if key_rows else 0.0
    threshold_exceeded = (
        duplicate_rate_threshold is not None and duplicate_key_rate > duplicate_rate_threshold
    )
    if encoding_errors:
        warnings.append("UTF-8 decoding errors were detected.")
    if malformed_json:
        warnings.append("Malformed JSON records were detected.")
    if threshold_exceeded:
        warnings.append("Configured duplicate-key rate threshold was exceeded.")
    if scan_pii and sum(pii_matches.values()):
        warnings.append("Likely PII patterns were detected locally; review the dataset before use.")

    ending_counts = {ending: newline_counts[ending] for ending in ("LF", "CRLF", "CR")}
    observed_endings = [ending for ending, count in ending_counts.items() if count]
    newline_style = "mixed" if len(observed_endings) > 1 else observed_endings[0] if observed_endings else "none"

    return {
        "format": "jsonl",
        "file_name": file_name,
        "file_size_bytes": size,
        "bytes_scanned": read_bytes,
        "lines_scanned": lines,
        "valid_rows": valid,
        "invalid_rows": invalid,
        "empty_rows": empty,
        "duplicate_rows": duplicates,
        "duplicate_key_summary": {
            "fields": list(duplicate_key_fields),
            "eligible_rows": key_rows,
            "duplicate_rows": key_duplicates,
            "rate": round(duplicate_key_rate, 6),
            "threshold": duplicate_rate_threshold,
            "threshold_exceeded": threshold_exceeded,
        },
        "overlong_rows": overlong,
        "diagnostics": {
            "encoding": "utf-8",
            "encoding_errors": encoding_errors,
            "malformed_json_rows": malformed_json,
            "non_object_json_rows": non_object_json,
            "newline_style": newline_style,
            "newline_counts": ending_counts,
            "delimiter": "not_applicable_jsonl",
        },
        "text_token_lengths": {
            name: {"observations": sum(histogram.values()), "histogram": dict(sorted(histogram.items()))}
            for name, histogram in sorted(token_lengths.items())
        },
        "unique_fields_tracked": len(fields),
        "truncated_field_names": truncated_field_names,
        "omitted_field_occurrences": omitted,
        "possibly_sensitive_fields": sensitive,
        "privacy_review_recommended": bool(sensitive),
        "sensitive_field_rules": list(sensitive_field_rules),
        "ignored_sensitive_fields": list(ignored_sensitive_fields),
        "pii_scan": {
            "status": "completed" if scan_pii else "not_run",
            "opted_in": scan_pii,
            "values_scanned": pii_values_scanned,
            "matches_by_type": {
                kind: pii_matches[kind] for kind in ("email", "phone_like", "payment_card")
            },
            "fields_by_type": {
                kind: sorted(pii_fields.get(kind, set()))
                for kind in ("email", "phone_like", "payment_card")
            },
            "long_values_skipped": pii_long_values_skipped,
            "rows_truncated_by_scan_limit": pii_rows_truncated,
            "max_text_bytes": MAX_PII_TEXT_BYTES,
            "max_values_per_row": MAX_PII_VALUES_PER_ROW,
        },
        "truncated": truncated,
        "truncation_reasons": sorted(truncation_reasons),
        "fields": fields,
        "warnings": warnings,
        "limits": {
            "max_bytes": max_bytes,
            "max_rows": max_rows,
            "max_line_bytes": max_line_bytes,
        },
    }
