"""Bounded, no-write preview of the existing JSONL cleaner's decisions."""
from __future__ import annotations

import json
from typing import Any, BinaryIO

from axiom.datasets.profiler import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_ROWS,
    _validate_limits,
)


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def preview_jsonl_cleaning(
    stream: BinaryIO,
    *,
    file_name: str,
    file_size_bytes: int,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
) -> dict[str, Any]:
    """Summarize cleaning decisions while never writing an output file."""
    _validate_limits(max_bytes, max_rows, max_line_bytes)
    if type(file_size_bytes) is not int or file_size_bytes < 0:
        raise ValueError("File size must be a non-negative integer.")

    read_bytes = lines = kept = empty = invalid = non_object = schema = duplicates = overlong = 0
    rewritten = estimated_output_bytes = 0
    truncated = False
    scan_budget_exhausted = False
    newline_semantics_mismatch = False
    encoding_semantics_mismatch = False
    truncation_reasons: set[str] = set()
    seen: set[str] = set()

    while lines < max_rows:
        remaining = max_bytes - read_bytes
        if remaining <= 0:
            if stream.read(1):
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
        newline_free_body = (
            raw.removesuffix(b"\r\n")
            .removesuffix(b"\n")
            .removesuffix(b"\r")
        )
        if b"\r" in newline_free_body:
            # The cleaner opens text files with universal-newline handling,
            # while this bounded binary preview splits on LF. Flag CR-only
            # separators so agents do not treat these counts as exact.
            newline_semantics_mismatch = True
        if len(raw) > max_line_bytes:
            overlong += 1
            truncated = True
            truncation_reasons.add("max_line_bytes")
            while raw and not raw.endswith(b"\n"):
                remaining = max_bytes - read_bytes
                if remaining <= 0:
                    if stream.read(1):
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
            if scan_budget_exhausted:
                break
            continue

        line_body = raw.rstrip(b"\r\n")
        try:
            line = line_body.decode("utf-8").strip()
        except UnicodeDecodeError:
            invalid += 1
            encoding_semantics_mismatch = True
            continue
        if not line:
            empty += 1
            continue
        try:
            item: Any = json.loads(line, parse_constant=_reject_non_json_constant)
        except (json.JSONDecodeError, ValueError):
            invalid += 1
            continue
        if not isinstance(item, dict):
            non_object += 1
            continue

        fingerprint = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)

        cleaned_line = json.dumps(item, ensure_ascii=False)
        cleaned_bytes = cleaned_line.encode("utf-8")
        newline = b"\r\n" if raw.endswith(b"\r\n") else b"\n" if raw.endswith(b"\n") else b"\r" if raw.endswith(b"\r") else b""
        if line_body != cleaned_bytes or newline != b"\n":
            rewritten += 1
        estimated_output_bytes += len(cleaned_bytes) + 1
        kept += 1

    if lines >= max_rows and read_bytes < file_size_bytes:
        truncated = True
        truncation_reasons.add("max_rows")
    if read_bytes >= max_bytes and read_bytes < file_size_bytes:
        truncated = True
        truncation_reasons.add("max_bytes")

    # Overlong rows are unknown, not necessarily removed: the cleaner has no
    # line-size cap and may keep them. Keep them outside the decision counts.
    removed_rows = empty + invalid + non_object + schema + duplicates
    changes = [
        {"operation": "remove_blank_rows", "rows": empty},
        {"operation": "remove_invalid_json_rows", "rows": invalid},
        {"operation": "remove_non_object_rows", "rows": non_object},
        {"operation": "remove_schema_mismatch_rows", "rows": schema},
        {"operation": "remove_duplicate_rows", "rows": duplicates},
        {"operation": "skip_overlong_rows_in_preview", "rows": overlong},
        {"operation": "normalize_kept_jsonl_rows", "rows": rewritten},
    ]
    warnings = ["Preview is partial because a scan limit was reached."] if truncated else []
    if truncation_reasons:
        warnings.append("Limit causes: " + ", ".join(sorted(truncation_reasons)) + ".")
    if overlong:
        warnings.append("Overlong rows were skipped by the preview limits; raise the line limit to inspect them.")
    if newline_semantics_mismatch:
        warnings.append(
            "CR-only line endings are interpreted differently by the cleaner; row counts may differ."
        )
    if encoding_semantics_mismatch:
        warnings.append(
            "Invalid UTF-8 makes the cleaner fail rather than remove individual rows; "
            "preview counts may differ."
        )

    return {
        "file_name": file_name,
        "preview_only": True,
        "output_written": False,
        "file_size_bytes": file_size_bytes,
        "bytes_scanned": read_bytes,
        "lines_scanned": lines,
        "kept_rows": kept,
        "classified_rows": kept + removed_rows,
        "removed_rows": removed_rows,
        "removed_empty_rows": empty,
        "removed_invalid_rows": invalid,
        "removed_non_object_rows": non_object,
        "removed_schema_rows": schema,
        "removed_duplicate_rows": duplicates,
        "overlong_rows": overlong,
        "unclassified_rows": overlong,
        "cleaner_semantics_match": not (
            truncated
            or overlong
            or newline_semantics_mismatch
            or encoding_semantics_mismatch
        ),
        "rewritten_rows": rewritten,
        "estimated_output_bytes": estimated_output_bytes,
        "truncated": truncated,
        "truncation_reasons": sorted(truncation_reasons),
        "changes": changes,
        "warnings": warnings,
        "limits": {
            "max_bytes": max_bytes,
            "max_rows": max_rows,
            "max_line_bytes": max_line_bytes,
        },
    }
