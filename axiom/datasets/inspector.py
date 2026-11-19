from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class DatasetInspection:
    def __init__(
        self,
        path: Path,
        format: str,
        samples: int,
        valid: int,
        invalid: int,
        empty: int,
        duplicates: int,
        fields: list[str],
        estimated_tokens: int,
        errors: list[tuple[int, str]],
    ):
        self.path = path
        self.format = format
        self.samples = samples
        self.valid = valid
        self.invalid = invalid
        self.empty = empty
        self.duplicates = duplicates
        self.fields = fields
        self.estimated_tokens = estimated_tokens
        self.errors = errors


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _estimate_tokens(value: Any) -> int:
    """Cheap token estimate for first-pass dataset analysis."""
    text = json.dumps(value, ensure_ascii=False)
    return max(1, len(text) // 4)


def inspect_jsonl(path: Path) -> DatasetInspection:
    total = 0
    valid = 0
    invalid = 0
    empty = 0
    token_estimate = 0
    field_counter: Counter[str] = Counter()
    fingerprints: Counter[str] = Counter()
    errors: list[tuple[int, str]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            total += 1
            line = raw_line.strip()

            if not line:
                empty += 1
                continue

            try:
                item = json.loads(
                    line,
                    parse_constant=_reject_non_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                invalid += 1
                if len(errors) < 50:
                    errors.append((line_number, str(exc)))
                continue

            if not isinstance(item, dict):
                invalid += 1
                if len(errors) < 50:
                    errors.append((line_number, "record is not a JSON object"))
                continue

            valid += 1
            token_estimate += _estimate_tokens(item)

            field_counter.update(item.keys())

            fingerprints.update([json.dumps(item, sort_keys=True, ensure_ascii=False)])

    duplicates = sum(count - 1 for count in fingerprints.values() if count > 1)

    return DatasetInspection(
        path=path,
        format="jsonl",
        samples=total,
        valid=valid,
        invalid=invalid,
        empty=empty,
        duplicates=duplicates,
        fields=sorted(field_counter),
        estimated_tokens=token_estimate,
        errors=errors,
    )


def inspect_dataset(path: str) -> DatasetInspection:
    raw_path = str(path)
    if not raw_path.strip() or "\x00" in raw_path:
        raise ValueError("Dataset path is empty or contains an invalid character.")
    requested_path = Path(raw_path).expanduser()
    try:
        # Work with a canonical path; daemon callers additionally restrict it to the
        # workspace in resolve_in_workspace before calling this inspector.
        dataset_path = requested_path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {requested_path}") from None
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Invalid dataset path: {requested_path}") from exc

    if not dataset_path.is_file():
        raise ValueError(f"Dataset path is not a file: {dataset_path}")

    suffix = dataset_path.suffix.lower()

    if suffix == ".jsonl":
        return inspect_jsonl(dataset_path)

    raise ValueError(
        f"Unsupported dataset format: {suffix or 'unknown'}. "
        "Currently supported: .jsonl"
    )
