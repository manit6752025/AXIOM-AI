"""Compare bounded dataset profiles without exposing row values."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def compare_dataset_schemas(
    datasets: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare field presence and observed JSON types across profile reports."""
    if not 2 <= len(datasets) <= 3:
        raise ValueError("Schema comparison requires two or three datasets.")

    names = [dataset["path"] for dataset in datasets]
    if len(set(names)) != len(names):
        raise ValueError("Each dataset must be selected only once.")

    by_dataset: list[dict[str, dict[str, Any]]] = []
    all_fields: set[str] = set()
    for dataset in datasets:
        fields = {field["name"]: field for field in dataset["profile"]["fields"]}
        by_dataset.append(fields)
        all_fields.update(fields)

    comparisons = []
    has_differences = False
    for name in sorted(all_fields):
        present = [
            path for path, fields in zip(names, by_dataset, strict=True)
            if name in fields
        ]
        missing = [path for path, fields in zip(names, by_dataset, strict=True) if name not in fields]
        types_by_dataset = {
            path: sorted(fields[name]["types"])
            for path, fields in zip(names, by_dataset, strict=True)
            if name in fields
        }
        observed_types = {
            value_type
            for value_types in types_by_dataset.values()
            for value_type in value_types
            if value_type != "null"
        }
        type_mismatch = len(observed_types) > 1
        has_differences = has_differences or bool(missing) or type_mismatch
        comparisons.append({
            "name": name,
            "present_in": present,
            "missing_from": missing,
            "types_by_dataset": types_by_dataset,
            "type_mismatch": type_mismatch,
            "possibly_sensitive": any(
                fields[name]["possibly_sensitive"]
                for fields in by_dataset
                if name in fields
            ),
        })

    return {
        "dataset_count": len(datasets),
        "datasets": [
            {
                "path": item["path"],
                "valid_rows": item["profile"]["valid_rows"],
                "truncated": item["profile"]["truncated"],
            }
            for item in datasets
        ],
        "has_differences": has_differences,
        "fields": comparisons,
    }


def validate_dataset_profile(
    profile: dict[str, Any],
    *,
    required_fields: Sequence[str] = (),
    expected_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check required fields and observed types against a bounded profile."""
    fields = {field["name"]: field for field in profile["fields"]}
    missing_fields = sorted(set(required_fields) - fields.keys())
    type_mismatches = []
    expected_types = expected_types or {}
    for name, expected in sorted(expected_types.items()):
        field = fields.get(name)
        observed = sorted(
            value_type for value_type in (field["types"] if field else {})
            if value_type != "null"
        )
        allowed = {"integer", "number"} if expected == "number" else {expected}
        if not observed or not set(observed).issubset(allowed):
            type_mismatches.append({
                "field": name,
                "expected": expected,
                "observed": observed,
            })

    return {
        "passed": not missing_fields and not type_mismatches,
        "missing_fields": missing_fields,
        "type_mismatches": type_mismatches,
    }
