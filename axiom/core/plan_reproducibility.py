"""Canonical rendering and comparison for immutable saved training-plan snapshots."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

PLAN_SNAPSHOT_SCHEMA_VERSION = 1
PLAN_SNAPSHOT_FORMAT = "application/vnd.axiom.training-plan+json;version=1"
PLANNER_ALGORITHM_VERSION = 1
_SCOPE_NOTE = (
    "This is the exact saved planning snapshot, not a regenerated estimate, benchmark, "
    "training job, or guarantee of runtime fit. Hardware and planner inputs are recorded "
    "for comparison; running the plan requires a separate explicit action."
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Saved plan contains values that cannot be represented as canonical JSON.") from exc


def reproducible_plan_document(record: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic, portable representation of a saved plan snapshot."""
    if not isinstance(record, dict):
        raise ValueError("Saved experiment record must be an object.")
    plan = record.get("plan")
    hardware = record.get("hardware")
    parameter_billions = record.get("parameter_billions")
    requested_method = record.get("requested_method")
    if not isinstance(plan, dict) or not plan:
        raise ValueError("Saved experiment has no complete training-plan snapshot.")
    if not isinstance(hardware, dict):
        raise ValueError("Saved experiment has no captured hardware snapshot.")
    if (
        isinstance(parameter_billions, bool)
        or not isinstance(parameter_billions, (int, float))
        or not math.isfinite(parameter_billions)
        or parameter_billions <= 0
    ):
        raise ValueError("Saved experiment has an invalid model parameter count.")
    if not isinstance(requested_method, str) or requested_method not in {"auto", "qlora", "lora", "full"}:
        raise ValueError("Saved experiment has an invalid requested training method.")

    algorithm_version = record.get("planner_algorithm_version")
    if algorithm_version is not None and (
        type(algorithm_version) is not int or algorithm_version < 1
    ):
        raise ValueError("Saved experiment has an invalid planner algorithm version.")
    axiom_version = record.get("axiom_version")
    if axiom_version is not None and not isinstance(axiom_version, str):
        raise ValueError("Saved experiment has an invalid AXIOM version.")

    document: dict[str, Any] = {
        "format": PLAN_SNAPSHOT_FORMAT,
        "schema_version": PLAN_SNAPSHOT_SCHEMA_VERSION,
        "planner": {
            "algorithm_version": algorithm_version,
            "axiom_version": axiom_version,
            "snapshot_kind": "saved_estimate",
        },
        "inputs": {
            "parameter_billions": parameter_billions,
            "requested_method": requested_method,
            "model": record.get("model"),
            "dataset": record.get("dataset"),
        },
        "hardware": hardware,
        "plan": plan,
        "execution": {
            "started": False,
            "scope_note": _SCOPE_NOTE,
        },
    }
    fingerprint = hashlib.sha256(_canonical_bytes(document)).hexdigest()
    return {
        **document,
        "fingerprint": {
            "algorithm": "sha256",
            "value": fingerprint,
        },
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], path))
    else:
        flattened[prefix] = value
    return flattened


def diff_plan_records(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
) -> dict[str, Any]:
    """Compare saved plan inputs, captured hardware, planner provenance, and estimates."""
    left = reproducible_plan_document(left_record)
    right = reproducible_plan_document(right_record)
    left_flat = _flatten({key: value for key, value in left.items() if key != "fingerprint" and key != "execution"})
    right_flat = _flatten({key: value for key, value in right.items() if key != "fingerprint" and key != "execution"})
    paths = sorted(set(left_flat) | set(right_flat))
    changes = []
    for path in paths:
        before_present = path in left_flat
        after_present = path in right_flat
        before = left_flat.get(path)
        after = right_flat.get(path)
        if before_present == after_present and before == after:
            continue
        top_level = path.split(".", 1)[0]
        category = {
            "inputs": "inputs",
            "hardware": "hardware",
            "planner": "planner",
            "plan": "estimate",
            "format": "format",
            "schema_version": "format",
        }.get(top_level, "metadata")
        changes.append({
            "path": path,
            "category": category,
            "before_present": before_present,
            "before": before,
            "after_present": after_present,
            "after": after,
        })

    left_fingerprint = left["fingerprint"]["value"]
    right_fingerprint = right["fingerprint"]["value"]
    return {
        "schema_version": 1,
        "left": {
            "id": left_record.get("id"),
            "name": left_record.get("name"),
            "fingerprint": left_fingerprint,
        },
        "right": {
            "id": right_record.get("id"),
            "name": right_record.get("name"),
            "fingerprint": right_fingerprint,
        },
        "identical": left_fingerprint == right_fingerprint,
        "changes": changes,
        "scope_note": (
            "Diff compares recorded inputs, hardware, planner provenance, and the exact saved "
            "estimate. It does not run a planner, benchmark a model, or compare training outcomes."
        ),
    }
