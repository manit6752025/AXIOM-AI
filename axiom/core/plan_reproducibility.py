"""Canonical rendering and comparison for immutable saved training-plan snapshots."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from typing import Any

PLAN_SNAPSHOT_SCHEMA_VERSION = 1
PLAN_SNAPSHOT_FORMAT = "application/vnd.axiom.training-plan+json;version=1"
PLANNER_ALGORITHM_VERSION = 1
SOFTWARE_MANIFEST_SCHEMA_VERSION = 1
SOFTWARE_DISTRIBUTIONS = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "numpy": "numpy",
    "safetensors": "safetensors",
}
_SCOPE_NOTE = (
    "This is the exact saved planning snapshot, not a regenerated estimate, benchmark, "
    "training job, or guarantee of runtime fit. Hardware and planner inputs are recorded "
    "for comparison; running the plan requires a separate explicit action."
)
_REPRODUCIBILITY_SCOPE_NOTE = (
    "The seed is recorded as provenance only. Exporting or comparing this plan does not "
    "seed Python, PyTorch, CUDA, or another random-number generator, start a job, or pass "
    "the seed to the local LoRA worker. A future workflow executor must explicitly consume "
    "this value before it can affect execution."
)


def capture_software_manifest(axiom_version: str) -> dict[str, Any]:
    """Capture bounded, non-secret software-version evidence at plan-save time."""
    if not isinstance(axiom_version, str) or not axiom_version or len(axiom_version) > 64:
        raise ValueError("AXIOM version must be a non-empty string of at most 64 characters.")
    packages: dict[str, str | None] = {}
    for name, distribution in SOFTWARE_DISTRIBUTIONS.items():
        try:
            packages[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": SOFTWARE_MANIFEST_SCHEMA_VERSION,
        "captured": True,
        "axiom_version": axiom_version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": packages,
    }


def _software_manifest(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("software_manifest")
    if raw is None:
        return {
            "schema_version": SOFTWARE_MANIFEST_SCHEMA_VERSION,
            "captured": False,
            "axiom_version": None,
            "python_version": None,
            "python_implementation": None,
            "platform": {"system": None, "machine": None},
            "packages": {},
        }
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw.get("schema_version") != SOFTWARE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Saved experiment has an unsupported software manifest.")
    captured = raw.get("captured")
    if type(captured) is not bool:
        raise ValueError("Saved experiment has an invalid software manifest status.")
    text_fields = ("axiom_version", "python_version", "python_implementation")
    normalized: dict[str, str | None] = {}
    for field in text_fields:
        value = raw.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > 128):
            raise ValueError(f"Saved experiment has an invalid software manifest {field}.")
        normalized[field] = value
    raw_platform = raw.get("platform", {})
    if not isinstance(raw_platform, dict):
        raise ValueError("Saved experiment has an invalid software manifest platform.")
    platform_values: dict[str, str | None] = {}
    for field in ("system", "machine"):
        value = raw_platform.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > 128):
            raise ValueError(f"Saved experiment has an invalid software manifest platform {field}.")
        platform_values[field] = value
    raw_packages = raw.get("packages", {})
    if not isinstance(raw_packages, dict) or len(raw_packages) > len(SOFTWARE_DISTRIBUTIONS):
        raise ValueError("Saved experiment has an invalid software package manifest.")
    packages: dict[str, str | None] = {}
    for name, version in raw_packages.items():
        if name not in SOFTWARE_DISTRIBUTIONS or (version is not None and (not isinstance(version, str) or len(version) > 128)):
            raise ValueError("Saved experiment has an invalid software package version.")
        packages[name] = version
    if captured and not normalized["axiom_version"]:
        raise ValueError("Captured software manifest is missing the AXIOM version.")
    return {
        "schema_version": SOFTWARE_MANIFEST_SCHEMA_VERSION,
        "captured": captured,
        **normalized,
        "platform": platform_values,
        "packages": packages,
    }


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
    required_plan_fields = {
        "method", "precision", "lora_rank", "batch_size",
        "gradient_accumulation", "sequence_length", "learning_rate",
        "estimated_vram_gb", "fits_hardware", "reason",
    }
    if not isinstance(plan, dict) or not required_plan_fields.issubset(plan):
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

    seed = record.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 4_294_967_295):
        raise ValueError("Saved experiment has an invalid reproducibility seed.")
    software = _software_manifest(record)
    reproducibility = {
        "schema_version": SOFTWARE_MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "seed_status": "recorded_only" if seed is not None else "not_recorded",
        "seed_applied": False,
        "software": software,
        "scope_note": _REPRODUCIBILITY_SCOPE_NOTE,
    }

    document: dict[str, Any] = {
        "format": PLAN_SNAPSHOT_FORMAT,
        "schema_version": PLAN_SNAPSHOT_SCHEMA_VERSION,
        "planner": {
            "algorithm_version": algorithm_version,
            "axiom_version": axiom_version,
            "snapshot_kind": "saved_estimate",
        },
        "inputs": {
            "parameter_billions": float(parameter_billions),
            "requested_method": requested_method,
            "model": record.get("model"),
            "dataset": record.get("dataset"),
        },
        "hardware": hardware,
        "plan": plan,
        "reproducibility": reproducibility,
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
    """Compare saved plan inputs, captured hardware, provenance, and estimates."""
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
            "reproducibility": "provenance",
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
            "Diff compares recorded inputs, hardware, software/seed provenance, and the exact saved "
            "estimate. It does not run a planner, apply a seed, benchmark a model, or execute training."
        ),
    }
