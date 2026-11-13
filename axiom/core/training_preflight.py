"""Read-only, bounded checks for a local AXIOM LoRA training setup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from axiom.api.safe_paths import safe_child
from axiom.core.fine_tuning import (
    iter_training_text_stream,
    runtime_status,
    validate_training_text_field,
)
from axiom.core.hardware import detect_hardware
from axiom.models.inspector import inspect_model
from axiom.path_safety import open_regular_file
from axiom.training.planner import create_training_plan


_SCOPE = {
    "read_only": True,
    "network_accessed": False,
    "files_written": False,
    "job_created": False,
    "model_weights_loaded": False,
    "dataset_values_returned": False,
    "output_directory_checked": False,
    "absolute_paths_returned": False,
    "hardware_detection": "Local detection may query nvidia-smi when available.",
}


def _path_failure(field: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "assessment": "blockers_found",
        "checks": {field: {"status": "fail", "reason": reason}},
        "blockers": [{"check": field, "reason": reason}],
        "warnings": [],
        "next_steps": ["Choose an existing workspace-relative path and try again."],
        "scope": dict(_SCOPE),
    }


def resolve_workspace_input(raw: str, root: Path, label: str) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or len(raw) > 4096
        or raw.startswith("/")
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError(f"{label} must be a workspace-relative path.")

    parts = raw.split("/")
    if (
        any(part in {"", ".", ".."} for part in parts)
        or (len(parts[0]) >= 2 and parts[0][1] == ":")
    ):
        raise ValueError(f"{label} must be a safe workspace-relative path.")

    try:
        workspace = Path(root).resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("The workspace root is unavailable.") from None
    if workspace != Path(root) or not workspace.is_dir():
        raise ValueError("The workspace root changed or is not a directory.")

    current = workspace
    for part in parts:
        current = current / part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise ValueError(f"{label} cannot contain symbolic links.")

    try:
        target = safe_child(workspace, raw, must_exist=True)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        raise ValueError(f"{label} was not found inside the workspace.") from None
    if not target.is_relative_to(workspace):
        raise ValueError(f"{label} must stay inside the workspace.")
    return target


def resolve_workspace_output(raw: str, root: Path) -> Path:
    """Resolve a new or empty adapter directory under an existing outputs folder."""
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or len(raw) > 4096
        or raw.startswith("/")
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError("Adapter output must be a workspace-relative path under outputs/.")

    parts = raw.split("/")
    if (
        len(parts) < 2
        or parts[0] != "outputs"
        or any(part in {"", ".", ".."} for part in parts)
        or (len(parts[0]) >= 2 and parts[0][1] == ":")
    ):
        raise ValueError("Adapter output must be a safe path below outputs/.")

    try:
        workspace = Path(root).resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("The workspace root is unavailable.") from None
    if workspace != Path(root) or not workspace.is_dir():
        raise ValueError("The workspace root changed or is not a directory.")
    parent = workspace
    for part in parts[:-1]:
        parent = parent / part
        is_junction = getattr(parent, "is_junction", lambda: False)
        if parent.is_symlink() or is_junction() or not parent.is_dir():
            raise ValueError("Adapter output parent directories must exist and cannot contain links.")

    target = parent / parts[-1]
    is_junction = getattr(target, "is_junction", lambda: False)
    if target.is_symlink() or is_junction():
        raise ValueError("Adapter output cannot be a symbolic link.")
    if target.exists():
        if not target.is_dir():
            raise ValueError("Adapter output must be a directory.")
        try:
            next(target.iterdir())
        except StopIteration:
            pass
        else:
            raise ValueError("Adapter output directory must be new or empty.")

    resolved = target.resolve(strict=target.exists())
    if not resolved.is_relative_to(workspace):
        raise ValueError("Adapter output must stay inside the workspace.")
    return resolved


def _model_metadata(inspection: Any, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "architecture": inspection.architecture,
        "model_type": inspection.model_type,
        "format": inspection.model_format,
        "parameters": inspection.parameter_count,
        "parameter_source": inspection.parameter_source,
        "weight_size_bytes": inspection.weight_size_bytes,
        "has_config": inspection.has_config,
        "has_tokenizer": inspection.has_tokenizer,
        "has_safetensors": inspection.has_safetensors,
        "has_pytorch_weights": inspection.has_pytorch_weights,
        "has_transformers_weights": inspection.has_transformers_weights,
        "transformers_weight_size_bytes": inspection.transformers_weight_size_bytes,
        "has_gguf": inspection.has_gguf,
        "capabilities": inspection.capabilities,
    }


def run_training_preflight(
    model_path: str,
    dataset_path: str,
    text_field: str = "text",
) -> dict[str, Any]:
    """Check static local prerequisites without loading weights or starting training."""
    try:
        root = Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError):
        return _path_failure("workspace", "The current workspace could not be resolved safely.")

    try:
        model_target = resolve_workspace_input(model_path, root, "Model path")
    except ValueError as exc:
        return _path_failure("model_path", str(exc))
    if not model_target.is_dir():
        return _path_failure("model_path", "Model path must refer to a local model directory.")

    try:
        dataset_target = resolve_workspace_input(dataset_path, root, "Dataset path")
    except ValueError as exc:
        return _path_failure("dataset_path", str(exc))
    if not dataset_target.is_file() or dataset_target.suffix.lower() != ".jsonl":
        return _path_failure("dataset_path", "Dataset path must refer to a local .jsonl file.")

    relative_model_path = model_target.relative_to(root).as_posix()
    relative_dataset_path = dataset_target.relative_to(root).as_posix()
    blockers: list[dict[str, str]] = []
    warnings: list[str] = []
    unknown_checks: list[str] = []

    try:
        inspection = inspect_model(str(model_target))
        model = _model_metadata(inspection, relative_model_path)
        model_blockers: list[str] = []
        model_unknown_reasons: list[str] = []
        if not inspection.has_config:
            model_blockers.append("The model directory has no config.json.")
        if not inspection.has_transformers_weights:
            model_blockers.append(
                "No non-empty Transformers weight files were detected; a GGUF-only model cannot use the LoRA trainer."
            )
        if not inspection.has_tokenizer:
            model_blockers.append("Tokenizer files were not detected in the model directory.")
        if "Text generation" not in inspection.capabilities:
            if "Unknown" in inspection.capabilities:
                unknown_checks.append("model_capability")
                model_unknown_reasons.append("Text-generation capability could not be inferred from the config.")
            else:
                model_blockers.append("The model metadata does not indicate text-generation capability.")
        if inspection.parameter_count is None or inspection.parameter_count <= 0:
            unknown_checks.append("parameter_count")
            model_unknown_reasons.append("Parameter count could not be inferred for hardware planning.")

        for reason in model_blockers:
            blockers.append({"check": "model", "reason": reason})
        if model_blockers:
            model_status = "fail"
        elif model_unknown_reasons:
            model_status = "unknown"
        else:
            model_status = "pass"
        model_check = {"status": model_status, "model": model}
        if model_blockers:
            model_check["reasons"] = model_blockers
        elif model_unknown_reasons:
            model_check["reasons"] = model_unknown_reasons
    except (OSError, ValueError, RecursionError):
        model = {"path": relative_model_path}
        model_check = {
            "status": "fail",
            "reason": "Local model metadata could not be inspected safely.",
        }
        blockers.append({"check": "model", "reason": model_check["reason"]})
        inspection = None

    dataset_size = 0
    accepted_records = 0
    dataset_scan_complete = False
    dataset_reason = None
    if not isinstance(text_field, str) or len(text_field) > 64:
        dataset_reason = "Text field must be a simple JSON field name of 64 characters or fewer."
        validated_text_field = None
    else:
        try:
            validate_training_text_field(text_field)
            validated_text_field = text_field
        except ValueError as exc:
            dataset_reason = str(exc)
            validated_text_field = None

    if dataset_reason is None:
        try:
            with open_regular_file(dataset_target) as stream:
                dataset_size = os.fstat(stream.fileno()).st_size
                for _ in iter_training_text_stream(
                    stream,
                    file_size_bytes=dataset_size,
                    text_field=validated_text_field,
                ):
                    accepted_records += 1
            dataset_scan_complete = True
            if not accepted_records:
                dataset_reason = "Dataset contains no non-empty training records."
        except (OSError, ValueError) as exc:
            dataset_reason = str(exc)
    if dataset_reason is not None:
        blockers.append({"check": "dataset", "reason": dataset_reason})

    dataset_check = {
        "status": "fail" if dataset_reason else "pass",
        "path": relative_dataset_path,
        "file_size_bytes": dataset_size,
        "text_field": validated_text_field,
        "accepted_records": accepted_records,
        "scan_complete": dataset_scan_complete,
        "reason": dataset_reason,
    }

    runtime = runtime_status()
    unverified_versions = [
        name
        for name, present in runtime["packages"].items()
        if present and runtime["versions"].get(name) is None
    ]
    if not runtime["can_start"]:
        runtime_check_status = "fail"
    elif unverified_versions:
        runtime_check_status = "unknown"
        unknown_checks.append("training_dependency_versions")
        warnings.append(
            "One or more installed training packages have no version metadata, so minimum versions could not be verified."
        )
    else:
        runtime_check_status = "pass"
    runtime_check = {
        "status": runtime_check_status,
        "check_type": "package_metadata_only",
        "packages": runtime["packages"],
        "versions": runtime["versions"],
        "minimum_versions": runtime["minimum_versions"],
        "unverified_versions": unverified_versions,
        "problems": runtime["problems"],
    }
    if not runtime["can_start"]:
        blockers.append({
            "check": "training_dependencies",
            "reason": "Required local training packages are missing or below minimum versions.",
        })

    hardware = detect_hardware()
    hardware_details = {
        "os_name": hardware.os_name,
        "architecture": hardware.architecture,
        "cpu_cores": hardware.cpu_cores,
        "ram_gb": hardware.ram_gb,
        "gpu_name": hardware.gpu_name,
        "vram_gb": hardware.vram_gb,
        "cuda_available": hardware.cuda_available,
    }
    estimate = None
    estimate_status = "unknown"
    if inspection is not None and inspection.parameter_count and inspection.parameter_count > 0:
        try:
            plan = create_training_plan(
                inspection.parameter_count / 1_000_000_000,
                hardware,
                method="lora",
            )
            estimate = {
                "selected_method": plan.method,
                "precision": plan.precision,
                "estimated_vram_gb": plan.estimated_vram_gb,
                "fits_hardware_estimate": plan.fits_hardware,
                "reason": plan.reason,
                "assumptions": (
                    "Generic planning heuristic only; the local trainer loads the checkpoint "
                    "without quantization and this estimate does not guarantee memory fit."
                ),
            }
            if plan.method == "lora" and plan.fits_hardware is True:
                estimate_status = "warn"
                warnings.append("Hardware fit is an estimate; actual device and model loading were not tested.")
            elif plan.method == "lora" and plan.fits_hardware is False:
                warnings.append(
                    "The generic planner estimates this LoRA plan above reported VRAM; actual fit remains unverified."
                )
                unknown_checks.append("hardware_fit")
            elif plan.method == "qlora":
                warnings.append(
                    "The generic planner fell back to QLoRA, which the current local trainer does not implement; actual LoRA fit is unknown."
                )
                unknown_checks.append("hardware_fit")
            else:
                warnings.append(
                    "NVIDIA VRAM was not available to the planner; CPU and Apple MPS fit are not estimated."
                )
                unknown_checks.append("hardware_fit")
        except (TypeError, ValueError):
            unknown_checks.append("hardware_fit")
            warnings.append("A hardware estimate could not be produced from the model metadata.")
    else:
        unknown_checks.append("hardware_fit")
        warnings.append("A hardware estimate is unavailable because parameter count is unknown.")

    checks = {
        "model": model_check,
        "dataset": dataset_check,
        "training_dependencies": runtime_check,
        "hardware": {
            "status": estimate_status,
            "detected": hardware_details,
            "estimate": estimate,
        },
    }
    if blockers:
        assessment = "blockers_found"
    elif unknown_checks:
        assessment = "review_needed"
    else:
        assessment = "no_known_blockers"

    next_steps = []
    if blockers:
        next_steps.append("Resolve the reported local model, dataset, or package blockers.")
    if unknown_checks:
        next_steps.append("Check model and device compatibility with the local training runtime before starting a job.")
    if not blockers and not unknown_checks:
        next_steps.append("Review the output directory and training settings before explicitly starting a LoRA job.")
    warnings.append(
        "Static checks do not load the model or prove architecture-specific LoRA compatibility; no-known-blockers does not guarantee training will start or fit."
    )

    return {
        "schema_version": 1,
        "assessment": assessment,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "next_steps": next_steps,
        "scope": dict(_SCOPE),
    }
