"""Read-only, bounded checks for a local AXIOM LoRA training setup."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from axiom.api.safe_paths import safe_child
from axiom.core.fine_tuning import (
    TRAINING_MAX_DATASET_BYTES,
    TRAINING_MAX_LINE_BYTES,
    TRAINING_MAX_RECORDS,
    TRAINING_MAX_TEXT_CHARS,
    iter_training_text_stream,
    runtime_status,
    validate_training_text_field,
)
from axiom.core.hardware import detect_hardware
from axiom.core.mcp_paths import open_mcp_workspace_file
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
_MAX_OVERLAP_EVALUATION_BYTES = 64 * 1024 * 1024
_MAX_OVERLAP_EVALUATION_ROWS = 100_000
_MAX_OVERLAP_EVALUATION_LINE_BYTES = 1024 * 1024
_MAX_OVERLAP_ROW_SAMPLES = 5


def _normalize_overlap_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _reject_overlap_json_constant(value: str) -> None:
    raise ValueError(f"Unsupported JSON constant: {value}")


def _file_snapshot_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


class _DigestingReader:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        return chunk


def _validate_input_snapshot(
    raw_path: str,
    root: Path,
    path: Path,
    expected_identity: tuple[int, int, int, int],
    expected_digest: str,
    max_bytes: int,
) -> bool:
    try:
        current_path, stream = open_mcp_workspace_file(raw_path, root)
        with stream:
            initial = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(initial.st_mode)
                or current_path != path
                or _file_snapshot_identity(initial) != expected_identity
                or initial.st_size > max_bytes
            ):
                return False
            digest = hashlib.sha256()
            bytes_read = 0
            while True:
                chunk = stream.read(min(64 * 1024, max_bytes - bytes_read + 1))
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    return False
                digest.update(chunk)
            final = os.fstat(stream.fileno())
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        _file_snapshot_identity(final) == expected_identity
        and bytes_read == initial.st_size
        and digest.hexdigest() == expected_digest
    )


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


def run_training_evaluation_overlap_preflight(
    training_dataset_path: str,
    evaluation_dataset_path: str,
    *,
    training_text_field: str = "text",
    evaluation_prompt_field: str = "prompt",
    include_reference: bool = False,
    evaluation_reference_field: str = "completion",
    sample_limit: int = _MAX_OVERLAP_ROW_SAMPLES,
) -> dict[str, Any]:
    """Check bounded, normalized exact overlap without returning text or hashes."""
    scope = {
        "read_only": True,
        "network_accessed": False,
        "files_written": False,
        "model_weights_loaded": False,
        "dataset_values_returned": False,
        "dataset_hashes_returned": False,
        "absolute_paths_returned": False,
        "dataset_content_rechecked_after_scan": False,
        "paraphrase_overlap_detected": False,
        "leakage_free_claimed": False,
    }

    def inconclusive(
        reason_code: str,
        *,
        training_records_scanned: int = 0,
        evaluation_rows_scanned: int = 0,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "assessment": "inconclusive",
            "reason_code": reason_code,
            "scan_complete": False,
            "training_records_scanned": training_records_scanned,
            "evaluation_rows_scanned": evaluation_rows_scanned,
            "scope": scope,
        }

    if type(include_reference) is not bool:
        return inconclusive("include_reference_must_be_boolean")
    if (
        isinstance(sample_limit, bool)
        or not isinstance(sample_limit, int)
        or not 1 <= sample_limit <= _MAX_OVERLAP_ROW_SAMPLES
    ):
        return inconclusive("sample_limit_out_of_range")
    try:
        validate_training_text_field(training_text_field)
        validate_training_text_field(evaluation_prompt_field)
        if include_reference:
            validate_training_text_field(evaluation_reference_field)
            if evaluation_reference_field == evaluation_prompt_field:
                return inconclusive("evaluation_fields_must_be_different")
    except ValueError:
        return inconclusive("field_name_invalid")

    try:
        root = Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError):
        return inconclusive("workspace_unavailable")
    try:
        training_has_jsonl_suffix = Path(training_dataset_path).suffix.lower() == ".jsonl"
        evaluation_has_jsonl_suffix = Path(evaluation_dataset_path).suffix.lower() == ".jsonl"
    except (TypeError, ValueError):
        return inconclusive("dataset_path_invalid")
    if not training_has_jsonl_suffix or not evaluation_has_jsonl_suffix:
        return inconclusive("datasets_must_be_jsonl_files")

    training_records = 0
    training_texts: set[str] = set()
    training_digest = ""
    try:
        training_target, stream = open_mcp_workspace_file(training_dataset_path, root)
        with stream:
            initial = os.fstat(stream.fileno())
            if not stat.S_ISREG(initial.st_mode):
                return inconclusive("training_dataset_not_regular")
            initial_identity = _file_snapshot_identity(initial)
            digesting_stream = _DigestingReader(stream)
            for text in iter_training_text_stream(
                digesting_stream,
                file_size_bytes=initial.st_size,
                text_field=training_text_field,
            ):
                training_records += 1
                normalized = _normalize_overlap_text(text)
                if normalized:
                    training_texts.add(normalized)
            training_digest = digesting_stream.digest.hexdigest()
            final = os.fstat(stream.fileno())
            if _file_snapshot_identity(final) != initial_identity:
                return inconclusive(
                    "training_dataset_changed_during_scan",
                    training_records_scanned=training_records,
                )
    except (OSError, RuntimeError, ValueError):
        return inconclusive("training_dataset_scan_incomplete", training_records_scanned=training_records)
    if not training_records or not training_texts:
        return inconclusive("training_dataset_has_no_records", training_records_scanned=training_records)

    evaluation_rows = 0
    evaluation_bytes = 0
    prompt_matches = 0
    prompt_match_rows: list[int] = []
    prompt_reference_matches = 0
    prompt_reference_match_rows: list[int] = []
    evaluation_digest = hashlib.sha256()
    evaluation_identity: tuple[int, int, int, int]
    try:
        evaluation_target, stream = open_mcp_workspace_file(evaluation_dataset_path, root)
        with stream:
            initial = os.fstat(stream.fileno())
            if not stat.S_ISREG(initial.st_mode):
                return inconclusive(
                    "evaluation_dataset_not_regular",
                    training_records_scanned=training_records,
                )
            if initial.st_size > _MAX_OVERLAP_EVALUATION_BYTES:
                return inconclusive(
                    "evaluation_dataset_size_limit",
                    training_records_scanned=training_records,
                )
            evaluation_identity = _file_snapshot_identity(initial)
            while True:
                remaining_bytes = _MAX_OVERLAP_EVALUATION_BYTES - evaluation_bytes
                read_limit = min(_MAX_OVERLAP_EVALUATION_LINE_BYTES, remaining_bytes) + 1
                raw_line = stream.readline(read_limit)
                if not raw_line:
                    break
                evaluation_rows += 1
                evaluation_bytes += len(raw_line)
                if evaluation_rows > _MAX_OVERLAP_EVALUATION_ROWS:
                    return inconclusive(
                        "evaluation_dataset_row_limit",
                        training_records_scanned=training_records,
                        evaluation_rows_scanned=evaluation_rows,
                    )
                if len(raw_line) > _MAX_OVERLAP_EVALUATION_LINE_BYTES:
                    return inconclusive(
                        "evaluation_dataset_line_limit",
                        training_records_scanned=training_records,
                        evaluation_rows_scanned=evaluation_rows,
                    )
                if evaluation_bytes > _MAX_OVERLAP_EVALUATION_BYTES:
                    return inconclusive(
                        "evaluation_dataset_size_limit",
                        training_records_scanned=training_records,
                        evaluation_rows_scanned=evaluation_rows,
                    )
                evaluation_digest.update(raw_line)
                if not raw_line.strip():
                    return inconclusive(
                        "evaluation_dataset_contains_blank_line",
                        training_records_scanned=training_records,
                        evaluation_rows_scanned=evaluation_rows,
                    )
                try:
                    row = json.loads(
                        raw_line.decode("utf-8"),
                        parse_constant=_reject_overlap_json_constant,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
                    return inconclusive(
                        "evaluation_dataset_invalid_json",
                        training_records_scanned=training_records,
                        evaluation_rows_scanned=evaluation_rows,
                    )
                if not isinstance(row, dict):
                    return inconclusive(
                        "evaluation_dataset_row_not_object",
                        training_records_scanned=training_records,
                        evaluation_rows_scanned=evaluation_rows,
                    )
                prompt = row.get(evaluation_prompt_field)
                if not isinstance(prompt, str) or not prompt.strip():
                    return inconclusive(
                        "evaluation_prompt_field_invalid",
                        training_records_scanned=training_records,
                        evaluation_rows_scanned=evaluation_rows,
                    )
                normalized_prompt = _normalize_overlap_text(prompt)
                if normalized_prompt in training_texts:
                    prompt_matches += 1
                    if len(prompt_match_rows) < sample_limit:
                        prompt_match_rows.append(evaluation_rows)

                if include_reference:
                    reference = row.get(evaluation_reference_field)
                    if not isinstance(reference, str):
                        return inconclusive(
                            "evaluation_reference_field_invalid",
                            training_records_scanned=training_records,
                            evaluation_rows_scanned=evaluation_rows,
                        )
                    normalized_pair = _normalize_overlap_text(f"{prompt}\n{reference}")
                    if normalized_pair in training_texts:
                        prompt_reference_matches += 1
                        if len(prompt_reference_match_rows) < sample_limit:
                            prompt_reference_match_rows.append(evaluation_rows)

            final = os.fstat(stream.fileno())
            if (
                _file_snapshot_identity(final) != evaluation_identity
                or evaluation_bytes != initial.st_size
            ):
                return inconclusive(
                    "evaluation_dataset_changed_during_scan",
                    training_records_scanned=training_records,
                    evaluation_rows_scanned=evaluation_rows,
                )
    except (OSError, RuntimeError, ValueError):
        return inconclusive(
            "evaluation_dataset_scan_incomplete",
            training_records_scanned=training_records,
            evaluation_rows_scanned=evaluation_rows,
        )
    if not evaluation_rows:
        return inconclusive(
            "evaluation_dataset_has_no_records",
            training_records_scanned=training_records,
            evaluation_rows_scanned=evaluation_rows,
        )
    if not _validate_input_snapshot(
        training_dataset_path,
        root,
        training_target,
        initial_identity,
        training_digest,
        initial_identity[2],
    ) or not _validate_input_snapshot(
        evaluation_dataset_path,
        root,
        evaluation_target,
        evaluation_identity,
        evaluation_digest.hexdigest(),
        _MAX_OVERLAP_EVALUATION_BYTES,
    ):
        return inconclusive(
            "dataset_path_changed_during_scan",
            training_records_scanned=training_records,
            evaluation_rows_scanned=evaluation_rows,
        )

    scope["dataset_content_rechecked_after_scan"] = True
    overlaps = {
        "prompt": {
            "matching_rows": prompt_matches,
            "sample_row_numbers": prompt_match_rows,
        },
    }
    if include_reference:
        overlaps["prompt_and_reference"] = {
            "matching_rows": prompt_reference_matches,
            "sample_row_numbers": prompt_reference_match_rows,
        }
    overlap_found = prompt_matches > 0 or (include_reference and prompt_reference_matches > 0)
    return {
        "schema_version": 1,
        "assessment": "exact_overlap_found" if overlap_found else "no_exact_overlap_detected",
        "scan_complete": True,
        "normalization": "casefold_and_collapse_whitespace",
        "scan_limits": {
            "training_dataset_max_bytes": TRAINING_MAX_DATASET_BYTES,
            "training_records_max": TRAINING_MAX_RECORDS,
            "training_line_max_bytes": TRAINING_MAX_LINE_BYTES,
            "training_record_text_max_chars": TRAINING_MAX_TEXT_CHARS,
            "evaluation_dataset_max_bytes": _MAX_OVERLAP_EVALUATION_BYTES,
            "evaluation_rows_max": _MAX_OVERLAP_EVALUATION_ROWS,
            "evaluation_line_max_bytes": _MAX_OVERLAP_EVALUATION_LINE_BYTES,
        },
        "training_dataset": {
            "path": training_target.relative_to(root).as_posix(),
            "text_field": training_text_field,
            "records_scanned": training_records,
        },
        "evaluation_dataset": {
            "path": evaluation_target.relative_to(root).as_posix(),
            "prompt_field": evaluation_prompt_field,
            "reference_checked": include_reference,
            "reference_field": evaluation_reference_field if include_reference else None,
            "rows_scanned": evaluation_rows,
        },
        "overlaps": overlaps,
        "limitations": [
            "Only case- and whitespace-normalized exact text is detected; paraphrases are not checked.",
            "No exact overlap does not prove that the evaluation data is held out or leakage-free.",
            "The result applies to the scanned file snapshots; files may change afterward.",
        ],
        "sample_limit_per_overlap": sample_limit,
        "scope": scope,
    }
