from __future__ import annotations

import math
import os
import platform
import re
from pathlib import Path

from mcp.server import MCPServer

from axiom.core.engine import (
    AxiomEngineError,
    check_local_adapter_compatibility,
    estimate_native_quantization,
    generate_local,
    inspect_native_model_context,
    refresh_local_engine_asset_index,
    resolve_local_model_directory,
    runtime_status,
)
from axiom.core.hardware import detect_hardware
from axiom.core.fine_tuning import (
    ActiveFineTuningJobError,
    FineTuningError,
    cancel_finetuning_job,
    get_job,
    list_jobs,
    start_finetuning_job,
    validate_local_model_tree,
)
from axiom.core.mcp_paths import (
    open_mcp_workspace_file,
    open_mcp_workspace_output_parent,
)
from axiom.core.tool_permissions import ToolPermissionStore
from axiom.core.training_preflight import (
    resolve_workspace_input,
    resolve_workspace_output,
    run_training_preflight,
)
from axiom.daemon import local_summary
from axiom.daemon_api import MAX_LISTED_FILES, list_workspace_files
from axiom.datasets.cleaner import clean_jsonl
from axiom.datasets.cleaning_preview import preview_jsonl_cleaning
from axiom.datasets.profiler import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_ROWS,
    profile_jsonl_stream,
)
from axiom.datasets.quality import validate_dataset_profile
from axiom.evaluation.jsonl import (
    DEFAULT_MAX_BYTES as DEFAULT_EVALUATION_MAX_BYTES,
    DEFAULT_MAX_LINE_BYTES as DEFAULT_EVALUATION_MAX_LINE_BYTES,
    DEFAULT_MAX_ROWS as DEFAULT_EVALUATION_MAX_ROWS,
    evaluate_jsonl_stream,
)
from axiom.evaluation.service import persist_evaluation_run
from axiom.evaluation.store import EvaluationStore
from axiom.models.inspector import inspect_model
from axiom.models.registry import ModelRegistry
from axiom.runtime.supercompress import compress_context, redacted_base_url
from axiom.training.planner import create_training_plan
from axiom.version import __version__

_MCP_ENGINE_MAX_PROMPT_CHARS = 8_000
_MCP_ENGINE_MAX_NEW_TOKENS = 256

mcp = MCPServer(
    f"AXIOM v{__version__}",
    instructions=(
        "AXIOM is an AI engineering platform. "
        "After the active AXIOM profile allows it, start with axiom_workspace_context "
        "for a bounded local workspace briefing. "
        "Use axiom_workspace_assets to find local model and dataset paths when allowed. "
        "Use axiom_dataset_inspect for bounded JSONL structure and schema checks. "
        "Sensitive field-name heuristics are included; content PII pattern scanning requires explicit opt-in. "
        "Use axiom_training_preflight for bounded, non-executing checks before local LoRA training. "
        "Use axiom_engine_preflight before local inference to check model/adapter structure, the selected device/precision, and advisory memory estimates; it does not load weights or run inference. Pass the same settings to axiom_engine_generate. "
        "Run preflight and show the exact configuration before calling axiom_training_job_start; "
        "call it only after the user explicitly requests that run. It uses compute and writes an adapter under outputs/. "
        "Use the separate status/cancel tools to manage the job. "
        "For local inference, use axiom_engine_generate only after the user requests a response; "
        "it runs one bounded prompt locally and AXIOM does not store the prompt or response. The MCP host may retain its tool transcript. "
        "Use AXIOM tools to inspect models, datasets, and hardware; preview dataset cleaning "
        "and show the counts and proposed output path before writing a new copy. "
        "Re-preview immediately before cleaning because previews are advisory. "
        "Use axiom_evaluation_run for bounded, offline lexical metrics on completed outputs; "
        "use axiom_evaluation_history to retrieve saved baselines; "
        "these metrics do not establish semantic quality. "
        "Create training plans and check optional AI runtime integrations. "
        "MCP tool access is denied by default and must "
        "be explicitly granted in an AXIOM tool profile."
    ),
)


def _validate_cleaning_limits(
    max_bytes: int,
    max_rows: int,
    max_line_bytes: int,
) -> None:
    if (
        type(max_bytes) is not int
        or not 1 <= max_bytes <= 256 * 1024 * 1024
        or type(max_rows) is not int
        or not 1 <= max_rows <= 500_000
        or type(max_line_bytes) is not int
        or not 1 <= max_line_bytes <= 8 * 1024 * 1024
    ):
        raise ValueError("Cleaning limits exceed the supported bounded scan range.")


def _validate_evaluation_limits(
    max_bytes: int,
    max_rows: int,
    max_line_bytes: int,
) -> None:
    if (
        type(max_bytes) is not int
        or not 1 <= max_bytes <= 64 * 1024 * 1024
        or type(max_rows) is not int
        or not 1 <= max_rows <= 100_000
        or type(max_line_bytes) is not int
        or not 1 <= max_line_bytes <= 1024 * 1024
    ):
        raise ValueError("Evaluation limits exceed the supported bounded scan range.")


def _evaluation_dataset_root(workspace: Path) -> Path:
    entry = workspace / "data"
    if entry.is_symlink():
        raise ValueError("The workspace data directory must not be a symlink.")
    try:
        data_root = entry.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("The workspace data directory is unavailable.") from exc
    if not data_root.is_dir() or not data_root.is_relative_to(workspace):
        raise ValueError("The workspace data directory is invalid.")
    return data_root


def _evaluation_record_summary(record: dict, *, include_report: bool = False) -> dict:
    report = record.get("report", {})
    if not isinstance(report, dict):
        report = {}
    counts = report.get("counts", {})
    metrics = report.get("metrics", {})
    manifest = report.get("dataset_manifest", {})
    if not isinstance(counts, dict):
        counts = {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(manifest, dict):
        manifest = {}
    summary = {
        "id": record.get("id"),
        "name": record.get("name"),
        "created_at": record.get("created_at"),
        "model_label": record.get("model_label"),
        "baseline": record.get("baseline"),
        "status": record.get("status"),
        "comparison": record.get("comparison"),
        "dataset_file": report.get("file_name"),
        "valid_samples": counts.get("valid_samples"),
        "metrics": metrics,
        "dataset_fully_hashed": manifest.get("full_content_hashed"),
    }
    if include_report:
        summary["run"] = _public_evaluation_record(record)
    return summary


def _public_evaluation_record(record: dict) -> dict:
    """Remove deterministic input fingerprints from data returned over MCP."""
    safe_record = dict(record)
    original_report = record.get("report", {})
    safe_report = dict(original_report) if isinstance(original_report, dict) else {}
    original_manifest = safe_report.get("dataset_manifest", {})
    safe_manifest = dict(original_manifest) if isinstance(original_manifest, dict) else {}
    for key in ("sha256", "partial_sha256", "comparison_corpus_sha256"):
        safe_manifest.pop(key, None)
    safe_report["dataset_manifest"] = safe_manifest
    safe_report["path"] = safe_report.get("file_name", "dataset.jsonl")
    safe_record["report"] = safe_report
    safe_record.pop("output_manifest", None)
    return safe_record


@mcp.tool()
def axiom_info() -> dict:
    """Return AXIOM release information."""
    ToolPermissionStore().require("axiom_info")
    return {
        "name": "AXIOM",
        "version": __version__,
        "release_channel": "beta",
        "status": "experimental",
        "capabilities": [
            "models",
            "datasets",
            "hardware",
            "training_planning",
            "offline_evaluation",
            "supercompress",
            "mcp",
            "agent_integrations",
        ],
    }


@mcp.tool()
def axiom_evaluation_run(
    path: str,
    prediction_field: str,
    reference_field: str,
    split_field: str | None = None,
    split_value: str | None = None,
    name: str | None = None,
    model_label: str | None = None,
    baseline: bool = False,
    baseline_run_id: str | None = None,
    regression_metric: str = "token_f1_micro",
    regression_tolerance: float = 0.05,
    max_bytes: int = DEFAULT_EVALUATION_MAX_BYTES,
    max_rows: int = DEFAULT_EVALUATION_MAX_ROWS,
    max_line_bytes: int = DEFAULT_EVALUATION_MAX_LINE_BYTES,
) -> dict:
    """Score precomputed JSONL predictions offline and save aggregate metrics."""
    ToolPermissionStore().require("axiom_evaluation_run")
    _validate_evaluation_limits(max_bytes, max_rows, max_line_bytes)

    if (
        not isinstance(prediction_field, str)
        or not prediction_field.strip()
        or len(prediction_field) > 160
        or not isinstance(reference_field, str)
        or not reference_field.strip()
        or len(reference_field) > 160
    ):
        raise ValueError("Prediction and reference fields must contain 1 to 160 characters.")
    prediction_field = prediction_field.strip()
    reference_field = reference_field.strip()
    if prediction_field == reference_field:
        raise ValueError("Prediction and reference fields must be different.")
    if (split_field is None) != (split_value is None):
        raise ValueError("Split field and split value must be supplied together.")
    if split_field is not None:
        if (
            not isinstance(split_field, str)
            or not split_field.strip()
            or len(split_field) > 160
            or not isinstance(split_value, str)
            or len(split_value) > 160
        ):
            raise ValueError("Split field and value must contain at most 160 characters.")
        split_field = split_field.strip()
    if name is not None:
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("Evaluation names must contain 1 to 120 characters.")
        name = name.strip()
    if model_label is not None and (
        not isinstance(model_label, str) or len(model_label) > 256
    ):
        raise ValueError("Model labels cannot exceed 256 characters.")
    if type(baseline) is not bool:
        raise ValueError("baseline must be a boolean.")
    if baseline_run_id is not None and (
        not isinstance(baseline_run_id, str)
        or not baseline_run_id.strip()
        or len(baseline_run_id) > 64
    ):
        raise ValueError("Baseline run IDs must contain 1 to 64 characters.")
    if baseline_run_id is not None:
        baseline_run_id = baseline_run_id.strip()
    if not isinstance(regression_metric, str) or regression_metric not in {
        "exact_match",
        "normalized_exact_match",
        "token_f1_micro",
    }:
        raise ValueError("Choose exact_match, normalized_exact_match, or token_f1_micro.")
    if isinstance(regression_tolerance, bool) or not isinstance(regression_tolerance, (int, float)):
        raise ValueError("Regression tolerance must be a finite number between 0 and 1.")
    try:
        regression_tolerance = float(regression_tolerance)
    except (OverflowError, ValueError):
        regression_tolerance = float("nan")
    if not math.isfinite(regression_tolerance) or not 0 <= regression_tolerance <= 1:
        raise ValueError("Regression tolerance must be a finite number between 0 and 1.")

    if not isinstance(path, str) or not path.strip():
        raise ValueError("Evaluation path cannot be empty.")
    relative_path = path[5:] if path.startswith("data/") else path
    root = Path.cwd().resolve(strict=True)
    data_root = _evaluation_dataset_root(root)
    source, source_stream = open_mcp_workspace_file(relative_path, data_root)
    if source.suffix.lower() != ".jsonl":
        source_stream.close()
        raise ValueError("Evaluation input must be a workspace-relative .jsonl file.")
    try:
        with source_stream:
            initial = os.fstat(source_stream.fileno())
            report = evaluate_jsonl_stream(
                source_stream,
                path=source.relative_to(data_root).as_posix(),
                file_name=source.name,
                file_size_bytes=initial.st_size,
                prediction_field=prediction_field,
                reference_field=reference_field,
                split_field=split_field,
                split_value=split_value,
                max_bytes=max_bytes,
                max_rows=max_rows,
                max_line_bytes=max_line_bytes,
            )
            final = os.fstat(source_stream.fileno())
        current_path, current_stream = open_mcp_workspace_file(relative_path, data_root)
        with current_stream:
            current = os.fstat(current_stream.fileno())
        initial_identity = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
        final_identity = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        if initial_identity != final_identity or initial_identity != current_identity:
            raise ValueError("Dataset changed during evaluation; retry against a stable file.")
        if current_path.relative_to(data_root).as_posix() != report["path"]:
            raise ValueError("Dataset path changed during evaluation; retry against a stable file.")
    except (OSError, ValueError) as exc:
        source_stream.close()
        raise ValueError(f"Offline evaluation could not complete safely: {exc}") from None

    try:
        record = persist_evaluation_run(
            report,
            name=name,
            model_label=model_label,
            baseline=baseline,
            baseline_run_id=baseline_run_id,
            regression_metric=regression_metric,
            regression_tolerance=regression_tolerance,
            python_version=platform.python_version(),
        )
    except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Offline evaluation could not be saved safely: {exc}") from None

    from axiom.api.workspace import record_workspace_event

    record_workspace_event("evaluation.run", f"Evaluated precomputed outputs from {report['path']}")
    return {"run": _public_evaluation_record(record)}


@mcp.tool()
def axiom_evaluation_history(
    run_id: str | None = None,
    limit: int = 20,
) -> dict:
    """Read aggregate evaluation history without exposing dataset row values or absolute paths."""
    ToolPermissionStore().require("axiom_evaluation_history")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Evaluation history limit must be between 1 and 100.")
    store = EvaluationStore()
    try:
        if run_id is not None:
            if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
                raise ValueError("Invalid evaluation run ID.")
            record = store.get(run_id)
            if record is None:
                raise ValueError("Evaluation run was not found.")
            return _evaluation_record_summary(record, include_report=True)

        page = store.list_page(limit=limit, offset=0)
    except (OSError, ValueError):
        raise ValueError("Evaluation history could not be read safely.") from None
    return {
        "runs": [_evaluation_record_summary(record) for record in page["runs"]],
        "total": page["total"],
        "limit": page["limit"],
        "has_more": page["has_more"],
    }


@mcp.tool()
def axiom_workspace_context() -> dict:
    """Read bounded local context; suggested commands are not run and need approval."""
    permission_store = ToolPermissionStore()
    permission_store.require("axiom_workspace_context")
    permission_state = permission_store.list()
    active_profile = next(
        profile
        for profile in permission_state["profiles"]
        if profile["name"] == permission_state["active_profile"]
    )
    summary = local_summary(Path.cwd())
    hardware = summary["hardware"]
    tools = summary["tools"]

    return {
        "schema_version": 1,
        "axiom_version": summary["version"],
        "workspace": {
            "project_detected": summary["project"]["detected"],
            "missing_paths": summary["project"]["missing_paths"],
        },
        "hardware": {
            "os_name": hardware["os_name"],
            "architecture": hardware["architecture"],
            "cpu_cores": hardware["cpu_cores"],
            "ram_gb": hardware["ram_gb"],
            "gpu_name": hardware["gpu_name"],
            "vram_gb": hardware["vram_gb"],
            "cuda_available": hardware["cuda_available"],
        },
        "developer_tools": {
            "available": tools["available"],
            "total": tools["total"],
            "available_names": tools["names"],
            "missing_names": tools["missing"],
            "catalog_scope": (
                "External developer tools in AXIOM's catalog; availability "
                "does not establish sign-in or credential status."
            ),
        },
        "mcp_access": {
            "active_profile": permission_state["active_profile"],
            "permitted_tools": active_profile["allowed_tools"],
        },
        "recommended_next_step": {
            **summary["recommended"],
            "requires_user_approval_before_execution": True,
        },
        "scope": {
            "read_only": True,
            "network_accessed": False,
            "file_contents_returned": False,
            "absolute_paths_returned": False,
            "hardware_detection": (
                "Local detection may query nvidia-smi when available."
            ),
        },
    }


@mcp.tool()
def axiom_workspace_assets() -> dict:
    """List bounded workspace-relative model and JSONL paths without reading file contents."""
    ToolPermissionStore().require("axiom_workspace_assets")
    model_listing = list_workspace_files("models")
    dataset_listing = list_workspace_files("datasets")
    models = model_listing["items"]
    datasets = dataset_listing["items"]

    return {
        "schema_version": 1,
        "model_directories": models,
        "jsonl_datasets": datasets,
        "max_results_per_category": MAX_LISTED_FILES,
        "result_limit_reached": {
            "models": model_listing["result_limit_reached"],
            "datasets": dataset_listing["result_limit_reached"],
        },
        "scan_truncated": {
            "models": model_listing["scan_truncated"],
            "datasets": dataset_listing["scan_truncated"],
        },
        "may_be_truncated": {
            "models": model_listing["result_limit_reached"]
            or model_listing["scan_truncated"],
            "datasets": dataset_listing["result_limit_reached"]
            or dataset_listing["scan_truncated"],
        },
        "scan_entry_limit_per_category": model_listing["scan_entry_limit"],
        "scope": {
            "read_only": True,
            "file_contents_read": False,
            "absolute_paths_returned": False,
            "symlinked_entries_followed": False,
        },
    }


@mcp.tool()
def axiom_training_preflight(
    model_path: str,
    dataset_path: str,
    text_field: str = "text",
) -> dict:
    """Check local LoRA prerequisites without loading weights or starting a job."""
    ToolPermissionStore().require("axiom_training_preflight")
    return run_training_preflight(model_path, dataset_path, text_field)


def _mcp_workspace_root() -> Path:
    try:
        return Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("The MCP workspace could not be resolved safely.") from None


def _training_job_matches_workspace(job: dict, root: Path) -> bool:
    config = job.get("config")
    if not isinstance(config, dict):
        return False
    for key in ("model_path", "dataset_path", "output_dir"):
        raw = config.get(key)
        if not isinstance(raw, str) or not raw:
            return False
        try:
            path = Path(raw).resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        if not path.is_relative_to(root):
            return False
    return True


def _training_job_summary(job: dict, root: Path) -> dict:
    config = job.get("config") or {}
    relative_paths = {}
    for key in ("model_path", "dataset_path", "output_dir"):
        resolved = Path(config[key]).resolve(strict=False)
        try:
            relative_paths[key] = resolved.relative_to(root).as_posix()
        except ValueError:
            raise ValueError("Saved training job paths no longer belong to this workspace.") from None

    result = job.get("result")
    safe_result = None
    if isinstance(result, dict):
        safe_result = {
            "dataset_rows": result.get("dataset_rows"),
            "max_steps": result.get("max_steps"),
            "device": result.get("device"),
            "engine": result.get("engine"),
            "target_modules": result.get("target_modules", []),
            "trainable_parameters": result.get("trainable_parameters"),
            "adapter_files": [Path(value).name for value in result.get("adapter_files", []) if isinstance(value, str)],
        }
    status = job.get("status")
    return {
        "schema_version": 1,
        "id": job.get("id"),
        "status": status,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "cancel_requested": bool(job.get("cancel_requested", False)),
        "config": {
            **relative_paths,
            "text_field": config.get("text_field"),
            "max_steps": config.get("max_steps"),
            "sequence_length": config.get("sequence_length"),
            "learning_rate": config.get("learning_rate"),
            "lora_rank": config.get("lora_rank"),
        },
        "completed_steps": job.get("completed_steps", 0),
        "loss": job.get("loss"),
        "result": safe_result,
        "error": (
            "Training failed; inspect the job in AXIOM for local details."
            if status == "failed"
            else None
        ),
    }


def _training_job_scope() -> dict:
    return {
        "dataset_values_returned": False,
        "logs_returned": False,
        "absolute_paths_returned": False,
        "network_accessed": False,
    }


def _resolve_engine_workspace_asset(path: Path, label: str) -> Path:
    try:
        return Path(resolve_local_model_directory(str(path), label))
    except ValueError:
        # A just-finished LoRA job may not be in the engine's short-lived index yet.
        refresh_local_engine_asset_index()
        return Path(resolve_local_model_directory(str(path), label))


def _safe_preflight_reason(reason: str, known_paths: tuple[str, ...] = ()) -> str:
    text = str(reason)
    for path in sorted((item for item in known_paths if item), key=len, reverse=True):
        text = text.replace(path, "[workspace path]")
    text = re.sub(
        r"(?i)(?<!\w)(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+[\\/]|/)[^\s,;]+",
        "[local path]",
        text,
    )
    return text[:512]


def _engine_preflight_error(
    check: str,
    reason: str,
    *,
    known_paths: tuple[str, ...] = (),
) -> dict:
    return {
        "schema_version": 1,
        "assessment": "blocked",
        "checks": {
            "model": {"status": "not_checked"},
            "adapter": {"status": "not_requested"},
            "runtime": {"status": "not_checked"},
            "resource_estimate": {"status": "not_checked"},
            check: {"status": "blocked", "reason": _safe_preflight_reason(reason, known_paths)},
        },
        "scope": {
            "read_only": True,
            "network_accessed": False,
            "files_written": False,
            "model_weights_loaded": False,
            "inference_run": False,
            "absolute_paths_returned": False,
        },
    }


@mcp.tool()
def axiom_engine_preflight(
    model_path: str,
    adapter_path: str | None = None,
    max_prompt_tokens: int = 4096,
    max_new_tokens: int = 128,
    device: str = "auto",
    precision: str = "auto",
    ram_budget_gib: float | None = None,
    vram_budget_gib: float | None = None,
) -> dict:
    """Check native model/adapter metadata and advisory memory estimates without inference."""
    ToolPermissionStore().require("axiom_engine_preflight")
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or not 1 <= max_prompt_tokens <= 8192
    ):
        return {"error": "max_prompt_tokens must be an integer from 1 through 8192."}
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= _MCP_ENGINE_MAX_NEW_TOKENS
    ):
        return {"error": f"max_new_tokens must be an integer from 1 through {_MCP_ENGINE_MAX_NEW_TOKENS}."}
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}:
        return {"error": "device must be auto, cpu, cuda, rocm, mps, or xpu."}
    if not isinstance(precision, str) or precision not in {"auto", "fp32", "fp16", "bf16"}:
        return {"error": "precision must be auto, fp32, fp16, or bf16."}
    for label, budget in (("RAM", ram_budget_gib), ("VRAM", vram_budget_gib)):
        if budget is not None and (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not 0.1 <= budget <= 2048
            or not math.isfinite(float(budget))
        ):
            return {"error": f"{label} budget must be between 0.1 and 2048 GiB."}

    root = _mcp_workspace_root()
    try:
        model_target = resolve_workspace_input(model_path, root, "Model path")
        if not model_target.is_dir():
            raise ValueError("Model path must refer to a local model directory.")
        validate_local_model_tree(model_target)
        trusted_model = _resolve_engine_workspace_asset(model_target, "Base model")
        if trusted_model != model_target:
            raise ValueError("Model path changed during validation.")
    except (AxiomEngineError, FineTuningError, OSError, RuntimeError, ValueError) as exc:
        return _engine_preflight_error("model", str(exc), known_paths=(str(root),))

    trusted_adapter: Path | None = None
    if adapter_path is not None:
        try:
            adapter_target = resolve_workspace_input(adapter_path, root, "Adapter path")
            if not adapter_target.is_dir():
                raise ValueError("Adapter path must refer to a local adapter directory.")
            validate_local_model_tree(adapter_target)
            trusted_adapter = _resolve_engine_workspace_asset(adapter_target, "LoRA adapter")
            if trusted_adapter != adapter_target:
                raise ValueError("Adapter path changed during validation.")
        except (AxiomEngineError, FineTuningError, OSError, RuntimeError, ValueError) as exc:
            result = _engine_preflight_error("adapter", str(exc), known_paths=(str(root),))
            result["checks"]["model"] = {"status": "path_validated"}
            return result

    total_context_tokens = max_prompt_tokens + max_new_tokens
    try:
        model_context = inspect_native_model_context(str(trusted_model))
    except (AxiomEngineError, OSError, RuntimeError, ValueError) as exc:
        return _engine_preflight_error(
            "model", str(exc), known_paths=(str(root), str(trusted_model))
        )
    effective_context_limit = model_context["effective_max_position_embeddings"]
    if total_context_tokens > effective_context_limit:
        result = _engine_preflight_error(
            "request_context",
            f"Requested total context is {total_context_tokens} tokens; this checkpoint supports at most {effective_context_limit}.",
        )
        result["request"] = {
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "requested_total_context_tokens": total_context_tokens,
        }
        result["checks"]["model"] = {
            "status": "config_valid",
            "model_type": model_context["model_type"],
            "architecture": model_context["architecture"],
            "checkpoint_format": model_context["checkpoint_format"],
            "effective_max_position_embeddings": effective_context_limit,
        }
        result["checks"]["adapter"] = {
            "status": "path_validated" if trusted_adapter is not None else "not_requested"
        }
        result["checks"]["request_context"] = {
            "status": "blocked",
            "requested_total_tokens": total_context_tokens,
            "supported_max_total_tokens": effective_context_limit,
            "suggestion": "Reduce max_prompt_tokens, max_new_tokens, or both so their sum fits the checkpoint context limit.",
        }
        return result

    try:
        estimate = estimate_native_quantization(
            str(trusted_model),
            context_length=total_context_tokens,
            batch_size=1,
            ram_budget_gib=ram_budget_gib,
            vram_budget_gib=vram_budget_gib,
        )
    except (AxiomEngineError, OSError, RuntimeError, ValueError) as exc:
        return _engine_preflight_error(
            "model", str(exc), known_paths=(str(root), str(trusted_model))
        )

    runtime = runtime_status()
    resolved_device = runtime.get("recommended_device") if device == "auto" else device
    runtime_devices = runtime.get("devices", [])
    runtime_devices = runtime_devices if isinstance(runtime_devices, list) else []
    selected_runtime_device = next(
        (item for item in runtime_devices if isinstance(item, dict) and item.get("id") == resolved_device),
        None,
    )
    selected_device_available = bool(selected_runtime_device and selected_runtime_device.get("available"))
    request_check = {
        "status": "pass" if selected_device_available else "review_needed",
        "device": device,
        "resolved_device": resolved_device,
        "precision": precision,
    }
    if device != "auto" and not selected_device_available:
        request_check.update({
            "status": "blocked",
            "reason": "The requested execution device is not available in the current AXIOM runtime.",
        })
    if precision == "bf16" and resolved_device == "mps":
        request_check.update({
            "status": "blocked",
            "reason": "The AXIOM native engine does not support BF16 precision on Metal/MPS.",
        })
    if precision == "bf16" and resolved_device == "mps":
        precision_check = {
            "status": "blocked",
            "reason": "The AXIOM native engine does not support BF16 precision on Metal/MPS.",
        }
    elif precision == "auto":
        precision_check = {
            "status": "unknown",
            "reason": "Auto precision follows the checkpoint dtype; the stored dtype has not been inspected by this estimate.",
        }
    elif precision == "bf16" and resolved_device in {"cuda", "xpu"}:
        precision_check = {
            "status": "review_needed",
            "reason": "BF16 support depends on the installed device backend and was not measured by this preflight.",
        }
    else:
        precision_check = {
            "status": "pass" if selected_device_available else "unknown",
            "precision": precision,
            "reason": "The requested precision is an AXIOM native precision option; checkpoint loading and device kernels remain unverified.",
        }
    model_issues: list[str] = []
    adapter_issues: list[str] = []
    adapter_check: dict = {"status": "not_requested", "compatible_on_selected_device": None}
    base_tensor_inventory_checked = False
    selected_adapter_compatible: bool | None = None
    compatibility: dict | None = None
    raw_model_issues: list[str] = []
    if trusted_adapter is not None and model_context.get("checkpoint_format") == "gguf":
        adapter_check = {
            "status": "not_checked",
            "compatible_on_selected_device": None,
            "reason": (
                "The metadata-only preflight cannot compare LoRA shapes with GGUF tensor metadata. "
                "AXIOM's native loader checks adapter shapes when inference is explicitly run."
            ),
        }
    elif trusted_adapter is not None:
        try:
            compatibility = check_local_adapter_compatibility(
                str(trusted_model), str(trusted_adapter)
            )
        except (AxiomEngineError, OSError, RuntimeError, ValueError) as exc:
            adapter_issues = [_safe_preflight_reason(str(exc), (str(root), str(trusted_model), str(trusted_adapter)))]
            adapter_check = {
                "status": "blocked",
                "structurally_compatible": False,
                "compatible_on_any_device": False,
                "compatible_on_selected_device": False,
                "issues": adapter_issues,
                "issue_count": 1,
                "issues_truncated": False,
            }
        if compatibility is not None:
            base_tensor_inventory_checked = True
            issues = compatibility.get("issues", [])
            issues = [item for item in issues if isinstance(item, str)] if isinstance(issues, list) else []
            raw_model_issues = [
                item for item in issues
                if item.startswith("Base model:") or item.startswith("Base checkpoint")
            ]
            raw_adapter_issues = [item for item in issues if item not in raw_model_issues]
            model_issues = [
                _safe_preflight_reason(item, (str(root), str(trusted_model), str(trusted_adapter)))
                for item in raw_model_issues[:32]
            ]
            adapter_issues = [
                _safe_preflight_reason(item, (str(root), str(trusted_model), str(trusted_adapter)))
                for item in raw_adapter_issues[:32]
            ]
            compatibility_devices = compatibility.get("devices", [])
            if not isinstance(compatibility_devices, list):
                compatibility_devices = []
            compatibility_device_name = {
                "cpu": "CPU",
                "cuda": "CUDA",
                "rocm": "ROCm",
                "mps": "Metal/MPS",
                "xpu": "Intel XPU",
            }.get(resolved_device)
            selected_compatibility_device = next(
                (
                    item for item in compatibility_devices
                    if isinstance(item, dict) and item.get("name") == compatibility_device_name
                ),
                None,
            )
            selected_adapter_compatible = bool(
                selected_compatibility_device and selected_compatibility_device.get("compatible")
            )
            adapter_metadata = compatibility.get("adapter")
            adapter_metadata = adapter_metadata if isinstance(adapter_metadata, dict) else {}
            safe_target_modules = adapter_metadata.get("target_modules", [])
            if not isinstance(safe_target_modules, list):
                safe_target_modules = []
            safe_target_modules = sorted({
                item for item in safe_target_modules
                if item in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
            })
            adapter_check = {
                "status": "pass" if compatibility.get("structurally_compatible") is True else "blocked",
                "structurally_compatible": compatibility.get("structurally_compatible") is True,
                "compatible_on_any_device": compatibility.get("compatible_on_any_device") is True,
                "compatible_on_selected_device": selected_adapter_compatible,
                "rank": adapter_metadata.get("rank"),
                "target_modules": safe_target_modules,
                "issues": adapter_issues,
                "issue_count": len(raw_adapter_issues),
                "issues_truncated": len(raw_adapter_issues) > 32,
                "devices": [
                    {
                        key: device.get(key)
                        for key in ("name", "available", "compatible", "status", "reason")
                        if key in device
                    }
                    for device in compatibility_devices
                    if isinstance(device, dict)
                ],
            }

    supported_candidates = [
        candidate
        for candidate in estimate.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("native_checkpoint_supported") is True
    ]
    supplied_budget = ram_budget_gib is not None or vram_budget_gib is not None
    budget_fitting_candidates = (
        [
            candidate
            for candidate in supported_candidates
            if (ram_budget_gib is None or candidate.get("fits_ram_budget") is True)
            and (vram_budget_gib is None or candidate.get("fits_vram_budget") is True)
        ]
        if supplied_budget
        else []
    )
    fitting_candidates = (
        budget_fitting_candidates
        if precision == "auto"
        else [candidate for candidate in budget_fitting_candidates if candidate.get("format") == precision]
    )
    accelerator_devices = {"cuda", "rocm", "xpu"}
    relevant_budget_available = (
        ram_budget_gib is not None
        if resolved_device in {"cpu", "mps"}
        else vram_budget_gib is not None
        if resolved_device in accelerator_devices
        else False
    )
    if precision == "auto" and supplied_budget:
        resource_status = "review_needed"
    elif fitting_candidates and relevant_budget_available:
        resource_status = "estimated_fit"
    elif supplied_budget:
        resource_status = "review_needed"
    else:
        resource_status = "unknown"
    resource_check = {
        "status": resource_status,
        "budget_assumptions_gib": {
            "ram": ram_budget_gib,
            "vram": vram_budget_gib,
        },
        "unbudgeted_dimensions": [
            dimension
            for dimension, budget in (("ram", ram_budget_gib), ("vram", vram_budget_gib))
            if budget is None
        ],
        "resolved_device": resolved_device,
        "selected_precision": precision,
        "budget_relevant_to_selected_device": relevant_budget_available,
        "estimated_native_precision_candidates": [
            {
                key: candidate.get(key)
                for key in (
                    "format",
                    "estimated_weight_gib",
                    "estimated_kv_cache_gib",
                    "estimated_workspace_gib",
                    "estimated_total_gib",
                    "estimated_range_gib",
                    "fits_ram_budget",
                    "fits_vram_budget",
                )
            }
            for candidate in supported_candidates
        ],
        "formats_within_supplied_budgets": [candidate.get("format") for candidate in fitting_candidates],
        "note": (
            "Advisory estimate only. It excludes adapter overhead, does not measure free memory, and does not account "
            "for every device or memory mode. CPU and Metal/MPS budgeting uses RAM because separate device-memory "
            "telemetry is unavailable for those paths. Auto precision follows the checkpoint dtype, so its actual candidate "
            "fit remains unknown. A fit does not guarantee successful loading or inference."
        ),
    }

    model_check = {
        "status": "blocked" if model_issues else "config_valid",
        "model_type": (estimate.get("model") or {}).get("model_type"),
        "architecture": (estimate.get("model") or {}).get("architecture"),
        "checkpoint_format": model_context["checkpoint_format"],
        "effective_max_position_embeddings": effective_context_limit,
        "estimated_parameter_count": (estimate.get("model") or {}).get("estimated_parameter_count"),
        "base_tensor_inventory_checked": base_tensor_inventory_checked,
        "issues": model_issues,
        "issue_count": len(raw_model_issues) if trusted_adapter is not None and compatibility is not None else len(model_issues),
        "issues_truncated": len(raw_model_issues) > 32 if trusted_adapter is not None and compatibility is not None else False,
    }
    runtime_check = {
        "status": "available" if runtime.get("ready") is True else "unavailable",
        "ready": runtime.get("ready") is True,
        "recommended_device": runtime.get("recommended_device"),
        "resolved_device": resolved_device,
        "selected_device_available": selected_device_available,
        "problems": runtime.get("problems", []),
        "devices": [
            {
                key: device.get(key)
                for key in ("id", "name", "available", "reason", "memory_free_gib", "memory_total_gib")
                if key in device
            }
            for device in runtime.get("devices", [])
            if isinstance(device, dict)
        ],
    }

    structurally_clear = not model_issues and (
        trusted_adapter is None or adapter_check.get("status") in {"pass", "not_checked"}
    )
    if not structurally_clear or request_check.get("status") == "blocked":
        assessment = "blocked"
    elif (
        trusted_adapter is not None
        and runtime_check["ready"]
        and selected_device_available
        and selected_adapter_compatible is True
        and precision_check.get("status") == "pass"
        and resource_status == "estimated_fit"
    ):
        assessment = "preflight_clear"
    else:
        assessment = "review_needed"

    return {
        "schema_version": 1,
        "assessment": assessment,
        "request": {
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "estimated_total_context_tokens": total_context_tokens,
            "device": device,
            "resolved_device": resolved_device,
            "precision": precision,
        },
        "checks": {
            "model": model_check,
            "adapter": adapter_check,
            "runtime": runtime_check,
            "request": request_check,
            "precision": precision_check,
            "resource_estimate": resource_check,
        },
        "scope": {
            "read_only": True,
            "network_accessed": False,
            "files_written": False,
            "model_weights_loaded": False,
            "adapter_tensor_values_loaded": False,
            "inference_run": False,
            "absolute_paths_returned": False,
            "memory_estimate_advisory": True,
            "mcp_host_transcript_retention": "controlled by the MCP host",
        },
        "scope_note": (
            "Preflight checks local model metadata and, for safetensors bases with an adapter, base tensor names "
            "plus adapter tensor shapes. GGUF adapter shape compatibility is not checked until AXIOM's native loader "
            "runs. Preflight does not load base-model weights or execute inference. Actual allocation, checkpoint dtype, "
            "device kernels, free memory, and output quality remain unverified."
        ),
    }
@mcp.tool()
def axiom_training_job_start(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    text_field: str = "text",
    max_steps: int = 20,
    sequence_length: int = 512,
    learning_rate: float = 0.0001,
    lora_rank: int = 8,
) -> dict:
    """Start a bounded local LoRA job using workspace inputs and an outputs/ destination."""
    ToolPermissionStore().require("axiom_training_job_start")
    root = _mcp_workspace_root()
    try:
        model_target = resolve_workspace_input(model_path, root, "Model path")
        dataset_target = resolve_workspace_input(dataset_path, root, "Dataset path")
        output_target = resolve_workspace_output(output_dir, root)
        if not model_target.is_dir():
            raise ValueError("Model path must refer to a local model directory.")
        if not dataset_target.is_file() or dataset_target.suffix.lower() != ".jsonl":
            raise ValueError("Dataset path must refer to a workspace .jsonl file.")
        created = start_finetuning_job({
            "model_path": str(model_target),
            "dataset_path": str(dataset_target),
            "output_dir": str(output_target),
            "text_field": text_field,
            "max_steps": max_steps,
            "sequence_length": sequence_length,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
        }, workspace_guard={
            "root": str(root),
            "paths": {
                "model_path": model_path,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
            },
        })
    except ActiveFineTuningJobError:
        return {"error": "Another local fine-tuning job is active; inspect its status before starting a new job."}
    except (FineTuningError, OSError, RuntimeError, ValueError):
        return {"error": "The local training job could not be started; check the workspace paths, dataset, and training runtime."}
    return {
        "job": _training_job_summary(created, root),
        "scope": {
            **_training_job_scope(),
            "job_created": True,
            "job_record_written_locally": True,
            "adapter_written": created.get("status") == "completed",
            "network_accessed": False,
            "model_downloaded": False,
            "remote_code_executed": False,
            "adapter_destination_under": "outputs/",
            "max_steps": 1000,
            "cancellation_available_until": "adapter finalization begins",
        },
    }


@mcp.tool()
def axiom_training_job_status(job_id: str | None = None, limit: int = 10) -> dict:
    """Read safe local LoRA job summaries for this workspace."""
    ToolPermissionStore().require("axiom_training_job_status")
    root = _mcp_workspace_root()
    if job_id is not None:
        try:
            job = get_job(job_id)
        except (FineTuningError, ValueError):
            job = None
        if job is None or not _training_job_matches_workspace(job, root):
            return {"jobs": [], "not_found": True, "scope": _training_job_scope()}
        return {"jobs": [_training_job_summary(job, root)], "scope": _training_job_scope()}
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 25:
        raise ValueError("Job list limit must be between 1 and 25.")
    jobs = list_jobs(limit=limit, workspace_root=root)
    jobs = [job for job in jobs if _training_job_matches_workspace(job, root)]
    return {"jobs": [_training_job_summary(job, root) for job in jobs], "scope": _training_job_scope()}


@mcp.tool()
def axiom_training_job_cancel(job_id: str) -> dict:
    """Request safe-boundary cancellation for a local LoRA job in this workspace."""
    ToolPermissionStore().require("axiom_training_job_cancel")
    root = _mcp_workspace_root()
    try:
        existing = get_job(job_id)
    except (FineTuningError, ValueError):
        existing = None
    if existing is None or not _training_job_matches_workspace(existing, root):
        return {"job": None, "not_found": True, "scope": _training_job_scope()}
    try:
        job = cancel_finetuning_job(job_id)
    except (FineTuningError, ValueError):
        job = None
    if job is None or not _training_job_matches_workspace(job, root):
        return {"job": None, "not_found": True, "scope": _training_job_scope()}
    status = job.get("status")
    if status in {"queued", "running", "cancelling"}:
        note = "Cancellation is cooperative and completes at the next safe training boundary."
    elif status == "finalizing":
        note = "Adapter finalization has started; cancellation is no longer available."
    elif status == "cancelled":
        note = "The job has already been cancelled."
    elif status == "completed":
        note = "The job completed before this cancellation request."
    elif status == "failed":
        note = "The job failed before this cancellation request."
    else:
        note = "Cancellation is unavailable for the current job state."
    return {
        "job": _training_job_summary(job, root),
        "scope": _training_job_scope(),
        "cancellation_accepted": status in {"queued", "running", "cancelling"},
        "note": note,
    }


@mcp.tool()
def axiom_engine_generate(
    model_path: str,
    prompt: str,
    adapter_path: str | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    seed: int | None = None,
    device: str = "auto",
    precision: str = "auto",
) -> dict:
    """Generate one bounded response; this synchronous call cannot be cancelled through MCP."""
    ToolPermissionStore().require("axiom_engine_generate")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > _MCP_ENGINE_MAX_PROMPT_CHARS:
        return {"error": f"Prompt must contain 1 to {_MCP_ENGINE_MAX_PROMPT_CHARS:,} characters."}
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= _MCP_ENGINE_MAX_NEW_TOKENS
    ):
        return {"error": f"max_new_tokens must be between 1 and {_MCP_ENGINE_MAX_NEW_TOKENS}."}
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
        or not math.isfinite(float(temperature))
    ):
        return {"error": "temperature must be between 0 and 2."}
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}:
        return {"error": "device must be auto, cpu, cuda, rocm, mps, or xpu."}
    if not isinstance(precision, str) or precision not in {"auto", "fp32", "fp16", "bf16"}:
        return {"error": "precision must be auto, fp32, fp16, or bf16."}
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1
    ):
        return {"error": "seed must be an integer between 0 and 2^63-1."}

    root = _mcp_workspace_root()
    try:
        model_target = resolve_workspace_input(model_path, root, "Model path")
        if not model_target.is_dir():
            raise ValueError("Model path must refer to a directory.")
        validate_local_model_tree(model_target)
        trusted_model = _resolve_engine_workspace_asset(model_target, "Base model")
        if trusted_model != model_target:
            raise ValueError("Model path changed during validation.")

        trusted_adapter = None
        adapter_target = None
        if adapter_path is not None:
            adapter_target = resolve_workspace_input(adapter_path, root, "Adapter path")
            if not adapter_target.is_dir():
                raise ValueError("Adapter path must refer to a directory.")
            validate_local_model_tree(adapter_target)
            trusted_adapter = _resolve_engine_workspace_asset(adapter_target, "LoRA adapter")
            if trusted_adapter != adapter_target:
                raise ValueError("Adapter path changed during validation.")

        result = generate_local(
            str(trusted_model),
            prompt,
            adapter_path=str(trusted_adapter) if trusted_adapter is not None else None,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            seed=seed,
            device_preference=device,
            precision=precision,
        )
    except (AxiomEngineError, OSError, RuntimeError, ValueError):
        return {"error": "Local generation could not run; check model compatibility, adapter paths, and available memory."}

    response = result.get("response")
    if not isinstance(response, str):
        return {"error": "Local generation ended without a text response."}
    context = result.get("context_management")
    context = context if isinstance(context, dict) else {}
    return {
        "response": response,
        "model": trusted_model.name,
        "adapter": adapter_target.name if adapter_target is not None else None,
        "prompt_tokens": result.get("prompt_tokens"),
        "prompt_truncated": context.get("token_truncation_applied") is True,
        "prompt_tokens_before_truncation": context.get("prompt_tokens_before_token_truncation"),
        "prompt_tokens_used": context.get("prompt_tokens_used"),
        "completion_tokens": result.get("completion_tokens"),
        "stop_reason": result.get("stop_reason"),
        "elapsed_ms": result.get("elapsed_ms"),
        "device": result.get("device"),
        "precision": result.get("precision"),
        "memory_mode": result.get("memory_mode"),
        "scope": {
            "engine": "axiom-native-engine",
            "network_accessed": False,
            "axiom_prompt_persisted": False,
            "axiom_response_persisted": False,
            "mcp_host_transcript_retention": "controlled by the MCP host",
            "prompt_returned": False,
            "response_returned": True,
            "cancellation_supported": False,
            "wall_clock_timeout_seconds": None,
            "max_prompt_characters": _MCP_ENGINE_MAX_PROMPT_CHARS,
            "max_new_tokens": _MCP_ENGINE_MAX_NEW_TOKENS,
        },
    }


@mcp.tool()
def axiom_model_list() -> list[dict]:
    """List models registered with AXIOM."""
    ToolPermissionStore().require("axiom_model_list")
    registry = ModelRegistry(create=False)

    return [model.to_dict() for model in registry.list()]


@mcp.tool()
def axiom_model_info(path: str) -> dict:
    """Inspect a local AI model directory."""
    ToolPermissionStore().require("axiom_model_info")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Model path cannot be empty.")

    result = inspect_model(path)

    return {
        "path": str(result.path),
        "architecture": result.architecture,
        "model_type": result.model_type,
        "parameters": result.parameter_count,
        "parameter_source": result.parameter_source,
        "format": result.model_format,
        "weight_size_bytes": result.weight_size_bytes,
        "has_config": result.has_config,
        "has_tokenizer": result.has_tokenizer,
        "has_safetensors": result.has_safetensors,
        "has_gguf": result.has_gguf,
        "has_pytorch_weights": result.has_pytorch_weights,
        "capabilities": result.capabilities,
        "estimated_vram_gb": result.estimated_vram_gb,
        "status": result.status,
    }


@mcp.tool()
def axiom_dataset_inspect(
    path: str,
    duplicate_key_fields: list[str] | None = None,
    duplicate_rate_threshold: float | None = None,
    text_fields: list[str] | None = None,
    required_fields: list[str] | None = None,
    expected_types: dict[str, str] | None = None,
    scan_pii: bool = False,
) -> dict:
    """Profile bounded workspace JSONL; content PII scanning is opt-in."""
    ToolPermissionStore().require("axiom_dataset_inspect")
    if not isinstance(path, str) or not path:
        raise ValueError("Dataset path must be workspace-relative.")
    if type(scan_pii) is not bool:
        raise ValueError("scan_pii must be a boolean; PII scanning is opt-in.")
    duplicate_key_fields = [] if duplicate_key_fields is None else duplicate_key_fields
    text_fields = [] if text_fields is None else text_fields
    required_fields = [] if required_fields is None else required_fields
    expected_types = {} if expected_types is None else expected_types
    if any(
        not isinstance(fields, list)
        for fields in (duplicate_key_fields, text_fields, required_fields)
    ):
        raise ValueError("Dataset profile field options must be lists of field names.")
    if any(
        len(fields) > 100
        or any(
            not isinstance(name, str) or not name.strip() or len(name) > 160
            for name in fields
        )
        for fields in (duplicate_key_fields, text_fields)
    ):
        raise ValueError("Profile fields must contain at most 100 non-empty names of 160 characters or fewer.")
    for label, fields in (
        ("duplicate_key_fields", duplicate_key_fields),
        ("text_fields", text_fields),
    ):
        normalized_fields = {name.strip() for name in fields}
        if len(normalized_fields) != len(fields):
            raise ValueError(f"{label} must contain unique names after trimming.")
    if duplicate_rate_threshold is not None:
        if isinstance(duplicate_rate_threshold, bool) or not isinstance(
            duplicate_rate_threshold, (int, float)
        ):
            raise ValueError("duplicate_rate_threshold must be a finite number between 0 and 1.")
        try:
            duplicate_rate_threshold = float(duplicate_rate_threshold)
        except (OverflowError, ValueError):
            duplicate_rate_threshold = float("nan")
        if not math.isfinite(duplicate_rate_threshold) or not 0 <= duplicate_rate_threshold <= 1:
            raise ValueError("duplicate_rate_threshold must be a finite number between 0 and 1.")
    if not isinstance(expected_types, dict) or len(expected_types) > 100:
        raise ValueError("expected_types must be an object with at most 100 field names.")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 160
        or not isinstance(value_type, str)
        or value_type not in {"string", "integer", "number", "boolean", "object", "array"}
        for name, value_type in expected_types.items()
    ):
        raise ValueError("Expected types must use supported JSON types and non-empty field names.")
    normalized_expected_types = {
        name.strip(): value_type for name, value_type in expected_types.items()
    }
    if len(normalized_expected_types) != len(expected_types):
        raise ValueError("Expected-type field names must be unique after trimming.")
    if any(
        not isinstance(name, str) or not name.strip() or len(name) > 160
        for name in required_fields
    ) or len(required_fields) > 100:
        raise ValueError("required_fields must contain at most 100 non-empty field names of 160 characters or fewer.")
    required_fields = [name.strip() for name in required_fields]
    if len(set(required_fields)) != len(required_fields):
        raise ValueError("required_fields must be unique after trimming.")

    root = Path.cwd().resolve(strict=True)
    source, source_stream = open_mcp_workspace_file(path, root)
    if source.suffix.lower() != ".jsonl":
        source_stream.close()
        raise ValueError("Dataset profiling supports workspace-relative .jsonl files only.")
    try:
        with source_stream:
            initial = os.fstat(source_stream.fileno())
            profile = profile_jsonl_stream(
                source_stream,
                file_name=source.name,
                file_size_bytes=initial.st_size,
                max_bytes=DEFAULT_MAX_BYTES,
                max_rows=DEFAULT_MAX_ROWS,
                max_line_bytes=DEFAULT_MAX_LINE_BYTES,
                duplicate_key_fields=duplicate_key_fields,
                duplicate_rate_threshold=duplicate_rate_threshold,
                text_fields=text_fields,
                scan_pii=scan_pii,
            )
            final = os.fstat(source_stream.fileno())
        current_path, current_stream = open_mcp_workspace_file(path, root)
        with current_stream:
            current = os.fstat(current_stream.fileno())
        initial_identity = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
        final_identity = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        if initial_identity != final_identity or initial_identity != current_identity:
            raise ValueError("Dataset changed during profiling; retry against a stable file.")
        if current_path.relative_to(root).as_posix() != path:
            raise ValueError("Dataset path changed during profiling; retry against a stable file.")
    except (OSError, ValueError, RecursionError):
        source_stream.close()
        raise ValueError("Dataset could not be profiled safely.") from None

    report = {
        "path": source.relative_to(root).as_posix(),
        "profile": profile,
        "scope": {
            "read_only": True,
            "network_accessed": False,
            "files_written": False,
            "absolute_paths_returned": False,
            "dataset_values_returned": False,
            "pii_scan_opted_in": scan_pii,
        },
    }
    if required_fields or expected_types:
        validation = validate_dataset_profile(
            profile,
            required_fields=required_fields,
            expected_types=normalized_expected_types,
        )
        validation_incomplete = (
            profile["truncated"]
            or profile["invalid_rows"] > 0
            or profile["truncated_field_names"] > 0
            or profile["omitted_field_occurrences"] > 0
        )
        validation_status = (
            "inconclusive"
            if validation_incomplete
            else "passed" if validation["passed"] else "failed"
        )
        report["validation"] = {
            **validation,
            "status": validation_status,
            "scope": "profiled_rows_only",
            "required_field_semantics": "present_at_least_once_not_every_row",
        }
        if validation_incomplete:
            report["validation"]["passed"] = None
            report["validation"]["observed_profile_passed"] = validation["passed"]
            if profile["truncated"]:
                reason = "profile_truncated"
            elif profile["invalid_rows"]:
                reason = "invalid_rows"
            elif profile["truncated_field_names"]:
                reason = "field_names_truncated"
            else:
                reason = "field_tracking_limit"
            report["validation"]["reason"] = reason
    return report


@mcp.tool()
def axiom_dataset_clean_preview(
    path: str,
    output: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
) -> dict:
    """Preview bounded JSONL cleaning counts without returning row values or writing files."""
    ToolPermissionStore().require("axiom_dataset_clean_preview")
    if output is not None and (not isinstance(output, str) or not output.strip()):
        raise ValueError("Output path cannot be empty.")
    root = Path.cwd().resolve(strict=True)
    source, source_stream = open_mcp_workspace_file(path, root)
    if source.suffix.lower() != ".jsonl":
        source_stream.close()
        raise ValueError("Cleaning previews currently support .jsonl files.")
    try:
        _validate_cleaning_limits(max_bytes, max_rows, max_line_bytes)
    except ValueError:
        source_stream.close()
        raise

    relative_output = (
        output
        if output
        else source.relative_to(root).with_name(f"{source.stem}.cleaned.jsonl").as_posix()
    )
    if not relative_output.lower().endswith(".jsonl"):
        source_stream.close()
        raise ValueError("Proposed cleaned output paths must end in .jsonl.")
    try:
        output_path, output_parent_fd, output_name, output_available = (
            open_mcp_workspace_output_parent(relative_output, root)
        )
    except (OSError, ValueError):
        source_stream.close()
        raise
    try:
        with source_stream:
            source_stat = os.fstat(source_stream.fileno())
            if not output_available:
                output_stat = (
                    os.stat(
                        output_name,
                        dir_fd=output_parent_fd,
                        follow_symlinks=False,
                    )
                    if output_parent_fd is not None
                    else os.stat(output_path, follow_symlinks=False)
                )
                if (output_stat.st_ino, output_stat.st_dev) == (
                    source_stat.st_ino,
                    source_stat.st_dev,
                ):
                    raise ValueError("Output path must differ from the source dataset.")
            preview = preview_jsonl_cleaning(
                source_stream,
                file_name=source.name,
                file_size_bytes=source_stat.st_size,
                max_bytes=max_bytes,
                max_rows=max_rows,
                max_line_bytes=max_line_bytes,
            )
    except (OSError, ValueError):
        if output_parent_fd is not None:
            os.close(output_parent_fd)
        raise ValueError("The dataset cleaning preview could not be created safely.") from None
    if output_parent_fd is not None:
        os.close(output_parent_fd)

    preview["path"] = source.relative_to(root).as_posix()
    preview["proposed_output_path"] = output_path.relative_to(root).as_posix()
    preview["proposed_output_available"] = output_available
    if not output_available:
        preview["warnings"].append(
            "The proposed output already exists; choose a new path because the cleaner will not overwrite it."
        )
    preview["source_modified"] = False
    preview["dataset_values_returned"] = False
    return preview


@mcp.tool()
def axiom_dataset_clean(
    path: str,
    output: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
) -> dict:
    """Write a cleaned JSONL copy to a new workspace-relative path."""
    ToolPermissionStore().require("axiom_dataset_clean")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Dataset path cannot be empty.")
    if output is not None and (not isinstance(output, str) or not output.strip()):
        raise ValueError("Output path cannot be empty.")
    _validate_cleaning_limits(max_bytes, max_rows, max_line_bytes)

    root = Path.cwd().resolve(strict=True)
    source, source_stream = open_mcp_workspace_file(path, root)
    if source.suffix.lower() != ".jsonl":
        source_stream.close()
        raise ValueError("Cleaning currently supports .jsonl files.")

    relative_output = (
        output
        if output
        else source.relative_to(root).with_name(f"{source.stem}.cleaned.jsonl").as_posix()
    )
    try:
        destination, output_parent_fd, output_name, output_available = (
            open_mcp_workspace_output_parent(relative_output, root)
        )
    except (OSError, ValueError):
        source_stream.close()
        raise
    if destination.suffix.lower() != ".jsonl":
        source_stream.close()
        if output_parent_fd is not None:
            os.close(output_parent_fd)
        raise ValueError("Cleaned output paths must end in .jsonl.")
    if not output_available:
        source_stream.close()
        if output_parent_fd is not None:
            os.close(output_parent_fd)
        raise ValueError("Choose a new output path; existing files are never overwritten.")

    try:
        source_stat = os.fstat(source_stream.fileno())
        clean_preview = preview_jsonl_cleaning(
            source_stream,
            file_name=source.name,
            file_size_bytes=source_stat.st_size,
            max_bytes=max_bytes,
            max_rows=max_rows,
            max_line_bytes=max_line_bytes,
        )
    except (OSError, ValueError):
        source_stream.close()
        if output_parent_fd is not None:
            os.close(output_parent_fd)
        raise ValueError("The dataset could not be previewed within the cleaning limits.") from None
    if clean_preview["truncated"] or not clean_preview["cleaner_semantics_match"]:
        source_stream.close()
        if output_parent_fd is not None:
            os.close(output_parent_fd)
        raise ValueError(
            "Cleaning requires a complete preview with matching cleaner semantics; "
            "raise the limits or correct the dataset first."
        )

    try:
        source_stream.seek(0)
        result = clean_jsonl(
            source,
            destination,
            overwrite=False,
            source_stream=source_stream,
            output_parent_fd=output_parent_fd,
            output_name=output_name if output_parent_fd is not None else None,
            max_bytes=max_bytes,
            max_rows=max_rows,
            max_line_bytes=max_line_bytes,
        )
    except (OSError, ValueError):
        source_stream.close()
        raise ValueError(
            "The cleaned dataset could not be written within the bounded preview limits."
        ) from None
    finally:
        if output_parent_fd is not None:
            os.close(output_parent_fd)

    return {
        "source": result.source.relative_to(root).as_posix(),
        "output": result.output.relative_to(root).as_posix(),
        "total_lines": result.total_lines,
        "kept": result.kept,
        "removed_invalid": result.removed_invalid,
        "removed_empty": result.removed_empty,
        "removed_duplicates": result.removed_duplicates,
        "removed_non_objects": result.removed_non_objects,
        "removed_schema": result.removed_schema,
    }


@mcp.tool()
def axiom_system_info() -> dict:
    """Return detected system hardware."""
    ToolPermissionStore().require("axiom_system_info")
    hardware = detect_hardware()

    return {
        "os": hardware.os_name,
        "architecture": hardware.architecture,
        "cpu_cores": hardware.cpu_cores,
        "ram_gb": hardware.ram_gb,
        "gpu": hardware.gpu_name,
        "vram_gb": hardware.vram_gb,
        "cuda_available": hardware.cuda_available,
    }


@mcp.tool()
def axiom_training_plan(
    parameters_billions: float,
    method: str = "auto",
) -> dict:
    """Generate a hardware-aware training plan."""
    ToolPermissionStore().require("axiom_training_plan")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("Training method cannot be empty.")

    hardware = detect_hardware()

    plan = create_training_plan(
        parameter_billions=parameters_billions,
        hardware=hardware,
        method=method.strip().lower(),
    )

    return {
        "method": plan.method,
        "precision": plan.precision,
        "lora_rank": plan.lora_rank,
        "batch_size": plan.batch_size,
        "gradient_accumulation": plan.gradient_accumulation,
        "sequence_length": plan.sequence_length,
        "learning_rate": plan.learning_rate,
        "estimated_vram_gb": plan.estimated_vram_gb,
        "fits_hardware": plan.fits_hardware,
        "reason": plan.reason,
    }


@mcp.tool()
def axiom_supercompress_status() -> dict:
    """Check whether SuperCompress is configured."""
    ToolPermissionStore().require("axiom_supercompress_status")

    return {
        "configured": bool(os.getenv("SUPERCOMPRESS_API_KEY")),
        "endpoint": redacted_base_url(),
    }


@mcp.tool()
def axiom_supercompress(
    context: str,
    query: str,
    budget_ratio: float = 0.35,
    ccr: bool = False,
) -> dict:
    """Compress context through SuperCompress before inference."""
    ToolPermissionStore().require("axiom_supercompress")
    result = compress_context(
        context,
        query,
        budget_ratio=budget_ratio,
        ccr=ccr,
    )

    return {
        "compressed_text": result.compressed_text,
        "original_tokens": result.original_tokens,
        "kept_tokens": result.kept_tokens,
        "tokens_saved": result.tokens_saved,
        "savings_pct": result.savings_pct,
        "compression_risk": result.compression_risk,
        "policy_name": result.policy_name,
        "mode": result.mode,
    }


if __name__ == "__main__":
    mcp.run()
