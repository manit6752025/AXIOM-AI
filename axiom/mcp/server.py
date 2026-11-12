from __future__ import annotations

import os
from pathlib import Path

from mcp.server import MCPServer

from axiom.core.hardware import detect_hardware
from axiom.core.mcp_paths import (
    open_mcp_workspace_file,
    open_mcp_workspace_output_parent,
)
from axiom.core.tool_permissions import ToolPermissionStore
from axiom.core.training_preflight import run_training_preflight
from axiom.daemon import local_summary
from axiom.daemon_api import MAX_LISTED_FILES, list_workspace_files
from axiom.datasets.cleaner import clean_jsonl
from axiom.datasets.cleaning_preview import preview_jsonl_cleaning
from axiom.datasets.inspector import inspect_dataset
from axiom.datasets.profiler import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINE_BYTES,
    DEFAULT_MAX_ROWS,
)
from axiom.models.inspector import inspect_model
from axiom.models.registry import ModelRegistry
from axiom.runtime.supercompress import compress_context, redacted_base_url
from axiom.training.planner import create_training_plan
from axiom.version import __version__

mcp = MCPServer(
    f"AXIOM v{__version__}",
    instructions=(
        "AXIOM is an AI engineering platform. "
        "After the active AXIOM profile allows it, start with axiom_workspace_context "
        "for a bounded local workspace briefing. "
        "Use axiom_workspace_assets to find local model and dataset paths when allowed. "
        "Use axiom_training_preflight for bounded, non-executing checks before local LoRA training. "
        "Use AXIOM tools to inspect models, datasets, and hardware; preview dataset cleaning "
        "and show the counts and proposed output path before writing a new copy. "
        "Re-preview immediately before cleaning because previews are advisory. "
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
            "supercompress",
            "mcp",
            "agent_integrations",
        ],
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
def axiom_dataset_inspect(path: str) -> dict:
    """Inspect an AXIOM-supported dataset."""
    ToolPermissionStore().require("axiom_dataset_inspect")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Dataset path cannot be empty.")

    result = inspect_dataset(path)

    return {
        "path": str(result.path),
        "format": result.format,
        "samples": result.samples,
        "valid": result.valid,
        "invalid": result.invalid,
        "empty": result.empty,
        "duplicates": result.duplicates,
        "estimated_tokens": result.estimated_tokens,
        "fields": result.fields,
        "errors": result.errors,
    }


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
