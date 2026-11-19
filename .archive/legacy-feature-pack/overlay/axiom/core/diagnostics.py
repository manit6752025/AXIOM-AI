"""Read-only workspace diagnostics and privacy-preserving support bundles.

Diagnostics never execute project code, mutate workspace files, or return raw
configuration values. Support bundles contain only a bounded JSON summary and
static documentation; they intentionally omit datasets, model files, env vars,
and the contents of axiom.yaml.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from axiom.version import __version__

_SCHEMA_VERSION = 1
_MAX_CONFIG_BYTES = 256 * 1024
_MIN_FREE_BYTES = 1024 * 1024 * 1024
_REQUIRED_MODULES = {
    "typer": "typer",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "yaml": "PyYAML",
    "mcp": "mcp",
}
_PROJECT_DIRECTORIES = ("data", "models", "experiments", "evaluations", "outputs")
_METADATA_DIRECTORIES = (".axiom",)


def discover_workspace_root(
    workspace: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a workspace without following a caller-provided symlink.

    Resolution order is explicit argument, ``AXIOM_WORKSPACE``, nearest parent
    containing ``axiom.yaml``, then the current directory. An invalid explicit
    or environment override is an error rather than a silent fallback.
    """
    env = os.environ if environ is None else environ
    requested: str | Path | None = workspace
    if requested is None:
        configured = env.get("AXIOM_WORKSPACE", "").strip()
        requested = configured or None

    if requested is not None:
        candidate = Path(requested).expanduser()
        if candidate.is_symlink():
            raise ValueError("Workspace path must not be a symbolic link.")
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Workspace path does not exist or cannot be resolved.") from exc
        if not resolved.is_dir():
            raise ValueError("Workspace path must be a directory.")
        return resolved

    start = Path(cwd) if cwd is not None else Path.cwd()
    if start.is_symlink():
        raise ValueError("Current workspace path must not be a symbolic link.")
    try:
        start = start.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Current workspace directory cannot be resolved.") from exc
    if not start.is_dir():
        raise ValueError("Current workspace path must be a directory.")

    for parent in (start, *start.parents):
        config = parent / "axiom.yaml"
        metadata = parent / ".axiom"
        if (config.exists() and not config.is_symlink()) or (
            metadata.is_dir() and not metadata.is_symlink()
        ):
            return parent
    return start


def _check(
    check_id: str,
    title: str,
    status: str,
    severity: str,
    detail: str,
    remediation: str | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "title": title,
        "status": status,
        "severity": severity,
        "detail": detail,
        "remediation": remediation,
    }


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def diagnose_workspace(
    workspace: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return stable, read-only checks for the local workspace and install."""
    root = discover_workspace_root(workspace, cwd=cwd, environ=environ)
    checks: list[dict[str, Any]] = []

    checks.append(
        _check(
            "workspace.root",
            "Workspace directory",
            "pass",
            "info",
            "A readable local workspace directory was resolved.",
        )
    )

    config = root / "axiom.yaml"
    config_present = config.exists() or config.is_symlink()
    if config.is_symlink():
        checks.append(
            _check(
                "workspace.config.symlink",
                "Project configuration",
                "fail",
                "error",
                "axiom.yaml is a symbolic link; diagnostics will not follow it.",
                "Replace it with a regular local configuration file after reviewing the target.",
            )
        )
    elif not config_present:
        checks.append(
            _check(
                "workspace.config.missing",
                "Project configuration",
                "warn",
                "warning",
                "No axiom.yaml file was found in the resolved workspace.",
                "Run `axiom init my-ai` for a new project, or choose the intended workspace.",
            )
        )
    elif not config.is_file():
        checks.append(
            _check(
                "workspace.config.type",
                "Project configuration",
                "fail",
                "error",
                "axiom.yaml exists but is not a regular file.",
                "Replace it with a regular YAML configuration file.",
            )
        )
    else:
        try:
            size = config.stat().st_size
            if size > _MAX_CONFIG_BYTES:
                raise ValueError("configuration exceeds the diagnostic size limit")
            raw_config = config.read_text(encoding="utf-8")
            parsed = yaml.safe_load(raw_config)
            if not raw_config.strip() or not isinstance(parsed, dict):
                raise ValueError("configuration must be a non-empty YAML mapping")
            malformed_sections = [
                name
                for name in ("project", "model", "dataset", "training", "evaluation", "deployment")
                if name in parsed and parsed[name] is not None and not isinstance(parsed[name], dict)
            ]
            if malformed_sections:
                raise ValueError("one or more known sections are not mappings")
        except (OSError, UnicodeError, yaml.YAMLError, ValueError):
            checks.append(
                _check(
                    "workspace.config.invalid",
                    "Project configuration",
                    "fail",
                    "error",
                    "axiom.yaml could not be parsed as a bounded, valid configuration mapping; values are withheld.",
                    "Run `axiom config validate --path axiom.yaml` and correct the reported format problem.",
                )
            )
        else:
            checks.append(
                _check(
                    "workspace.config.valid",
                    "Project configuration",
                    "pass",
                    "info",
                    "axiom.yaml is a valid YAML mapping with compatible known sections.",
                )
            )

    for directory_name in (*_METADATA_DIRECTORIES, *_PROJECT_DIRECTORIES):
        directory = root / directory_name
        if directory.is_symlink():
            checks.append(
                _check(
                    f"workspace.path.{directory_name.replace('.', 'dot').replace('/', '_')}",
                    f"{directory_name}/ path safety",
                    "fail",
                    "error",
                    f"The {directory_name}/ entry is a symbolic link; AXIOM diagnostics do not follow it.",
                    "Review the path and replace the symlink with an approved in-workspace directory if needed.",
                )
            )
        elif directory.exists() and not directory.is_dir():
            checks.append(
                _check(
                    f"workspace.path.{directory_name.replace('.', 'dot').replace('/', '_')}",
                    f"{directory_name}/ directory",
                    "fail",
                    "error",
                    f"The {directory_name}/ entry exists but is not a directory.",
                    "Rename or remove the conflicting entry, then create a directory if this feature needs it.",
                )
            )
        elif not directory.exists():
            if directory_name in _METADATA_DIRECTORIES:
                detail = "Local metadata has not been created yet; AXIOM creates it when needed."
                severity = "info"
                status = "info"
            elif config_present:
                detail = f"The optional {directory_name}/ project directory is not present."
                severity = "warning"
                status = "warn"
            else:
                detail = f"The optional {directory_name}/ directory is not present in this workspace."
                severity = "info"
                status = "info"
            checks.append(
                _check(
                    f"workspace.path.{directory_name.replace('.', 'dot').replace('/', '_')}",
                    f"{directory_name}/ directory",
                    status,
                    severity,
                    detail,
                    "Create this directory when the corresponding AXIOM workflow needs it."
                    if status == "warn"
                    else None,
                )
            )
        else:
            checks.append(
                _check(
                    f"workspace.path.{directory_name.replace('.', 'dot').replace('/', '_')}",
                    f"{directory_name}/ directory",
                    "pass",
                    "info",
                    f"The {directory_name}/ entry is a directory and is not a symlink.",
                )
            )

    if os.access(root, os.R_OK | os.X_OK):
        checks.append(_check("workspace.permissions.read", "Workspace read access", "pass", "info", "The current user can traverse and read the workspace directory."))
    else:
        checks.append(_check("workspace.permissions.read", "Workspace read access", "fail", "error", "The current user cannot read or traverse the workspace directory.", "Review filesystem permissions for the selected workspace."))

    if os.access(root, os.W_OK | os.X_OK):
        checks.append(_check("workspace.permissions.write", "Workspace write access", "pass", "info", "The current user appears able to create local workspace files."))
    else:
        checks.append(_check("workspace.permissions.write", "Workspace write access", "warn", "warning", "The current user does not appear to have write access to the workspace directory.", "Choose a writable workspace or adjust its ownership and permissions."))

    try:
        free_bytes = shutil.disk_usage(root).free
    except OSError:
        checks.append(_check("workspace.disk", "Free disk space", "warn", "warning", "Free disk space could not be measured for this workspace."))
    else:
        if free_bytes < _MIN_FREE_BYTES:
            checks.append(_check("workspace.disk", "Free disk space", "warn", "warning", f"Less than 1 GiB of free disk space is available ({free_bytes} bytes).", "Free disk space before downloading large models or writing datasets."))
        else:
            checks.append(_check("workspace.disk", "Free disk space", "pass", "info", f"At least 1 GiB of free disk space is available ({free_bytes} bytes)."))

    python_ok = sys.version_info >= (3, 11)
    checks.append(
        _check(
            "runtime.python",
            "Python version",
            "pass" if python_ok else "fail",
            "info" if python_ok else "error",
            f"Python {platform.python_version()} is " + ("supported." if python_ok else "older than the supported Python 3.11 minimum."),
            None if python_ok else "Install Python 3.11 or newer and recreate the virtual environment.",
        )
    )

    missing_modules: list[str] = []
    for module_name, distribution_name in _REQUIRED_MODULES.items():
        available = _module_available(module_name)
        if not available:
            missing_modules.append(distribution_name)
        checks.append(
            _check(
                f"dependency.{module_name}",
                f"Dependency: {distribution_name}",
                "pass" if available else "fail",
                "info" if available else "error",
                f"{distribution_name} is {'available' if available else 'not importable'} in the active Python environment.",
                None if available else f"Install project dependencies in the active environment (missing: {distribution_name}).",
            )
        )

    failed = sum(1 for item in checks if item["status"] == "fail")
    warnings = sum(1 for item in checks if item["status"] == "warn")
    passed = sum(1 for item in checks if item["status"] == "pass")
    informational = sum(1 for item in checks if item["status"] == "info")
    overall = "fail" if failed else "warn" if warnings else "pass"
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": overall,
        "workspace": {
            "name": root.name or "workspace",
            "project_config_present": config_present,
            "is_project": config.is_file() and not config.is_symlink(),
        },
        "summary": {
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "informational": informational,
            "missing_dependencies": missing_modules,
        },
        "checks": checks,
    }


def create_support_bundle(workspace: str | Path | None = None) -> bytes:
    """Create a small support ZIP without collecting project data or secrets."""
    report = diagnose_workspace(workspace)
    # Exclude workspace basename and all local paths from the exported report.
    safe_report = {
        "schema_version": report["schema_version"],
        "generated_at": report["generated_at"],
        "status": report["status"],
        "summary": report["summary"],
        "checks": report["checks"],
        "privacy": {
            "config_values_included": False,
            "environment_variables_included": False,
            "dataset_or_model_contents_included": False,
            "absolute_paths_included": False,
        },
    }
    runtime: dict[str, Any] = {
        "axiom_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os_family": platform.system(),
        "os_release_major": platform.release().split(".", maxsplit=1)[0],
        "architecture": platform.machine(),
    }
    package_versions: dict[str, str] = {}
    for distribution_name in _REQUIRED_MODULES.values():
        try:
            package_versions[distribution_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution_name] = "not-installed"
    runtime["dependency_versions"] = package_versions

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("diagnostics.json", json.dumps(safe_report, indent=2, sort_keys=True) + "\n")
        archive.writestr("environment.json", json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        archive.writestr(
            "README.txt",
            "AXIOM privacy-preserving support bundle\n\n"
            "Included: aggregate workspace checks, AXIOM/Python versions, OS family, and dependency versions.\n"
            "Excluded: axiom.yaml contents, environment variables, credentials, personal paths, model weights, datasets, logs, and user-generated records.\n"
            "Review this archive before sharing it with anyone.\n",
        )
    return buffer.getvalue()
