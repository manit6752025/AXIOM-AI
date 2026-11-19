"""Passive discovery of installed local AI runtime tooling.

No executables are launched, no sockets are opened, no models are loaded, and
no network endpoints are probed. Presence means only that an executable or
Python package can be discovered on this machine; it is not a connection test.
"""
from __future__ import annotations

import importlib.util
import shutil
from datetime import UTC, datetime
from typing import Any

_CONNECTORS: tuple[dict[str, Any], ...] = (
    {"id": "ollama", "name": "Ollama", "kind": "executable", "executable": "ollama", "package": None, "potential_capabilities": ["local_model_management", "local_inference"], "note": "Executable presence only; no daemon or model endpoint was contacted."},
    {"id": "llama-server", "name": "llama.cpp server", "kind": "executable", "executable": "llama-server", "package": None, "potential_capabilities": ["local_inference", "http_api"], "note": "Executable presence only; model readiness and API compatibility are unverified."},
    {"id": "lm-studio", "name": "LM Studio CLI", "kind": "executable", "executable": "lms", "package": None, "potential_capabilities": ["local_model_management", "local_inference"], "note": "Executable presence only; no CLI command was run."},
    {"id": "vllm-cli", "name": "vLLM CLI", "kind": "executable", "executable": "vllm", "package": None, "potential_capabilities": ["local_inference", "serving"], "note": "Executable presence only; no server was started."},
    {"id": "llamafile", "name": "llamafile", "kind": "executable", "executable": "llamafile", "package": None, "potential_capabilities": ["local_inference"], "note": "Executable presence only; no model file was inspected or executed."},
    {"id": "python-vllm", "name": "vLLM Python package", "kind": "python-package", "executable": None, "package": "vllm", "potential_capabilities": ["local_inference", "serving"], "note": "Package metadata discovery only; the package was not imported."},
    {"id": "llama-cpp-python", "name": "llama-cpp-python", "kind": "python-package", "executable": None, "package": "llama_cpp", "potential_capabilities": ["local_inference"], "note": "Package metadata discovery only; the package was not imported."},
    {"id": "mlx-lm", "name": "MLX-LM", "kind": "python-package", "executable": None, "package": "mlx_lm", "potential_capabilities": ["apple_silicon_inference"], "note": "Package metadata discovery only; hardware compatibility was not tested."},
)


def _package_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def discover_runtime_connectors() -> dict[str, Any]:
    """Return an evidence-labelled inventory of potential runtime adapters."""
    connectors: list[dict[str, Any]] = []
    for definition in _CONNECTORS:
        executable = definition["executable"]
        package = definition["package"]
        present = bool(shutil.which(executable)) if executable else _package_present(package)
        connectors.append({
            "id": definition["id"],
            "name": definition["name"],
            "kind": definition["kind"],
            "present": present,
            "status": "discovered_not_connected" if present else "not_detected",
            "potential_capabilities": list(definition["potential_capabilities"]),
            "evidence": "found on PATH" if executable and present else (
                "Python package discoverable" if package and present else "not present in the current environment"
            ),
            "connection_checked": False,
            "note": definition["note"],
        })
    return {
        "schema_version": 1,
        "discovery_mode": "passive_local_discovery",
        "checked_at": datetime.now(UTC).isoformat(),
        "connectors": connectors,
        "scope_note": "Presence is not a connection, readiness, compatibility, performance, or security attestation.",
    }
