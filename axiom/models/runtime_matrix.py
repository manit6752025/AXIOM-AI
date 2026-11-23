from __future__ import annotations

import copy
import importlib.metadata
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from threading import Lock
from typing import Any

PROBE_TIMEOUT_SECONDS = 2.0
CACHE_TTL_SECONDS = 30
_MAX_VERSION_EVIDENCE = 180
_CACHE_LOCK = Lock()
_CACHE_CREATED_AT = 0.0
_CACHE_VALUE: dict[str, Any] | None = None

_RUNTIME_SPECS = (
    {
        "id": "ollama",
        "name": "Ollama",
        "kind": "command",
        "commands": (("ollama", "--version"),),
        "formats": ("GGUF",),
        "evidence_method": "ollama --version",
    },
    {
        "id": "llama.cpp",
        "name": "llama.cpp",
        "kind": "command",
        "commands": (
            ("llama-cli", "--version"),
            ("llama-server", "--version"),
            ("main", "--version"),
        ),
        "formats": ("GGUF",),
        "evidence_method": "llama.cpp CLI --version",
    },
    {
        "id": "transformers",
        "name": "Hugging Face Transformers",
        "kind": "python-package",
        "package": "transformers",
        "formats": ("Safetensors", "PyTorch"),
        "evidence_method": "importlib.metadata.version('transformers')",
    },
)

_LIMITATIONS = [
    "Runtime discovery checks local PATH entries and installed Python package metadata only; it performs no network requests.",
    "Version evidence is a local installation snapshot, not a tested compatibility range or a security/support endorsement.",
    "Format/runtime entries are candidates based on format-family expectations. AXIOM does not load a model or validate its architecture, tokenizer, adapters, operators, or dependencies.",
    "A missing runtime or unavailable version probe is reported explicitly and is not interpreted as proof that a format is unsupported.",
    "Runtime version output is bounded and sanitized before it is returned; executable paths and environment values are never included.",
]


def _safe_version_evidence(value: str) -> str | None:
    """Return one bounded version/build line without exposing local paths."""
    for line in value.splitlines():
        cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", line).strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"(?i)(token|password|api[_-]?key)(\s*[:=]\s*)\S+", r"\1\2<redacted>", cleaned)
        cleaned = re.sub(r"[A-Za-z]:\\[^\s]+", "<path>", cleaned)
        cleaned = re.sub(r"(?<!:)\/(?:[^\s]+)", "<path>", cleaned)
        return cleaned[:_MAX_VERSION_EVIDENCE]
    return None


def _probe_command(spec: dict[str, Any]) -> dict[str, Any]:
    selected: tuple[str, str] | None = None
    for executable, argument in spec["commands"]:
        resolved = shutil.which(executable)
        if resolved:
            selected = (resolved, f"{executable} {argument}")
            break

    if selected is None:
        return {
            "id": spec["id"],
            "name": spec["name"],
            "availability": "not_detected",
            "version": None,
            "version_evidence": None,
            "evidence_method": spec["evidence_method"],
            "candidate_formats": list(spec["formats"]),
        }

    executable_path, command_label = selected
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        result = subprocess.run(
            [executable_path, "--version"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
            env=safe_env,
        )
    except subprocess.TimeoutExpired:
        return {
            "id": spec["id"],
            "name": spec["name"],
            "availability": "version_probe_timed_out",
            "version": None,
            "version_evidence": None,
            "evidence_method": command_label,
            "candidate_formats": list(spec["formats"]),
        }
    except OSError:
        return {
            "id": spec["id"],
            "name": spec["name"],
            "availability": "version_probe_failed",
            "version": None,
            "version_evidence": None,
            "evidence_method": command_label,
            "candidate_formats": list(spec["formats"]),
        }

    evidence = _safe_version_evidence(
        f"{result.stdout or ''}\n{result.stderr or ''}"
    )
    if result.returncode == 0:
        availability = "installed"
    else:
        availability = "installed_version_unavailable"
    return {
        "id": spec["id"],
        "name": spec["name"],
        "availability": availability,
        "version": evidence,
        "version_evidence": command_label if evidence else None,
        "evidence_method": command_label,
        "candidate_formats": list(spec["formats"]),
    }


def _probe_python_package(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(spec["package"])
    except importlib.metadata.PackageNotFoundError:
        availability = "not_detected"
        version = None
    except (OSError, ValueError):
        availability = "version_probe_failed"
        version = None
    else:
        availability = "installed"
    return {
        "id": spec["id"],
        "name": spec["name"],
        "availability": availability,
        "version": _safe_version_evidence(version) if version else None,
        "version_evidence": spec["evidence_method"] if version else None,
        "evidence_method": spec["evidence_method"],
        "candidate_formats": list(spec["formats"]),
    }


def _build_matrix() -> dict[str, Any]:
    runtimes: list[dict[str, Any]] = []
    for spec in _RUNTIME_SPECS:
        if spec["kind"] == "command":
            runtimes.append(_probe_command(spec))
        else:
            runtimes.append(_probe_python_package(spec))

    formats = sorted({format_name for runtime in runtimes for format_name in runtime["candidate_formats"]})
    matrix = []
    for format_name in formats:
        candidates = []
        for runtime in runtimes:
            if format_name not in runtime["candidate_formats"]:
                continue
            candidates.append({
                "runtime_id": runtime["id"],
                "runtime_name": runtime["name"],
                "availability": runtime["availability"],
                "version": runtime["version"],
                "version_evidence": runtime["version_evidence"],
                "status": "candidate_not_verified",
            })
        matrix.append({"format": format_name, "candidates": candidates})

    return {
        "schema_version": 1,
        "probed_at": datetime.now(UTC).isoformat(),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "network_access": False,
        "model_load_tested": False,
        "runtimes": runtimes,
        "formats": matrix,
        "limitations": list(_LIMITATIONS),
        "scope_note": (
            "Local runtime and version evidence only. Format matches are candidates, "
            "not proof that a particular model can be loaded."
        ),
    }


def probe_local_runtime_matrix(*, force_refresh: bool = False) -> dict[str, Any]:
    """Return a bounded, cached, no-network runtime/version compatibility matrix."""
    global _CACHE_CREATED_AT, _CACHE_VALUE

    now = time.monotonic()
    with _CACHE_LOCK:
        if (
            not force_refresh
            and _CACHE_VALUE is not None
            and now - _CACHE_CREATED_AT < CACHE_TTL_SECONDS
        ):
            return copy.deepcopy(_CACHE_VALUE)

        value = _build_matrix()
        _CACHE_VALUE = value
        _CACHE_CREATED_AT = time.monotonic()
        return copy.deepcopy(value)
