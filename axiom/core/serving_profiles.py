"""Safe, workspace-local configuration profiles for AXIOM native model serving."""
from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.resource_ids import is_resource_id, new_resource_id
from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import workspace_metadata_dir

SCHEMA_VERSION = 1
MAX_PROFILES = 25
MAX_STORE_BYTES = 128 * 1024
PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,47}$")
_DEVICES = frozenset({"auto", "cpu", "cuda", "rocm", "mps", "xpu"})
_MEMORY_MODES = frozenset({"auto", "full", "layer_streaming", "disk_streaming"})
_PRECISIONS = frozenset({"auto", "fp32", "fp16", "bf16"})
_PROFILE_KEYS = frozenset({
    "id", "name", "model_path", "adapter_path", "device_preference",
    "memory_mode", "precision", "context_length", "max_new_tokens",
    "max_prompt_chars", "max_concurrent_requests", "max_queued_requests",
    "request_timeout_seconds", "created_at", "updated_at",
})
_STORE_LOCK = threading.Lock()
_ACTIVE_PROFILE_REQUESTS: set[str] = set()


class ServingProfileError(ValueError):
    """A serving profile is invalid, unavailable, or unsafe to persist."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ServingProfileError(f"{label} must be an integer from {minimum} through {maximum}.")
    return value


def _path(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or not os.path.isabs(value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or ".." in Path(value).parts
    ):
        raise ServingProfileError(f"{label} must be an absolute trusted local path.")
    return value


def normalize_profile_config(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the bounded public profile schema without touching model files."""
    if not isinstance(value, dict):
        raise ServingProfileError("Serving profile configuration must be an object.")
    allowed = {
        "name", "model_path", "adapter_path", "device_preference", "memory_mode",
        "precision", "context_length", "max_new_tokens", "max_prompt_chars",
        "max_concurrent_requests", "max_queued_requests", "request_timeout_seconds",
    }
    if set(value) != allowed:
        raise ServingProfileError("Serving profile configuration has missing or unsupported fields.")
    name = value["name"]
    if not isinstance(name, str) or not PROFILE_NAME.fullmatch(name.strip()):
        raise ServingProfileError("Serving profile names must be 1–48 safe characters and start with a letter or number.")
    device = value["device_preference"]
    memory_mode = value["memory_mode"]
    precision = value["precision"]
    if not isinstance(device, str) or device not in _DEVICES:
        raise ServingProfileError("Device preference must be auto, cpu, cuda, rocm, mps, or xpu.")
    if not isinstance(memory_mode, str) or memory_mode not in _MEMORY_MODES:
        raise ServingProfileError("Memory mode must be auto, full, layer_streaming, or disk_streaming.")
    if not isinstance(precision, str) or precision not in _PRECISIONS:
        raise ServingProfileError("Precision must be auto, fp32, fp16, or bf16.")
    concurrency = _bounded_int(value["max_concurrent_requests"], "Maximum concurrent requests", 1, 1)
    queue_depth = _bounded_int(value["max_queued_requests"], "Maximum queued requests", 0, 0)
    timeout = _bounded_int(value["request_timeout_seconds"], "Request timeout", 5, 900)
    return {
        "name": name.strip(),
        "model_path": _path(value["model_path"], "Base model path"),
        "adapter_path": _path(value["adapter_path"], "Adapter path", optional=True),
        "device_preference": device,
        "memory_mode": memory_mode,
        "precision": precision,
        "context_length": _bounded_int(value["context_length"], "Context length", 1, 131072),
        "max_new_tokens": _bounded_int(value["max_new_tokens"], "Maximum output tokens", 1, 2048),
        "max_prompt_chars": _bounded_int(value["max_prompt_chars"], "Maximum prompt characters", 1, 12000),
        "max_concurrent_requests": concurrency,
        "max_queued_requests": queue_depth,
        "request_timeout_seconds": timeout,
    }


class ServingProfileStore:
    """Persist bounded, secret-free serving configuration in the active workspace."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else workspace_metadata_dir()
        self.path = self.root / "serving-profiles.json"
        self.lock_path = self.root / ".serving-profiles.lock"

    def _check_paths(self, *, create: bool = False) -> None:
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise ServingProfileError("The serving profile store must not use symbolic links.")
        if self.root.exists() and not self.root.is_dir():
            raise ServingProfileError("The workspace metadata path is not a directory.")
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise ServingProfileError("The serving profile store path changed unsafely.")
        if self.root.exists() and os.name != "nt":
            try:
                os.chmod(self.root, 0o700)
            except OSError as exc:
                raise ServingProfileError("Could not restrict serving profile directory permissions.") from exc

    def _read(self) -> list[dict[str, Any]]:
        self._check_paths()
        if not self.path.exists():
            return []
        try:
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                raise ServingProfileError("The serving profile store is not a bounded regular file.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except ServingProfileError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ServingProfileError("The serving profile store could not be read safely.") from exc
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "profiles"}:
            raise ServingProfileError("The serving profile document is malformed.")
        version, profiles = raw.get("schema_version"), raw.get("profiles")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise ServingProfileError("Unsupported serving profile schema version.")
        if not isinstance(profiles, list) or len(profiles) > MAX_PROFILES:
            raise ServingProfileError("The serving profile list is invalid or exceeds its limit.")
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for item in profiles:
            if not isinstance(item, dict) or set(item) != _PROFILE_KEYS:
                raise ServingProfileError("A serving profile contains unsupported or secret-bearing fields.")
            if not is_resource_id(item.get("id")):
                raise ServingProfileError("The serving profile store contains an invalid ID.")
            config = normalize_profile_config({key: item[key] for key in (
                "name", "model_path", "adapter_path", "device_preference", "memory_mode",
                "precision", "context_length", "max_new_tokens", "max_prompt_chars",
                "max_concurrent_requests", "max_queued_requests", "request_timeout_seconds",
            )})
            created_at, updated_at = item.get("created_at"), item.get("updated_at")
            if (
                not isinstance(created_at, str) or not created_at or len(created_at) > 40
                or not isinstance(updated_at, str) or not updated_at or len(updated_at) > 40
            ):
                raise ServingProfileError("The serving profile timestamps are invalid.")
            profile_id = item["id"]
            name_key = config["name"].casefold()
            if profile_id in seen_ids or name_key in seen_names:
                raise ServingProfileError("Serving profile IDs and names must be unique.")
            seen_ids.add(profile_id)
            seen_names.add(name_key)
            normalized.append({
                "id": profile_id, **config,
                "created_at": created_at, "updated_at": updated_at,
            })
        return normalized

    def _write(self, profiles: list[dict[str, Any]]) -> None:
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "profiles": profiles},
            ensure_ascii=False, indent=2, allow_nan=False,
        ) + "\n"
        if len(payload.encode("utf-8")) > MAX_STORE_BYTES:
            raise ServingProfileError("The serving profile store exceeds its 128 KiB limit.")
        atomic_write_text(self.path, payload)
        if os.name != "nt":
            try:
                os.chmod(self.path, 0o600)
            except OSError as exc:
                raise ServingProfileError("Could not secure serving profile file permissions.") from exc

    def list(self) -> dict[str, Any]:
        profiles = self._read()
        return {
            "schema_version": SCHEMA_VERSION,
            "profiles": profiles,
            "total": len(profiles),
            "max_profiles": MAX_PROFILES,
            "credentials_persisted": False,
            "network_used": False,
            "weights_loaded": False,
        }

    def get(self, profile_id: str) -> dict[str, Any] | None:
        if not is_resource_id(profile_id):
            raise ServingProfileError("Invalid serving profile ID.")
        return next((item for item in self._read() if item["id"] == profile_id), None)

    def save(self, config: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_profile_config(config)
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            profiles = self._read()
            existing = next((item for item in profiles if item["name"].casefold() == normalized["name"].casefold()), None)
            now = _now()
            if existing is None:
                if len(profiles) >= MAX_PROFILES:
                    raise ServingProfileError(f"The serving profile limit of {MAX_PROFILES} has been reached.")
                profile = {"id": new_resource_id(), **normalized, "created_at": now, "updated_at": now}
                profiles.insert(0, profile)
            else:
                profile = {**existing, **normalized, "updated_at": now}
                profiles = [profile if item["id"] == existing["id"] else item for item in profiles]
            self._write(profiles)
            return dict(profile)

    def delete(self, profile_id: str) -> dict[str, Any] | None:
        if not is_resource_id(profile_id):
            raise ServingProfileError("Invalid serving profile ID.")
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            profiles = self._read()
            removed = next((item for item in profiles if item["id"] == profile_id), None)
            if removed is None:
                return None
            self._write([item for item in profiles if item["id"] != profile_id])
            return dict(removed)


def readiness_probe(profile: dict[str, Any]) -> dict[str, Any]:
    """Check dependency/device/checkpoint readiness without loading model weights."""
    from axiom.core.engine import inspect_native_model_context, runtime_status

    checks: dict[str, Any] = {}
    problems: list[str] = []
    try:
        context = inspect_native_model_context(profile["model_path"])
        checks["model"] = {
            "available": True,
            "model_type": context["model_type"],
            "architecture": context["architecture"],
            "checkpoint_format": context["checkpoint_format"],
            "effective_context_length": context["effective_max_position_embeddings"],
        }
    except (OSError, RuntimeError, ValueError) as exc:
        context = None
        checks["model"] = {"available": False}
        problems.append(f"Model checkpoint is not ready: {exc}")

    try:
        runtime = runtime_status()
    except Exception:
        runtime = {"ready": False, "problems": ["Runtime status inspection failed."], "devices": []}
    checks["runtime"] = {
        "ready": runtime.get("ready") is True,
        "problems": runtime.get("problems", []),
    }
    if runtime.get("ready") is not True:
        problems.extend(str(item) for item in runtime.get("problems", []) if isinstance(item, str))
        if not runtime.get("problems"):
            problems.append("AXIOM Native Engine dependencies are not ready.")

    devices = runtime.get("devices")
    devices = devices if isinstance(devices, list) else []
    available = {
        item.get("id"): item for item in devices
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    requested_device = profile["device_preference"]
    selected_device = runtime.get("recommended_device") if requested_device == "auto" else requested_device
    device = available.get(selected_device) if isinstance(selected_device, str) else None
    device_ready = bool(device and device.get("available") is True)
    checks["device"] = {
        "requested": requested_device,
        "selected": selected_device,
        "available": device_ready,
        "name": device.get("name") if device else None,
    }
    if not device_ready:
        problems.append(f"Requested execution device {selected_device or requested_device} is not available.")

    if context is not None:
        if profile["context_length"] > context["effective_max_position_embeddings"]:
            checks["context"] = {
                "available": False,
                "requested": profile["context_length"],
                "maximum": context["effective_max_position_embeddings"],
            }
            problems.append("Configured context length exceeds the checkpoint's supported context.")
        else:
            checks["context"] = {
                "available": True,
                "requested": profile["context_length"],
                "maximum": context["effective_max_position_embeddings"],
            }

    adapter_path = profile.get("adapter_path")
    if adapter_path:
        from axiom.core.engine import check_local_adapter_compatibility

        try:
            compatibility = check_local_adapter_compatibility(profile["model_path"], adapter_path)
            compatibility_rows = compatibility.get("devices")
            compatibility_rows = compatibility_rows if isinstance(compatibility_rows, list) else []
            selected_compatibility = next(
                (
                    item for item in compatibility_rows
                    if isinstance(item, dict) and item.get("id") == selected_device
                ),
                None,
            )
            compatible = (
                compatibility.get("structurally_compatible") is True
                and selected_compatibility is not None
                and selected_compatibility.get("compatible") is True
            )
            checks["adapter"] = {
                "available": compatible,
                "compatible": compatible,
                "structurally_compatible": compatibility.get("structurally_compatible") is True,
                "selected_device": selected_device,
                "device_compatibility": compatibility_rows,
            }
            if not compatible:
                problems.append("Configured LoRA adapter is not compatible with the selected base model.")
        except (OSError, RuntimeError, ValueError) as exc:
            checks["adapter"] = {"available": False, "compatible": False}
            problems.append(f"LoRA adapter readiness check failed: {exc}")
    else:
        checks["adapter"] = {"available": True, "configured": False}

    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile["id"],
        "profile_name": profile["name"],
        "ready": not problems,
        "checks": checks,
        "problems": problems,
        "weights_loaded": False,
        "service_started": False,
        "network_used": False,
        "probed_at": _now(),
        "scope_note": (
            "Readiness checks inspect trusted local checkpoint metadata, AXIOM dependencies and available devices. "
            "They do not load model weights or start a listener; the first profile generation request loads the model. "
            "A successful probe is not a guarantee that real inference will fit available memory or support every operator."
        ),
    }


def validate_serving_prompt(
    profile: dict[str, Any],
    prompt: str,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Enforce profile character, output-token and checkpoint-context limits."""
    from axiom.core.engine import inspect_native_prompt_batch

    if not isinstance(prompt, str) or not prompt.strip():
        raise ServingProfileError("Prompt must contain non-whitespace text.")
    if len(prompt) > profile["max_prompt_chars"]:
        raise ServingProfileError(
            f"Prompt exceeds this profile's {profile['max_prompt_chars']}-character limit."
        )
    output_tokens = profile["max_new_tokens"] if max_new_tokens is None else _bounded_int(
        max_new_tokens, "Maximum output tokens", 1, profile["max_new_tokens"]
    )
    # The model's native prompt inspector bounds its own tokenizer probe to 64 output tokens.
    report = inspect_native_prompt_batch(
        profile["model_path"], [prompt], max_new_tokens=min(output_tokens, 64),
    )
    if report.get("available") is not True or report.get("prompts_truncated") != 0:
        raise ServingProfileError(
            "The native tokenizer could not verify the complete prompt within this profile's safe input budget."
        )
    prompt_tokens = report.get("prompt_token_counts", {}).get("max")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 1:
        raise ServingProfileError("Native prompt token inspection did not return a valid token count.")
    required_context = prompt_tokens + output_tokens
    if required_context > profile["context_length"]:
        raise ServingProfileError(
            f"Prompt and requested output exceed the serving profile context limit: {required_context} tokens required; configured limit is {profile['context_length']}."
        )
    return {
        "prompt_characters": len(prompt),
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": output_tokens,
        "requested_context_tokens": required_context,
        "context_limit_tokens": profile["context_length"],
    }


def reserve_profile_request(profile_id: str) -> None:
    """Enforce the native engine's single-request execution contract without queueing."""
    if not is_resource_id(profile_id):
        raise ServingProfileError("Invalid serving profile ID.")
    with _STORE_LOCK:
        if _ACTIVE_PROFILE_REQUESTS:
            raise ServingProfileError("AXIOM Native Engine is busy; this profile does not queue requests.")
        _ACTIVE_PROFILE_REQUESTS.add(profile_id)


def release_profile_request(profile_id: str) -> None:
    with _STORE_LOCK:
        _ACTIVE_PROFILE_REQUESTS.discard(profile_id)


def generate_from_profile(
    profile: dict[str, Any],
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
    cancel_event: threading.Event | None = None,
    prompt_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one inference request under a saved profile's validated limits."""
    from axiom.core.engine import generate_local

    budget = prompt_budget or validate_serving_prompt(profile, prompt, max_new_tokens)
    return generate_local(
        profile["model_path"],
        prompt,
        adapter_path=profile.get("adapter_path"),
        max_new_tokens=budget["max_new_tokens"],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop_sequences=stop_sequences,
        device_preference=profile["device_preference"],
        memory_mode=profile["memory_mode"],
        precision=profile["precision"],
        cancel_event=cancel_event,
    )
