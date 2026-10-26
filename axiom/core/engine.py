"""AXIOM-owned native Llama/Mistral inference engine.

PyTorch supplies device tensor kernels; AXIOM implements the supported model
forward pass, attention/cache path, LoRA application, and token-by-token decode.
Local weights and tokenizer assets are required. The native path does not call
Transformers model generation or an Ollama service.
"""
from __future__ import annotations

import gc
from contextlib import contextmanager, nullcontext
from functools import lru_cache
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

_MAX_PROMPT_CHARS = 12_000
_MAX_CHAT_CHARS = 24_000
_MAX_NEW_TOKENS = 2_048
_MAX_ENGINE_QUEUE_WAIT_SECONDS = 30.0
_MAX_INPUT_TOKENS = 8_192
_MAX_CONFIG_BYTES = 1_048_576
_LOG = logging.getLogger(__name__)
# A plain Lock is deliberate: the streaming generator may resume on a different worker thread.
_ENGINE_LOCK = threading.Lock()


@contextmanager
def _engine_request_lock(cancel_event: threading.Event | None = None) -> Iterator[None]:
    """Bound admission wait for the single loaded native runtime.

    The current engine executes one request per loaded runtime. Waiting
    requests are bounded and cooperatively cancellable instead of accumulating
    indefinitely behind the runtime lock.
    """
    deadline = time.monotonic() + _MAX_ENGINE_QUEUE_WAIT_SECONDS
    acquired = False
    while not acquired:
        if cancel_event is not None and cancel_event.is_set():
            raise AxiomEngineError("Native generation was cancelled while waiting for the engine.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AxiomEngineError(
                "AXIOM Native Engine is busy with another request. Retry shortly or use a second AXIOM process for concurrent inference."
            )
        acquired = _ENGINE_LOCK.acquire(timeout=min(0.1, remaining))
    try:
        yield
    finally:
        _ENGINE_LOCK.release()
_LOADED_RUNTIME: dict[str, Any] | None = None
_LOADED_ARCHITECTURE: str | None = None
_MODEL_ROOT_ENV = "AXIOM_MODEL_ROOTS"


class AxiomEngineError(RuntimeError):
    """Raised when the AXIOM local model engine cannot run a request."""


def _version_key(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", value)[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def _device_capability_rows(torch_module: Any | None = None) -> list[dict[str, Any]]:
    """Return observed PyTorch devices without assuming a GPU exists."""
    try:
        if torch_module is None:
            import torch as torch_module
    except Exception:
        reason = "PyTorch could not be imported; install the native-inference extra for device execution."
        return [
            {"id": "cpu", "name": "CPU", "available": False, "reason": reason, "memory_free_gib": None, "memory_total_gib": None},
            {"id": "cuda", "name": "NVIDIA CUDA", "available": False, "reason": reason, "memory_free_gib": None, "memory_total_gib": None},
            {"id": "rocm", "name": "AMD ROCm", "available": False, "reason": reason, "memory_free_gib": None, "memory_total_gib": None},
            {"id": "mps", "name": "Apple Metal (MPS)", "available": False, "reason": reason, "memory_free_gib": None, "memory_total_gib": None},
            {"id": "xpu", "name": "Intel XPU", "available": False, "reason": reason, "memory_free_gib": None, "memory_total_gib": None},
        ]
    gib = 1024 ** 3
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:
        cuda_available = False
    is_rocm = bool(getattr(getattr(torch_module, "version", None), "hip", None))
    free_gib = total_gib = None
    accelerator_name = "AMD ROCm" if is_rocm else "NVIDIA CUDA"
    if cuda_available:
        try:
            accelerator_name = str(torch_module.cuda.get_device_name(0)) or accelerator_name
        except Exception:
            pass
        try:
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
            free_gib, total_gib = round(float(free_bytes) / gib, 3), round(float(total_bytes) / gib, 3)
        except Exception:
            # Some drivers do not expose memory telemetry; availability is still useful.
            pass
    rows: list[dict[str, Any]] = [
        {"id": "cuda", "name": accelerator_name if cuda_available and not is_rocm else "NVIDIA CUDA", "available": cuda_available and not is_rocm,
         "reason": "CUDA is available in this PyTorch environment." if cuda_available and not is_rocm else
            ("This PyTorch build targets AMD ROCm; select ROCm instead." if is_rocm else "CUDA is not available in this PyTorch environment."),
         "memory_free_gib": free_gib if cuda_available and not is_rocm else None,
         "memory_total_gib": total_gib if cuda_available and not is_rocm else None},
        {"id": "rocm", "name": accelerator_name if cuda_available and is_rocm else "AMD ROCm", "available": cuda_available and is_rocm,
         "reason": "ROCm is available through PyTorch's CUDA-compatible device API." if cuda_available and is_rocm else "ROCm is not available in this PyTorch environment.",
         "memory_free_gib": free_gib if cuda_available and is_rocm else None,
         "memory_total_gib": total_gib if cuda_available and is_rocm else None},
    ]
    try:
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
    except Exception:
        mps_available = False
    rows.append({"id": "mps", "name": "Apple Metal (MPS)", "available": mps_available,
        "reason": "PyTorch MPS is available; portable free-memory telemetry is not exposed here." if mps_available else "Apple Metal/MPS is not available in this PyTorch environment.",
        "memory_free_gib": None, "memory_total_gib": None})
    try:
        xpu_backend = getattr(torch_module, "xpu", None)
        xpu_available = bool(xpu_backend and xpu_backend.is_available())
    except Exception:
        xpu_available, xpu_backend = False, None
    xpu_name = "Intel XPU"
    xpu_free_gib = xpu_total_gib = None
    if xpu_available:
        try:
            xpu_name = str(xpu_backend.get_device_name(0)) or xpu_name
        except Exception:
            pass
        try:
            free_bytes, total_bytes = xpu_backend.mem_get_info()
            xpu_free_gib, xpu_total_gib = round(float(free_bytes) / gib, 3), round(float(total_bytes) / gib, 3)
        except Exception:
            pass
    rows.append({"id": "xpu", "name": xpu_name, "available": xpu_available,
        "reason": "PyTorch XPU is available; validate model/operator support on this driver." if xpu_available else "Intel XPU is not available in this PyTorch environment.",
        "memory_free_gib": xpu_free_gib, "memory_total_gib": xpu_total_gib})
    rows.append({"id": "cpu", "name": "CPU", "available": True,
        "reason": "PyTorch CPU tensor execution is available.", "memory_free_gib": None, "memory_total_gib": None})
    return rows


def _resolve_execution_device(torch_module: Any, preference: str = "auto") -> tuple[Any, dict[str, str]]:
    """Resolve an explicit backend or select the highest-priority available one."""
    choices = {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}
    if not isinstance(preference, str) or preference.strip().lower() not in choices:
        raise ValueError("Execution device must be one of: auto, cpu, cuda, rocm, mps, or xpu.")
    requested = preference.strip().lower()
    rows = _device_capability_rows(torch_module)
    by_id = {item["id"]: item for item in rows}
    if requested == "auto":
        selected = next((by_id[item] for item in ("cuda", "rocm", "mps", "xpu", "cpu") if by_id.get(item, {}).get("available")), None)
        if selected is None:
            raise AxiomEngineError("PyTorch could not initialize a supported execution device.")
        reason = f"Auto-selected {selected['name']} because it is the highest-priority available backend."
    else:
        selected = by_id.get(requested)
        if selected is None or not selected.get("available"):
            explanation = selected.get("reason") if selected else "This device is not supported by AXIOM."
            raise AxiomEngineError(f"Requested execution device {requested!r} is unavailable. {explanation}")
        reason = f"Using explicitly selected device: {selected['name']}."
    # PyTorch uses device='cuda' for both CUDA and ROCm builds.
    torch_name = "cuda" if selected["id"] == "rocm" else selected["id"]
    return torch_module.device(torch_name), {"requested": requested, "selected": selected["id"], "reason": reason}


def _clear_device_cache(torch_module: Any | None, device: Any | None) -> None:
    """Best-effort release of the inactive allocator cache for the model's device."""
    if torch_module is None or device is None:
        return
    device_type = str(getattr(device, "type", device)).split(":", 1)[0]
    try:
        if device_type == "cuda":
            torch_module.cuda.empty_cache()
        elif device_type == "mps":
            mps_backend = getattr(torch_module, "mps", None)
            if mps_backend is not None and callable(getattr(mps_backend, "empty_cache", None)):
                mps_backend.empty_cache()
        elif device_type == "xpu":
            xpu_backend = getattr(torch_module, "xpu", None)
            if xpu_backend is not None and callable(getattr(xpu_backend, "empty_cache", None)):
                xpu_backend.empty_cache()
    except Exception:
        _LOG.debug("Device allocator cache cleanup was unavailable for %s.", device_type, exc_info=True)



def _reset_device_peak_memory(torch_module: Any | None, device: Any | None) -> bool:
    """Reset device allocator peak counters when the selected backend exposes them."""
    if torch_module is None or device is None:
        return False
    device_type = str(getattr(device, "type", device)).split(":", 1)[0]
    backend = getattr(torch_module, "cuda", None) if device_type == "cuda" else getattr(torch_module, "xpu", None) if device_type == "xpu" else None
    reset = getattr(backend, "reset_peak_memory_stats", None)
    if not callable(reset):
        return False
    try:
        reset(device)
        return True
    except TypeError:
        try:
            reset()
            return True
        except Exception:
            return False
    except Exception:
        _LOG.debug("Peak memory counters could not be reset for %s.", device_type, exc_info=True)
        return False


def _process_peak_rss_bytes() -> int | None:
    """Return a process high-water RSS counter when the operating system exposes one.

    Windows exposes peak working-set size through psutil. Unix resource usage
    is reported in bytes on macOS and KiB on Linux and other supported Unix
    platforms. Unsupported platforms deliberately return None.
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        for attribute in ("peak_wset", "peak_rss", "peak_working_set_size"):
            value = getattr(info, attribute, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                return int(value)
    except Exception:
        _LOG.debug("Peak process RSS is unavailable through psutil.", exc_info=True)

    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            return int(value) if sys.platform == "darwin" else int(value * 1024)
    except Exception:
        _LOG.debug("Peak process RSS is unavailable through resource usage.", exc_info=True)
    return None


def _device_memory_snapshot(
    torch_module: Any | None,
    device: Any | None,
    *,
    selected_backend: str | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read allocator counters where supported; unavailable metrics remain null, never guessed."""
    device_type = str(getattr(device, "type", device)).split(":", 1)[0] if device is not None else "unknown"
    result: dict[str, Any] = {
        "backend": selected_backend or device_type,
        "current_allocated_bytes": None,
        "current_reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "peak_increment_bytes": None,
        "process_rss_bytes": None,
        "process_rss_delta_bytes": None,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "process_peak_rss_delta_bytes": None,
        "peak_reset_supported": False,
        "device_allocator_metrics_supported": False,
    }
    if torch_module is not None and device is not None:
        try:
            if device_type in {"cuda", "xpu"}:
                backend = getattr(torch_module, device_type, None)
                methods = {
                    "current_allocated_bytes": "memory_allocated",
                    "current_reserved_bytes": "memory_reserved",
                    "peak_allocated_bytes": "max_memory_allocated",
                    "peak_reserved_bytes": "max_memory_reserved",
                }
                for target, name in methods.items():
                    function = getattr(backend, name, None)
                    if not callable(function):
                        continue
                    try:
                        value = function(device)
                    except TypeError:
                        value = function()
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        result[target] = max(0, int(value))
                result["device_allocator_metrics_supported"] = any(
                    result[key] is not None for key in methods
                )
            elif device_type == "mps":
                backend = getattr(torch_module, "mps", None)
                current = getattr(backend, "current_allocated_memory", None)
                reserved = getattr(backend, "driver_allocated_memory", None)
                if callable(current):
                    value = current()
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        result["current_allocated_bytes"] = max(0, int(value))
                if callable(reserved):
                    value = reserved()
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        result["current_reserved_bytes"] = max(0, int(value))
                result["device_allocator_metrics_supported"] = (
                    result["current_allocated_bytes"] is not None
                    or result["current_reserved_bytes"] is not None
                )
        except Exception:
            _LOG.debug("Live allocator memory telemetry unavailable for %s.", device_type, exc_info=True)
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
        result["process_rss_bytes"] = max(0, int(rss))
        if baseline and isinstance(baseline.get("process_rss_bytes"), int):
            result["process_rss_delta_bytes"] = int(rss) - baseline["process_rss_bytes"]
    except (ImportError, OSError, RuntimeError, AttributeError):
        pass
    if baseline and isinstance(baseline.get("current_allocated_bytes"), int) and isinstance(result.get("peak_allocated_bytes"), int):
        result["peak_increment_bytes"] = max(0, result["peak_allocated_bytes"] - baseline["current_allocated_bytes"])
    if baseline and isinstance(baseline.get("process_peak_rss_bytes"), int) and isinstance(result.get("process_peak_rss_bytes"), int):
        result["process_peak_rss_delta_bytes"] = max(
            0, result["process_peak_rss_bytes"] - baseline["process_peak_rss_bytes"]
        )
    if baseline:
        result["peak_reset_supported"] = bool(baseline.get("peak_reset_supported"))
    return result


def runtime_status() -> dict[str, Any]:
    packages: dict[str, bool] = {}
    versions: dict[str, str | None] = {}
    minimum_versions = {"torch": "2.2.0", "tokenizers": "0.19.0", "safetensors": "0.4.0"}
    problems: list[str] = []
    for name in ("torch", "tokenizers", "safetensors"):
        try:
            packages[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            packages[name] = False
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if not packages[name]:
            problems.append(f"{name} is not installed")
        elif versions[name] is not None and _version_key(versions[name]) < _version_key(minimum_versions[name]):
            problems.append(f"{name} {versions[name]} is too old (need >= {minimum_versions[name]})")
    # GGUF stays optional: environments without gguf-py can still use
    # safetensors checkpoints. GGUF is dequantized into floating-point compute
    # tensors when enabled; source quantization does not provide low-bit kernels.
    for name in ("gguf", "numpy"):
        try:
            packages[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            packages[name] = False
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    devices = _device_capability_rows()
    if packages["torch"] and not any(item["available"] for item in devices):
        problems.append("PyTorch is installed but no supported execution device could be initialized")
    ready = not problems
    recommended_device = next(
        (item["id"] for item in devices if item["available"] and item["id"] in {"cuda", "rocm", "mps", "xpu"}),
        "cpu" if any(item["id"] == "cpu" and item["available"] for item in devices) else None,
    )
    return {
        "engine": "axiom-native-pytorch-causal-lm",
        "ready": ready,
        "packages": packages,
        "versions": versions,
        "minimum_versions": minimum_versions,
        "problems": problems,
        "loaded": _LOADED_RUNTIME is not None,
        "architecture": _LOADED_ARCHITECTURE,
        "loaded_device": str(_LOADED_RUNTIME.get("device")) if _LOADED_RUNTIME is not None else None,
        "loaded_memory_mode": _LOADED_RUNTIME["model"].memory_mode if _LOADED_RUNTIME is not None else None,
        "loaded_precision": _LOADED_RUNTIME["details"].get("precision") if _LOADED_RUNTIME is not None else None,
        "devices": devices,
        "recommended_device": recommended_device,
        "model_loading": "local_safetensors_and_gguf_if_available",
        "execution_backend": "pytorch-native-axiom",
        "transformers_generation_used": False,
        "ollama_used": False,
        "remote_code": False,
        "supported_model_types": ["llama", "mistral", "qwen2", "qwen3"],
        "supported_weights": (
            ["safetensors", "gguf-dequantized-float"]
            if packages["gguf"] and packages["numpy"] else
            ["safetensors"]
        ),
        "setup_note": (
            "AXIOM Native Engine is ready for local Llama/Mistral/Qwen2/Qwen3 safetensors checkpoints"
            + (
                " and complete single-file or split-shard GGUF checkpoints (dequantized to floating-point; no low-bit kernels)."
                if packages["gguf"] and packages["numpy"] else "."
            )
            if ready else
            "AXIOM Native Engine setup required: " + "; ".join(problems)
            + ". Install with: python -m pip install -e '.[native-inference]'"
        ),
        "scope_note": (
            "AXIOM implements its own Llama/Mistral/Qwen2/Qwen3 forward pass, RMSNorm, RoPE, grouped-query attention, KV cache, "
            "LoRA application and token-by-token decoder using PyTorch tensor operations. It supports full-weight, CPU-staged "
            "layer streaming and disk-resident layer streaming with CPU KV-cache offload. GGUF files are dequantized to the chosen "
            "floating-point compute dtype; this is not low-bit inference and can use much more memory than the source quantized file. "
            "Single-file and metadata-validated split-shard GGUF are supported when gguf-py and NumPy are installed; EXL2/GPTQ/AWQ kernels, additional "
            "architectures and training remain unsupported. It does not call Transformers model.generate or an Ollama server."
        ),
        "checked_at": datetime.now(UTC).isoformat(),
    }

def runtime_telemetry() -> dict[str, Any]:
    """Return a privacy-bounded, point-in-time view of local runtime resources."""
    sampled_at = datetime.now(UTC).isoformat()
    cpu: dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "system_percent": None,
        "process_percent": None,
    }
    memory: dict[str, Any] = {
        "total_bytes": None,
        "available_bytes": None,
        "used_bytes": None,
        "percent": None,
    }
    process: dict[str, Any] = {
        "pid": os.getpid(),
        "rss_bytes": None,
        "peak_rss_bytes": _process_peak_rss_bytes(),
        "threads": None,
    }
    try:
        import psutil
        cpu["system_percent"] = float(psutil.cpu_percent(interval=None))
        process_handle = psutil.Process()
        process["rss_bytes"] = int(process_handle.memory_info().rss)
        process["threads"] = int(process_handle.num_threads())
        cpu["process_percent"] = float(process_handle.cpu_percent(interval=None))
        virtual = psutil.virtual_memory()
        memory.update({
            "total_bytes": int(virtual.total),
            "available_bytes": int(virtual.available),
            "used_bytes": int(virtual.used),
            "percent": float(virtual.percent),
        })
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
        _LOG.debug("psutil telemetry unavailable; trying basic operating-system memory counters.", exc_info=True)

    if memory["total_bytes"] is None and os.name == "posix":
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            total_bytes = total_pages * page_size
            available_bytes = available_pages * page_size
            if total_bytes > 0 and available_bytes >= 0:
                memory.update({
                    "total_bytes": total_bytes,
                    "available_bytes": available_bytes,
                    "used_bytes": max(0, total_bytes - available_bytes),
                    "percent": round((total_bytes - available_bytes) * 100 / total_bytes, 2),
                })
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    loaded = _LOADED_RUNTIME
    loaded_details = loaded.get("details", {}) if isinstance(loaded, dict) else {}
    loaded_model = loaded.get("model") if isinstance(loaded, dict) else None
    loaded_device = loaded.get("device") if isinstance(loaded, dict) else None
    loaded_torch = loaded.get("torch") if isinstance(loaded, dict) else None
    selected_backend = loaded_details.get("device_selection", {}).get("selected") if isinstance(loaded_details, dict) else None
    try:
        device_rows = [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "available": bool(item.get("available")),
                "memory_free_gib": item.get("memory_free_gib"),
                "memory_total_gib": item.get("memory_total_gib"),
            }
            for item in _device_capability_rows(loaded_torch)
        ]
    except Exception:
        _LOG.debug("Device capability telemetry could not be collected.", exc_info=True)
        device_rows = []

    allocator = None
    if loaded_torch is not None and loaded_device is not None:
        allocator = _device_memory_snapshot(loaded_torch, loaded_device, selected_backend=selected_backend)

    generation = None
    if loaded_model is not None:
        last = getattr(loaded_model, "last_generation", {})
        if isinstance(last, dict):
            allowed = {
                "prompt_tokens", "completion_tokens", "prefill_ms", "decode_ms",
                "time_to_first_token_ms", "generation_tokens_per_second",
                "stop_reason", "cache_layers",
            }
            generation = {
                key: value for key, value in last.items()
                if key in allowed and isinstance(value, (str, int, float))
            }

    disks: list[dict[str, Any]] = []
    try:
        roots = _allowed_model_roots()
    except (OSError, ValueError):
        roots = ()
    for index, root in enumerate(roots[:12], start=1):
        try:
            usage = shutil.disk_usage(root)
            disks.append({
                "label": f"trusted_model_root_{index}",
                "total_bytes": int(usage.total),
                "free_bytes": int(usage.free),
                "used_bytes": int(usage.used),
            })
        except OSError:
            continue

    return {
        "schema_version": 1,
        "sampled_at": sampled_at,
        "sample_type": "point_in_time",
        "cpu": cpu,
        "memory": memory,
        "process": process,
        "devices": device_rows,
        "loaded_runtime": {
            "loaded": loaded_model is not None,
            "model": loaded_details.get("model_directory_name") if isinstance(loaded_details, dict) else None,
            "architecture": loaded_details.get("architecture") if isinstance(loaded_details, dict) else None,
            "device": str(loaded_device) if loaded_device is not None else None,
            "memory_mode": getattr(loaded_model, "memory_mode", None) if loaded_model is not None else None,
            "precision": loaded_details.get("precision") if isinstance(loaded_details, dict) else None,
            "last_generation": generation,
            "device_allocator": allocator,
        },
        "disks": disks,
        "privacy_scope": {
            "prompt_content_included": False,
            "generated_content_included": False,
            "model_paths_included": False,
            "environment_values_included": False,
            "credentials_included": False,
            "network_access_performed": False,
        },
        "scope_note": (
            "This is a local point-in-time resource snapshot, not a persisted time series. "
            "Poll this endpoint for samples; unsupported driver metrics remain null."
        ),
    }


def _allowed_model_roots() -> tuple[Path, ...]:
    """Return canonical roots from trusted defaults and AXIOM_MODEL_ROOTS.

    Set AXIOM_MODEL_ROOTS to one or more trusted directories separated by the
    platform path separator to permit models stored outside the defaults.
    """
    home = Path.home()
    defaults = (
        Path.cwd() / "models",
        Path.cwd() / "runs",
        Path.cwd() / "outputs",
        home / "models",
        home / "Documents" / "models",
        home / "Documents" / "AXIOM-AI" / "models",
        home / "Documents" / "AXIOM-AI" / "runs",
        home / ".cache" / "huggingface" / "hub",
        home / ".local" / "share" / "axiom" / "models",
        home / ".local" / "share" / "axiom" / "fine-tuning",
    )
    configured = os.environ.get(_MODEL_ROOT_ENV, "").strip()
    custom_roots = tuple(Path(item).expanduser() for item in configured.split(os.pathsep) if item.strip())
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in (*defaults, *custom_roots):
        try:
            canonical = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            if candidate in custom_roots:
                raise ValueError("AXIOM_MODEL_ROOTS entries must be existing local directories.") from None
            continue
        if not canonical.is_dir():
            if candidate in custom_roots:
                raise ValueError("AXIOM_MODEL_ROOTS entries must be existing local directories.")
            continue
        key = str(canonical)
        if key not in seen:
            seen.add(key)
            roots.append(canonical)
    return tuple(roots)


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@lru_cache(maxsize=8)
def _trusted_file_entry(path: str, root_text: str) -> bool:
    try:
        resolved = os.path.realpath(path)
        return os.path.isfile(resolved) and os.path.commonpath((resolved, root_text)) == root_text
    except (OSError, ValueError):
        return False


def _trusted_directory_index(
    root_strings: tuple[str, ...],
    freshness_bucket: int,
) -> tuple[tuple[str, bool, bool, bool], ...]:
    """Index model directories by walking only configured trusted roots.

    Request strings are compared to this filesystem-derived index; they are
    never fed into Path, open, stat, resolve, or another filesystem operation.
    The time bucket expires the index every 30 seconds so newly installed models
    become selectable without an API restart.
    """
    del freshness_bucket
    entries: dict[str, tuple[str, bool, bool, bool]] = {}
    max_dirs_per_root = 50_000
    max_depth = 12
    for root_text in root_strings:
        visited = 0
        for directory, subdirs, filenames in os.walk(root_text, topdown=True, followlinks=False):
            visited += 1
            if visited > max_dirs_per_root:
                break
            try:
                relative = os.path.relpath(directory, root_text)
                depth = 0 if relative == os.curdir else len(Path(relative).parts)
            except (OSError, ValueError):
                subdirs[:] = []
                continue
            blocked_subdirs: list[str] = []
            allowed_subdirs: list[str] = []
            for name in subdirs:
                child = os.path.join(directory, name)
                if os.path.islink(child):
                    blocked_subdirs.append(child)
                else:
                    allowed_subdirs.append(name)
            subdirs[:] = allowed_subdirs if depth < max_depth else []
            for blocked in blocked_subdirs:
                key = os.path.normcase(os.path.normpath(blocked))
                entries[key] = (blocked, False, False, True)

            if depth > max_depth:
                continue
            has_config = "config.json" in filenames
            adapter_config = "adapter_config.json" in filenames
            gguf_names = tuple(name for name in filenames if name.lower().endswith(".gguf"))
            model_config = has_config or bool(gguf_names)
            if not model_config and not adapter_config:
                continue
            marker_names = (
                (("config.json",) if has_config else ())
                + (("adapter_config.json",) if adapter_config else ())
                + gguf_names
            )
            # HF cache metadata may be symlinked to blobs. Only accept targets
            # whose canonical path remains below this root.
            if any(not _trusted_file_entry(os.path.join(directory, marker), root_text) for marker in marker_names):
                continue
            try:
                canonical = os.path.realpath(directory)
                if os.path.commonpath((canonical, root_text)) != root_text:
                    continue
            except (OSError, ValueError):
                continue
            key = os.path.normcase(os.path.normpath(canonical))
            entries[key] = (canonical, model_config, adapter_config, False)
    return tuple(entries[key] for key in sorted(entries))


def _resolve_model_directory(value: str, label: str) -> Path:
    """Resolve a request by matching it to an inventoried trusted model folder."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required.")
    if len(value) > 4096 or any(character in value for character in ("\\x00", "\\r", "\\n")):
        raise ValueError(f"{label} contains invalid control characters or exceeds the path limit.")

    roots = _allowed_model_roots()
    if not roots:
        raise ValueError(
            "No allowed model directories were found. Create a model directory under ~/models "
            "or set AXIOM_MODEL_ROOTS to trusted local directories."
        )

    # Normalization here is lexical only. All filesystem reads are performed
    # against paths obtained independently from a walk of trusted roots.
    raw_value = os.path.expanduser(value.strip())
    # Reject traversal components up front. These are string operations only;
    # the request value is never passed to a filesystem API or Path constructor.
    components = raw_value.replace("\\", "/").split("/")
    if os.pardir in components:
        raise ValueError(f"{label} must not contain parent-directory traversal components.")

    root_strings = tuple(str(root) for root in roots)
    candidate_keys = {os.path.normcase(os.path.normpath(raw_value))}
    if not os.path.isabs(raw_value):
        # Relative CLI/UI input is first interpreted from the current working
        # directory; matching still happens against the inventory before any
        # filesystem operation uses a model path.
        cwd_key = os.getcwd().rstrip("/\\") + os.sep + raw_value
        candidate_keys.add(os.path.normcase(os.path.normpath(cwd_key)))
        candidate_keys.update(
            os.path.normcase(os.path.normpath(root_text.rstrip("/\\") + os.sep + raw_value))
            for root_text in root_strings
        )
    freshness_bucket = int(time.monotonic() // 30)
    entries = _trusted_directory_index(root_strings, freshness_bucket)
    for discovered, is_model, is_adapter, is_symlink in entries:
        key = os.path.normcase(os.path.normpath(discovered))
        if key not in candidate_keys:
            continue
        if is_symlink:
            raise ValueError(f"{label} cannot be a symlink; trusted-root escapes are not allowed.")
        lowered = label.casefold()
        if lowered == "base model" and not is_model:
            continue
        if lowered == "lora adapter" and not is_adapter:
            continue
        # The returned path is from the trusted directory inventory, never
        # constructed from the request-supplied path.
        return Path(discovered)
    raise ValueError(
        f"{label} must match an existing checkpoint directory discovered under a trusted model root. "
        "Set AXIOM_MODEL_ROOTS to add a trusted location and retry."
    )

def resolve_local_model_directory(value: str, label: str = "Model") -> str:
    """Validate an API-supplied path and return its canonical allowed location."""
    return str(_resolve_model_directory(value, label))


def check_local_adapter_compatibility(model_path: str, adapter_path: str) -> dict[str, Any]:
    """Preflight a local LoRA adapter against the native model and available device backends."""
    model_dir = _resolve_model_directory(model_path, "Base model")
    adapter_dir = _resolve_model_directory(adapter_path, "LoRA adapter")
    from axiom.core.native_engine import inspect_native_adapter_compatibility

    return inspect_native_adapter_compatibility(
        model_dir,
        adapter_dir,
        trusted_roots=_allowed_model_roots(),
    )


def estimate_native_quantization(
    model_path: str,
    *,
    context_length: int = 4096,
    batch_size: int = 1,
    ram_budget_gib: float | None = None,
    vram_budget_gib: float | None = None,
) -> dict[str, Any]:
    """Estimate model/KV/workspace memory using config dimensions without loading tensors.

    Quantized candidates are planning-only until an AXIOM-native quantized kernel
    and checkpoint loader are implemented. The estimate is never presented as a
    measured memory or model-quality result.
    """
    if isinstance(context_length, bool) or not isinstance(context_length, int) or not 1 <= context_length <= 131072:
        raise ValueError("Context length must be an integer from 1 through 131072.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
        raise ValueError("Batch size must be an integer from 1 through 16.")
    for label, budget in (("RAM", ram_budget_gib), ("VRAM", vram_budget_gib)):
        if budget is not None and (
            not isinstance(budget, (int, float))
            or isinstance(budget, bool)
            or not math.isfinite(float(budget))
            or not 0.1 <= float(budget) <= 2048
        ):
            raise ValueError(f"{label} budget must be between 0.1 and 2048 GiB.")

    model_dir = _resolve_model_directory(model_path, "Base model")
    inspected = _inspect_model_config(model_dir)
    model_config = inspected["config"]
    from axiom.core.native_engine import NativeLlamaConfig
    native_config = NativeLlamaConfig.from_mapping(model_config)
    if context_length > native_config.effective_max_position_embeddings:
        raise ValueError(
            f"Context length {context_length} exceeds this checkpoint's limit "
            f"of {native_config.effective_max_position_embeddings} tokens (including supported RoPE scaling)."
        )

    hidden = native_config.hidden_size
    intermediate = native_config.intermediate_size
    heads = native_config.num_attention_heads
    kv_heads = native_config.num_key_value_heads
    head_dim = hidden // heads
    layers = native_config.num_hidden_layers
    vocab = native_config.vocab_size
    kv_width = kv_heads * head_dim
    kv_cache_context = (
        min(context_length, native_config.sliding_window)
        if native_config.sliding_window
        else context_length
    )
    kv_prefill_context = (
        context_length
        if native_config.sliding_window and context_length > native_config.sliding_window
        else 0
    )

    embedding_params = vocab * hidden
    lm_head_params = 0 if native_config.tie_word_embeddings else vocab * hidden
    per_layer_params = (
        2 * hidden * hidden
        + 2 * hidden * kv_width
        + 3 * hidden * intermediate
        + 2 * hidden
    )
    if model_config.get("attention_bias") is True:
        per_layer_params += 2 * hidden + 2 * kv_width
    if model_config.get("mlp_bias") is True:
        per_layer_params += 2 * intermediate + hidden
    estimated_parameters = embedding_params + lm_head_params + layers * per_layer_params + hidden

    gib = 1024 ** 3
    workspace_bytes = max(
        batch_size * context_length * intermediate * 4,
        batch_size * heads * context_length * context_length * 4,
    )
    candidates = [
        ("int4", 4, 0.5, 1.15, 2, False, "Quantized weights are not supported by AXIOM Native Engine yet."),
        ("int8", 8, 1.0, 1.05, 2, False, "Quantized weights are not supported by AXIOM Native Engine yet."),
        ("bf16", 16, 2.0, 1.0, 2, True, None),
        ("fp16", 16, 2.0, 1.0, 2, True, None),
        ("fp32", 32, 4.0, 1.0, 4, True, None),
    ]
    candidate_results: list[dict[str, Any]] = []
    for name, bits, bytes_per_parameter, storage_overhead, kv_bytes, native_supported, reason in candidates:
        weight_bytes = estimated_parameters * bytes_per_parameter * storage_overhead
        kv_bytes_total = 2 * layers * kv_heads * head_dim * kv_cache_context * batch_size * kv_bytes
        kv_prefill_temporary_bytes = (
            2 * kv_heads * head_dim * kv_prefill_context * batch_size * kv_bytes
        )
        unbuffered_total = (
            weight_bytes + kv_bytes_total + kv_prefill_temporary_bytes + workspace_bytes
        )
        total_bytes = unbuffered_total * 1.20
        total_gib = total_bytes / gib
        uncertainty_low = total_gib * 0.85
        uncertainty_high = total_gib * 1.25
        candidate_results.append({
            "format": name,
            "nominal_weight_bits": bits,
            # Preserve enough precision that small test/checkpoint models do
            # not collapse all candidate formats to 0.000 GiB.
            "estimated_weight_gib": round(weight_bytes / gib, 9),
            "estimated_kv_cache_gib": round(kv_bytes_total / gib, 9),
            "estimated_kv_prefill_temporary_gib": round(kv_prefill_temporary_bytes / gib, 9),
            "estimated_workspace_gib": round(workspace_bytes / gib, 9),
            "estimated_total_gib": round(total_gib, 9),
            "estimated_range_gib": [round(uncertainty_low, 9), round(uncertainty_high, 9)],
            "fits_ram_budget": None if ram_budget_gib is None else total_gib <= float(ram_budget_gib),
            "fits_vram_budget": None if vram_budget_gib is None else total_gib <= float(vram_budget_gib),
            "native_checkpoint_supported": native_supported,
            "native_support_note": reason,
        })

    native_candidates = [item for item in candidate_results if item["native_checkpoint_supported"]]
    fitting_candidates = [
        item for item in native_candidates
        if (ram_budget_gib is None or item["fits_ram_budget"] is True)
        and (vram_budget_gib is None or item["fits_vram_budget"] is True)
    ]
    recommended = min(fitting_candidates, key=lambda item: item["estimated_total_gib"]) if fitting_candidates else None
    recommendation = {
        "format": recommended["format"] if recommended else None,
        "status": "fits_budget" if recommended else "no_native_candidate_fits",
        "note": (
            "Lowest estimated-memory native format that fits every supplied budget. This is a planning suggestion, "
            "not a conversion and not a guarantee that the checkpoint's stored dtype or device kernels are compatible."
            if recommended else
            "No currently native-supported precision is estimated to fit every supplied budget. Reduce context/batch "
            "or increase the budget; INT4/INT8 are planning-only and cannot be loaded by the native engine."
        ),
    }

    return {
        "schema_version": 1,
        "model": {
            "name": model_dir.name,
            "model_type": native_config.model_type,
            "architecture": inspected["architecture"],
            "estimated_parameter_count": estimated_parameters,
            "parameter_count_method": "analytic estimate from config dimensions; not a tensor-index count",
        },
        "request": {
            "context_length": context_length,
            "batch_size": batch_size,
            "ram_budget_gib": ram_budget_gib,
            "vram_budget_gib": vram_budget_gib,
        },
        "assumptions": {
            "weight_storage": {
                "fp32": "4 bytes/parameter",
                "fp16": "2 bytes/parameter",
                "bf16": "2 bytes/parameter",
                "int8": "1 byte/parameter plus 5% scales/metadata",
                "int4": "0.5 bytes/parameter plus 15% scales/metadata",
            },
            "kv_cache_context_length": kv_cache_context,
            "kv_prefill_temporary_context_length": kv_prefill_context,
            "kv_cache_formula": "2 × layers × KV heads × head dimension × retained cache context × batch × KV element bytes; Mistral retained context is bounded by sliding_window",
            "kv_prefill_temporary_formula": "For Mistral prefill beyond sliding_window: 2 × KV heads × head dimension × full prefill context × batch × KV element bytes for the active layer",
            "kv_cache_precision": "FP32 candidate uses 4-byte cache elements; all other candidates assume 2-byte cache elements",
            "workspace_formula": "max(batch × context × intermediate × 4, batch × attention heads × context² × 4)",
            "safety_margin_percent": 20,
            "uncertainty_range_percent": {"low": -15, "high": 25},
            "quality_and_speed": "Not estimated; must be measured through model evaluation and runtime benchmarks.",
        },
        "candidates": candidate_results,
        "recommendation": recommendation,
        "scope_note": (
            "Planning estimates only. They do not inspect tensor contents or measure physical memory, and do not predict model quality/speed. "
            "INT4/INT8 are sizing-only and cannot be loaded by the current AXIOM Native Engine. "
            "FP16/BF16/FP32 availability still depends on the actual checkpoint dtype, device kernels and free memory. "
            "Estimated fit is not a guarantee of successful loading."
        ),
    }


def discover_local_engine_assets() -> dict[str, Any]:
    """List metadata for native-compatible local folders without loading model weights."""
    roots = _allowed_model_roots()
    if not roots:
        return {
            "schema_version": 1,
            "models": [],
            "adapters": [],
            "truncated": False,
            "scope_note": "No trusted model roots exist yet. Create ~/models or configure AXIOM_MODEL_ROOTS.",
        }

    root_strings = tuple(str(root) for root in roots)
    entries = _trusted_directory_index(root_strings, int(time.monotonic() // 30))
    models: list[dict[str, Any]] = []
    adapters: list[dict[str, Any]] = []
    max_results = 500
    for discovered, is_model, is_adapter, is_symlink in entries:
        if is_symlink:
            continue
        directory = Path(discovered)
        if is_model and len(models) < max_results:
            try:
                inspected = _inspect_model_config(directory)
                from axiom.core.native_engine import NativeLlamaConfig

                native_config = NativeLlamaConfig.from_mapping(inspected["config"])
                if inspected.get("checkpoint_format") == "gguf":
                    if not inspected.get("tokenizer_source"):
                        continue
                    models.append({
                        "path": discovered,
                        "name": directory.name,
                        "model_type": native_config.model_type,
                        "architecture": inspected["architecture"],
                        "weight_format": "GGUF (AXIOM dequantizes to floating point)",
                        "max_position_embeddings": native_config.max_position_embeddings,
                        "effective_max_position_embeddings": native_config.effective_max_position_embeddings,
                        "rope_scaling_type": native_config.rope_scaling_type,
                        "ready_for_native_attempt": True,
                    })
                    continue
                tokenizer_ok = any(_trusted_file_entry(str(directory / "tokenizer.json"), str(root)) for root in roots)
                single_weights = any(_trusted_file_entry(str(directory / "model.safetensors"), str(root)) for root in roots)
                indexed_weights = any(_trusted_file_entry(str(directory / "model.safetensors.index.json"), str(root)) for root in roots)
                if not tokenizer_ok or not (single_weights or indexed_weights):
                    continue
                models.append({
                    "path": discovered,
                    "name": directory.name,
                    "model_type": native_config.model_type,
                    "architecture": inspected["architecture"],
                    "weight_format": "safetensors" if single_weights else "sharded safetensors",
                    "max_position_embeddings": native_config.max_position_embeddings,
                    "effective_max_position_embeddings": native_config.effective_max_position_embeddings,
                    "rope_scaling_type": native_config.rope_scaling_type,
                    "ready_for_native_attempt": True,
                })
            except (ValueError, OSError, RuntimeError):
                # Incompatible or incomplete folders are not offered as runnable models.
                continue
        if is_adapter and len(adapters) < max_results:
            adapter_config = directory / "adapter_config.json"
            adapter_weights = directory / "adapter_model.safetensors"
            config_trusted = any(_trusted_file_entry(str(adapter_config), str(root)) for root in roots)
            weights_trusted = any(_trusted_file_entry(str(adapter_weights), str(root)) for root in roots)
            if config_trusted and weights_trusted:
                adapters.append({
                    "path": discovered,
                    "name": directory.name,
                    "weight_format": "safetensors LoRA",
                })
        if len(models) >= max_results and len(adapters) >= max_results:
            break

    models.sort(key=lambda item: (item["name"].casefold(), item["path"]))
    adapters.sort(key=lambda item: (item["name"].casefold(), item["path"]))
    return {
        "schema_version": 1,
        "models": models,
        "adapters": adapters,
        "truncated": len(models) >= max_results or len(adapters) >= max_results,
        "scope_note": (
            "Metadata-only discovery under trusted local roots. Weight tensors are not loaded or fully validated "
            "until an inference request runs. AXIOM does not download models during discovery."
        ),
    }




def _inspect_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    safetensors_present = (
        (model_dir / "model.safetensors").is_file()
        or (model_dir / "model.safetensors.index.json").is_file()
    )
    try:
        gguf_present = any(
            item.is_file() and item.suffix.lower() == ".gguf"
            for item in model_dir.iterdir()
        )
    except OSError:
        gguf_present = False
    # The runtime prefers safetensors when they exist; otherwise GGUF metadata
    # is authoritative even if a stale/foreign config.json sidecar is present.
    if not safetensors_present and (gguf_present or not config_path.exists()):
        from axiom.core.native_gguf import inspect_gguf_checkpoint
        return inspect_gguf_checkpoint(model_dir, trusted_roots=_allowed_model_roots())
    try:
        resolved_config = config_path.resolve(strict=True)
        if (
            not resolved_config.is_file()
            or resolved_config.stat().st_size > _MAX_CONFIG_BYTES
            or not any(_path_is_within_root(resolved_config, root) for root in _allowed_model_roots())
        ):
            raise ValueError("Base model directory must contain a config.json inside a trusted root and no larger than 1 MiB.")
        config = json.loads(resolved_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("Local model config.json could not be read as valid UTF-8 JSON.") from None
    if not isinstance(config, dict) or not isinstance(config.get("model_type"), str):
        raise ValueError("Local model config.json must contain a model_type string.")
    architectures = config.get("architectures", [])
    architecture = architectures[0] if isinstance(architectures, list) and architectures and isinstance(architectures[0], str) else config["model_type"]
    return {
        "config": config,
        "model_type": config["model_type"],
        "architecture": architecture,
        "checkpoint_format": "safetensors",
    }


def validate_generation(prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, int, float]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must contain at least one non-whitespace character.")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds the {_MAX_PROMPT_CHARS:,}-character limit.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= _MAX_NEW_TOKENS:
        raise ValueError(f"Maximum generated tokens must be between 1 and {_MAX_NEW_TOKENS}.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    return prompt, max_new_tokens, float(temperature)


def validate_sampling_parameters(
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int | None,
) -> tuple[float, int, float, int | None]:
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < float(top_p) <= 1:
        raise ValueError("Top-p must be greater than 0 and at most 1.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 100_000:
        raise ValueError("Top-k must be an integer from 0 through 100000; 0 disables top-k.")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not 0.1 <= float(repetition_penalty) <= 5
    ):
        raise ValueError("Repetition penalty must be between 0.1 and 5.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1)
    ):
        raise ValueError("Sampling seed must be an integer from 0 through 2^63-1.")
    return float(top_p), top_k, float(repetition_penalty), seed


def validate_min_p(min_p: float) -> float:
    """Validate the probability-relative min-p sampler threshold."""
    if isinstance(min_p, bool) or not isinstance(min_p, (int, float)) or not math.isfinite(float(min_p)) or not 0 <= float(min_p) <= 1:
        raise ValueError("Min-p must be between 0 and 1; 0 disables min-p.")
    return float(min_p)


def validate_stop_sequences(stop_sequences: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Validate bounded stop strings while preserving intentional surrounding spaces."""
    if stop_sequences is None:
        return ()
    if not isinstance(stop_sequences, (list, tuple)):
        raise ValueError("Stop sequences must be a list of text strings.")
    if len(stop_sequences) > 16:
        raise ValueError("At most 16 stop sequences may be configured per inference request.")
    validated: list[str] = []
    for sequence in stop_sequences:
        if not isinstance(sequence, str):
            raise ValueError("Every stop sequence must be text.")
        if not sequence or not sequence.strip():
            raise ValueError("Stop sequences must contain at least one non-whitespace character.")
        if len(sequence) > 256:
            raise ValueError("Each stop sequence is limited to 256 characters.")
        if sequence not in validated:
            validated.append(sequence)
    return tuple(validated)


def _pending_stop_prefix_length(text: str, stop_sequences: tuple[str, ...]) -> int:
    """Length of the longest text suffix that could become a stop sequence."""
    longest = 0
    for sequence in stop_sequences:
        upper = min(len(sequence) - 1, len(text))
        for size in range(upper, 0, -1):
            if size > longest and text.endswith(sequence[:size]):
                longest = size
                break
    return longest


def _available_system_memory_bytes() -> int | None:
    """Get observable available system RAM without assuming psutil is installed."""
    try:
        import psutil
        return max(0, int(psutil.virtual_memory().available))
    except (ImportError, OSError, RuntimeError, AttributeError):
        pass
    if os.name == "posix":
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return None


def _choose_memory_mode(
    model_dir: Path,
    inspected: dict[str, Any],
    torch_module: Any,
    device: Any,
    device_selection: dict[str, str],
    adapter_dir: Path | None = None,
    precision: str = "auto",
) -> tuple[str, dict[str, Any]]:
    """Choose a native loading strategy from observed RAM/VRAM and model dimensions.

    Estimates intentionally use a safety margin and return their assumptions.
    A choice is advisory, not a guarantee that the selected kernel/model will fit.
    """
    from axiom.core.native_engine import NativeLlamaConfig, _resolve_safetensor_files, _resolve_trusted_file

    checkpoint_format = inspected.get("checkpoint_format", "safetensors")
    device_type = str(getattr(device, "type", device)).split(":", 1)[0]
    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}.get(precision, 4)
    checkpoint_source_bytes = 0
    try:
        config = NativeLlamaConfig.from_mapping(inspected["config"])
        if checkpoint_format == "gguf":
            from axiom.core.native_gguf import _find_files

            weight_files = _find_files(model_dir, _allowed_model_roots())
            checkpoint_source_bytes = sum(int(path.stat().st_size) for path in weight_files)
            weight_elements = inspected.get("gguf_weight_elements")
            if isinstance(weight_elements, bool) or not isinstance(weight_elements, int) or weight_elements <= 0:
                raise ValueError("GGUF tensor element count is unavailable for safe memory planning")
            # GGUF is loaded as floating-point tensors, not in its source
            # quantization. Auto precision mirrors the native GGUF loader:
            # FP32 on CPU, FP16 on supported accelerator devices.
            if precision == "auto":
                dtype_bytes = 4 if device_type == "cpu" else 2
            weight_bytes = weight_elements * dtype_bytes
        else:
            weight_files, _weight_map = _resolve_safetensor_files(model_dir, _allowed_model_roots())
            checkpoint_source_bytes = sum(int(path.stat().st_size) for path in weight_files)
            weight_bytes = checkpoint_source_bytes
            if precision == "auto":
                try:
                    from safetensors import safe_open
                    for shard in weight_files:
                        with safe_open(str(shard), framework="pt", device="cpu") as handle:
                            if "model.embed_tokens.weight" not in handle.keys():
                                continue
                            dtype_name = handle.get_slice("model.embed_tokens.weight").get_dtype()
                            dtype_bytes = {"F16": 2, "BF16": 2, "F32": 4}.get(str(dtype_name).upper(), 4)
                            break
                except Exception:
                    # Header inspection is an optional precision improvement.
                    # Keep the conservative FP32 estimate on format/driver errors.
                    _LOG.debug("Could not inspect checkpoint dtype for memory planning.", exc_info=True)

        if checkpoint_source_bytes <= 0 or weight_bytes <= 0:
            raise ValueError("the checkpoint weight size is unavailable")
        adapter_weight_bytes = 0
        if adapter_dir is not None:
            adapter_file = _resolve_trusted_file(
                adapter_dir / "adapter_model.safetensors",
                _allowed_model_roots(),
                "LoRA safetensors checkpoint",
            )
            adapter_weight_bytes = int(adapter_file.stat().st_size)
            if adapter_weight_bytes <= 0:
                raise ValueError("the LoRA adapter weight file is empty")
    except Exception as exc:
        raise AxiomEngineError(f"AXIOM could not safely estimate this checkpoint for automatic memory selection: {exc}") from None

    gib = 1024 ** 3
    context = min(config.effective_max_position_embeddings, _MAX_INPUT_TOKENS + _MAX_NEW_TOKENS)
    # The retained Mistral KV cache is bounded by sliding_window, but prefill
    # still materializes K/V for the full prompt while the active layer runs.
    # Budget both: persistent cache uses the bounded length; peak estimates add
    # one full-context K/V temporary rather than pretending every layer retains it.
    kv_cache_context = min(context, config.sliding_window) if config.sliding_window else context
    head_dim = config.hidden_size // config.num_attention_heads
    embedding_bytes = config.vocab_size * config.hidden_size * dtype_bytes
    head_bytes = 0 if config.tie_word_embeddings else embedding_bytes
    norm_bytes = config.hidden_size * dtype_bytes
    resident_non_layer = embedding_bytes + head_bytes + norm_bytes
    resident_with_adapter = resident_non_layer + adapter_weight_bytes
    total_weight_bytes = weight_bytes + adapter_weight_bytes
    kv_total = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * kv_cache_context * dtype_bytes
    kv_prefill_per_layer = (
        2 * config.num_key_value_heads * head_dim * context * dtype_bytes
        if config.sliding_window and context > config.sliding_window
        else 0
    )
    workspace = max(context * config.intermediate_size * dtype_bytes * 2, config.hidden_size * context * dtype_bytes * 4)
    non_layer_weight_bytes = resident_with_adapter
    layer_bytes = max(0, weight_bytes - resident_non_layer) / max(1, config.num_hidden_layers)
    active_layer_bytes = int(layer_bytes * 1.25)
    full_required = int((total_weight_bytes + kv_total + kv_prefill_per_layer + workspace) * 1.25)
    cpu_layer_required = int((total_weight_bytes + kv_total + kv_prefill_per_layer + workspace) * 1.15)
    disk_cpu_required = int((resident_with_adapter + active_layer_bytes + kv_total + kv_prefill_per_layer + workspace) * 1.30)
    streamed_device_required = int((resident_with_adapter + active_layer_bytes + kv_prefill_per_layer + workspace) * 1.25)
    system_available = _available_system_memory_bytes()
    try:
        disk_free = int(shutil.disk_usage(model_dir).free)
    except OSError:
        disk_free = None
    rows = _device_capability_rows(torch_module)
    selected_id = device_selection.get("selected", str(getattr(device, "type", device)))
    device_row = next((row for row in rows if row.get("id") == selected_id), None)
    device_free = None
    if device_row and isinstance(device_row.get("memory_free_gib"), (int, float)):
        device_free = int(float(device_row["memory_free_gib"]) * gib)
    device_type = str(getattr(device, "type", device)).split(":", 1)[0]
    base = {
        "requested": "auto",
        "selected": "full",
        "reason": "",
        "checkpoint_format": checkpoint_format,
        "estimated_checkpoint_weight_bytes": weight_bytes,
        "estimated_source_checkpoint_bytes": checkpoint_source_bytes,
        "estimated_adapter_weight_bytes": adapter_weight_bytes,
        "estimated_total_weight_bytes": total_weight_bytes,
        "estimated_non_layer_weight_bytes": non_layer_weight_bytes,
        "estimated_kv_cache_bytes": kv_total,
        "estimated_kv_cache_context_tokens": kv_cache_context,
        "estimated_kv_prefill_temporary_bytes": kv_prefill_per_layer,
        "estimated_context_tokens": context,
        "estimated_dtype_bytes": dtype_bytes,
        "estimated_full_execution_bytes": full_required,
        "estimated_streaming_device_bytes": streamed_device_required,
        "estimated_disk_streaming_system_ram_bytes": disk_cpu_required,
        "available_system_ram_bytes": system_available,
        "available_device_memory_bytes": device_free,
        "available_checkpoint_disk_bytes": disk_free,
        "estimates_are_advisory": True,
    }
    if device_type == "cpu":
        if system_available is not None and system_available >= full_required:
            mode, reason = "full", "The model plus estimated KV/workspace fits available system RAM with a safety margin."
        elif system_available is not None and system_available >= disk_cpu_required:
            mode, reason = "disk_streaming", "Full checkpoint residency is too large for available RAM; disk-resident layer loading has a lower estimated RAM footprint."
        else:
            if system_available is None:
                raise AxiomEngineError(
                    "Automatic memory planning cannot safely select CPU execution because available system RAM could not be measured. "
                    "Install psutil or choose an explicit memory mode after checking your machine's memory."
                )
            raise AxiomEngineError(
                "Automatic memory planning found no estimated CPU strategy that fits available RAM. "
                f"Estimated full RAM need: {full_required / gib:.2f} GiB; "
                f"disk-streaming RAM need: {disk_cpu_required / gib:.2f} GiB; "
                f"available RAM: {system_available / gib:.2f} GiB. "
                "Free memory or select a smaller model/context."
            )
    elif device_free is None:
        if system_available is None:
            raise AxiomEngineError(
                "Automatic memory planning cannot safely choose a streaming mode because neither free accelerator memory "
                "nor available system RAM was measurable. Install psutil or choose an explicit memory mode after checking resources."
            )
        if system_available >= cpu_layer_required:
            mode, reason = "layer_streaming", "PyTorch did not expose free VRAM; available system RAM is estimated to hold the model's layer weights and KV/workspace."
        elif system_available >= disk_cpu_required:
            mode, reason = "disk_streaming", "PyTorch did not expose free VRAM; available system RAM is estimated to be insufficient for all layer weights, so on-demand disk loading is selected."
        else:
            raise AxiomEngineError(
                "Automatic memory planning found no estimated streaming strategy that fits available system RAM. "
                f"Estimated in-RAM streaming need: {cpu_layer_required / gib:.2f} GiB; "
                f"disk-streaming need: {disk_cpu_required / gib:.2f} GiB; available RAM: {system_available / gib:.2f} GiB."
            )
    elif device_free >= full_required:
        mode, reason = "full", "The checkpoint plus estimated KV/workspace fits observed free accelerator memory with a safety margin."
    elif device_free >= streamed_device_required and system_available is not None and system_available >= cpu_layer_required:
        mode, reason = "layer_streaming", "The full checkpoint exceeds observed accelerator memory, while CPU RAM can hold layer weights and the estimated KV/workspace."
    elif device_free >= streamed_device_required and system_available is not None and system_available >= disk_cpu_required:
        mode, reason = "disk_streaming", "The full checkpoint exceeds accelerator memory and RAM cannot comfortably hold every layer; selected on-demand disk layer reads."
    else:
        ram_text = (
            f"{system_available / gib:.2f} GiB"
            if system_available is not None
            else "not measurable"
        )
        raise AxiomEngineError(
            "Automatic memory planning could not find an estimated strategy that fits this device. "
            f"Estimated full accelerator need: {full_required / gib:.2f} GiB; "
            f"streamed-device need: {streamed_device_required / gib:.2f} GiB; "
            f"observed free accelerator memory: {device_free / gib:.2f} GiB; "
            f"available system RAM: {ram_text}. Try a smaller model/context, free memory, or use CPU."
        )
    base["selected"] = mode
    base["reason"] = reason
    return mode, base

def _load_local_model(
    model_path: str,
    adapter_path: str | None,
    device_preference: str = "auto",
    memory_mode: str = "auto",
    precision: str = "auto",
):
    global _LOADED_RUNTIME, _LOADED_ARCHITECTURE
    try:
        import torch
    except (ImportError, OSError):
        raise AxiomEngineError(
            "AXIOM Native Engine requires PyTorch. Install with: python -m pip install -e '.[native-inference]'"
        ) from None

    if memory_mode not in {"auto", "full", "layer_streaming", "disk_streaming"}:
        raise ValueError("Memory mode must be auto, full, layer_streaming, or disk_streaming.")
    if precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("Precision must be auto, fp32, fp16, or bf16.")
    requested_memory_mode = memory_mode
    base_dir = _resolve_model_directory(model_path, "Base model")
    inspected = _inspect_model_config(base_dir)
    resolved_adapter: Path | None = None
    if adapter_path and adapter_path.strip():
        adapter_dir = _resolve_model_directory(adapter_path, "LoRA adapter")
        if not (adapter_dir / "adapter_config.json").is_file() or not (adapter_dir / "adapter_model.safetensors").is_file():
            raise ValueError(
                "Native LoRA adapter must contain adapter_config.json and adapter_model.safetensors; "
                "pickle-based adapter weights are not supported."
            )
        resolved_adapter = adapter_dir

    device, device_selection = _resolve_execution_device(torch, device_preference)
    memory_selection: dict[str, Any] = {
        "requested": requested_memory_mode,
        "selected": memory_mode,
        "reason": "Memory strategy was explicitly selected by the caller." if requested_memory_mode != "auto" else "",
        "estimates_are_advisory": True,
    }
    if requested_memory_mode == "auto":
        try:
            memory_mode, memory_selection = _choose_memory_mode(
                base_dir, inspected, torch, device, device_selection, resolved_adapter, precision
            )
        except AxiomEngineError as planning_error:
            # If the accelerator is already estimated not to fit, do not wait
            # for a real allocation failure before considering CPU. Explicit
            # device choices remain strict, and malformed/unsupported checkpoints
            # must not be disguised as a memory fallback.
            device_type = str(getattr(device, "type", device)).split(":", 1)[0]
            no_accelerator_fit = (
                "could not find an estimated strategy that fits this device"
                in str(planning_error).lower()
            )
            if (
                device_preference.strip().lower() != "auto"
                or device_type not in {"cuda", "mps", "xpu"}
                or not no_accelerator_fit
            ):
                raise
            cpu_device = torch.device("cpu")
            cpu_selection = {
                "requested": "auto",
                "selected": "cpu",
                "reason": (
                    "The accelerator memory planner found no estimated fitting strategy; "
                    "AXIOM is checking whether a CPU memory strategy fits."
                ),
            }
            try:
                memory_mode, memory_selection = _choose_memory_mode(
                    base_dir, inspected, torch, cpu_device, cpu_selection, resolved_adapter, precision
                )
            except AxiomEngineError as cpu_planning_error:
                raise AxiomEngineError(
                    f"{planning_error} CPU fallback was also unavailable: {cpu_planning_error}"
                ) from None
            device = cpu_device
            device_selection = {
                "requested": "auto",
                "selected": "cpu",
                "reason": (
                    "The accelerator memory estimate had no fitting strategy; "
                    "CPU fallback was selected after its memory planner succeeded."
                ),
            }
            memory_selection = {
                **memory_selection,
                "fallback_from": device_type,
                "fallback_reason": "No estimated accelerator strategy fit; CPU memory planning succeeded.",
            }
    key = (str(base_dir), str(resolved_adapter) if resolved_adapter else None, str(device), device_preference, memory_mode, requested_memory_mode, precision)
    if _LOADED_RUNTIME is not None and _LOADED_RUNTIME["key"] == key:
        return (
            _LOADED_RUNTIME["model"],
            _LOADED_RUNTIME["tokenizer"],
            _LOADED_RUNTIME["torch"],
            _LOADED_RUNTIME["device"],
            _LOADED_RUNTIME["details"],
        )

    previous_runtime = _LOADED_RUNTIME
    _LOADED_RUNTIME = None
    _LOADED_ARCHITECTURE = None
    if previous_runtime is not None:
        previous_torch = previous_runtime.get("torch")
        previous_device = previous_runtime.get("device")
        del previous_runtime
        gc.collect()
        _clear_device_cache(previous_torch, previous_device)

    from axiom.core.native_engine import NativeDependencyError, NativeEngineError, load_native_runtime
    try:
        model, tokenizer, native_details = load_native_runtime(
            base_dir, torch, device, adapter_dir=resolved_adapter, trusted_roots=_allowed_model_roots(), memory_mode=memory_mode, precision=precision
        )
    except NativeDependencyError as exc:
        raise AxiomEngineError(str(exc)) from None
    except NativeEngineError as exc:
        # Auto mode makes one transparent CPU retry only for an accelerator OOM.
        # Explicit backend choices never silently switch devices.
        device_type = str(getattr(device, "type", device)).split(":", 1)[0]
        if device_preference.strip().lower() != "auto" or device_type not in {"cuda", "mps", "xpu"} or "out of memory" not in str(exc).lower():
            raise ValueError(str(exc)) from None
        _LOG.warning("Native accelerator load ran out of memory; retrying this checkpoint on CPU.")
        _clear_device_cache(torch, device)
        gc.collect()
        device = torch.device("cpu")
        device_selection = {
            "requested": "auto",
            "selected": "cpu",
            "reason": "The preferred accelerator ran out of memory during model loading; AXIOM retried on CPU.",
        }
        if requested_memory_mode == "auto":
            memory_mode, memory_selection = _choose_memory_mode(base_dir, inspected, torch, device, device_selection, resolved_adapter, precision)
        try:
            model, tokenizer, native_details = load_native_runtime(
                base_dir, torch, device, adapter_dir=resolved_adapter, trusted_roots=_allowed_model_roots(), memory_mode=memory_mode, precision=precision
            )
        except NativeDependencyError as retry_exc:
            raise AxiomEngineError(str(retry_exc)) from None
        except NativeEngineError as retry_exc:
            raise ValueError(f"Accelerator memory was insufficient and CPU retry also failed: {retry_exc}") from None
        except (OSError, RuntimeError):
            _LOG.debug("CPU retry after accelerator OOM failed.", exc_info=True)
            raise AxiomEngineError("AXIOM Native Engine could not load the checkpoint after an accelerator memory failure and CPU retry.") from None
    except (OSError, RuntimeError) as exc:
        # Handle native device allocation failures that aren't wrapped by the loader.
        device_type = str(getattr(device, "type", device)).split(":", 1)[0]
        if device_preference.strip().lower() == "auto" and device_type in {"cuda", "mps", "xpu"} and "out of memory" in str(exc).lower():
            _LOG.warning("Native accelerator load ran out of memory; retrying this checkpoint on CPU.")
            _clear_device_cache(torch, device)
            gc.collect()
            device = torch.device("cpu")
            device_selection = {"requested": "auto", "selected": "cpu", "reason": "The preferred accelerator ran out of memory during model loading; AXIOM retried on CPU."}
            if requested_memory_mode == "auto":
                memory_mode, memory_selection = _choose_memory_mode(base_dir, inspected, torch, device, device_selection, resolved_adapter, precision)
            try:
                model, tokenizer, native_details = load_native_runtime(
                    base_dir, torch, device, adapter_dir=resolved_adapter, trusted_roots=_allowed_model_roots(), memory_mode=memory_mode, precision=precision
                )
            except (NativeDependencyError, NativeEngineError, OSError, RuntimeError) as retry_exc:
                _LOG.debug("CPU retry after accelerator OOM failed.", exc_info=True)
                raise AxiomEngineError(f"AXIOM Native Engine could not load the checkpoint after an accelerator memory failure and CPU retry: {retry_exc}") from None
        else:
            _LOG.debug("Native checkpoint load failed.", exc_info=True)
            raise AxiomEngineError(
                "AXIOM Native Engine could not allocate or load this checkpoint. "
                "Confirm supported safetensors weights and available memory."
            ) from None

    # The selected backend may change after a transparent auto-mode CPU retry.
    key = (str(base_dir), str(resolved_adapter) if resolved_adapter else None, str(device), device_preference, memory_mode, requested_memory_mode, precision)
    details = {
        **inspected,
        **native_details,
        "device_selection": device_selection,
        "memory_selection": memory_selection,
        "precision_selection": {"requested": precision, "selected": native_details.get("precision", precision)},
        "config": {
            "max_position_embeddings": model.config.max_position_embeddings,
            "effective_max_position_embeddings": getattr(
                model.config, "effective_max_position_embeddings", model.config.max_position_embeddings
            ),
        },
    }
    _LOADED_RUNTIME = {
        "key": key,
        "model": model,
        "tokenizer": tokenizer,
        "torch": torch,
        "device": device,
        "details": details,
    }
    _LOADED_ARCHITECTURE = native_details["architecture"]
    return model, tokenizer, torch, device, details

def unload_local_model() -> dict[str, Any]:
    """Release the loaded local model and best-effort accelerator cache."""
    global _LOADED_RUNTIME, _LOADED_ARCHITECTURE
    with _ENGINE_LOCK:
        previous_runtime = _LOADED_RUNTIME
        was_loaded = previous_runtime is not None
        _LOADED_RUNTIME = None
        _LOADED_ARCHITECTURE = None
        if previous_runtime is not None:
            previous_torch = previous_runtime.get("torch")
            previous_device = previous_runtime.get("device")
            del previous_runtime
            gc.collect()
            _clear_device_cache(previous_torch, previous_device)
        else:
            gc.collect()
        return {
            "schema_version": 1,
            "unloaded": was_loaded,
            "loaded": False,
            "scope_note": "The loaded model and tokenizer references were released by AXIOM. Framework or OS-level allocators may retain some reserved memory.",
        }



def _format_prompt_for_native_model(
    prompt: str,
    model_dir: Path,
    model_type: str,
    tokenizer: Any,
    chat_messages: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    """Apply a checkpoint's safe chat template when present; keep raw completion for base models.

    Qwen2 text-instruct models have a ChatML fallback because the standard native
    decoder has no Transformers dependency and many repositories omit template metadata.
    Model-provided templates run only inside Jinja's sandbox with a small, explicit context.
    """
    tokenizer_config: dict[str, Any] = {}
    config_path = model_dir / "tokenizer_config.json"
    if config_path.is_file():
        try:
            if config_path.stat().st_size > _MAX_CONFIG_BYTES:
                raise AxiomEngineError("tokenizer_config.json exceeds AXIOM's 1 MiB chat-template limit.")
            value = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise AxiomEngineError("tokenizer_config.json must be a JSON object.")
            tokenizer_config = value
        except AxiomEngineError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise AxiomEngineError("tokenizer_config.json could not be read while resolving the model's chat template.") from None

    template = tokenizer_config.get("chat_template")
    if isinstance(template, list):
        named = [entry for entry in template if isinstance(entry, dict) and isinstance(entry.get("template"), str)]
        preferred = next((entry for entry in named if entry.get("name") in {"default", "chat"}), named[0] if named else None)
        template = preferred["template"] if preferred else None
    if isinstance(template, str) and template.strip():
        try:
            from jinja2.sandbox import SandboxedEnvironment
        except ImportError:
            if model_type in {"qwen2", "qwen3"}:
                fallback_messages = chat_messages or [{"role": "user", "content": prompt}]
                rendered = "".join(
                    f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                    for message in fallback_messages
                ) + "<|im_start|>assistant\n"
                return rendered, f"{model_type}-chatml-fallback"
            raise AxiomEngineError(
                "This checkpoint supplies a chat template, but Jinja2 is unavailable. Install the template runtime dependency or use a raw/completion checkpoint."
            ) from None
        def special_token(key: str) -> str:
            raw = tokenizer_config.get(key)
            if isinstance(raw, dict):
                raw = raw.get("content")
            if not isinstance(raw, str):
                return ""
            return raw if tokenizer.token_to_id(raw) is not None else ""
        environment = SandboxedEnvironment(autoescape=False, keep_trailing_newline=True)
        environment.globals.clear()
        environment.globals.update({
            "raise_exception": lambda message: (_ for _ in ()).throw(ValueError(str(message))),
            "strftime_now": lambda fmt: datetime.now(UTC).strftime(fmt),
        })
        environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
        template_messages = chat_messages or [{"role": "user", "content": prompt}]
        context = {
            "messages": template_messages,
            "add_generation_prompt": True,
            "bos_token": special_token("bos_token"),
            "eos_token": special_token("eos_token"),
            "tools": None,
            "documents": None,
        }
        try:
            rendered = environment.from_string(template).render(**context)
        except Exception as exc:
            raise AxiomEngineError(
                f"AXIOM could not safely render this checkpoint's chat template: {type(exc).__name__}. "
                "Use a compatible tokenizer_config.json or a base/completion checkpoint."
            ) from None
        if not isinstance(rendered, str) or not rendered.strip():
            raise AxiomEngineError("The checkpoint's chat template produced an empty prompt.")
        return rendered, "tokenizer-config-template"

    if chat_messages:
        if model_type in {"qwen2", "qwen3"}:
            rendered = "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                for message in chat_messages
            ) + "<|im_start|>assistant\n"
            return rendered, f"{model_type}-chatml-fallback"
        rendered = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}" for message in chat_messages
        ) + "\n\nASSISTANT:\n"
        return rendered, "role-labelled-chat-fallback"
    if model_type in {"qwen2", "qwen3"}:
        # Standard Qwen2/Qwen3 text-instruct checkpoints use ChatML markers even when
        # older/custom releases omit an exported chat_template.
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", f"{model_type}-chatml-fallback"
    return prompt, "raw-completion"


def generate_local_stream(
    model_path: str,
    prompt: str,
    *,
    adapter_path: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
    include_token_events: bool = True,
    cancel_event: threading.Event | None = None,
    device_preference: str = "auto",
    memory_mode: str = "auto",
    chat_messages: list[dict[str, str]] | None = None,
    precision: str = "auto",
    _lock_held: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield cumulative decoded text updates and one final metrics event.

    The engine lock is held for the duration of the stream so another request
    cannot replace or unload the model while its KV cache is active. Closing
    the iterator closes the native token iterator and releases that lock.
    """
    global _LOADED_RUNTIME, _LOADED_ARCHITECTURE
    prompt, max_new_tokens, temperature = validate_generation(prompt, max_new_tokens, temperature)
    if chat_messages is not None:
        if not isinstance(chat_messages, list) or not 1 <= len(chat_messages) <= 128:
            raise ValueError("Chat messages must contain between 1 and 128 entries.")
        normalized_messages: list[dict[str, str]] = []
        total_message_chars = 0
        valid_roles = {"system", "user", "assistant"}
        for item in chat_messages:
            if not isinstance(item, dict) or item.get("role") not in valid_roles or not isinstance(item.get("content"), str):
                raise ValueError("Native chat messages support only system, user, and assistant roles with string content.")
            role = item["role"]
            text_value = item["content"]
            if not text_value.strip() or len(text_value) > _MAX_PROMPT_CHARS:
                raise ValueError("Each chat message must contain non-blank text within the prompt limit.")
            total_message_chars += len(text_value)
            if total_message_chars > _MAX_CHAT_CHARS:
                raise ValueError(f"Combined chat messages exceed the {_MAX_CHAT_CHARS:,}-character limit.")
            normalized_messages.append({"role": role, "content": text_value})
        if not any(message["role"] == "user" for message in normalized_messages):
            raise ValueError("Native chat requests must contain at least one user message.")
        chat_messages = normalized_messages
        prompt = next(message["content"] for message in reversed(chat_messages) if message["role"] == "user")
    top_p, top_k, repetition_penalty, seed = validate_sampling_parameters(top_p, top_k, repetition_penalty, seed)
    min_p = validate_min_p(min_p)
    stop_sequences = validate_stop_sequences(stop_sequences)
    with (nullcontext() if _lock_held else _engine_request_lock(cancel_event)):
        model, tokenizer, torch, device, details = _load_local_model(model_path, adapter_path, device_preference, memory_mode, precision)
        selected_backend = details.get("device_selection", {}).get("selected")
        device_memory_before = _device_memory_snapshot(torch, device, selected_backend=selected_backend)
        device_memory_before["peak_reset_supported"] = _reset_device_peak_memory(torch, device)
        context_cap = max(2, min(
            getattr(model.config, "effective_max_position_embeddings", model.config.max_position_embeddings),
            _MAX_INPUT_TOKENS + max_new_tokens,
        ))
        effective_max_new_tokens = min(max_new_tokens, context_cap - 1)
        input_limit = max(1, min(_MAX_INPUT_TOKENS, context_cap - effective_max_new_tokens))
        original_message_count = len(chat_messages or [])
        original_message_chars = sum(len(message["content"]) for message in (chat_messages or []))
        dropped_history_messages = 0
        bos_id = model.config.bos_token_id

        def encode_current_context() -> tuple[list[int], str]:
            formatted, source = _format_prompt_for_native_model(
                prompt,
                Path(details.get("model_directory_path", ".")),
                details["model_type"],
                tokenizer,
                chat_messages=chat_messages,
            )
            try:
                encoded = tokenizer.encode(formatted, add_special_tokens=False)
                ids = list(encoded.ids)
            except Exception:
                raise ValueError("Prompt could not be tokenized by the local tokenizer.json.") from None
            if details.get("add_bos_token") and bos_id is not None and (not ids or ids[0] != bos_id):
                ids.insert(0, bos_id)
            return ids, source

        prompt_ids, prompt_format = encode_current_context()
        # Preserve system instructions and the newest user request when the context
        # budget is exceeded. Drop the oldest non-system turn first, re-rendering the
        # model's real chat template so message boundaries remain valid.
        while chat_messages and len(prompt_ids) > input_limit:
            latest_user_index = max(
                (index for index, message in enumerate(chat_messages) if message["role"] == "user"),
                default=-1,
            )
            drop_index = next(
                (index for index, message in enumerate(chat_messages)
                 if message["role"] != "system" and index != latest_user_index),
                None,
            )
            if drop_index is None:
                break
            del chat_messages[drop_index]
            dropped_history_messages += 1
            prompt_ids, prompt_format = encode_current_context()

        prompt_tokens_before_truncation = len(prompt_ids)
        token_truncation_applied = prompt_tokens_before_truncation > input_limit
        if token_truncation_applied:
            prompt_ids = prompt_ids[-input_limit:]
            # Reserve the leading token for BOS, rather than duplicating it.
            if details.get("add_bos_token") and bos_id is not None and prompt_ids:
                prompt_ids[0] = bos_id
        if not prompt_ids:
            raise ValueError("Prompt tokenization returned no tokens.")
        context_management = {
            "chat_history_supplied": bool(original_message_count),
            "original_message_count": original_message_count,
            "original_message_characters": original_message_chars,
            "messages_dropped_to_fit_context": dropped_history_messages,
            "prompt_tokens_before_token_truncation": prompt_tokens_before_truncation,
            "prompt_tokens_used": len(prompt_ids),
            "token_truncation_applied": token_truncation_applied,
            "system_prompt_may_be_truncated": bool(
                token_truncation_applied
                and any(message["role"] == "system" for message in (chat_messages or []))
            ),
        }
        effective_max_new_tokens = min(effective_max_new_tokens, max(1, context_cap - len(prompt_ids)))
        started = time.perf_counter()
        generated_ids: list[int] = []
        response = ""
        if cancel_event is not None and cancel_event.is_set():
            return
        iterator = model.iter_generate(
            prompt_ids,
            max_new_tokens=effective_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            eos_token_ids=model.config.eos_token_ids,
            cancel_event=cancel_event,
        )
        stop_reason = "max_new_tokens"
        try:
            for token_id in iterator:
                if cancel_event is not None and cancel_event.is_set():
                    stop_reason = "cancelled"
                    break
                generated_ids.append(token_id)
                if not include_token_events and not stop_sequences:
                    continue
                try:
                    decoded_response = tokenizer.decode(generated_ids, skip_special_tokens=True)[:100_000]
                except Exception:
                    raise AxiomEngineError("Generated token stream could not be decoded by tokenizer.json.") from None

                matches = [
                    (decoded_response.find(sequence), sequence)
                    for sequence in stop_sequences
                    if sequence in decoded_response
                ]
                if matches:
                    stop_at = min(position for position, _sequence in matches)
                    response = decoded_response[:stop_at]
                    stop_reason = "stop_sequence"
                    if include_token_events:
                        yield {
                            "type": "token",
                            "response": response,
                            "completion_tokens": len(generated_ids),
                        }
                        live_metrics = getattr(model, "live_generation", None)
                        if isinstance(live_metrics, dict):
                            yield {
                                "type": "telemetry",
                                "metrics": dict(live_metrics),
                                "memory_mode": getattr(model, "memory_mode", memory_mode),
                                "memory_selection": details.get("memory_selection"),
                                "memory_stats": dict(getattr(model, "memory_stats", {})),
                                "device_memory": _device_memory_snapshot(
                                    torch, device, selected_backend=selected_backend, baseline=device_memory_before
                                ),
                            }
                    # Closing now finalizes the native generator's metrics and
                    # releases the KV cache before the completion event is built.
                    close_iterator = getattr(iterator, "close", None)
                    if callable(close_iterator):
                        close_iterator()
                    break

                pending = _pending_stop_prefix_length(decoded_response, stop_sequences)
                response = decoded_response[:-pending] if pending else decoded_response
                if include_token_events:
                    yield {
                        "type": "token",
                        "response": response,
                        "completion_tokens": len(generated_ids),
                    }
                    live_metrics = getattr(model, "live_generation", None)
                    if isinstance(live_metrics, dict):
                        yield {
                            "type": "telemetry",
                            "metrics": dict(live_metrics),
                            "memory_mode": getattr(model, "memory_mode", memory_mode),
                            "memory_selection": details.get("memory_selection"),
                            "memory_stats": dict(getattr(model, "memory_stats", {})),
                            "device_memory": _device_memory_snapshot(
                                torch, device, selected_backend=selected_backend, baseline=device_memory_before
                            ),
                        }

            if stop_reason != "stop_sequence":
                # Flush any partial stop-prefix back into the final text if no
                # complete stop sequence was ever observed.
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)[:100_000]
                if cancel_event is not None and cancel_event.is_set():
                    stop_reason = "cancelled"
                else:
                    stop_reason = model.last_generation.get("stop_reason", stop_reason)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            result = {
                "engine": "axiom-native-engine",
                "execution_backend": "pytorch-native-axiom",
                "transformers_generation_used": False,
                "ollama_used": False,
                "model": details["model_directory_name"] or "local-model",
                "architecture": details["architecture"],
                "model_type": details["model_type"],
                "prompt_format": prompt_format,
                "context_management": context_management,
                "adapter": details["adapter_directory_name"],
                "device": str(device),
                "device_selection": details.get("device_selection"),
                "precision_selection": details.get("precision_selection"),
                "precision": details.get("precision"),
                "memory_mode": getattr(model, "memory_mode", memory_mode),
                "memory_selection": details.get("memory_selection"),
                "memory_stats": dict(getattr(model, "memory_stats", {})),
                "device_memory": _device_memory_snapshot(
                    torch, device, selected_backend=selected_backend, baseline=device_memory_before
                ),
                "checkpoint_format": details.get("checkpoint_format", "safetensors"),
                "decoder": model.last_generation.get("decoder", "axiom-autoregressive-kv-cache"),
                "attention_backend": model.last_attention_backend,
                "prefill_ms": model.last_generation.get("prefill_ms", 0.0),
                "decode_ms": model.last_generation.get("decode_ms", 0.0),
                "time_to_first_token_ms": model.last_generation.get("time_to_first_token_ms", 0.0),
                "generation_tokens_per_second": model.last_generation.get("generation_tokens_per_second", 0.0),
                "sampling": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "seed": seed,
                },
                "stop_sequences": list(stop_sequences),
                "stop_reason": stop_reason,
                "response": response,
                "prompt_tokens": model.last_generation.get("prompt_tokens", len(prompt_ids)),
                "completion_tokens": len(generated_ids),
                "elapsed_ms": elapsed_ms,
                "generated_at": datetime.now(UTC).isoformat(),
                "local_only": True,
            }
            yield {"type": "complete", "result": result}
        except (RuntimeError, OSError, MemoryError) as exc:
            error_text = str(exc).casefold()
            out_of_memory = isinstance(exc, MemoryError) or "out of memory" in error_text
            device_type = str(getattr(device, "type", device)).split(":", 1)[0].casefold()
            may_retry_on_cpu = (
                out_of_memory
                and not generated_ids
                and (cancel_event is None or not cancel_event.is_set())
                and isinstance(device_preference, str)
                and device_preference.strip().casefold() == "auto"
                and device_type in {"cuda", "mps", "xpu"}
            )
            if may_retry_on_cpu:
                _LOG.warning(
                    "AXIOM Native Engine hit accelerator OOM before the first output token; "
                    "releasing the failed runtime and retrying this request on CPU."
                )
                close_iterator = getattr(iterator, "close", None)
                if callable(close_iterator):
                    try:
                        close_iterator()
                    except (RuntimeError, ValueError):
                        _LOG.debug("OOM-failed native iterator was already closed.", exc_info=True)
                iterator = None
                model = None
                tokenizer = None
                _LOADED_RUNTIME = None
                _LOADED_ARCHITECTURE = None
                gc.collect()
                _clear_device_cache(torch, device)
                fallback_reason = (
                    f"The {device_type.upper()} backend ran out of memory before output began; "
                    "AXIOM retried this request on CPU."
                )
                fallback_iterator = generate_local_stream(
                    model_path,
                    prompt,
                    adapter_path=adapter_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    seed=seed,
                    stop_sequences=stop_sequences,
                    include_token_events=include_token_events,
                    cancel_event=cancel_event,
                    device_preference="cpu",
                    memory_mode=memory_mode,
                    chat_messages=chat_messages,
                    precision=precision,
                    _lock_held=True,
                )
                try:
                    for fallback_event in fallback_iterator:
                        if fallback_event.get("type") == "complete" and isinstance(fallback_event.get("result"), dict):
                            fallback_result = fallback_event["result"]
                            selection = fallback_result.get("device_selection")
                            if isinstance(selection, dict):
                                fallback_result["device_selection"] = {
                                    **selection,
                                    "requested": "auto",
                                    "selected": "cpu",
                                    "reason": fallback_reason,
                                    "fallback_from": device_type,
                                    "fallback_reason": "Accelerator OOM occurred before any token was emitted.",
                                }
                            fallback_result["device_fallback"] = {
                                "fallback_from": device_type,
                                "selected": "cpu",
                                "reason": fallback_reason,
                            }
                            fallback_event["result"] = fallback_result
                        yield fallback_event
                finally:
                    close_fallback = getattr(fallback_iterator, "close", None)
                    if callable(close_fallback):
                        close_fallback()
                return
            if out_of_memory:
                raise AxiomEngineError(
                    "AXIOM Native Engine ran out of memory during inference. "
                    "Automatic CPU retry is available only before any output token is emitted; "
                    "use a smaller model/context or select CPU explicitly."
                ) from None
            _LOG.debug("AXIOM Native Engine token generation failed.", exc_info=True)
            raise AxiomEngineError(
                "AXIOM Native Engine generation failed. The prompt/context may exceed memory, "
                "or the checkpoint may be incompatible with the implemented architecture."
            ) from None
        except (ValueError, GeneratorExit):
            raise
        except Exception:
            _LOG.debug("AXIOM Native Engine token generation failed.", exc_info=True)
            raise AxiomEngineError(
                "AXIOM Native Engine generation failed. The prompt/context may exceed memory, "
                "or the checkpoint may be incompatible with the implemented architecture."
            ) from None
        finally:
            close_iterator = getattr(iterator, "close", None)
            if callable(close_iterator):
                close_iterator()


def generate_local(
    model_path: str,
    prompt: str,
    *,
    adapter_path: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
    device_preference: str = "auto",
    memory_mode: str = "auto",
    precision: str = "auto",
    cancel_event: threading.Event | None = None,
    _lock_held: bool = False,
) -> dict[str, Any]:
    """Generate a complete local response through the streaming decoder."""
    completed: dict[str, Any] | None = None
    for event in generate_local_stream(
        model_path,
        prompt,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop_sequences=stop_sequences,
        include_token_events=False,
        device_preference=device_preference,
        memory_mode=memory_mode,
        precision=precision,
        cancel_event=cancel_event,
        _lock_held=_lock_held,
    ):
        if event.get("type") == "complete" and isinstance(event.get("result"), dict):
            completed = event["result"]
    if completed is None:
        raise AxiomEngineError("Native generation ended before a completion event was produced.")
    return completed




def benchmark_local(
    model_path: str,
    prompt: str,
    *,
    adapter_path: str | None = None,
    warmup_runs: int = 1,
    iterations: int = 3,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
    device_preference: str = "auto",
    memory_mode: str = "auto",
    precision: str = "auto",
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run bounded local benchmarks and discard all generated text."""
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or not 0 <= warmup_runs <= 2:
        raise ValueError("Warm-up runs must be an integer from 0 through 2.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 5:
        raise ValueError("Measured iterations must be an integer from 1 through 5.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
        raise ValueError("Benchmark output length must be an integer from 1 through 128 tokens.")
    valid_prompt, max_new_tokens, temperature = validate_generation(
        prompt, max_new_tokens, temperature
    )
    top_p, top_k, repetition_penalty, seed = validate_sampling_parameters(
        top_p, top_k, repetition_penalty, seed
    )
    min_p = validate_min_p(min_p)
    stops = validate_stop_sequences(stop_sequences)

    started = time.perf_counter()
    measured: list[dict[str, Any]] = []
    warmups_completed = 0
    last_outcome: dict[str, Any] | None = None
    with _engine_request_lock(cancel_event):
        for index in range(warmup_runs + iterations):
            if cancel_event is not None and cancel_event.is_set():
                raise AxiomEngineError("Native benchmark was cancelled.")
            outcome = generate_local(
                model_path,
                valid_prompt,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_sequences=stops,
                device_preference=device_preference,
                memory_mode=memory_mode,
                precision=precision,
                cancel_event=cancel_event,
                _lock_held=True,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise AxiomEngineError("Native benchmark was cancelled.")
            last_outcome = outcome
            if index < warmup_runs:
                warmups_completed += 1
                continue
            measured.append({
                "iteration": len(measured) + 1,
                "elapsed_ms": float(outcome.get("elapsed_ms", 0.0)),
                "prefill_ms": float(outcome.get("prefill_ms", 0.0)),
                "decode_ms": float(outcome.get("decode_ms", 0.0)),
                "time_to_first_token_ms": float(outcome.get("time_to_first_token_ms", 0.0)),
                "generation_tokens_per_second": float(outcome.get("generation_tokens_per_second", 0.0)),
                "prompt_tokens": int(outcome.get("prompt_tokens", 0)),
                "completion_tokens": int(outcome.get("completion_tokens", 0)),
                "stop_reason": str(outcome.get("stop_reason", "unknown")),
            })

    if last_outcome is None or not measured:
        raise AxiomEngineError("Native benchmark ended before a measured run was completed.")
    total_wall_ms = round((time.perf_counter() - started) * 1000, 2)

    def metric_median(key: str) -> float:
        values = [item[key] for item in measured if math.isfinite(item[key]) and item[key] >= 0]
        return round(float(statistics.median(values)), 2) if values else 0.0

    wall_times = sorted(item["elapsed_ms"] for item in measured)
    p95_wall_ms = round(wall_times[min(len(wall_times) - 1, math.ceil(0.95 * len(wall_times)) - 1)], 2)
    measured_wall_ms = sum(item["elapsed_ms"] for item in measured)
    completion_tokens = sum(item["completion_tokens"] for item in measured)
    prompt_tokens = sum(item["prompt_tokens"] for item in measured)
    return {
        "schema_version": 1,
        "engine": "axiom-native-engine",
        "model": last_outcome.get("model", "local-model"),
        "architecture": last_outcome.get("architecture"),
        "model_type": last_outcome.get("model_type"),
        "device": last_outcome.get("device"),
        "precision": last_outcome.get("precision"),
        "memory_mode": last_outcome.get("memory_mode", memory_mode),
        "checkpoint_format": last_outcome.get("checkpoint_format", "safetensors"),
        "warmup_runs_requested": warmup_runs,
        "warmup_runs_completed": warmups_completed,
        "warmups_excluded": True,
        "iterations_requested": iterations,
        "iterations_completed": len(measured),
        "max_new_tokens": max_new_tokens,
        "sampling": {
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "min_p": min_p, "repetition_penalty": repetition_penalty, "seed": seed,
        },
        "metrics": {
            "total_wall_time_ms": total_wall_ms,
            "measured_wall_time_ms": round(measured_wall_ms, 2),
            "median_elapsed_ms": metric_median("elapsed_ms"),
            "p95_elapsed_ms": p95_wall_ms,
            "median_prefill_ms": metric_median("prefill_ms"),
            "median_decode_ms": metric_median("decode_ms"),
            "median_time_to_first_token_ms": metric_median("time_to_first_token_ms"),
            "median_generation_tokens_per_second": metric_median("generation_tokens_per_second"),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "aggregate_tokens_per_second": round(completion_tokens * 1000 / measured_wall_ms, 2) if measured_wall_ms > 0 else 0.0,
        },
        "runs": measured,
        "device_memory": last_outcome.get("device_memory"),
        "memory_selection": last_outcome.get("memory_selection"),
        "scope_note": (
            "Measured local AXIOM inference only. Warm-ups are excluded from measured statistics. "
            "Prompts, generated responses and model paths are not returned or persisted. "
            "Latency and throughput are specific to the observed machine, model and workload."
        ),
        "privacy": {
            "prompt_returned": False,
            "response_returned": False,
            "results_persisted": False,
            "cloud_provider_used": False,
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }

def generate_local_batch(
    model_path: str,
    prompts: list[str],
    *,
    adapter_path: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
    device_preference: str = "auto",
    memory_mode: str = "auto",
    precision: str = "auto",
    cancel_event: threading.Event | None = None,
    chat_messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate a bounded prompt batch with one AXIOM-owned runtime.

    Prompts may be independent text items or multiple choices for one chat
    context. The native runtime buckets by token length without padding.
    """
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Batch generation requires at least one prompt.")
    if len(prompts) > 128:
        raise ValueError("Batch generation is limited to 128 prompts per request.")
    normalized_prompts: list[str] = []
    combined_chars = 0
    for item in prompts:
        valid_prompt, _token_limit, _temperature = validate_generation(item, max_new_tokens, temperature)
        combined_chars += len(valid_prompt)
        normalized_prompts.append(valid_prompt)

    normalized_chat_messages: list[dict[str, str]] | None = None
    if chat_messages is not None:
        if not isinstance(chat_messages, list) or not 1 <= len(chat_messages) <= 128:
            raise ValueError("Chat messages must contain between 1 and 128 entries.")
        normalized_chat_messages = []
        total_message_chars = 0
        valid_roles = {"system", "user", "assistant"}
        for item in chat_messages:
            if (
                not isinstance(item, dict)
                or item.get("role") not in valid_roles
                or not isinstance(item.get("content"), str)
            ):
                raise ValueError("Native chat messages support only system, user, and assistant roles with string content.")
            role, content = item["role"], item["content"]
            if not content.strip() or len(content) > _MAX_PROMPT_CHARS:
                raise ValueError("Each chat message must contain non-blank text within the prompt limit.")
            total_message_chars += len(content)
            if total_message_chars > _MAX_CHAT_CHARS:
                raise ValueError(f"Combined chat messages exceed the {_MAX_CHAT_CHARS:,}-character limit.")
            normalized_chat_messages.append({"role": role, "content": content})
        if not any(item["role"] == "user" for item in normalized_chat_messages):
            raise ValueError("Native chat requests must contain at least one user message.")
    elif combined_chars > _MAX_CHAT_CHARS:
        raise ValueError(f"Combined batch prompts exceed the {_MAX_CHAT_CHARS:,}-character limit.")
    top_p, top_k, repetition_penalty, seed = validate_sampling_parameters(
        top_p, top_k, repetition_penalty, seed
    )
    min_p = validate_min_p(min_p)
    stop_sequences = validate_stop_sequences(stop_sequences)
    requested_started = time.perf_counter()

    with _engine_request_lock(cancel_event):
        model, tokenizer, torch, device, details = _load_local_model(
            model_path, adapter_path, device_preference, memory_mode, precision
        )
        selected_backend = details.get("device_selection", {}).get("selected")
        device_memory_before = _device_memory_snapshot(
            torch, device, selected_backend=selected_backend
        )
        device_memory_before["peak_reset_supported"] = _reset_device_peak_memory(torch, device)
        context_cap = max(
            2,
            min(
                getattr(model.config, "effective_max_position_embeddings", model.config.max_position_embeddings),
                _MAX_INPUT_TOKENS + max_new_tokens,
            ),
        )
        effective_max_new_tokens = min(max_new_tokens, context_cap - 1)
        input_limit = max(1, min(_MAX_INPUT_TOKENS, context_cap - effective_max_new_tokens))
        prompt_ids_batch: list[list[int]] = []
        prompt_formats: list[str] = []
        context_rows: list[dict[str, Any]] = []
        bos_id = model.config.bos_token_id
        model_dir = Path(details.get("model_directory_path", model_path))

        for prompt in normalized_prompts:
            active_messages = list(normalized_chat_messages) if normalized_chat_messages is not None else None

            def encode_context(messages: list[dict[str, str]] | None) -> tuple[list[int], str]:
                formatted, source = _format_prompt_for_native_model(
                    prompt,
                    model_dir,
                    details["model_type"],
                    tokenizer,
                    chat_messages=messages,
                )
                try:
                    ids = list(tokenizer.encode(formatted, add_special_tokens=False).ids)
                except Exception:
                    raise ValueError("A batch prompt could not be tokenized by the local tokenizer.") from None
                if details.get("add_bos_token") and bos_id is not None and (not ids or ids[0] != bos_id):
                    ids.insert(0, bos_id)
                if not ids:
                    raise ValueError("Batch prompt tokenization returned no tokens.")
                return ids, source

            ids, prompt_format = encode_context(active_messages)
            before = len(ids)
            messages_dropped = 0
            # Match single-chat context management: preserve system instructions
            # and the newest user request by dropping the oldest non-system turn
            # before cutting tokens from a still-oversized template.
            while active_messages is not None and len(ids) > input_limit:
                latest_user_index = max(
                    (index for index, message in enumerate(active_messages) if message["role"] == "user"),
                    default=-1,
                )
                drop_index = next(
                    (
                        index for index, message in enumerate(active_messages)
                        if message["role"] != "system" and index != latest_user_index
                    ),
                    -1,
                )
                if drop_index < 0:
                    break
                del active_messages[drop_index]
                messages_dropped += 1
                ids, prompt_format = encode_context(active_messages)

            tail_truncated = len(ids) > input_limit
            truncated = messages_dropped > 0 or tail_truncated
            if tail_truncated:
                ids = ids[-input_limit:]
                if details.get("add_bos_token") and bos_id is not None and ids:
                    ids[0] = bos_id
            prompt_ids_batch.append(ids)
            prompt_formats.append(prompt_format)
            context_row = {
                "prompt_tokens_before_truncation": before,
                "prompt_tokens_used": len(ids),
                "token_truncation_applied": truncated,
            }
            if normalized_chat_messages is not None:
                context_row["messages_dropped_to_fit_context"] = messages_dropped
                context_row["system_prompt_may_be_truncated"] = (
                    tail_truncated and any(
                        message["role"] == "system" for message in normalized_chat_messages
                    )
                )
            context_rows.append(context_row)
        effective_max_new_tokens = min(
            effective_max_new_tokens,
            max(1, min(
                getattr(model.config, "effective_max_position_embeddings", model.config.max_position_embeddings) - len(ids)
                for ids in prompt_ids_batch
            )),
        )

        def should_stop(_row: int, generated_ids: list[int]) -> bool:
            if not stop_sequences:
                return False
            try:
                decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)[:100_000]
            except Exception:
                raise AxiomEngineError("Generated batch tokens could not be decoded by the local tokenizer.") from None
            return any(sequence in decoded for sequence in stop_sequences)

        outputs = model.generate_batch(
            prompt_ids_batch,
            max_new_tokens=effective_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            eos_token_ids=model.config.eos_token_ids,
            cancel_event=cancel_event,
            stop_checker=should_stop if stop_sequences else None,
        )
        batch_metrics = dict(getattr(model, "last_batch_generation", {}))
        sequence_reasons = batch_metrics.get("sequence_stop_reasons", [])
        results: list[dict[str, Any]] = []
        for index, generated_ids in enumerate(outputs):
            try:
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)[:100_000]
            except Exception:
                raise AxiomEngineError("Generated batch response could not be decoded by the local tokenizer.") from None
            stop_reason = (
                sequence_reasons[index]
                if isinstance(sequence_reasons, list) and index < len(sequence_reasons)
                else "max_new_tokens"
            )
            if stop_sequences and response:
                matches = [
                    (response.find(sequence), sequence)
                    for sequence in stop_sequences
                    if sequence in response
                ]
                if matches:
                    response = response[:min(position for position, _sequence in matches)]
                    stop_reason = "stop_sequence"
            results.append({
                "index": index + 1,
                "prompt_format": prompt_formats[index],
                "context_management": context_rows[index],
                "prompt_tokens": len(prompt_ids_batch[index]),
                "completion_tokens": len(generated_ids),
                "stop_reason": stop_reason,
                "response": response,
            })

        return {
            "engine": "axiom-native-engine",
            "execution_backend": "pytorch-native-axiom",
            "transformers_generation_used": False,
            "ollama_used": False,
            "model": details.get("model_directory_name") or "local-model",
            "architecture": details["architecture"],
            "model_type": details["model_type"],
            "adapter": details.get("adapter_directory_name"),
            "device": str(device),
            "device_selection": details.get("device_selection"),
            "precision": details.get("precision"),
            "precision_selection": details.get("precision_selection"),
            "memory_mode": getattr(model, "memory_mode", memory_mode),
            "memory_selection": details.get("memory_selection"),
            "memory_stats": dict(getattr(model, "memory_stats", {})),
            "device_memory": _device_memory_snapshot(
                torch, device, selected_backend=selected_backend, baseline=device_memory_before
            ),
            "checkpoint_format": details.get("checkpoint_format", "safetensors"),
            "decoder": batch_metrics.get("decoder", "axiom-batched-autoregressive-kv-cache"),
            "attention_backend": getattr(model, "last_attention_backend", "native"),
            "sampling": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
            },
            "stop_sequences": list(stop_sequences),
            "batch_size": len(results),
            "batch_generation": batch_metrics,
            "results": results,
            "prompt_tokens": sum(result["prompt_tokens"] for result in results),
            "completion_tokens": sum(result["completion_tokens"] for result in results),
            "elapsed_ms": round((time.perf_counter() - requested_started) * 1000, 2),
            "generated_at": datetime.now(UTC).isoformat(),
            "local_only": True,
        }
