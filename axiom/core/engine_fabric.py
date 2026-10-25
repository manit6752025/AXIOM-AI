"""Local engine-fabric discovery and safe dispatch across local inference runtimes.

AXIOM Native Engine remains its own transformer runtime. This module is an
orchestration layer: it inventories useful local execution backends, explains
their trade-offs, recommends an engine for a workload, and dispatches to
fixed loopback-only APIs when the user explicitly chooses an external runtime.
No remote hosts, environment proxies, credentials, downloads, or subprocess
launches are involved in discovery or dispatch.
"""
from __future__ import annotations

import importlib.util
import json
import math
import platform
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from axiom.core.inference import validate_model_name

_SCHEMA_VERSION = 1
_PROBE_TIMEOUT = 0.55
_REQUEST_TIMEOUT = 300
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_MESSAGES = 64
_MAX_MESSAGE_CHARS = 12_000
_MAX_TOTAL_CHARS = 24_000
_MAX_OUTPUT_TOKENS = 2048

_BACKENDS: tuple[dict[str, Any], ...] = (
    {
        "id": "ollama", "name": "Ollama", "kind": "ollama",
        "endpoint": "http://127.0.0.1:11434", "models_path": "/api/tags",
        "chat_path": "/api/chat", "executable": "ollama", "package": None,
        "supports_chat": True, "formats": ["Ollama library models", "GGUF through Ollama"],
        "strengths": ["Simple model pull and lifecycle", "Broad packaged model catalogue", "Straightforward local API"],
        "tradeoffs": ["Requires a separate daemon", "Less control over low-level scheduler and memory policies"],
        "best_for": ["easy_setup", "general_local"],
    },
    {
        "id": "llama-cpp", "name": "llama.cpp server", "kind": "openai-compatible",
        "endpoint": "http://127.0.0.1:8080", "models_path": "/v1/models",
        "chat_path": "/v1/chat/completions", "executable": "llama-server", "package": None,
        "supports_chat": True, "formats": ["GGUF", "F16/BF16 GGUF", "Quantized GGUF"],
        "strengths": ["Small native server", "Fine-grained CPU/GPU offload", "Broad CPU and accelerator backends"],
        "tradeoffs": ["Binary/build options affect available acceleration", "Model templates and server flags need deliberate setup"],
        "best_for": ["low_memory", "portable", "quantized"],
    },
    {
        "id": "lm-studio", "name": "LM Studio local server", "kind": "openai-compatible",
        "endpoint": "http://127.0.0.1:1234", "models_path": "/v1/models",
        "chat_path": "/v1/chat/completions", "executable": "lms", "package": None,
        "supports_chat": True, "formats": ["Formats supported by the installed LM Studio runtime"],
        "strengths": ["Desktop model discovery and lifecycle", "Convenient local OpenAI-compatible endpoint"],
        "tradeoffs": ["Desktop/server lifecycle is an additional dependency", "Capabilities depend on installed build and hardware"],
        "best_for": ["easy_setup", "desktop"],
    },
    {
        "id": "vllm", "name": "vLLM local server", "kind": "openai-compatible",
        "endpoint": "http://127.0.0.1:8000", "models_path": "/v1/models",
        "chat_path": "/v1/chat/completions", "executable": "vllm", "package": "vllm",
        "supports_chat": True, "formats": ["Hugging Face model formats supported by installed vLLM"],
        "strengths": ["High-throughput serving", "Continuous batching and KV-cache scheduling", "Large deployment and API surface"],
        "tradeoffs": ["More involved environment and accelerator setup", "Optimized mainly for capable GPU serving, not smallest installs"],
        "best_for": ["throughput", "multi_user"],
    },
    {
        "id": "llama-cpp-python", "name": "llama-cpp-python server", "kind": "openai-compatible",
        "endpoint": "http://127.0.0.1:8001", "models_path": "/v1/models",
        "chat_path": "/v1/chat/completions", "executable": None, "package": "llama_cpp",
        "supports_chat": True, "formats": ["GGUF"],
        "strengths": ["Python integration and GGUF support", "CPU and accelerator-specific builds"],
        "tradeoffs": ["Wheel/backend compatibility can require careful installation", "Server may use a custom port or launch configuration"],
        "best_for": ["python", "quantized"],
    },
)

_BACKEND_BY_ID = {item["id"]: item for item in _BACKENDS}
_PROFILES: dict[str, tuple[str, ...]] = {
    "balanced": ("axiom-native", "llama-cpp", "ollama", "lm-studio", "llama-cpp-python", "vllm"),
    "low-memory": ("llama-cpp", "axiom-native", "ollama", "llama-cpp-python", "lm-studio", "vllm"),
    "throughput": ("vllm", "llama-cpp", "lm-studio", "ollama", "llama-cpp-python", "axiom-native"),
    "easy-setup": ("ollama", "lm-studio", "axiom-native", "llama-cpp", "llama-cpp-python", "vllm"),
    "portable": ("llama-cpp", "axiom-native", "llama-cpp-python", "ollama", "lm-studio", "vllm"),
    "minimal-daemons": ("axiom-native", "llama-cpp", "llama-cpp-python", "ollama", "lm-studio", "vllm"),
}


class EngineFabricError(RuntimeError):
    """A selected local backend cannot satisfy a bounded engine-fabric request."""


def _public_failure_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "authentication" in text:
        return "Local runtime requires authentication; AXIOM did not send credentials."
    if "expected api route" in text:
        return "Local runtime did not expose the expected API route or model."
    if "invalid model inventory" in text:
        return "Local runtime returned an invalid model inventory."
    if "not responding" in text:
        return "Local runtime is not responding on its configured loopback endpoint."
    if "safety limit" in text:
        return "Local runtime response exceeded the 2 MB safety limit."
    if "invalid json" in text or "unexpected response shape" in text or "without a text response" in text:
        return "Local runtime returned an unusable response."
    if "http" in text:
        return "Local runtime returned an HTTP error."
    return "The backend could not complete this request."


def _package_present(package: str | None) -> bool:
    if not package:
        return False
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _request_json(backend_id: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float) -> dict[str, Any]:
    """Call one fixed, allowlisted endpoint on a local runtime."""
    if backend_id not in _BACKEND_BY_ID:
        raise ValueError("Unknown local backend.")
    backend = _BACKEND_BY_ID[backend_id]
    allowed_paths = {backend["models_path"], backend["chat_path"]}
    if path not in allowed_paths:
        raise ValueError("Unsupported local runtime route.")
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        backend["endpoint"] + path,
        data=data,
        method="GET" if data is None else "POST",
        headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data is not None else {})},
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise EngineFabricError("Local runtime requires authentication; AXIOM did not send credentials.") from None
        if exc.code == 404:
            raise EngineFabricError("Local runtime did not expose the expected API route or model.") from None
        raise EngineFabricError(f"Local runtime returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError, OSError):
        raise EngineFabricError("Local runtime is not responding on its configured loopback endpoint.") from None
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise EngineFabricError("Local runtime response exceeded the 2 MB safety limit.")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise EngineFabricError("Local runtime returned invalid JSON.") from None
    if not isinstance(result, dict):
        raise EngineFabricError("Local runtime returned an unexpected response shape.")
    return result


def _model_ids(backend_id: str, payload: dict[str, Any]) -> list[str]:
    raw = payload.get("models") if backend_id == "ollama" else payload.get("data")
    if not isinstance(raw, list):
        raise EngineFabricError("Local runtime returned an invalid model inventory.")
    output: list[str] = []
    for item in raw[:200]:
        if not isinstance(item, dict):
            continue
        candidate = item.get("name") if backend_id == "ollama" else item.get("id")
        if not isinstance(candidate, str):
            continue
        try:
            name = validate_model_name(candidate)
        except ValueError:
            continue
        if name not in output:
            output.append(name)
    return output


def _generation_capabilities(backend_id: str) -> dict[str, Any]:
    """Describe the request dialect and separate extensions from common controls."""
    if backend_id == "axiom-native":
        return {
            "transport": "in_process",
            "parameter_mapping": {
                "max_new_tokens": "max_new_tokens",
                "temperature": "temperature",
                "top_p": "top_p",
                "top_k": "top_k",
                "min_p": "min_p",
                "repetition_penalty": "repetition_penalty",
                "seed": "seed",
                "stop_sequences": "stop_sequences",
            },
            "native_sampling_controls": [
                "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed", "stop_sequences",
            ],
            "extension_sampling_controls": [],
            "note": "Controls are passed into AXIOM's own sampler; reproducibility still depends on device and kernel behavior.",
        }
    if backend_id == "ollama":
        return {
            "transport": "ollama_options",
            "parameter_mapping": {
                "max_new_tokens": "options.num_predict",
                "temperature": "options.temperature",
                "top_p": "options.top_p",
                "top_k": "options.top_k",
                "min_p": "options.min_p",
                "repetition_penalty": "options.repeat_penalty",
                "seed": "options.seed",
                "stop_sequences": "options.stop",
            },
            "native_sampling_controls": [
                "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed", "stop_sequences",
            ],
            "extension_sampling_controls": [],
            "note": "Uses Ollama's /api/chat options dialect; exact output behavior depends on model and installed runtime version.",
        }
    if backend_id not in _BACKEND_BY_ID:
        raise ValueError("Unknown engine backend for generation capability lookup.")
    penalty_field = "repetition_penalty" if backend_id == "vllm" else "repeat_penalty"
    notes = {
        "llama-cpp": "Uses llama.cpp's OpenAI-compatible endpoint and server-specific top-k and repeat-penalty fields.",
        "lm-studio": "Advanced sampler extensions depend on the installed server and model backend.",
        "vllm": "Uses vLLM's repetition_penalty extension; API-version support can differ.",
        "llama-cpp-python": "Advanced sampler extensions depend on package version and server configuration.",
    }
    return {
        "transport": "openai_compatible",
        "parameter_mapping": {
            "max_new_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k (extension)",
            "repetition_penalty": penalty_field,
            "seed": "seed",
            "stop_sequences": "stop",
        },
        "native_sampling_controls": ["temperature", "top_p", "seed", "stop_sequences"],
        "extension_sampling_controls": ["top_k", "repetition_penalty"],
        "note": notes[backend_id],
    }


def _probe_backend(definition: dict[str, Any]) -> dict[str, Any]:
    backend_id = definition["id"]
    executable = definition["executable"]
    package = definition["package"]
    installed = bool(shutil.which(executable)) if executable else _package_present(package)
    try:
        inventory = _request_json(backend_id, definition["models_path"], timeout=_PROBE_TIMEOUT)
        models = _model_ids(backend_id, inventory)
        status = "ready"
        error = None
        reachable = True
    except EngineFabricError as exc:
        models = []
        reachable = "authentication" in str(exc).lower()
        if reachable:
            status, error = "auth_required", _public_failure_reason(exc)
        elif "expected api route" in str(exc).lower() or "invalid model inventory" in str(exc).lower():
            status, error = "incompatible", _public_failure_reason(exc)
        else:
            status = "offline"
            error = "No compatible local API responded at the configured loopback endpoint."
    return {
        **{key: definition[key] for key in ("id", "name", "kind", "endpoint", "supports_chat", "formats", "strengths", "tradeoffs", "best_for")},
        "status": status,
        "installed": installed,
        "reachable": reachable,
        "models": models,
        "model_count": len(models),
        "error": error,
        "scope": "fixed_loopback_only",
        "generation": _generation_capabilities(backend_id),
    }



def _native_backend() -> dict[str, Any]:
    try:
        from axiom.core.engine import runtime_status
        from axiom.core.llm_client import _native_models

        runtime = runtime_status()
        native_models = _native_models()
        names: dict[str, int] = {}
        for item in native_models:
            name = item.get("name")
            if isinstance(name, str):
                names[name] = names.get(name, 0) + 1
        models = sorted(name for name, count in names.items() if count == 1)
        count = len(models)
        ready = bool(runtime.get("ready")) and count > 0
        status = "ready" if ready else "setup_required"
        error = None if ready else (
            runtime.get("setup_note") if not runtime.get("ready")
            else "Add a uniquely named compatible Llama, Mistral, Qwen2 or Qwen3 safetensors checkpoint under a trusted model root."
        )
        installed = bool(runtime.get("ready"))
    except Exception:
        status, error, installed, models, count = (
            "setup_required",
            "Install AXIOM's native-inference extra and add a supported local checkpoint.",
            False,
            [],
            0,
        )
    return {
        "id": "axiom-native",
        "name": "AXIOM Native Engine",
        "kind": "native",
        "endpoint": None,
        "supports_chat": status == "ready",
        "formats": ["Floating-point safetensors"],
        "strengths": [
            "AXIOM-owned transformer forward pass and autoregressive KV-cache decode",
            "No separate inference daemon and no Transformers model.generate",
            "Explicit device, precision, memory-mode, cancellation and telemetry controls",
        ],
        "tradeoffs": [
            "Current native architecture support is Llama, Mistral, Qwen2 and Qwen3",
            "GGUF and other quantized formats need a compatible adapter/runtime",
            "Throughput may trail specialized C++ or batched serving engines",
        ],
        "best_for": ["privacy", "minimal-daemons", "balanced"],
        "status": status,
        "installed": installed,
        "reachable": True,
        "models": models,
        "model_count": count,
        "error": error,
        "scope": "in_process_native_engine",
        "generation": _generation_capabilities("axiom-native"),
    }


def _recommendation(backends: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    preference = _PROFILES[profile]
    by_id = {item["id"]: item for item in backends}
    for candidate_id in preference:
        item = by_id.get(candidate_id)
        if item and item["status"] == "ready" and (item["model_count"] > 0 or item["id"] != "axiom-native"):
            reasons = {
                "axiom-native": "Best fit when you want AXIOM-owned execution, a trusted safetensors checkpoint and no separate model daemon.",
                "ollama": "Best fit for convenient local model management and broad model choice with a simple loopback API.",
                "llama-cpp": "Best fit for GGUF quantization, controllable CPU/GPU offload and a compact local server.",
                "lm-studio": "Best fit when a desktop model manager and local OpenAI-compatible endpoint are already part of your workflow.",
                "vllm": "Best fit for throughput and multiple concurrent clients on a suitably provisioned accelerator.",
                "llama-cpp-python": "Best fit when Python integration and an existing GGUF runtime are priorities.",
            }
            return {"profile": profile, "backend_id": candidate_id, "name": item["name"], "status": "ready", "reason": reasons.get(candidate_id, "A compatible local runtime responded and is a good fit for this profile.")}
    for candidate_id in preference:
        item = by_id.get(candidate_id)
        if item and (item.get("installed") or item["id"] == "axiom-native"):
            return {
                "profile": profile,
                "backend_id": candidate_id,
                "name": item["name"],
                "status": "setup_required",
                "reason": "No selected runtime is ready yet. This is the most relevant installed/setup candidate for the chosen profile; verify model compatibility before running.",
            }
    return {
        "profile": profile,
        "backend_id": "axiom-native",
        "name": "AXIOM Native Engine",
        "status": "setup_required",
        "reason": "No ready local runtime was detected. Start with the AXIOM Native Engine setup and trusted local model inventory.",
    }


def discover_engine_backends(profile: str = "balanced") -> dict[str, Any]:
    """Return a live, evidence-labelled local engine matrix and a profile recommendation."""
    if profile not in _PROFILES:
        raise ValueError("Profile must be one of: " + ", ".join(_PROFILES))
    with ThreadPoolExecutor(max_workers=len(_BACKENDS), thread_name_prefix="axiom-engine-probe") as pool:
        futures = [pool.submit(_probe_backend, definition) for definition in _BACKENDS]
        backends = [future.result() for future in futures]
    backends.insert(0, _native_backend())
    return {
        "schema_version": _SCHEMA_VERSION,
        "checked_at": datetime.now(UTC).isoformat(),
        "host": {"os": platform.system(), "architecture": platform.machine()},
        "profile": profile,
        "profiles": [
            {"id": key, "label": label}
            for key, label in (
                ("balanced", "Balanced"),
                ("low-memory", "Low memory"),
                ("throughput", "High throughput"),
                ("easy-setup", "Easy setup"),
                ("portable", "Portable / quantized"),
                ("minimal-daemons", "Minimal background services"),
            )
        ],
        "backends": backends,
        "recommendation": _recommendation(backends, profile),
        "security": {
            "network_scope": "IPv4 loopback only",
            "environment_proxies_used": False,
            "remote_endpoints_configurable": False,
            "credentials_sent": False,
            "model_downloads_started": False,
            "processes_launched": False,
        },
        "scope_note": "Probe results describe fixed loopback APIs on this machine. An installed executable is not a running service, and availability does not prove model quality or hardware performance. AXIOM Native Engine stays independent of Ollama and other external runtimes.",
    }


def _validate_engine_chat_inputs(
    model: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, list[dict[str, str]]]:
    model = validate_model_name(model)
    if not isinstance(messages, list) or not 1 <= len(messages) <= _MAX_MESSAGES:
        raise ValueError("Provide between 1 and 64 conversation messages.")
    normalized: list[dict[str, str]] = []
    total_chars = 0
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Every conversation message must be an object.")
        role, content = item.get("role"), item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError("Conversation messages require system, user, or assistant roles and non-empty text.")
        if len(content) > _MAX_MESSAGE_CHARS:
            raise ValueError("Each conversation message is limited to 12,000 characters.")
        total_chars += len(content)
        normalized.append({"role": role, "content": content})
    if total_chars > _MAX_TOTAL_CHARS:
        raise ValueError("Combined conversation text is limited to 24,000 characters.")
    if not any(item["role"] == "user" for item in normalized):
        raise ValueError("At least one user message is required.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= _MAX_OUTPUT_TOKENS:
        raise ValueError("Maximum output tokens must be between 1 and 2,048.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    return model, normalized


def _validate_sampling_options(
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    seed: int | None,
    stop_sequences: list[str] | tuple[str, ...],
) -> tuple[float, int, float, float, int | None, tuple[str, ...]]:
    """Validate shared generation settings before contacting any runtime."""
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(float(top_p)) or not 0 < float(top_p) <= 1:
        raise ValueError("Top-p must be a finite number greater than 0 and at most 1.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 100_000:
        raise ValueError("Top-k must be an integer between 0 and 100,000; 0 disables the limit.")
    if isinstance(min_p, bool) or not isinstance(min_p, (int, float)) or not math.isfinite(float(min_p)) or not 0 <= float(min_p) <= 1:
        raise ValueError("Min-p must be a finite number between 0 and 1.")
    if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)) or not math.isfinite(float(repetition_penalty)) or not 0.1 <= float(repetition_penalty) <= 5:
        raise ValueError("Repetition penalty must be a finite number between 0.1 and 5.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 9_223_372_036_854_775_807):
        raise ValueError("Seed must be an integer between 0 and 9,223,372,036,854,775,807.")
    if not isinstance(stop_sequences, (list, tuple)) or len(stop_sequences) > 16:
        raise ValueError("Provide no more than 16 stop sequences.")
    stops: list[str] = []
    for stop in stop_sequences:
        if not isinstance(stop, str) or not stop or not stop.strip():
            raise ValueError("Stop sequences must be non-empty text.")
        if len(stop) > 256:
            raise ValueError("Each stop sequence is limited to 256 characters.")
        if stop not in stops:
            stops.append(stop)
    return float(top_p), top_k, float(min_p), float(repetition_penalty), seed, tuple(stops)


def run_engine_backend_chat(
    backend_id: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
    request_timeout: float | None = None,
) -> dict[str, Any]:
    """Run a bounded conversation through a chosen, fixed loopback-only backend."""
    if backend_id not in _BACKEND_BY_ID and backend_id != "axiom-native":
        raise ValueError("Choose a supported local backend from the engine inventory.")
    model, normalized = _validate_engine_chat_inputs(model, messages, max_new_tokens, temperature)
    top_p, top_k, min_p, repetition_penalty, seed, stops = _validate_sampling_options(
        top_p, top_k, min_p, repetition_penalty, seed, stop_sequences
    )
    if min_p > 0 and backend_id not in {"axiom-native", "ollama"}:
        raise ValueError("Min-p is supported only by AXIOM Native Engine and Ollama backends.")
    if backend_id == "axiom-native":
        if request_timeout is not None:
            raise ValueError("AXIOM Native Engine runs in-process and does not accept an external HTTP request timeout.")
        from axiom.core import llm_client

        started = time.perf_counter()
        try:
            native_result = llm_client.chat(
                "axiom-local",
                model,
                normalized,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_sequences=stops,
            )
        except (ValueError, llm_client.LLMClientError):
            raise EngineFabricError("AXIOM Native Engine could not complete this request.") from None
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        return {
            "schema_version": _SCHEMA_VERSION,
            "backend_id": "axiom-native",
            "backend_name": "AXIOM Native Engine",
            "model": model,
            "response": str(native_result.get("response", ""))[:100_000],
            "done": True,
            "prompt_eval_count": native_result.get("prompt_tokens"),
            "eval_count": native_result.get("completion_tokens"),
            "total_duration_ms": round(elapsed_ms, 2),
            "generated_at": native_result.get("generated_at") or datetime.now(UTC).isoformat(),
            "execution": "in_process_native",
            "device": native_result.get("device"),
            "precision": native_result.get("precision"),
            "privacy": {"prompt_returned": False, "response_persisted": False, "cloud_provider_used": False},
        }
    if request_timeout is not None and (
        isinstance(request_timeout, bool)
        or not isinstance(request_timeout, (int, float))
        or not 0.1 <= float(request_timeout) <= _REQUEST_TIMEOUT
    ):
        raise ValueError(f"Request timeout must be between 0.1 and {_REQUEST_TIMEOUT} seconds.")
    timeout = _REQUEST_TIMEOUT if request_timeout is None else float(request_timeout)
    started = time.perf_counter()
    if backend_id == "ollama":
        options: dict[str, Any] = {
            "num_predict": max_new_tokens,
            "temperature": float(temperature),
            "top_p": top_p,
        }
        if top_k:
            options["top_k"] = top_k
        if min_p > 0:
            options["min_p"] = min_p
        if repetition_penalty != 1.0:
            options["repeat_penalty"] = repetition_penalty
        if seed is not None:
            options["seed"] = seed
        if stops:
            options["stop"] = list(stops)
        payload = {
            "model": model,
            "messages": normalized,
            "stream": False,
            "options": options,
        }
        result = _request_json(backend_id, _BACKEND_BY_ID[backend_id]["chat_path"], payload, timeout=timeout)
        message = result.get("message")
        response_text = message.get("content") if isinstance(message, dict) else None
        prompt_tokens = result.get("prompt_eval_count")
        completion_tokens = result.get("eval_count")
        duration_ns = result.get("total_duration")
        finished = bool(result.get("done", True))
    else:
        # OpenAI-compatible runtimes share top_p/stop/seed; top-k and
        # repetition-penalty extensions differ in spelling.
        payload = {
            "model": model,
            "messages": normalized,
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": float(temperature),
            "top_p": top_p,
        }
        if stops:
            payload["stop"] = list(stops)
        if seed is not None:
            payload["seed"] = seed
        if top_k:
            payload["top_k"] = top_k
        if repetition_penalty != 1.0:
            payload["repetition_penalty" if backend_id == "vllm" else "repeat_penalty"] = repetition_penalty
        result = _request_json(backend_id, _BACKEND_BY_ID[backend_id]["chat_path"], payload, timeout=timeout)
        choices = result.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        response_text = message.get("content")
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        duration_ns = None
        finished = choice.get("finish_reason") not in {"length", "content_filter"}
    if not isinstance(response_text, str):
        raise EngineFabricError("Local backend completed without a text response.")
    elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
    duration_ms = round(duration_ns / 1_000_000, 2) if isinstance(duration_ns, (int, float)) and duration_ns >= 0 else round(elapsed_ms, 2)
    return {
        "schema_version": _SCHEMA_VERSION,
        "backend_id": backend_id,
        "backend_name": _BACKEND_BY_ID[backend_id]["name"],
        "model": model,
        "response": response_text[:100_000],
        "done": finished,
        "prompt_eval_count": prompt_tokens if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) else None,
        "eval_count": completion_tokens if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) else None,
        "total_duration_ms": duration_ms,
        "generated_at": datetime.now(UTC).isoformat(),
        "execution": "local_loopback",
        "privacy": {"prompt_returned": False, "response_persisted": False, "cloud_provider_used": False},
    }


_AUTO_ROUTE_MAX_ATTEMPTS = 3
_AUTO_ROUTE_TIMEOUT_SECONDS = 45.0
_AUTO_ROUTE_SCOPE = (
    "Auto-routing uses fixed loopback services only and matches exact model IDs in live inventories. "
    "Matching names do not prove identical checkpoint weights or quantization across runtimes."
)


def run_engine_fabric_chat(
    profile: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Select a local backend by workload profile and fail over only to exact-ID matches."""
    if profile not in _PROFILES:
        raise ValueError("Profile must be one of: " + ", ".join(_PROFILES))
    model, normalized = _validate_engine_chat_inputs(model, messages, max_new_tokens, temperature)
    top_p, top_k, min_p, repetition_penalty, seed, stops = _validate_sampling_options(
        top_p, top_k, min_p, repetition_penalty, seed, stop_sequences
    )
    report = discover_engine_backends(profile)
    inventory = {backend.get("id"): backend for backend in report.get("backends", []) if isinstance(backend, dict)}
    candidates: list[dict[str, Any]] = []
    for backend_id in _PROFILES[profile]:
        # Filter incompatible runtimes before applying the attempt limit. Otherwise
        # three earlier unsupported backends could hide a later native/Ollama match.
        if min_p > 0 and backend_id not in {"axiom-native", "ollama"}:
            continue
        backend = inventory.get(backend_id)
        if (
            isinstance(backend, dict)
            and backend.get("status") == "ready"
            and backend.get("supports_chat") is True
            and model in backend.get("models", [])
        ):
            candidates.append(backend)
        if len(candidates) >= _AUTO_ROUTE_MAX_ATTEMPTS:
            break
    if not candidates:
        if min_p > 0:
            raise EngineFabricError("Min-p requires a ready AXIOM Native Engine or Ollama backend with this exact model ID.")
        raise EngineFabricError(
            f"No ready local backend advertises the exact model ID {model!r}. "
            "Load/configure that model on a supported local service and refresh discovery."
        )

    attempts: list[dict[str, Any]] = []
    for backend in candidates:
        backend_id = str(backend["id"])
        backend_name = str(backend["name"])
        try:
            dispatch_options: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
                "stop_sequences": stops,
            }
            if backend_id != "axiom-native":
                dispatch_options["request_timeout"] = _AUTO_ROUTE_TIMEOUT_SECONDS
            result = run_engine_backend_chat(backend_id, model, normalized, **dispatch_options)
        except EngineFabricError as exc:
            attempts.append({
                "backend_id": backend_id,
                "backend_name": backend_name,
                "outcome": "failed",
                "reason": _public_failure_reason(exc),
            })
            continue
        attempts.append({"backend_id": backend_id, "backend_name": backend_name, "outcome": "selected"})
        result["routing"] = {
            "strategy": "profile_order_exact_model_id",
            "profile": profile,
            "requested_model": model,
            "fallback_used": len(attempts) > 1,
            "attempts": attempts,
            "max_attempts": _AUTO_ROUTE_MAX_ATTEMPTS,
            "scope_note": _AUTO_ROUTE_SCOPE,
        }
        return result

    failures = "; ".join(f"{item['backend_name']}: {item['reason']}" for item in attempts)
    raise EngineFabricError("All matching local backends failed: " + failures)



def run_engine_backend_benchmark(
    backend_id: str,
    model: str,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    warmup_runs: int = 1,
    iterations: int = 3,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Measure bounded local end-to-end latency without returning or saving generated text."""
    import statistics

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Benchmark prompt must contain non-whitespace text.")
    if len(prompt) > _MAX_MESSAGE_CHARS:
        raise ValueError("Benchmark prompt is limited to 12,000 characters.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 256:
        raise ValueError("Benchmark output-token limit must be between 1 and 256.")
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or not 0 <= warmup_runs <= 1:
        raise ValueError("Benchmark warm-up count must be 0 or 1.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 5:
        raise ValueError("Benchmark iterations must be between 1 and 5.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
        raise ValueError("Benchmark temperature must be between 0 and 2.")
    if backend_id not in _BACKEND_BY_ID and backend_id != "axiom-native":
        raise ValueError("Choose a supported local backend.")
    top_p, top_k, min_p, repetition_penalty, seed, stops = _validate_sampling_options(
        top_p, top_k, min_p, repetition_penalty, seed, stop_sequences
    )

    messages = [{"role": "user", "content": prompt}]
    for _ in range(warmup_runs):
        warmup = run_engine_backend_chat(
            backend_id, model, messages, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed, stop_sequences=stops,
        )
        del warmup

    latencies: list[float] = []
    completion_counts: list[int | None] = []
    token_counted_durations_ms: list[float] = []
    started = time.perf_counter()
    for _ in range(iterations):
        request_started = time.perf_counter()
        measured = run_engine_backend_chat(
            backend_id, model, messages, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed, stop_sequences=stops,
        )
        latency_ms = max(0.0, (time.perf_counter() - request_started) * 1000.0)
        latencies.append(round(latency_ms, 2))
        count = measured.get("eval_count")
        completion_count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None
        completion_counts.append(completion_count)
        if completion_count is not None:
            token_counted_durations_ms.append(latency_ms)
        del measured
    wall_time_ms = round(max(0.0, (time.perf_counter() - started) * 1000.0), 2)
    ordered = sorted(latencies)
    median = round(statistics.median(ordered), 2)
    p95_index = max(0, min(len(ordered) - 1, int((0.95 * len(ordered) + 0.999999)) - 1))
    known_counts = [value for value in completion_counts if value is not None]
    total_tokens = sum(known_counts) if known_counts else None
    known_duration_ms = sum(token_counted_durations_ms)
    return {
        "schema_version": _SCHEMA_VERSION,
        "backend_id": backend_id,
        "backend_name": "AXIOM Native Engine" if backend_id == "axiom-native" else _BACKEND_BY_ID[backend_id]["name"],
        "model": model,
        "configuration": {
            "prompt_characters": len(prompt),
            "max_new_tokens": max_new_tokens,
            "temperature": float(temperature),
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
            "stop_sequence_count": len(stops),
            "warmup_runs": warmup_runs,
            "iterations": iterations,
            "measured_requests": iterations,
        },
        "measurements": {
            "wall_time_ms": wall_time_ms,
            "requests_per_second": round(iterations * 1000 / wall_time_ms, 3) if wall_time_ms > 0 else 0.0,
            "completion_tokens_known": len(known_counts),
            "total_completion_tokens": total_tokens,
            "completion_tokens_per_second": round(total_tokens * 1000 / known_duration_ms, 3) if total_tokens is not None and known_duration_ms > 0 else None,
            "latency_ms": {
                "min": min(ordered),
                "median": median,
                "p95": ordered[p95_index],
                "max": max(ordered),
            },
            "requests": [
                {"latency_ms": latency, "completion_tokens": count}
                for latency, count in zip(latencies, completion_counts, strict=True)
            ],
        },
        "privacy": {"prompt_returned": False, "responses_returned": False, "results_persisted": False},
        "scope_note": "Measured end-to-end calls on this machine. Token throughput is available only when the backend reports token counts; warm-ups are excluded, and benchmark results are not persisted.",
    }


__all__ = ["EngineFabricError", "discover_engine_backends", "run_engine_backend_chat", "run_engine_backend_benchmark"]
