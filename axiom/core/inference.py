"""Bounded local inference adapter for an Ollama instance on loopback only."""
from __future__ import annotations

import json
import math
import re
import shutil
from datetime import UTC, datetime
from typing import Any, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
_STATUS_TIMEOUT_SECONDS = 2.5
_GENERATION_TIMEOUT_SECONDS = 300
_MAX_PROMPT_CHARS = 12_000
_MAX_MODEL_NAME_CHARS = 120
_MAX_GENERATED_TOKENS = 2_048
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$")


class InferenceRuntimeError(RuntimeError):
    """Raised when the local inference runtime cannot complete a request."""


def validate_model_name(model: str) -> str:
    value = model.strip() if isinstance(model, str) else ""
    if not value or len(value) > _MAX_MODEL_NAME_CHARS or "://" in value or not _MODEL_NAME.fullmatch(value):
        raise ValueError(
            "Model name must be 1–120 characters and contain only letters, digits, '.', '_', ':', '/', or '-'."
        )
    return value


def validate_generation(prompt: str, num_predict: int, temperature: float) -> tuple[str, int, float]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must contain at least one non-whitespace character.")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds the {_MAX_PROMPT_CHARS:,}-character limit.")
    if isinstance(num_predict, bool) or not isinstance(num_predict, int) or not 1 <= num_predict <= _MAX_GENERATED_TOKENS:
        raise ValueError(f"Maximum generated tokens must be between 1 and {_MAX_GENERATED_TOKENS}.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    return prompt, num_predict, float(temperature)


def validate_sampling_options(
    *,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float | None = None,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] | None = None,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Validate optional Ollama generation settings and return Ollama option names."""
    options: dict[str, Any] = {}
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < float(top_p) <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1.")
        options["top_p"] = float(top_p)
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 100_000:
            raise ValueError("top_k must be between 0 and 100,000.")
        options["top_k"] = top_k
    if min_p is not None:
        if isinstance(min_p, bool) or not isinstance(min_p, (int, float)) or not math.isfinite(float(min_p)) or not 0 <= float(min_p) <= 1:
            raise ValueError("min_p must be a finite number between 0 and 1.")
        if float(min_p) > 0:
            options["min_p"] = float(min_p)
    if repetition_penalty is not None:
        if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)) or not math.isfinite(float(repetition_penalty)) or not 0.1 <= float(repetition_penalty) <= 5:
            raise ValueError("repetition_penalty must be between 0.1 and 5.")
        options["repeat_penalty"] = float(repetition_penalty)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 9_223_372_036_854_775_807:
            raise ValueError("seed must be a non-negative signed 64-bit integer.")
        options["seed"] = seed
    if num_ctx is not None:
        if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or not 512 <= num_ctx <= 131_072:
            raise ValueError("num_ctx must be between 512 and 131,072 tokens.")
        options["num_ctx"] = num_ctx
    if stop_sequences is not None:
        if not isinstance(stop_sequences, (list, tuple)) or len(stop_sequences) > 16:
            raise ValueError("At most 16 stop sequences may be supplied.")
        stops: list[str] = []
        for value in stop_sequences:
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError("Each stop sequence must contain 1–256 non-whitespace characters.")
            if value not in stops:
                stops.append(value)
        if stops:
            options["stop"] = stops
    return options


def _open_loopback(request: Request, timeout: float):
    # Explicitly disable environment-configured proxies. The request destination
    # is a constant IPv4 loopback URL; callers cannot provide a hostname or URL.
    return build_opener(ProxyHandler({})).open(request, timeout=timeout)


def _request_json(path: str, *, payload: dict[str, Any] | None = None, timeout: float) -> dict[str, Any]:
    if path not in {"/api/tags", "/api/generate", "/api/chat", "/api/show", "/api/ps"}:
        raise ValueError("Unsupported local inference endpoint.")
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{OLLAMA_ENDPOINT}{path}",
        data=data,
        method="GET" if data is None else "POST",
        headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data is not None else {})},
    )
    try:
        with _open_loopback(request, timeout) as response:
            raw = response.read(2_000_000)
    except HTTPError as exc:
        try:
            error_raw = exc.read(8192).decode("utf-8", errors="replace")
            parsed = json.loads(error_raw) if error_raw else {}
            detail = parsed.get("error", "") if isinstance(parsed, dict) else ""
        except (ValueError, UnicodeError):
            detail = ""
        detail = str(detail).replace("\n", " ").strip()[:240]
        if exc.code == 404 and path in {"/api/generate", "/api/show"}:
            raise InferenceRuntimeError(detail or "Ollama could not find that model. Pull the model locally first.") from None
        raise InferenceRuntimeError(f"Local Ollama request failed (HTTP {exc.code}). {detail}".strip()) from None
    except (URLError, TimeoutError, OSError) as exc:
        # Do not reveal machine-local paths or proxy/config details.
        raise InferenceRuntimeError(
            "Ollama is not responding at 127.0.0.1:11434. Start the local Ollama service and retry."
        ) from None
    try:
        result = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise InferenceRuntimeError("The local inference runtime returned invalid JSON.") from None
    if not isinstance(result, dict):
        raise InferenceRuntimeError("The local inference runtime returned an unexpected response shape.")
    return result


def runtime_status() -> dict[str, Any]:
    """Probe only the fixed Ollama loopback API; never launch or contact remote services."""
    base: dict[str, Any] = {
        "backend": "ollama",
        "endpoint": OLLAMA_ENDPOINT,
        "runtime_installed": shutil.which("ollama") is not None,
        "status": "offline",
        "models": [],
        "error": None,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    try:
        result = _request_json("/api/tags", timeout=_STATUS_TIMEOUT_SECONDS)
        raw_models = result.get("models", [])
        if not isinstance(raw_models, list):
            raise InferenceRuntimeError("Ollama returned an invalid model inventory.")
        names: list[str] = []
        for item in raw_models[:100]:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str):
                    try:
                        names.append(validate_model_name(name))
                    except ValueError:
                        continue
        base.update(status="online", models=names)
    except InferenceRuntimeError:
        base["error"] = f"Local inference runtime is unavailable at {OLLAMA_ENDPOINT}."
    return base





def remove_local_model(model: str) -> dict[str, Any]:
    """Remove one installed Ollama model tag after explicit caller confirmation."""
    import os
    import subprocess

    model = validate_model_name(model)
    before = runtime_status()
    if before.get("status") != "online":
        raise InferenceRuntimeError("Ollama must be online before removing a local model.")
    if model not in before.get("models", []):
        raise ValueError("That exact model tag is not installed in the local Ollama inventory.")
    executable = shutil.which("ollama")
    if not executable:
        raise InferenceRuntimeError("Ollama's command-line executable was not found on PATH.")
    environment = os.environ.copy()
    environment["OLLAMA_HOST"] = "127.0.0.1:11434"
    try:
        result = subprocess.run(
            [executable, "rm", model],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise InferenceRuntimeError("Ollama model removal exceeded the 120-second time limit.") from None
    except OSError:
        raise InferenceRuntimeError("AXIOM could not run the local Ollama removal command.") from None
    if result.returncode != 0:
        raise InferenceRuntimeError(f"Ollama could not remove this model tag (exit status {result.returncode}).")
    after = runtime_status()
    still_installed = model in after.get("models", [])
    if after.get("status") == "online" and still_installed:
        raise InferenceRuntimeError("Ollama returned success, but the selected model tag is still in the inventory.")
    return {
        "schema_version": 1,
        "model": model,
        "removed": not still_installed,
        "model_files_deleted": not still_installed,
        "remaining_models": after.get("models", []) if after.get("status") == "online" else [],
        "scope_note": "The requested Ollama model tag was passed to the local removal command. Shared model blobs may remain when other installed tags reference them.",
    }



def list_running_local_models() -> dict[str, Any]:
    """List loaded Ollama models and their reported CPU/GPU memory footprint."""
    result = _request_json("/api/ps", timeout=_STATUS_TIMEOUT_SECONDS)
    raw_models = result.get("models")
    if not isinstance(raw_models, list):
        raise InferenceRuntimeError("Ollama returned an invalid loaded-model inventory.")

    models: list[dict[str, Any]] = []
    for item in raw_models[:100]:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if not isinstance(name, str):
            continue
        try:
            name = validate_model_name(name)
        except ValueError:
            continue
        size = item.get("size")
        size_vram = item.get("size_vram")
        expires_at = item.get("expires_at")
        digest = item.get("digest")
        details = item.get("details")
        details = details if isinstance(details, dict) else {}

        def nonnegative_bytes(value: Any) -> int | None:
            return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**63 - 1 else None

        def safe_text(value: Any, limit: int = 100) -> str | None:
            return value[:limit] if isinstance(value, str) and value.strip() else None

        size_bytes = nonnegative_bytes(size)
        vram_bytes = nonnegative_bytes(size_vram)
        models.append({
            "name": name,
            "size_bytes": size_bytes,
            "size_vram_bytes": vram_bytes,
            "size_ram_bytes": max(0, size_bytes - vram_bytes) if size_bytes is not None and vram_bytes is not None else None,
            "memory_placement": "mixed" if vram_bytes and size_bytes and vram_bytes < size_bytes else "gpu" if vram_bytes else "cpu_or_unreported",
            "expires_at": expires_at if isinstance(expires_at, str) else None,
            "digest_prefix": digest[:12] if isinstance(digest, str) and len(digest) >= 12 else None,
            "format": safe_text(details.get("format")),
            "family": safe_text(details.get("family")),
            "parameter_size": safe_text(details.get("parameter_size")),
            "quantization_level": safe_text(details.get("quantization_level")),
        })
    return {
        "schema_version": 1,
        "backend": "ollama",
        "endpoint": OLLAMA_ENDPOINT,
        "models": models,
        "count": len(models),
        "scope_note": "Point-in-time local Ollama process inventory. Memory sizes are runtime-reported estimates, not guaranteed allocator measurements. Prompts and generated text are never requested.",
    }


def inspect_local_model(model: str) -> dict[str, Any]:
    """Return a deliberately small metadata summary for a model already in Ollama."""
    model = validate_model_name(model)
    result = _request_json(
        "/api/show",
        payload={"model": model, "verbose": False},
        timeout=15,
    )
    raw_details = result.get("details")
    details = raw_details if isinstance(raw_details, dict) else {}
    raw_info = result.get("model_info")
    model_info = raw_info if isinstance(raw_info, dict) else {}

    context_length: int | None = None
    parameter_count: int | None = None
    for key, value in model_info.items():
        lowered = str(key).lower()
        if context_length is None and ("context_length" in lowered or lowered == "context_length"):
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 10_000_000:
                context_length = value
        if parameter_count is None and ("parameter_count" in lowered or lowered == "parameter_count"):
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 10**15:
                parameter_count = value

    tags = _request_json("/api/tags", timeout=_STATUS_TIMEOUT_SECONDS)
    model_entry: dict[str, Any] = {}
    raw_models = tags.get("models")
    if isinstance(raw_models, list):
        for item in raw_models[:500]:
            if isinstance(item, dict) and item.get("name") == model:
                model_entry = item
                break

    def safe_text(value: Any, limit: int = 100) -> str | None:
        return value[:limit] if isinstance(value, str) and value.strip() else None

    size = model_entry.get("size")
    modified_at = model_entry.get("modified_at")
    families = details.get("families")
    capabilities = result.get("capabilities")
    return {
        "schema_version": 1,
        "model": model,
        "installed": bool(model_entry),
        "size_bytes": size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None,
        "modified_at": modified_at if isinstance(modified_at, str) else None,
        "format": safe_text(details.get("format")),
        "family": safe_text(details.get("family")),
        "families": [value[:60] for value in families[:16] if isinstance(value, str)] if isinstance(families, list) else [],
        "parameter_size": safe_text(details.get("parameter_size")),
        "parameter_count": parameter_count,
        "quantization_level": safe_text(details.get("quantization_level")),
        "context_length": context_length,
        "capabilities": [value[:60] for value in capabilities[:16] if isinstance(value, str)] if isinstance(capabilities, list) else [],
        "template_configured": bool(safe_text(result.get("template"), 1)),
        "license_present": bool(safe_text(result.get("license"), 1)),
        "system_prompt_configured": bool(safe_text(result.get("system"), 1)),
        "scope_note": "Metadata only from the local Ollama API. Templates, license text, modelfile contents, prompts, and generated responses are not returned.",
    }


def unload_local_model(model: str) -> dict[str, Any]:
    """Ask Ollama to unload one model from memory without removing its downloaded files."""
    model = validate_model_name(model)
    _request_json(
        "/api/generate",
        payload={
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        },
        timeout=30,
    )
    return {
        "schema_version": 1,
        "model": model,
        "unloaded": True,
        "model_files_deleted": False,
        "scope_note": "Ollama was asked to release this model from memory. Downloaded model files remain installed.",
    }



def stream_local_inference(
    model: str,
    prompt: str,
    *,
    num_predict: int = 256,
    temperature: float = 0.7,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float | None = None,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] | None = None,
    num_ctx: int | None = None,
    cancel_event: Any = None,
    on_response: Callable[[Any], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream Ollama NDJSON token events from the fixed loopback service.

    The iterator does not persist prompts or output. The on_response callback
    lets the API layer close the upstream socket when its caller disconnects.
    """
    model = validate_model_name(model)
    prompt, num_predict, temperature = validate_generation(prompt, num_predict, temperature)
    sampling_options = validate_sampling_options(top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed, stop_sequences=stop_sequences, num_ctx=num_ctx)
    if cancel_event is not None and cancel_event.is_set():
        return
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict, "temperature": temperature, **sampling_options},
    }
    request = Request(
        f"{OLLAMA_ENDPOINT}/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Accept": "application/x-ndjson, application/json", "Content-Type": "application/json"},
    )
    chars_returned = 0
    response_parts: list[str] = []
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration_ms: float | None = None
    done_reason: str | None = None
    completed = False
    started_at = datetime.now(UTC).isoformat()

    try:
        with _open_loopback(request, timeout=_GENERATION_TIMEOUT_SECONDS) as response:
            if on_response is not None:
                on_response(response)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return
                raw_line = response.readline(2_000_000)
                if not raw_line:
                    break
                if len(raw_line) >= 2_000_000 and not raw_line.endswith(b"\n"):
                    raise InferenceRuntimeError("Ollama returned an oversized streaming event.")
                try:
                    item = json.loads(raw_line.decode("utf-8"))
                except (ValueError, UnicodeError):
                    raise InferenceRuntimeError("Ollama returned an invalid streaming event.") from None
                if not isinstance(item, dict):
                    raise InferenceRuntimeError("Ollama returned an unexpected streaming event shape.")
                error = item.get("error")
                if isinstance(error, str) and error.strip():
                    raise InferenceRuntimeError(error.replace("\n", " ")[:240])
                token = item.get("response")
                if isinstance(token, str) and token:
                    remaining = max(0, 100_000 - chars_returned)
                    token = token[:remaining]
                    if token:
                        chars_returned += len(token)
                        response_parts.append(token)
                        yield {"type": "token", "token": token}
                if isinstance(item.get("prompt_eval_count"), int) and not isinstance(item.get("prompt_eval_count"), bool):
                    if item["prompt_eval_count"] >= 0:
                        prompt_eval_count = item["prompt_eval_count"]
                if isinstance(item.get("eval_count"), int) and not isinstance(item.get("eval_count"), bool):
                    if item["eval_count"] >= 0:
                        eval_count = item["eval_count"]
                duration = item.get("total_duration")
                if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
                    total_duration_ms = round(float(duration) / 1_000_000, 2)
                if isinstance(item.get("done_reason"), str):
                    done_reason = item["done_reason"][:80]
                if item.get("done") is True:
                    completed = True
                    break
    except HTTPError as exc:
        try:
            body = exc.read(8192).decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            detail = parsed.get("error", "") if isinstance(parsed, dict) else ""
        except (ValueError, UnicodeError):
            detail = ""
        detail = str(detail).replace("\n", " ").strip()[:240]
        if exc.code == 404:
            raise InferenceRuntimeError(detail or "Ollama could not find that model or endpoint. Check the local model name and Ollama version.") from None
        raise InferenceRuntimeError(f"Local Ollama stream failed (HTTP {exc.code}). {detail}".strip()) from None
    except (URLError, TimeoutError, OSError):
        if cancel_event is not None and cancel_event.is_set():
            return
        raise InferenceRuntimeError("Ollama stopped responding while streaming. Check the local service and retry.") from None

    if cancel_event is not None and cancel_event.is_set():
        return
    if not completed:
        raise InferenceRuntimeError("Ollama closed the generation stream before reporting completion.")
    yield {
        "type": "complete",
        "result": {
            "model": model,
            "response": "".join(response_parts),
            "done": True,
            "done_reason": done_reason,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "total_duration_ms": total_duration_ms,
            "generated_at": started_at,
            "execution": "local",
        },
    }



def run_local_inference(
    model: str,
    prompt: str,
    num_predict: int = 256,
    temperature: float = 0.7,
    *,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float | None = None,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] | None = None,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Generate text through a local Ollama service with bounded input/output."""
    model = validate_model_name(model)
    prompt, num_predict, temperature = validate_generation(prompt, num_predict, temperature)
    sampling_options = validate_sampling_options(top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty, seed=seed, stop_sequences=stop_sequences, num_ctx=num_ctx)
    result = _request_json(
        "/api/generate",
        payload={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": temperature, **sampling_options},
        },
        timeout=_GENERATION_TIMEOUT_SECONDS,
    )
    response = result.get("response")
    if not isinstance(response, str):
        raise InferenceRuntimeError("The local inference runtime did not return generated text.")
    # Output is limited by Ollama's num_predict option; truncate defensively in
    # case a compatible runtime ignores the option or returns an oversized body.
    response = response[:100_000]
    total_duration = result.get("total_duration")
    return {
        "model": model,
        "response": response,
        "done": bool(result.get("done", True)),
        "prompt_eval_count": result.get("prompt_eval_count") if isinstance(result.get("prompt_eval_count"), int) else None,
        "eval_count": result.get("eval_count") if isinstance(result.get("eval_count"), int) else None,
        "total_duration_ms": round(total_duration / 1_000_000, 2) if isinstance(total_duration, (int, float)) and total_duration >= 0 else None,
        "generated_at": datetime.now(UTC).isoformat(),
    }



def run_local_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    num_predict: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run a bounded multi-turn chat through the fixed loopback Ollama API."""
    model = validate_model_name(model)
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ValueError("Provide between 1 and 64 chat messages.")
    normalized: list[dict[str, str]] = []
    total_chars = 0
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Every chat message must be an object.")
        role, content = item.get("role"), item.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError("Chat messages require system, user, or assistant roles with non-empty text.")
        if len(content) > _MAX_PROMPT_CHARS:
            raise ValueError(f"Each chat message must be at most {_MAX_PROMPT_CHARS:,} characters.")
        total_chars += len(content)
        normalized.append({"role": role, "content": content})
    if total_chars > 24_000:
        raise ValueError("Combined chat content must not exceed 24,000 characters.")
    if not any(item["role"] == "user" for item in normalized):
        raise ValueError("At least one user message is required.")
    last_user = next(item["content"] for item in reversed(normalized) if item["role"] == "user")
    _, num_predict, temperature = validate_generation(last_user, num_predict, temperature)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < float(top_p) <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 100_000:
        raise ValueError("top_k must be between 0 and 100,000.")
    if isinstance(min_p, bool) or not isinstance(min_p, (int, float)) or not math.isfinite(float(min_p)) or not 0 <= float(min_p) <= 1:
        raise ValueError("min_p must be a finite number between 0 and 1.")
    if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)) or not math.isfinite(float(repetition_penalty)) or not 0.1 <= float(repetition_penalty) <= 5:
        raise ValueError("repetition_penalty must be between 0.1 and 5.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 9_223_372_036_854_775_807):
        raise ValueError("seed must be a non-negative signed 64-bit integer.")
    if not isinstance(stop_sequences, (list, tuple)) or len(stop_sequences) > 16:
        raise ValueError("At most 16 stop sequences may be supplied.")
    stops: list[str] = []
    for value in stop_sequences:
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError("Each stop sequence must contain 1–256 non-whitespace characters.")
        if value not in stops:
            stops.append(value)
    options: dict[str, Any] = {
        "num_predict": num_predict,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repeat_penalty": float(repetition_penalty),
    }
    if top_k > 0:
        options["top_k"] = top_k
    if min_p > 0:
        options["min_p"] = float(min_p)
    if seed is not None:
        options["seed"] = seed
    if stops:
        options["stop"] = stops
    result = _request_json(
        "/api/chat",
        payload={
            "model": model,
            "messages": normalized,
            "stream": False,
            "options": options,
        },
        timeout=_GENERATION_TIMEOUT_SECONDS,
    )
    message = result.get("message")
    response = message.get("content") if isinstance(message, dict) else None
    if not isinstance(response, str):
        raise InferenceRuntimeError("The local inference runtime did not return chat text.")
    total_duration = result.get("total_duration")
    return {
        "model": model,
        "response": response[:100_000],
        "done": bool(result.get("done", True)),
        "done_reason": result.get("done_reason") if isinstance(result.get("done_reason"), str) else None,
        "prompt_eval_count": result.get("prompt_eval_count") if isinstance(result.get("prompt_eval_count"), int) else None,
        "eval_count": result.get("eval_count") if isinstance(result.get("eval_count"), int) else None,
        "total_duration_ms": round(total_duration / 1_000_000, 2) if isinstance(total_duration, (int, float)) and total_duration >= 0 else None,
        "generated_at": datetime.now(UTC).isoformat(),
        "execution": "local",
    }


def run_local_inference_benchmark(
    model: str,
    prompt: str,
    *,
    num_predict: int = 128,
    temperature: float = 0.2,
    warmup_runs: int = 1,
    iterations: int = 3,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Measure real loopback inference latency and throughput without retaining text.

    The benchmark is intentionally bounded to at most ten measured generations
    and two warm-up generations. It uses the same loopback-only adapter as normal
    inference. Generated prompt/response contents are not returned or persisted.
    """
    import statistics
    import time
    from concurrent.futures import ThreadPoolExecutor

    model = validate_model_name(model)
    prompt, num_predict, temperature = validate_generation(prompt, num_predict, temperature)
    if num_predict > 512:
        raise ValueError("Benchmark output-token limit must be between 1 and 512.")
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or not 0 <= warmup_runs <= 2:
        raise ValueError("Benchmark warm-up runs must be between 0 and 2.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 10:
        raise ValueError("Benchmark iterations must be between 1 and 10.")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or not 1 <= concurrency <= 4:
        raise ValueError("Benchmark concurrency must be between 1 and 4.")
    measured_requests = iterations * concurrency
    if measured_requests > 10:
        raise ValueError("Benchmark iterations × concurrency may not exceed 10 measured requests.")

    for _ in range(warmup_runs):
        run_local_inference(model, prompt, num_predict=num_predict, temperature=temperature)

    def measured_call() -> dict[str, Any]:
        started = time.perf_counter()
        result = run_local_inference(model, prompt, num_predict=num_predict, temperature=temperature)
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        return {
            "latency_ms": round(elapsed_ms, 2),
            "prompt_tokens": result.get("prompt_eval_count") if isinstance(result.get("prompt_eval_count"), int) else None,
            "completion_tokens": result.get("eval_count") if isinstance(result.get("eval_count"), int) else None,
            "done": result.get("done") is True,
        }

    samples: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    # Each iteration is a measured wave. Requests inside a wave run concurrently
    # at the caller's explicitly selected concurrency level.
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="axiom-inference-bench") as executor:
        for _ in range(iterations):
            futures = [executor.submit(measured_call) for _ in range(concurrency)]
            samples.extend(future.result() for future in futures)
    total_wall_seconds = max(time.perf_counter() - benchmark_started, 1e-9)

    latencies = sorted(sample["latency_ms"] for sample in samples)
    completion_counts = [sample["completion_tokens"] for sample in samples if sample["completion_tokens"] is not None]
    prompt_counts = [sample["prompt_tokens"] for sample in samples if sample["prompt_tokens"] is not None]

    def percentile(values: list[float], percentage: float) -> float | None:
        if not values:
            return None
        index = max(0, min(len(values) - 1, int((percentage / 100.0) * len(values) + 0.999999) - 1))
        return round(values[index], 2)

    total_completion_tokens = sum(completion_counts)
    total_prompt_tokens = sum(prompt_counts)
    total_wall_ms = round(total_wall_seconds * 1000.0, 2)
    return {
        "schema_version": 1,
        "backend": "ollama",
        "endpoint": OLLAMA_ENDPOINT,
        "model": model,
        "configuration": {
            "prompt_characters": len(prompt),
            "num_predict": num_predict,
            "temperature": temperature,
            "warmup_runs": warmup_runs,
            "iterations": iterations,
            "concurrency": concurrency,
            "measured_requests": measured_requests,
        },
        "measurements": {
            "wall_time_ms": total_wall_ms,
            "requests_per_second": round(measured_requests / total_wall_seconds, 3),
            "completion_tokens_known": len(completion_counts),
            "prompt_tokens_known": len(prompt_counts),
            "total_prompt_tokens": total_prompt_tokens if prompt_counts else None,
            "total_completion_tokens": total_completion_tokens if completion_counts else None,
            "completion_tokens_per_second": round(total_completion_tokens / total_wall_seconds, 3) if completion_counts else None,
            "latency_ms": {
                "min": round(latencies[0], 2) if latencies else None,
                "median": round(statistics.median(latencies), 2) if latencies else None,
                "p95": percentile(latencies, 95),
                "max": round(latencies[-1], 2) if latencies else None,
            },
            "requests": samples,
        },
        "privacy": {
            "prompt_returned": False,
            "responses_returned": False,
            "results_persisted": False,
        },
        "scope_note": "These are measured timings from this run against the local loopback Ollama service. They are not hardware-independent scores. Compare runs only with the same model, prompt, token limit, runtime configuration, and concurrency; no model quality metric is computed.",
    }
