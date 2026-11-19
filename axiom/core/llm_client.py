"""Unified local/cloud LLM client used by AXIOM's API, CLI, and Control Center.

Secrets are request-scoped only: this module never writes provider credentials or
conversation content to the workspace, logs, traces, or a persistent settings store.
"""
from __future__ import annotations

import http.client
import ipaddress
import json
import math
import re
import socket
import ssl
from datetime import UTC, datetime
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from axiom.core.engine import AxiomEngineError, discover_local_engine_assets, generate_local_stream
from axiom.core.inference import InferenceRuntimeError, run_local_chat, runtime_status as ollama_runtime_status
from axiom.core.model_studio import ModelStudioError, validate_remote_base_url

_MAX_KEY_CHARS = 512
_MAX_MESSAGES = 64
_MAX_MESSAGE_CHARS = 12_000
_MAX_TOTAL_CHARS = 24_000
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_OUTPUT_TOKENS = 2_048
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")

# Cloud endpoints are fixed per provider. Only the explicitly labelled
# OpenAI-compatible provider accepts a user-supplied endpoint.
_PROVIDER_CONFIG: dict[str, dict[str, Any]] = {
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "kind": "openai"},
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "kind": "openai"},
    "moonshot": {"label": "Moonshot AI (Kimi)", "base_url": "https://api.moonshot.ai/v1", "kind": "openai"},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "kind": "openai"},
    "anthropic": {"label": "Anthropic", "base_url": "https://api.anthropic.com/v1", "kind": "anthropic"},
    "gemini": {"label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta", "kind": "gemini"},
    "openai-compatible": {"label": "Custom OpenAI-compatible", "base_url": None, "kind": "openai"},
}
_LOCAL_PROVIDER = "axiom-local"
_OLLAMA_PROVIDER = "ollama-local"


class LLMClientError(RuntimeError):
    """A provider request or local generation failed without exposing credentials."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, new_url: str) -> None:
        return None


def provider_catalog() -> dict[str, Any]:
    """Describe supported clients without requiring credentials or network access."""
    providers = [{
        "id": _LOCAL_PROVIDER,
        "label": "AXIOM Native Engine",
        "kind": "local",
        "requires_api_key": False,
        "supports_model_listing": True,
        "privacy": "local-only",
        "note": "Runs supported local Llama, Mistral, Qwen2, and Qwen3 safetensors checkpoints.",
    }]
    providers.append({
        "id": _OLLAMA_PROVIDER,
        "label": "Ollama (local)",
        "kind": "local",
        "requires_api_key": False,
        "supports_model_listing": True,
        "privacy": "local-only",
        "note": "Uses the fixed 127.0.0.1:11434 Ollama API; supports locally installed Ollama formats such as GGUF.",
    })
    for provider_id, config in _PROVIDER_CONFIG.items():
        providers.append({
            "id": provider_id,
            "label": config["label"],
            "kind": "cloud",
            "requires_api_key": True,
            "supports_model_listing": True,
            "privacy": "remote-consent-required",
            "note": "Credentials are used for this request only and are not saved by AXIOM.",
        })
    return {"schema_version": 1, "providers": providers, "credentials_persisted": False}


def _validate_key(api_key: str | None) -> str:
    value = api_key.strip() if isinstance(api_key, str) else ""
    if not value or len(value) > _MAX_KEY_CHARS or any(ord(char) < 32 for char in value):
        raise ValueError("Enter a single-line provider API key no longer than 512 characters.")
    return value


def _native_models() -> list[dict[str, Any]]:
    try:
        report = discover_local_engine_assets()
    except Exception:
        raise LLMClientError("AXIOM could not inspect the trusted local model inventory.") from None
    models = report.get("models", []) if isinstance(report, dict) else []
    compatible = [
        item for item in models
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("path"), str)
        and item.get("model_type") in {"llama", "mistral", "qwen2", "qwen3"}
    ]
    # Model aliases are public CLI/API identifiers. Never silently choose one
    # checkpoint when several trusted directories share the same alias.
    name_counts: dict[str, int] = {}
    for item in compatible:
        name_counts[item["name"]] = name_counts.get(item["name"], 0) + 1
    return [item for item in compatible if name_counts[item["name"]] == 1]


def _base_url(provider: str, base_url: str | None) -> str:
    config = _PROVIDER_CONFIG.get(provider)
    if config is None:
        raise ValueError(f"Unsupported LLM provider: {provider!r}.")
    if provider == "openai-compatible":
        try:
            return validate_remote_base_url(base_url or "")
        except ModelStudioError as exc:
            raise LLMClientError(str(exc)) from None
        except ValueError:
            raise
    if base_url and base_url.strip():
        raise ValueError("A custom base URL is accepted only for the OpenAI-compatible provider.")
    return str(config["base_url"])


def _resolve_public_addresses(base_url: str) -> tuple[str, ...]:
    """Resolve a custom HTTPS endpoint once, reject any non-public result, and pin the connection to these IPs."""
    parsed = urlsplit(base_url)
    host = parsed.hostname
    try:
        port = parsed.port or 443
        if not host:
            raise ValueError("hostname missing")
        results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, ValueError, IndexError):
        raise LLMClientError("The custom HTTPS provider hostname could not be resolved safely.") from None
    addresses: list[str] = []
    for result in results:
        try:
            address = ipaddress.ip_address(result[4][0].split("%", 1)[0])
        except (ValueError, IndexError, TypeError):
            raise LLMClientError("The custom HTTPS provider returned an invalid network address.") from None
        if not address.is_global:
            raise LLMClientError("The custom HTTPS provider resolved to a non-public address; private, loopback, link-local, and reserved targets are blocked.")
        normalized = address.compressed
        if normalized not in addresses:
            addresses.append(normalized)
    if not addresses:
        raise LLMClientError("The custom HTTPS provider hostname has no public addresses.")
    return tuple(addresses)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection with DNS resolution pinned to prevalidated public addresses."""

    def __init__(self, host: str, *, addresses: tuple[str, ...], **kwargs: Any) -> None:
        self._pinned_addresses = addresses
        super().__init__(host, **kwargs)

    def connect(self) -> None:
        # Proxies are deliberately disabled by the only caller; do not add tunnel
        # support that could change the destination after the address check.
        if self._tunnel_host:
            raise OSError("HTTP tunnels are not supported for custom provider requests.")
        failures: list[OSError] = []
        for address in self._pinned_addresses:
            sock = None
            try:
                sock = socket.create_connection((address, self.port), self.timeout, self.source_address)
                # Validate the certificate against the original DNS hostname, not its pinned IP.
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                return
            except OSError as exc:
                if sock is not None:
                    sock.close()
                failures.append(exc)
        if failures:
            raise failures[-1]
        raise OSError("No validated HTTPS destination address was available.")


def _pinned_public_request(
    base_url: str,
    endpoint: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, bytes]:
    """Send an HTTP request to a validated public IP while retaining TLS hostname verification."""
    parsed = urlsplit(base_url)
    host = parsed.hostname
    if not host:
        raise LLMClientError("The custom HTTPS provider hostname is missing.")
    try:
        port = parsed.port or 443
    except ValueError:
        raise LLMClientError("The custom HTTPS provider port is invalid.") from None
    addresses = _resolve_public_addresses(base_url)
    path = parsed.path.rstrip("/") + endpoint
    connection = _PinnedHTTPSConnection(
        host,
        port=port,
        timeout=timeout,
        context=ssl.create_default_context(),
        addresses=addresses,
    )
    try:
        connection.request("GET" if body is None else "POST", path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read(_MAX_RESPONSE_BYTES + 1)
    except (OSError, TimeoutError, http.client.HTTPException):
        raise LLMClientError("Could not connect safely to the configured public HTTPS model endpoint.") from None
    finally:
        connection.close()


def _provider_request(
    provider: str,
    api_key: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    key = _validate_key(api_key)
    base = _base_url(provider, base_url)
    allowed = {"/models", "/chat/completions", "/messages", "/models?limit=100", "/models?pageSize=100"}
    if endpoint not in allowed and not re.fullmatch(r"/models/[A-Za-z0-9._-]{1,128}:generateContent", endpoint):
        raise ValueError("Unsupported provider operation.")
    if provider == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    elif provider == "gemini":
        headers = {"x-goog-api-key": key}
    else:
        headers = {"Authorization": f"Bearer {key}"}
    headers.update({"Accept": "application/json", "User-Agent": "AXIOM-Unified-LLM/1"})
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if body is not None:
        headers["Content-Type"] = "application/json"
    timeout = 30 if body is None else 180
    if provider == "openai-compatible":
        status, raw = _pinned_public_request(base, endpoint, body, headers, timeout)
        if status in {301, 302, 303, 307, 308}:
            raise LLMClientError("The online provider redirected the request. Redirects are blocked to protect the API key; enter its final HTTPS /v1 endpoint.")
        if status in {401, 403}:
            raise LLMClientError(f"The provider rejected the API key or denied access (HTTP {status}).")
        if not 200 <= status < 300:
            raise LLMClientError(f"Online provider request failed (HTTP {status}). Check the endpoint, model access, and provider status.")
    else:
        req = Request(base + endpoint, data=body, method="GET" if body is None else "POST", headers=headers)
        opener = build_opener(ProxyHandler({}), _RejectRedirects())
        try:
            with opener.open(req, timeout=timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                raise LLMClientError("The provider redirected the request. Redirects are blocked to protect credentials; use its final HTTPS endpoint.") from None
            if exc.code in {401, 403}:
                raise LLMClientError(f"The provider rejected the API key or denied access (HTTP {exc.code}).") from None
            raise LLMClientError(f"The provider request failed (HTTP {exc.code}); check the model ID, access, and provider status.") from None
        except (URLError, TimeoutError, OSError):
            raise LLMClientError("Could not connect to the configured HTTPS provider.") from None
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise LLMClientError("The provider response exceeded AXIOM's 2 MB response limit.")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LLMClientError("The provider returned invalid JSON.") from None
    if not isinstance(data, dict):
        raise LLMClientError("The provider returned an unexpected response shape.")
    return data


def _model_ids(provider: str, data: dict[str, Any]) -> list[str]:
    raw = data.get("models") if provider == "gemini" else data.get("data", data.get("models", []))
    models: list[str] = []
    if isinstance(raw, list):
        for item in raw[:200]:
            if not isinstance(item, (dict, str)):
                continue
            if isinstance(item, dict):
                name = item.get("name") if provider == "gemini" else item.get("id", item.get("name"))
                if provider == "gemini":
                    methods = item.get("supportedGenerationMethods")
                    if isinstance(methods, list) and "generateContent" not in methods:
                        continue
            else:
                name = item
            if not isinstance(name, str):
                continue
            if provider == "gemini" and name.startswith("models/"):
                name = name[len("models/"):]
            if _MODEL_ID.fullmatch(name):
                models.append(name)
    # Remove duplicate IDs while preserving provider ordering.
    return list(dict.fromkeys(models))[:100]


def list_models(provider: str, api_key: str | None = None, base_url: str | None = None) -> dict[str, Any]:
    """List local trusted models or models offered by a selected cloud provider."""
    if provider == _LOCAL_PROVIDER:
        models = [
            {"id": item["name"], "model_type": item.get("model_type"), "architecture": item.get("architecture")}
            for item in _native_models()
        ]
        return {
            "provider": provider,
            "models": [item["id"] for item in models],
            "model_details": models,
            "credential_storage": "not_applicable",
            "execution": "local",
        }
    if provider == _OLLAMA_PROVIDER:
        if api_key or base_url:
            raise ValueError("The local Ollama provider does not accept API keys or remote base URLs.")
        status = ollama_runtime_status()
        models = status.get("models", []) if isinstance(status, dict) else []
        safe_models = [name for name in models if isinstance(name, str) and name]
        return {
            "provider": provider,
            "models": safe_models,
            "count": len(safe_models),
            "credential_storage": "not_applicable",
            "execution": "local",
            "runtime_status": status.get("status", "offline"),
            "note": status.get("error") or (
                "Ollama is online but no models are installed yet; pull a model locally first."
                if not safe_models else
                "Models are discovered from the fixed local Ollama loopback API."
            ),
        }
    if provider not in _PROVIDER_CONFIG:
        raise ValueError(f"Unsupported LLM provider: {provider!r}.")
    key = _validate_key(api_key)
    if provider == "anthropic":
        data = _provider_request(provider, key, "/models?limit=100", base_url=base_url)
    elif provider == "gemini":
        data = _provider_request(provider, key, "/models?pageSize=100", base_url=base_url)
    else:
        data = _provider_request(provider, key, "/models", base_url=base_url)
    models = _model_ids(provider, data)
    return {
        "provider": provider,
        "models": models,
        "count": len(models),
        "credential_storage": "not_saved",
        "execution": "remote",
        "note": "The API key was used for this request only and was not saved by AXIOM.",
    }


def _validate_messages(messages: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not 1 <= len(messages) <= _MAX_MESSAGES:
        raise ValueError(f"Provide between 1 and {_MAX_MESSAGES} chat messages.")
    normalized: list[dict[str, str]] = []
    total = 0
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Each message must be an object with role and content.")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError("Messages must use system, user, or assistant roles and non-empty text.")
        if len(content) > _MAX_MESSAGE_CHARS:
            raise ValueError("Each message must be 12,000 characters or fewer.")
        total += len(content)
        normalized.append({"role": str(role), "content": content})
    if total > _MAX_TOTAL_CHARS:
        raise ValueError(f"Combined chat content must not exceed {_MAX_TOTAL_CHARS} characters.")
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError("At least one user message is required.")
    return normalized


def _validate_generation(model: str, max_tokens: int, temperature: float) -> tuple[str, int, float]:
    value = model.strip() if isinstance(model, str) else ""
    if not _MODEL_ID.fullmatch(value):
        raise ValueError("Enter a valid model ID returned by the selected provider.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= _MAX_OUTPUT_TOKENS:
        raise ValueError(f"Maximum output tokens must be between 1 and {_MAX_OUTPUT_TOKENS}.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    return value, max_tokens, float(temperature)


def _validate_sampling_options(
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    seed: int | None,
    stop_sequences: list[str] | tuple[str, ...],
) -> tuple[float, int, float, float, int | None, tuple[str, ...]]:
    """Validate shared controls for local inference engines."""
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
    for value in stop_sequences:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Stop sequences must be non-empty text.")
        if len(value) > 256:
            raise ValueError("Each stop sequence is limited to 256 characters.")
        if value not in stops:
            stops.append(value)
    return float(top_p), top_k, float(min_p), float(repetition_penalty), seed, tuple(stops)


def _extract_openai(data: dict[str, Any]) -> tuple[str, int | None, int | None]:
    choices = data.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    if not isinstance(content, str):
        raise LLMClientError("The provider response did not contain text content.")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return content[:100_000], _int_or_none(usage.get("prompt_tokens")), _int_or_none(usage.get("completion_tokens"))


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _extract_anthropic(data: dict[str, Any]) -> tuple[str, int | None, int | None]:
    parts = data.get("content")
    content = "".join(part.get("text", "") for part in parts if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)) if isinstance(parts, list) else ""
    if not content:
        raise LLMClientError("The provider response did not contain text content.")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return content[:100_000], _int_or_none(usage.get("input_tokens")), _int_or_none(usage.get("output_tokens"))


def _extract_gemini(data: dict[str, Any]) -> tuple[str, int | None, int | None]:
    candidates = data.get("candidates")
    first = candidates[0] if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) else {}
    content_data = first.get("content") if isinstance(first, dict) else {}
    parts = content_data.get("parts") if isinstance(content_data, dict) else []
    content = "".join(part.get("text", "") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)) if isinstance(parts, list) else ""
    if not content:
        raise LLMClientError("The provider response did not contain text content.")
    usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
    return content[:100_000], _int_or_none(usage.get("promptTokenCount")), _int_or_none(usage.get("candidatesTokenCount"))


def chat(
    provider: str,
    model: str,
    messages: list[Mapping[str, Any]],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_sequences: list[str] | tuple[str, ...] = (),
    consent: bool = False,
    device: str = "auto",
    memory_mode: str = "auto",
    precision: str = "auto",
) -> dict[str, Any]:
    """Run one local or explicitly consented remote chat completion."""
    normalized = _validate_messages(messages)
    model_id, token_limit, temp = _validate_generation(model, max_tokens, temperature)
    top_p, top_k, min_p, repetition_penalty, seed, stops = _validate_sampling_options(
        top_p, top_k, min_p, repetition_penalty, seed, stop_sequences
    )
    if provider == _LOCAL_PROVIDER:
        if base_url or api_key:
            raise ValueError("The local AXIOM provider does not accept API keys or remote base URLs.")
        choices = {item["name"]: item for item in _native_models()}
        local_model = choices.get(model_id)
        if local_model is None:
            raise ValueError("That model is not present as a unique compatible checkpoint inside AXIOM's trusted local model roots.")
        if device not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}:
            raise ValueError("Choose auto, cpu, cuda, rocm, mps, or xpu for the local execution device.")
        if memory_mode not in {"auto", "full", "layer_streaming", "disk_streaming"}:
            raise ValueError("Choose auto, full, layer_streaming, or disk_streaming for local model memory mode.")
        if precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("Choose auto, fp32, fp16, or bf16 for local compute precision.")
        last_user = next(message["content"] for message in reversed(normalized) if message["role"] == "user")
        final: dict[str, Any] | None = None
        try:
            for event in generate_local_stream(
                local_model["path"],
                last_user,
                max_new_tokens=token_limit,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_sequences=stops,
                chat_messages=normalized,
                include_token_events=False,
                device_preference=device,
                memory_mode=memory_mode,
                precision=precision,
            ):
                if event.get("type") == "complete" and isinstance(event.get("result"), dict):
                    final = event["result"]
        except (AxiomEngineError, ValueError) as exc:
            raise LLMClientError(str(exc)) from None
        if final is None:
            raise LLMClientError("AXIOM Native Engine did not return a completion result.")
        return {
            "provider": _LOCAL_PROVIDER,
            "model": model_id,
            "response": str(final.get("response", ""))[:100_000],
            "prompt_tokens": _int_or_none(final.get("prompt_tokens")),
            "completion_tokens": _int_or_none(final.get("completion_tokens")),
            "credential_storage": "not_applicable",
            "execution": "local",
            "device": final.get("device"),
            "precision": final.get("precision"),
            "precision_selection": final.get("precision_selection"),
            "context_management": final.get("context_management"),
            "generated_at": datetime.now(UTC).isoformat(),
        }
    if provider == _OLLAMA_PROVIDER:
        if api_key or base_url:
            raise ValueError("The local Ollama provider does not accept API keys or remote base URLs.")
        try:
            final = run_local_chat(
                model_id,
                normalized,
                num_predict=token_limit,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_sequences=stops,
            )
        except (InferenceRuntimeError, ValueError) as exc:
            raise LLMClientError(str(exc)) from None
        return {
            "provider": _OLLAMA_PROVIDER,
            "model": model_id,
            "response": str(final.get("response", ""))[:100_000],
            "prompt_tokens": _int_or_none(final.get("prompt_eval_count")),
            "completion_tokens": _int_or_none(final.get("eval_count")),
            "credential_storage": "not_applicable",
            "execution": "local",
            "device": "ollama",
            "generated_at": final.get("generated_at") or datetime.now(UTC).isoformat(),
        }
    if provider not in _PROVIDER_CONFIG:
        raise ValueError(f"Unsupported LLM provider: {provider!r}.")
    if top_p != 0.9 or top_k != 0 or min_p != 0.0 or repetition_penalty != 1.0 or seed is not None or stops:
        raise ValueError("Advanced sampling controls currently support the local AXIOM Native Engine and Ollama providers only; no settings were sent to the cloud provider.")
    if not consent:
        raise ValueError("Explicitly confirm that the selected cloud provider will receive your conversation before sending it.")
    key = _validate_key(api_key)
    kind = _PROVIDER_CONFIG[provider]["kind"]
    if kind == "openai":
        data = _provider_request(provider, key, "/chat/completions", {
            "model": model_id,
            "messages": normalized,
            "max_tokens": token_limit,
            "temperature": temp,
            "stream": False,
        }, base_url=base_url)
        content, prompt_tokens, completion_tokens = _extract_openai(data)
    elif kind == "anthropic":
        system_text = "\n\n".join(message["content"] for message in normalized if message["role"] == "system")
        cloud_messages = [{"role": message["role"], "content": message["content"]} for message in normalized if message["role"] != "system"]
        body: dict[str, Any] = {"model": model_id, "max_tokens": token_limit, "temperature": temp, "messages": cloud_messages}
        if system_text:
            body["system"] = system_text
        data = _provider_request(provider, key, "/messages", body, base_url=base_url)
        content, prompt_tokens, completion_tokens = _extract_anthropic(data)
    else:
        system_text = "\n\n".join(message["content"] for message in normalized if message["role"] == "system")
        cloud_messages = [
            {"role": "model" if message["role"] == "assistant" else "user", "parts": [{"text": message["content"]}]}
            for message in normalized if message["role"] != "system"
        ]
        body = {"contents": cloud_messages, "generationConfig": {"maxOutputTokens": token_limit, "temperature": temp}}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        endpoint = "/models/" + quote(model_id, safe="._-") + ":generateContent"
        data = _provider_request(provider, key, endpoint, body, base_url=base_url)
        content, prompt_tokens, completion_tokens = _extract_gemini(data)
    return {
        "provider": provider,
        "model": model_id,
        "response": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "credential_storage": "not_saved",
        "execution": "remote",
        "generated_at": datetime.now(UTC).isoformat(),
    }
