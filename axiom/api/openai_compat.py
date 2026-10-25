"""Token-protected OpenAI-compatible chat for AXIOM Native Engine and loopback Ollama.

The bridge serves trusted native checkpoints and already-installed Ollama models only. It
never forwards the request to an external inference provider or exposes workspace endpoints.
"""
from __future__ import annotations

import asyncio
import json
import hmac
import os
import time
import uuid
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, AsyncIterator, Iterator, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.responses import JSONResponse, StreamingResponse

from axiom.core.engine import AxiomEngineError, discover_local_engine_assets, generate_local_batch, generate_local_stream
from axiom.core.inference import InferenceRuntimeError, run_local_chat, runtime_status as ollama_runtime_status

router = APIRouter()
_STREAM_END = object()
_STREAM_QUEUE_SIZE = 2


def _authorization_error(request: Request) -> JSONResponse | None:
    """Enforce an optional bearer token for clients exposed beyond loopback."""
    expected = os.environ.get("AXIOM_OPENAI_API_TOKEN", "").strip()
    if not expected:
        return None
    authorization = request.headers.get("Authorization", "")
    scheme, separator, supplied = authorization.partition(" ")
    matched = bool(separator and scheme.lower() == "bearer" and hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")))
    if matched:
        return None
    return JSONResponse(
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
        content={"error": {"message": "A valid AXIOM bearer token is required.", "type": "authentication_error", "param": None, "code": "invalid_api_key"}},
    )


def _openai_error(message: str, status_code: int, *, code: str | None = None,
                  error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": code}},
    )


def _native_models() -> list[dict[str, Any]]:
    report = discover_local_engine_assets()
    models = report.get("models", []) if isinstance(report, dict) else []
    return [
        item for item in models
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("path"), str)
        and item.get("model_type") in {"llama", "mistral", "qwen2", "qwen3"}
    ]


def _resolve_model(model_id: str) -> dict[str, Any] | None:
    matches = [item for item in _native_models() if item.get("name") == model_id]
    return matches[0] if len(matches) == 1 else None


def _ollama_model_names() -> list[str]:
    """Return installed Ollama model identifiers from the fixed loopback-only inventory."""
    try:
        status = ollama_runtime_status()
    except Exception:
        return []
    names = status.get("models", []) if isinstance(status, dict) and status.get("status") == "online" else []
    return [
        name for name in names[:100]
        if isinstance(name, str) and name and len(name) <= 120 and not any(ord(char) < 32 for char in name)
    ]


def _resolve_ollama_model(model_id: str) -> str | None:
    """Resolve the explicit ollama/ namespace without colliding with native model IDs."""
    if not model_id.startswith("ollama/"):
        return None
    candidate = model_id[len("ollama/"):]
    return candidate if candidate and candidate in _ollama_model_names() else ""


def _ollama_completion_payload(request: "OpenAIChatCompletionRequest", model_name: str) -> dict[str, Any]:
    """Execute one bounded local Ollama chat and normalize its result to OpenAI's schema."""
    if request.adapter_path:
        raise ValueError("LoRA adapter paths apply only to AXIOM-native checkpoints, not Ollama models.")
    result = run_local_chat(
        model_name,
        [message.model_dump() for message in request.messages],
        num_predict=request.max_completion_tokens or request.max_tokens or 256,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        seed=request.seed,
        stop_sequences=_stop_sequences(request),
    )
    prompt_tokens = result.get("prompt_eval_count")
    completion_tokens = result.get("eval_count")
    prompt_tokens = prompt_tokens if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) else 0
    completion_tokens = completion_tokens if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) else 0
    finish_reason = "length" if result.get("done_reason") in {"length", "max_tokens"} else "stop"
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": str(result.get("response", ""))[:100_000]},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": "ollama-local",
    }


def _run_ollama_nonstream(request: "OpenAIChatCompletionRequest", model_name: str) -> Any:
    try:
        return _ollama_completion_payload(request, model_name)
    except ValueError as exc:
        return _openai_error(str(exc), 422, code="invalid_request")
    except InferenceRuntimeError as exc:
        return _openai_error(str(exc), 503, code="ollama_local_error", error_type="server_error")


def _ollama_streaming_response(
    request: "OpenAIChatCompletionRequest",
    model_name: str,
    http_request: Request,
) -> StreamingResponse:
    """SSE-compatible response for Ollama; output arrives in one final content delta."""
    stream_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    async def events() -> AsyncIterator[str]:
        yield "data: " + json.dumps({
            "id": stream_id, "object": "chat.completion.chunk", "created": created, "model": request.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }, ensure_ascii=False) + "\n\n"
        if await http_request.is_disconnected():
            return
        result = await asyncio.to_thread(_run_ollama_nonstream, request, model_name)
        if isinstance(result, JSONResponse):
            try:
                payload = json.loads(result.body.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                payload = {"error": {"message": "Local Ollama generation failed.", "type": "server_error", "code": "ollama_local_error"}}
            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        choice = result["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if content:
            yield "data: " + json.dumps({
                "id": stream_id, "object": "chat.completion.chunk", "created": created, "model": request.model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }, ensure_ascii=False) + "\n\n"
        yield "data: " + json.dumps({
            "id": stream_id, "object": "chat.completion.chunk", "created": created, "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
        }, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chat message content must not be blank.")
        return value


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=256)
    messages: list[OpenAIChatMessage] = Field(min_length=1, max_length=128)
    max_tokens: int | None = Field(default=256, ge=1, le=2048)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100000)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
    seed: int | None = Field(default=None, ge=0, le=9223372036854775807)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=8)
    user: str | None = Field(default=None, max_length=256)
    adapter_path: str | None = Field(default=None, max_length=4096)
    device: Literal["auto", "cpu", "cuda", "rocm", "mps", "xpu"] = "auto"
    memory_mode: Literal["auto", "full", "layer_streaming", "disk_streaming"] = "auto"
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"

    @model_validator(mode="after")
    def validate_messages_and_stop(self) -> "OpenAIChatCompletionRequest":
        combined = sum(len(message.content) for message in self.messages)
        if combined > 12000:
            raise ValueError("Combined chat message content must not exceed 12000 characters.")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("At least one user message is required.")
        if isinstance(self.stop, list):
            if len(self.stop) > 16 or any(not item or len(item) > 256 for item in self.stop):
                raise ValueError("stop must contain at most 16 non-empty strings of 256 characters or fewer.")
        return self


def _stop_sequences(request: OpenAIChatCompletionRequest) -> list[str]:
    if request.stop is None:
        return []
    return [request.stop] if isinstance(request.stop, str) else request.stop


def _generation_args(request: OpenAIChatCompletionRequest, model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_path": model["path"],
        "prompt": next(message.content for message in reversed(request.messages) if message.role == "user"),
        "adapter_path": request.adapter_path,
        "max_new_tokens": request.max_completion_tokens or request.max_tokens or 256,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "repetition_penalty": request.repetition_penalty,
        "seed": request.seed,
        "stop_sequences": _stop_sequences(request),
        "device_preference": request.device,
        "memory_mode": request.memory_mode,
        "precision": request.precision,
        "chat_messages": [message.model_dump() for message in request.messages],
    }


def _finish_reason(stop_reason: str) -> str:
    if stop_reason == "max_new_tokens":
        return "length"
    return "stop"


def _produce_stream(iterator: Iterator[dict[str, Any]], output: Queue, cancelled: Event) -> None:
    try:
        for item in iterator:
            while not cancelled.is_set():
                try:
                    output.put(item, timeout=0.1)
                    break
                except Full:
                    continue
            else:
                break
    except (ValueError, AxiomEngineError) as exc:
        while not cancelled.is_set():
            try:
                output.put({"type": "error", "message": str(exc)}, timeout=0.1)
                break
            except Full:
                continue
    except Exception:
        while not cancelled.is_set():
            try:
                output.put({"type": "error", "message": "AXIOM Native Engine failed during chat completion."}, timeout=0.1)
                break
            except Full:
                continue
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except (RuntimeError, ValueError):
                pass
        while not cancelled.is_set():
            try:
                output.put(_STREAM_END, timeout=0.1)
                break
            except Full:
                continue


@router.get("/v1/models")
def openai_list_models(request: Request) -> Any:
    """List native models as OpenAI-compatible model objects."""
    auth_error = _authorization_error(request)
    if auth_error is not None:
        return auth_error
    created = int(time.time())
    models = _native_models()
    data = [
        {"id": model["name"], "object": "model", "created": created, "owned_by": "axiom"}
        for model in models
    ]
    data.extend(
        {"id": "ollama/" + model_name, "object": "model", "created": created, "owned_by": "ollama"}
        for model_name in _ollama_model_names()
    )
    return {"object": "list", "data": data}



def _run_batched_nonstream(request: OpenAIChatCompletionRequest, model: dict[str, Any]) -> Any:
    """Return independent OpenAI choices through one AXIOM native microbatch."""
    try:
        last_user = next(message.content for message in reversed(request.messages) if message.role == "user")
        batch = generate_local_batch(
            model["path"],
            [last_user] * request.n,
            adapter_path=request.adapter_path,
            max_new_tokens=request.max_completion_tokens or request.max_tokens or 256,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stop_sequences=_stop_sequences(request),
            device_preference=request.device,
            memory_mode=request.memory_mode,
            precision=request.precision,
            chat_messages=[message.model_dump() for message in request.messages],
        )
    except ValueError as exc:
        return _openai_error(str(exc), 422, code="invalid_request")
    except AxiomEngineError as exc:
        return _openai_error(str(exc), 503, code="native_engine_error", error_type="server_error")
    results = batch.get("results", [])
    if not isinstance(results, list) or len(results) != request.n or any(not isinstance(row, dict) for row in results):
        return _openai_error("AXIOM Native Engine did not return every requested choice.", 503, code="native_engine_error", error_type="server_error")
    choices = [
        {
            "index": index,
            "message": {"role": "assistant", "content": str(row.get("response", ""))},
            "finish_reason": _finish_reason(str(row.get("stop_reason", "stop"))),
        }
        for index, row in enumerate(results)
    ]
    usage_prompt = int(results[0].get("prompt_tokens", 0))
    usage_completion = sum(
        int(row.get("completion_tokens", 0))
        for row in results
        if isinstance(row.get("completion_tokens", 0), int) and not isinstance(row.get("completion_tokens", 0), bool)
    )
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": usage_prompt,
            "completion_tokens": usage_completion,
            "total_tokens": usage_prompt + usage_completion,
        },
        "system_fingerprint": "axiom-native",
    }


def _run_nonstream(request: OpenAIChatCompletionRequest, model: dict[str, Any]) -> Any:
    try:
        final: dict[str, Any] | None = None
        for event in generate_local_stream(**_generation_args(request, model), include_token_events=False):
            if event.get("type") == "complete":
                final = event.get("result")
        if not isinstance(final, dict):
            return _openai_error("AXIOM Native Engine did not return a completion result.", 503, code="native_engine_error", error_type="server_error")
    except ValueError as exc:
        return _openai_error(str(exc), 422, code="invalid_request")
    except AxiomEngineError as exc:
        return _openai_error(str(exc), 503, code="native_engine_error", error_type="server_error")
    created = int(time.time())
    prompt_tokens = int(final.get("prompt_tokens", 0))
    completion_tokens = int(final.get("completion_tokens", 0))
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final.get("response", "")},
            "finish_reason": _finish_reason(str(final.get("stop_reason", "stop"))),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": "axiom-native",
    }


def _streaming_response(
    request: OpenAIChatCompletionRequest,
    model: dict[str, Any],
    http_request: Request,
) -> StreamingResponse:
    args = _generation_args(request, model)
    cancelled = Event()
    output: Queue = Queue(maxsize=_STREAM_QUEUE_SIZE)
    iterator = generate_local_stream(**args, include_token_events=True, cancel_event=cancelled)
    worker = Thread(target=_produce_stream, args=(iterator, output, cancelled), name="axiom-openai-chat-stream", daemon=True)
    stream_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    async def events() -> AsyncIterator[str]:
        previous = ""
        completed = False
        yield "data: " + json.dumps({
            "id": stream_id, "object": "chat.completion.chunk", "created": created, "model": request.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }, ensure_ascii=False) + "\n\n"
        worker.start()
        try:
            while True:
                if await http_request.is_disconnected():
                    cancelled.set()
                    return
                try:
                    item = await asyncio.to_thread(output.get, True, 0.1)
                except Empty:
                    continue
                if item is _STREAM_END:
                    break
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "error":
                    yield "data: " + json.dumps({
                        "error": {"message": item.get("message", "Native inference failed."), "type": "server_error", "code": "native_engine_error"}
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                if item.get("type") == "token":
                    cumulative = str(item.get("response", ""))
                    delta = cumulative[len(previous):] if cumulative.startswith(previous) else cumulative
                    previous = cumulative
                    if delta:
                        yield "data: " + json.dumps({
                            "id": stream_id, "object": "chat.completion.chunk", "created": created, "model": request.model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }, ensure_ascii=False) + "\n\n"
                if item.get("type") == "complete":
                    result = item.get("result", {})
                    yield "data: " + json.dumps({
                        "id": stream_id, "object": "chat.completion.chunk", "created": created, "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": _finish_reason(str(result.get("stop_reason", "stop")))}],
                    }, ensure_ascii=False) + "\n\n"
                    completed = True
                    break
            yield "data: [DONE]\n\n"
        finally:
            cancelled.set()
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest, http_request: Request) -> Any:
    """Create OpenAI-compatible native chat completions, either JSON or SSE."""
    auth_error = _authorization_error(http_request)
    if auth_error is not None:
        return auth_error
    if request.model.startswith("ollama/"):
        if request.n > 1:
            return _openai_error("Multiple n choices are supported for AXIOM-native models only; the Ollama compatibility path requires n=1.", 422, code="unsupported_parameter")
        ollama_model = _resolve_ollama_model(request.model)
        if not ollama_model:
            return _openai_error(f"Installed local Ollama model {request.model!r} was not found.", 404, code="model_not_found")
        if request.stream:
            return _ollama_streaming_response(request, ollama_model, http_request)
        return await asyncio.to_thread(_run_ollama_nonstream, request, ollama_model)
    model = _resolve_model(request.model)
    if model is None:
        return _openai_error(f"Native model {request.model!r} was not found or its ID is ambiguous.", 404, code="model_not_found")
    if request.stream and request.n > 1:
        return _openai_error("Streaming n>1 is not supported yet; set stream=false to use native batched choices.", 422, code="unsupported_parameter")
    if request.stream:
        return _streaming_response(request, model, http_request)
    if request.n > 1:
        return await asyncio.to_thread(_run_batched_nonstream, request, model)
    return await asyncio.to_thread(_run_nonstream, request, model)
