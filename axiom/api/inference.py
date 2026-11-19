"""Local inference API. Requests are served only through the loopback Ollama adapter."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from time import perf_counter
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from starlette.responses import StreamingResponse

from axiom.core.inference import (
    InferenceRuntimeError,
    inspect_local_model,
    list_running_local_models,
    remove_local_model,
    unload_local_model,
    run_local_inference,
    stream_local_inference,
    run_local_inference_benchmark,
    runtime_status,
    validate_model_name,
)
from axiom.core.runtime_traces import record_runtime_trace
from axiom.core.ollama_service import (
    OllamaServiceError,
    cancel_model_pull,
    get_model_pull,
    service_status as managed_service_status,
    start_model_pull,
    start_service,
    stop_service,
)

router = APIRouter(prefix="/api/inference", tags=["local-inference"])


class GenerateRequest(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=12_000)
    num_predict: int = Field(default=256, ge=1, le=2_048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0, le=100_000)
    min_p: float | None = Field(default=None, ge=0, le=1)
    repetition_penalty: float | None = Field(default=None, ge=0.1, le=5)
    seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807)
    num_ctx: int | None = Field(default=None, ge=512, le=131_072)
    stop_sequences: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("stop_sequences")
    @classmethod
    def stop_sequences_must_be_bounded(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            if not value.strip() or len(value) > 256:
                raise ValueError("Each stop sequence must contain 1–256 non-whitespace characters.")
            if value not in cleaned:
                cleaned.append(value)
        return cleaned

    @field_validator("model")
    @classmethod
    def model_must_be_a_name_not_a_url(cls, value: str) -> str:
        return validate_model_name(value)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must contain at least one non-whitespace character.")
        return value


def _sampling_kwargs(request: GenerateRequest) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for source, target in (("top_p", "top_p"), ("top_k", "top_k"), ("min_p", "min_p"), ("repetition_penalty", "repetition_penalty"), ("seed", "seed"), ("num_ctx", "num_ctx")):
        value = getattr(request, source)
        if value is not None:
            values[target] = value
    if request.stop_sequences:
        values["stop_sequences"] = request.stop_sequences
    return values


class PullModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=120)

    @field_validator("model")
    @classmethod
    def model_must_be_a_tag_not_a_url(cls, value: str) -> str:
        return validate_model_name(value)


class BenchmarkRequest(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=12_000)
    num_predict: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0, le=2)
    warmup_runs: int = Field(default=1, ge=0, le=2)
    iterations: int = Field(default=3, ge=1, le=10)
    concurrency: int = Field(default=1, ge=1, le=4)

    @field_validator("model")
    @classmethod
    def model_must_be_a_name_not_a_url(cls, value: str) -> str:
        return validate_model_name(value)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must contain at least one non-whitespace character.")
        return value

def _require_local_runtime_management(request: Request) -> None:
    """Keep process and download controls local unless an operator opts in."""
    import os
    from urllib.parse import urlparse

    peer = request.client.host if request.client else ""
    allowed_peer = peer in {"127.0.0.1", "::1", "localhost", "testclient"}
    allowed_remote = os.environ.get("AXIOM_ALLOW_REMOTE_RUNTIME_MANAGEMENT") == "1"
    if not allowed_peer and not allowed_remote:
        raise HTTPException(
            status_code=403,
            detail="Ollama service management is limited to local clients. Set AXIOM_ALLOW_REMOTE_RUNTIME_MANAGEMENT=1 only on a trusted network.",
        )
    origin = request.headers.get("origin")
    if origin:
        hostname = (urlparse(origin).hostname or "").lower()
        if hostname not in {"127.0.0.1", "::1", "localhost", "testserver"} and not allowed_remote:
            raise HTTPException(status_code=403, detail="Ollama service management requires a local browser origin.")


@router.get("/status")
def inference_status() -> dict[str, Any]:
    return managed_service_status()


@router.post("/service/start")
def inference_service_start(request: Request) -> dict[str, Any]:
    _require_local_runtime_management(request)
    try:
        return start_service()
    except OllamaServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/service/stop")
def inference_service_stop(request: Request) -> dict[str, Any]:
    _require_local_runtime_management(request)
    try:
        return stop_service()
    except OllamaServiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.get("/models/running")
def inference_running_models() -> dict[str, Any]:
    try:
        return list_running_local_models()
    except InferenceRuntimeError:
        raise HTTPException(status_code=503, detail="The local Ollama API could not list loaded models.") from None


@router.post("/models/inspect")
def inference_model_inspect(request: Request, payload: PullModelRequest) -> dict[str, Any]:
    _require_local_runtime_management(request)
    try:
        return inspect_local_model(payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except InferenceRuntimeError:
        raise HTTPException(status_code=503, detail="The local Ollama API could not inspect this model.") from None


@router.post("/models/remove")
def inference_model_remove(request: Request, payload: PullModelRequest) -> dict[str, Any]:
    _require_local_runtime_management(request)
    try:
        return remove_local_model(payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except InferenceRuntimeError:
        raise HTTPException(status_code=503, detail="The local Ollama API could not remove this model tag.") from None


@router.post("/models/unload")
def inference_model_unload(request: Request, payload: PullModelRequest) -> dict[str, Any]:
    _require_local_runtime_management(request)
    try:
        return unload_local_model(payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except InferenceRuntimeError:
        raise HTTPException(status_code=503, detail="Ollama could not unload this model from memory.") from None


@router.post("/models/pull")
def inference_model_pull(request: Request, payload: PullModelRequest) -> dict[str, Any]:
    _require_local_runtime_management(request)
    try:
        return start_model_pull(payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except OllamaServiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.get("/models/pull/{job_id}")
def inference_model_pull_status(request: Request, job_id: str) -> dict[str, Any]:
    _require_local_runtime_management(request)
    try:
        return get_model_pull(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from None


@router.post("/models/pull/{job_id}/cancel")
def inference_model_pull_cancel(request: Request, job_id: str) -> dict[str, Any]:
    _require_local_runtime_management(request)
    try:
        return cancel_model_pull(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from None


@router.post("/benchmark")
def inference_benchmark(request: BenchmarkRequest, response: Response) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    response.headers["X-Request-ID"] = request_id
    started_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    if request.iterations * request.concurrency > 10:
        record_runtime_trace(
            request_id=request_id, operation="inference.benchmark", model=request.model,
            status="error", started_at=started_at, duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(request.prompt), requested_output_tokens=request.num_predict,
            temperature=request.temperature, warmup_runs=request.warmup_runs,
            iterations=request.iterations, concurrency=request.concurrency,
            error_code="request_limit_exceeded",
        )
        raise HTTPException(
            status_code=422,
            detail="Benchmark iterations × concurrency may not exceed 10 measured requests.",
            headers={"X-Request-ID": request_id},
        )
    try:
        result = run_local_inference_benchmark(
            request.model,
            request.prompt,
            num_predict=request.num_predict,
            temperature=request.temperature,
            warmup_runs=request.warmup_runs,
            iterations=request.iterations,
            concurrency=request.concurrency,
        )
        measurements = result.get("measurements", {})
        configuration = result.get("configuration", {})
        record_runtime_trace(
            request_id=request_id,
            operation="inference.benchmark",
            model=request.model,
            status="success",
            started_at=started_at,
            duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(request.prompt),
            requested_output_tokens=request.num_predict,
            temperature=request.temperature,
            warmup_runs=request.warmup_runs,
            iterations=request.iterations,
            concurrency=request.concurrency,
            measured_requests=configuration.get("measured_requests"),
            prompt_tokens=None,
            completion_tokens=measurements.get("total_completion_tokens"),
            completion_tokens_known=measurements.get("completion_tokens_known"),
            completion_tokens_per_second=measurements.get("completion_tokens_per_second"),
        )
        return {**result, "request_id": request_id}
    except ValueError as exc:
        record_runtime_trace(
            request_id=request_id, operation="inference.benchmark", model=request.model,
            status="error", started_at=started_at, duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(request.prompt), requested_output_tokens=request.num_predict,
            temperature=request.temperature, warmup_runs=request.warmup_runs,
            iterations=request.iterations, concurrency=request.concurrency,
            error_code="invalid_benchmark_configuration",
        )
        raise HTTPException(status_code=422, detail=str(exc), headers={"X-Request-ID": request_id}) from None
    except InferenceRuntimeError:
        record_runtime_trace(
            request_id=request_id, operation="inference.benchmark", model=request.model,
            status="error", started_at=started_at, duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(request.prompt), requested_output_tokens=request.num_predict,
            temperature=request.temperature, warmup_runs=request.warmup_runs,
            iterations=request.iterations, concurrency=request.concurrency,
            error_code="runtime_unavailable",
        )
        raise HTTPException(status_code=503, detail="Local inference benchmark failed because the runtime did not complete the request.", headers={"X-Request-ID": request_id}) from None




_STREAM_QUEUE_SIZE = 32
_STREAM_END = object()


def _enqueue_stream_event(output: Queue, item: Any, cancel_event: Event) -> bool:
    while not cancel_event.is_set():
        try:
            output.put(item, timeout=0.1)
            return True
        except Full:
            continue
    return False


def _safe_stream_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if "not found" in message.lower() and "model" in message.lower():
        return "Ollama could not find that model. Pull it locally first."
    if "prompt" in message.lower() or "token" in message.lower() or "temperature" in message.lower():
        return message[:240]
    return "Local Ollama generation failed. Check the local service and model, then retry."


def _produce_ollama_stream(
    payload: GenerateRequest,
    request_id: str,
    started_at: str,
    started: float,
    output: Queue,
    cancel_event: Event,
    upstream_state: dict[str, Any],
    upstream_lock: Lock,
) -> None:
    trace_recorded = False
    completed = False

    def register_upstream(response: Any) -> None:
        with upstream_lock:
            upstream_state["response"] = response
            should_close = cancel_event.is_set()
        if should_close:
            try:
                response.close()
            except OSError:
                pass

    iterator = stream_local_inference(
        payload.model,
        payload.prompt,
        num_predict=payload.num_predict,
        temperature=payload.temperature,
        **_sampling_kwargs(payload),
        cancel_event=cancel_event,
        on_response=register_upstream,
    )
    try:
        for item in iterator:
            if cancel_event.is_set():
                break
            if item.get("type") == "complete":
                result = item.get("result", {})
                measurements = result if isinstance(result, dict) else {}
                record_runtime_trace(
                    request_id=request_id,
                    operation="inference.generate",
                    model=payload.model,
                    status="success",
                    started_at=started_at,
                    duration_ms=(perf_counter() - started) * 1000.0,
                    prompt_characters=len(payload.prompt),
                    requested_output_tokens=payload.num_predict,
                    temperature=payload.temperature,
                    prompt_tokens=measurements.get("prompt_eval_count"),
                    completion_tokens=measurements.get("eval_count"),
                )
                trace_recorded = True
                completed = True
                # Tokens were already streamed in prior events; don't duplicate
                # the entire answer in the terminal metadata event.
                public_result = {
                    **measurements,
                    "response": "",
                    "request_id": request_id,
                }
                item = {"type": "complete", "result": public_result}
            if not _enqueue_stream_event(output, item, cancel_event):
                break
    except (ValueError, InferenceRuntimeError) as exc:
        message = _safe_stream_error(exc)
        if not cancel_event.is_set():
            _enqueue_stream_event(output, {"type": "error", "message": message, "request_id": request_id}, cancel_event)
        record_runtime_trace(
            request_id=request_id,
            operation="inference.generate",
            model=payload.model,
            status="error",
            started_at=started_at,
            duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(payload.prompt),
            requested_output_tokens=payload.num_predict,
            temperature=payload.temperature,
            error_code="invalid_generation_request" if isinstance(exc, ValueError) else "runtime_unavailable",
        )
        trace_recorded = True
    except Exception:
        # Avoid forwarding provider error strings or response headers to clients.
        if not cancel_event.is_set():
            _enqueue_stream_event(
                output,
                {"type": "error", "message": "Local Ollama stream failed unexpectedly.", "request_id": request_id},
                cancel_event,
            )
        record_runtime_trace(
            request_id=request_id,
            operation="inference.generate",
            model=payload.model,
            status="error",
            started_at=started_at,
            duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(payload.prompt),
            requested_output_tokens=payload.num_predict,
            temperature=payload.temperature,
            error_code="stream_failed",
        )
        trace_recorded = True
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except (RuntimeError, ValueError):
                pass
        if not trace_recorded:
            record_runtime_trace(
                request_id=request_id,
                operation="inference.generate",
                model=payload.model,
                status="error",
                started_at=started_at,
                duration_ms=(perf_counter() - started) * 1000.0,
                prompt_characters=len(payload.prompt),
                requested_output_tokens=payload.num_predict,
                temperature=payload.temperature,
                error_code="cancelled" if cancel_event.is_set() else "stream_incomplete",
            )
        if not cancel_event.is_set():
            while not cancel_event.is_set():
                try:
                    output.put(_STREAM_END, timeout=0.1)
                    break
                except Full:
                    continue


@router.post("/generate/stream")
async def inference_generate_stream(
    request: GenerateRequest,
    http_request: Request,
) -> StreamingResponse:
    """Stream Ollama token deltas and cooperatively stop on client disconnect."""
    request_id = uuid.uuid4().hex
    started_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    cancel_event = Event()
    output: Queue = Queue(maxsize=_STREAM_QUEUE_SIZE)
    upstream_state: dict[str, Any] = {"response": None}
    upstream_lock = Lock()
    producer = Thread(
        target=_produce_ollama_stream,
        args=(request, request_id, started_at, started, output, cancel_event, upstream_state, upstream_lock),
        name="axiom-ollama-inference-stream",
        daemon=True,
    )

    async def events() -> AsyncIterator[str]:
        async def watch_disconnect() -> None:
            while not cancel_event.is_set():
                if await http_request.is_disconnected():
                    cancel_event.set()
                    with upstream_lock:
                        upstream = upstream_state.get("response")
                    if upstream is not None:
                        try:
                            upstream.close()
                        except OSError:
                            pass
                    return
                await asyncio.sleep(0.1)

        monitor = asyncio.create_task(watch_disconnect())
        producer.start()
        try:
            while not cancel_event.is_set():
                try:
                    item = await asyncio.to_thread(output.get, True, 0.2)
                except Empty:
                    if not producer.is_alive() and output.empty():
                        break
                    continue
                if item is _STREAM_END:
                    break
                if cancel_event.is_set():
                    break
                yield json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        finally:
            cancel_event.set()
            with upstream_lock:
                upstream = upstream_state.get("response")
            if upstream is not None:
                try:
                    upstream.close()
                except OSError:
                    pass
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
            await asyncio.to_thread(producer.join, 2.0)

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


@router.post("/generate")
def inference_generate(request: GenerateRequest, response: Response) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    response.headers["X-Request-ID"] = request_id
    started_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    try:
        result = run_local_inference(
            request.model,
            request.prompt,
            num_predict=request.num_predict,
            temperature=request.temperature,
            **_sampling_kwargs(request),
        )
        record_runtime_trace(
            request_id=request_id, operation="inference.generate", model=request.model,
            status="success", started_at=started_at, duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(request.prompt), requested_output_tokens=request.num_predict,
            temperature=request.temperature, prompt_tokens=result.get("prompt_eval_count"),
            completion_tokens=result.get("eval_count"),
        )
        return {**result, "request_id": request_id}
    except ValueError:
        record_runtime_trace(
            request_id=request_id, operation="inference.generate", model=request.model,
            status="error", started_at=started_at, duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(request.prompt), requested_output_tokens=request.num_predict,
            temperature=request.temperature, error_code="invalid_generation_request",
        )
        raise HTTPException(status_code=422, detail="Local inference request could not be validated.", headers={"X-Request-ID": request_id}) from None
    except InferenceRuntimeError:
        record_runtime_trace(
            request_id=request_id, operation="inference.generate", model=request.model,
            status="error", started_at=started_at, duration_ms=(perf_counter() - started) * 1000.0,
            prompt_characters=len(request.prompt), requested_output_tokens=request.num_predict,
            temperature=request.temperature, error_code="runtime_unavailable",
        )
        raise HTTPException(status_code=503, detail="Local inference runtime did not complete the request.", headers={"X-Request-ID": request_id}) from None
