"""HTTP API for AXIOM's direct local model engine."""
from __future__ import annotations

import asyncio
import json
import logging
from queue import Full, Queue
from threading import Event, Thread
from typing import Any, AsyncIterator, Iterator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.responses import StreamingResponse
from axiom.core import model_studio
from axiom.core.engine_fabric import EngineFabricError, discover_engine_backends, run_engine_backend_benchmark, run_engine_backend_chat, run_engine_fabric_chat

from axiom.core.engine import (
    AxiomEngineError,
    check_local_adapter_compatibility,
    discover_local_engine_assets,
    estimate_native_quantization,
    generate_local,
    generate_local_batch,
    generate_local_stream,
    resolve_local_model_directory,
    runtime_status,
    runtime_telemetry,
    unload_local_model,
)

router = APIRouter(prefix="/api/engine", tags=["axiom-engine"])


def _validated_path(value: Any, label: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Path must be a string.")
    candidate = value.strip()
    if not candidate:
        raise ValueError("Path must not be blank.")
    # The core matches user input against filesystem-discovered directories
    # under explicitly configured trusted roots before returning a usable path.
    return resolve_local_model_directory(candidate, label)


class NativePullRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=192)

    @field_validator("repo_id")
    @classmethod
    def repository_must_be_validated(cls, value: str) -> str:
        return model_studio.validate_repo_id(value)


class GenerateRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=4096)
    adapter_path: str | None = Field(default=None, max_length=4096)
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100000)
    min_p: float = Field(default=0.0, ge=0, le=1)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
    seed: int | None = Field(default=None, ge=0, le=9223372036854775807)
    stop_sequences: list[str] = Field(default_factory=list, max_length=16)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|rocm|mps|xpu)$")
    memory_mode: str = Field(default="auto", pattern="^(auto|full|layer_streaming|disk_streaming)$")
    precision: str = Field(default="auto", pattern="^(auto|fp32|fp16|bf16)$")

    @field_validator("model_path", mode="before")
    @classmethod
    def base_model_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "Base model")

    @field_validator("adapter_path", mode="before")
    @classmethod
    def adapter_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "LoRA adapter")

    @field_validator("stop_sequences")
    @classmethod
    def stop_sequences_must_be_bounded(cls, value: list[str]) -> list[str]:
        unique: list[str] = []
        for sequence in value:
            if not isinstance(sequence, str) or not sequence or not sequence.strip():
                raise ValueError("Stop sequences must be non-empty text strings.")
            if len(sequence) > 256:
                raise ValueError("Each stop sequence is limited to 256 characters.")
            if sequence not in unique:
                unique.append(sequence)
        return unique

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must contain at least one non-whitespace character.")
        return value


class BatchGenerateRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=4096)
    adapter_path: str | None = Field(default=None, max_length=4096)
    prompts: list[str] = Field(min_length=1, max_length=128)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100000)
    min_p: float = Field(default=0.0, ge=0, le=1)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
    seed: int | None = Field(default=None, ge=0, le=9223372036854775807)
    stop_sequences: list[str] = Field(default_factory=list, max_length=16)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|rocm|mps|xpu)$")
    memory_mode: str = Field(default="auto", pattern="^(auto|full|layer_streaming|disk_streaming)$")
    precision: str = Field(default="auto", pattern="^(auto|fp32|fp16|bf16)$")

    @field_validator("model_path", mode="before")
    @classmethod
    def base_model_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "Base model")

    @field_validator("adapter_path", mode="before")
    @classmethod
    def adapter_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "LoRA adapter")

    @field_validator("prompts")
    @classmethod
    def prompts_must_be_nonblank_and_bounded(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Batch generation requires at least one prompt.")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in value):
            raise ValueError("Every batch prompt must contain non-whitespace text.")
        if any(len(prompt) > 12000 for prompt in value):
            raise ValueError("Each batch prompt is limited to 12000 characters.")
        if sum(len(prompt) for prompt in value) > 24000:
            raise ValueError("Combined batch prompts exceed the 24000-character limit.")
        return value

    @field_validator("stop_sequences")
    @classmethod
    def stop_sequences_must_be_bounded(cls, value: list[str]) -> list[str]:
        unique: list[str] = []
        for sequence in value:
            if not isinstance(sequence, str) or not sequence or not sequence.strip():
                raise ValueError("Stop sequences must be non-empty text strings.")
            if len(sequence) > 256:
                raise ValueError("Each stop sequence is limited to 256 characters.")
            if sequence not in unique:
                unique.append(sequence)
        return unique


class QuantizationEstimateRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=4096)
    context_length: int = Field(default=4096, ge=1, le=131072)
    batch_size: int = Field(default=1, ge=1, le=16)
    ram_budget_gib: float | None = Field(default=None, ge=0.1, le=2048)
    vram_budget_gib: float | None = Field(default=None, ge=0.1, le=2048)

    @field_validator("model_path", mode="before")
    @classmethod
    def base_model_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "Base model")

    @model_validator(mode="after")
    def must_set_at_least_one_memory_budget(self) -> "QuantizationEstimateRequest":
        if self.ram_budget_gib is None and self.vram_budget_gib is None:
            raise ValueError("Provide a RAM and/or VRAM budget to compare estimates.")
        return self


class AdapterCompatibilityRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=4096)
    adapter_path: str = Field(min_length=1, max_length=4096)

    @field_validator("model_path", mode="before")
    @classmethod
    def base_model_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "Base model")

    @field_validator("adapter_path", mode="before")
    @classmethod
    def adapter_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "LoRA adapter")


@router.post("/adapters/compatibility")
def engine_adapter_compatibility(request: AdapterCompatibilityRequest) -> dict[str, Any]:
    """Check adapter structure and per-device readiness without loading the full base model."""
    try:
        return check_local_adapter_compatibility(request.model_path, request.adapter_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except AxiomEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/quantization/estimate")
def engine_quantization_estimate(request: QuantizationEstimateRequest) -> dict[str, Any]:
    """Estimate memory for candidate weight formats without loading weights or quantizing."""
    try:
        return estimate_native_quantization(
            request.model_path,
            context_length=request.context_length,
            batch_size=request.batch_size,
            ram_budget_gib=request.ram_budget_gib,
            vram_budget_gib=request.vram_budget_gib,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except AxiomEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("/models")
def engine_models() -> dict[str, Any]:
    """List metadata-only native-compatible models and LoRA adapters under trusted roots."""
    return discover_local_engine_assets()


@router.get("/status")
def engine_status() -> dict[str, Any]:
    return runtime_status()


@router.get("/telemetry")
def engine_telemetry() -> dict[str, Any]:
    """Return local CPU, RAM, device allocator and trusted-model-storage telemetry."""
    return runtime_telemetry()


@router.post("/unload")
def engine_unload() -> dict[str, Any]:
    """Release AXIOM's currently loaded local model and its cache references."""
    return unload_local_model()


@router.post("/pull", status_code=202)
def engine_pull(request: NativePullRequest) -> dict[str, Any]:
    """Start a bounded public-Hub pull for a compatible AXIOM-native checkpoint."""
    try:
        return model_studio.start_job({"operation": "pull_native", "repo_id": request.repo_id})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except model_studio.ModelStudioError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/generate")
def engine_generate(request: GenerateRequest) -> dict[str, Any]:
    try:
        return generate_local(
            request.model_path,
            request.prompt,
            adapter_path=request.adapter_path,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stop_sequences=request.stop_sequences,
            device_preference=request.device,
            memory_mode=request.memory_mode,
            precision=request.precision,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except AxiomEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/generate/batch")
async def engine_generate_batch(
    request: BatchGenerateRequest,
    raw_request: Request,
) -> dict[str, Any]:
    """Run native batch inference with client-disconnect cancellation."""
    cancel_event = Event()
    worker = asyncio.create_task(asyncio.to_thread(
        generate_local_batch,
        request.model_path,
        request.prompts,
        adapter_path=request.adapter_path,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        seed=request.seed,
        stop_sequences=request.stop_sequences,
        device_preference=request.device,
        memory_mode=request.memory_mode,
        precision=request.precision,
        cancel_event=cancel_event,
    ))
    while not worker.done():
        if await raw_request.is_disconnected():
            cancel_event.set()
            # The model loop observes this event at decode boundaries. Do not
            # block the server response on a checkpoint still loading; consume
            # a later exception if the worker finishes after the client leaves.
            def consume_background_result(task: asyncio.Task[Any]) -> None:
                if not task.cancelled():
                    try:
                        task.exception()
                    except Exception:
                        pass
            worker.add_done_callback(consume_background_result)
            raise HTTPException(status_code=499, detail="Batch generation cancelled because the client disconnected.")
        await asyncio.sleep(0.05)

    try:
        return await worker
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except AxiomEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


_STREAM_END = object()
_STREAM_QUEUE_SIZE = 2
_LOG = logging.getLogger(__name__)


def _enqueue_stream_item(output: Queue, item: object, cancel_event: Event) -> bool:
    while not cancel_event.is_set():
        try:
            output.put(item, timeout=0.1)
            return True
        except Full:
            continue
    return False


def _produce_native_stream(
    iterator: Iterator[dict[str, Any]],
    output: Queue,
    cancel_event: Event,
) -> None:
    """Advance a model stream on one dedicated thread and apply backpressure."""
    try:
        for event in iterator:
            if not _enqueue_stream_item(output, event, cancel_event):
                break
    except ValueError as exc:
        _enqueue_stream_item(output, {
            "type": "error",
            "code": "invalid_request",
            "message": str(exc),
        }, cancel_event)
    except AxiomEngineError as exc:
        _enqueue_stream_item(output, {
            "type": "error",
            "code": "native_engine_error",
            "message": str(exc),
        }, cancel_event)
    except Exception:
        _LOG.exception("Unexpected failure in AXIOM Native Engine stream producer.")
        _enqueue_stream_item(output, {
            "type": "error",
            "code": "native_engine_error",
            "message": "AXIOM Native Engine stream failed unexpectedly.",
        }, cancel_event)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except (RuntimeError, ValueError):
                _LOG.debug("Native stream iterator cleanup was already in progress.")
        # On a healthy stream, wait for buffer capacity so the consumer always
        # receives an explicit end marker. After disconnect, the consumer exits
        # from the shared cancellation event and no sentinel is required.
        while not cancel_event.is_set():
            try:
                output.put(_STREAM_END, timeout=0.1)
                break
            except Full:
                continue
        # On disconnect the async consumer exits via cancel_event; it does not
        # require a sentinel, especially when the bounded queue is already full.


@router.post("/generate/stream")
async def engine_generate_stream(
    request: GenerateRequest,
    http_request: Request,
) -> StreamingResponse:
    """Stream native token snapshots while detecting disconnects cooperatively."""
    cancel_event = Event()
    output: Queue = Queue(maxsize=_STREAM_QUEUE_SIZE)
    iterator = generate_local_stream(
        request.model_path,
        request.prompt,
        adapter_path=request.adapter_path,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        seed=request.seed,
        stop_sequences=request.stop_sequences,
        cancel_event=cancel_event,
        device_preference=request.device,
        memory_mode=request.memory_mode,
        precision=request.precision,
    )
    producer = Thread(
        target=_produce_native_stream,
        args=(iterator, output, cancel_event),
        name="axiom-native-inference-stream",
        daemon=True,
    )

    async def events() -> AsyncIterator[str]:
        async def watch_disconnect() -> None:
            while not cancel_event.is_set():
                if await http_request.is_disconnected():
                    cancel_event.set()
                    return
                await asyncio.sleep(0.1)

        monitor = asyncio.create_task(watch_disconnect())
        producer.start()
        try:
            while True:
                item = await asyncio.to_thread(output.get)
                if item is _STREAM_END:
                    break
                if cancel_event.is_set():
                    break
                yield json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        finally:
            cancel_event.set()
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                # monitor.cancel() makes cancellation the expected cleanup path.
                if not monitor.cancelled():
                    raise
            # The model checks cancellation between transformer layers/tokens.
            # Never wait indefinitely for a slow device kernel during disconnect.
            await asyncio.to_thread(producer.join, 2.0)

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


class EngineFabricMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=12_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Conversation messages must not be blank.")
        return value


class EngineFabricChatRequest(BaseModel):
    backend_id: str = Field(pattern="^(axiom-native|ollama|llama-cpp|lm-studio|vllm|llama-cpp-python)$")
    model: str = Field(min_length=1, max_length=120)
    messages: list[EngineFabricMessage] = Field(min_length=1, max_length=64)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)

    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100_000, strict=True)
    min_p: float = Field(default=0.0, ge=0, le=1)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
    seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807, strict=True)
    stop_sequences: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("model")
    @classmethod
    def model_must_be_a_name_not_a_url(cls, value: str) -> str:
        from axiom.core.inference import validate_model_name
        return validate_model_name(value)


class EngineFabricAutoChatRequest(BaseModel):
    profile: str = Field(pattern="^(balanced|low-memory|throughput|easy-setup|portable|minimal-daemons)$")
    model: str = Field(min_length=1, max_length=120)
    messages: list[EngineFabricMessage] = Field(min_length=1, max_length=64)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)

    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100_000, strict=True)
    min_p: float = Field(default=0.0, ge=0, le=1)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
    seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807, strict=True)
    stop_sequences: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("model")
    @classmethod
    def model_must_be_a_name_not_a_url(cls, value: str) -> str:
        from axiom.core.inference import validate_model_name
        return validate_model_name(value)


@router.get("/backends")
def engine_backends(profile: str = "balanced") -> dict[str, Any]:
    """Inventory local execution backends and recommend one using an explicit workload profile."""
    try:
        return discover_engine_backends(profile)
    except ValueError:
        raise HTTPException(status_code=422, detail="The workload profile is invalid or unsupported.") from None


@router.post("/backends/chat")
def engine_backend_chat(request: EngineFabricChatRequest) -> dict[str, Any]:
    """Dispatch a bounded chat request to a selected fixed-loopback backend."""
    try:
        return run_engine_backend_chat(
            request.backend_id,
            request.model,
            [message.model_dump() for message in request.messages],
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stop_sequences=request.stop_sequences,
        )
    except ValueError:
        raise HTTPException(status_code=422, detail="The selected backend request is invalid or unsupported.") from None
    except EngineFabricError:
        raise HTTPException(status_code=503, detail="The selected local backend could not complete this request.") from None


@router.post("/backends/auto-chat")
def engine_backend_auto_chat(request: EngineFabricAutoChatRequest) -> dict[str, Any]:
    """Route a chat request across ready local services by profile and exact model ID."""
    try:
        return run_engine_fabric_chat(
            request.profile,
            request.model,
            [message.model_dump() for message in request.messages],
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stop_sequences=request.stop_sequences,
        )
    except ValueError:
        raise HTTPException(status_code=422, detail="The routing request is invalid or unsupported.") from None
    except EngineFabricError:
        raise HTTPException(status_code=503, detail="No matching local backend could complete this request.") from None


class EngineFabricBenchmarkRequest(BaseModel):
    backend_id: str = Field(pattern="^(axiom-native|ollama|llama-cpp|lm-studio|vllm|llama-cpp-python)$")
    model: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=12_000)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    warmup_runs: int = Field(default=1, ge=0, le=1)
    iterations: int = Field(default=3, ge=1, le=5)
    temperature: float = Field(default=0.2, ge=0, le=2)

    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100_000, strict=True)
    min_p: float = Field(default=0.0, ge=0, le=1)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
    seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807, strict=True)
    stop_sequences: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("model")
    @classmethod
    def model_must_be_a_name_not_a_url(cls, value: str) -> str:
        from axiom.core.inference import validate_model_name
        return validate_model_name(value)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Benchmark prompt must not be blank.")
        return value


@router.post("/backends/benchmark")
def engine_backend_benchmark(request: EngineFabricBenchmarkRequest) -> dict[str, Any]:
    """Run a small real-workload benchmark and retain only timing/token metadata."""
    try:
        return run_engine_backend_benchmark(
            request.backend_id,
            request.model,
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            warmup_runs=request.warmup_runs,
            iterations=request.iterations,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stop_sequences=request.stop_sequences,
        )
    except ValueError:
        raise HTTPException(status_code=422, detail="The benchmark request is invalid or unsupported.") from None
    except EngineFabricError:
        raise HTTPException(status_code=503, detail="The selected local backend could not complete the benchmark.") from None
