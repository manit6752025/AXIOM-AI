"""HTTP API for AXIOM's direct local model engine."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from queue import Full, Queue
from threading import Event, Thread
import time
from typing import Any, AsyncIterator, Iterator

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.responses import StreamingResponse
from axiom.core import model_studio
from axiom.core.engine_fabric import EngineFabricBenchmarkRequirementError, EngineFabricError, discover_engine_backends, inspect_engine_backend_benchmark_requirements, run_engine_backend_benchmark, run_engine_backend_chat, run_engine_fabric_chat
from axiom.core.engine_fabric_profiles import EngineFabricProfileError, EngineFabricProfileStore
from axiom.core.benchmark_profiles import BenchmarkProfileError, BenchmarkProfileStore
from axiom.core.resource_telemetry import ResourceTelemetryError, ResourceTelemetryOptInRequired, ResourceTelemetryStore
from axiom.core.embeddings import EmbeddingEngineError, embed_local_texts, unload_local_embeddings
from axiom.core.reranking import RerankerError, rerank_local_documents, unload_local_reranker
from axiom.core.request_admission import (
    BoundedRequestAdmission,
    RequestAdmissionCancelled,
    RequestAdmissionQueueFull,
    RequestAdmissionTimeout,
)
from axiom.core.serving_profiles import (
    ServingProfileError,
    ServingProfileStore,
    generate_from_profile,
    readiness_probe,
    release_profile_request,
    reserve_profile_request,
    validate_serving_prompt,
)

from axiom.core.engine import (
    AxiomEngineError,
    benchmark_local,
    check_local_adapter_compatibility,
    discover_local_engine_assets,
    estimate_native_quantization,
    generate_local,
    generate_local_batch,
    generate_local_stream,
    inspect_native_model_context,
    resolve_local_model_directory,
    runtime_status,
    runtime_telemetry,
    unload_local_model,
)

router = APIRouter(prefix="/api/engine", tags=["axiom-engine"])
NATIVE_PROFILE_ADMISSION = BoundedRequestAdmission(max_active=1, max_queue=32)


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


class ResourceTelemetryHistorySettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(strict=True)
    max_samples: int = Field(default=120, ge=1, le=500, strict=True)
    max_age_days: int = Field(default=30, ge=1, le=3650, strict=True)


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



class NativeBenchmarkRequest(GenerateRequest):
    warmup_runs: int = Field(default=1, ge=0, le=2)
    iterations: int = Field(default=3, ge=1, le=5)
    max_new_tokens: int = Field(default=64, ge=1, le=128)
    profile_name: str | None = Field(default=None, min_length=1, max_length=64, pattern="^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")


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


@router.get("/telemetry-history")
def engine_telemetry_history(limit: int = Query(default=500, ge=1, le=500, strict=True)) -> dict[str, Any]:
    """Read explicit, bounded workspace resource history without probing hardware."""
    try:
        return ResourceTelemetryStore().list(limit=limit)
    except ResourceTelemetryError:
        raise HTTPException(status_code=409, detail="Resource telemetry history is invalid or unavailable.") from None


@router.put("/telemetry-history/settings")
def configure_engine_telemetry_history(request: ResourceTelemetryHistorySettingsRequest) -> dict[str, Any]:
    """Enable or disable persistent resource history and set its retention limits."""
    try:
        return ResourceTelemetryStore().configure(
            enabled=request.enabled,
            max_samples=request.max_samples,
            max_age_days=request.max_age_days,
        )
    except ResourceTelemetryError:
        raise HTTPException(status_code=422, detail="Resource telemetry retention settings are invalid or could not be saved.") from None


@router.post("/telemetry-history", status_code=201)
def record_engine_telemetry_history() -> dict[str, Any]:
    """Capture one allowlisted local resource sample only after explicit opt-in."""
    store = ResourceTelemetryStore()
    try:
        if not store.list(limit=1)["settings"]["enabled"]:
            raise ResourceTelemetryOptInRequired(
                "Persistent resource history is disabled. Enable it explicitly first."
            )
        return store.record(runtime_telemetry())
    except ResourceTelemetryOptInRequired:
        raise HTTPException(status_code=409, detail="Persistent resource history is disabled. Enable it explicitly first.") from None
    except ResourceTelemetryError:
        raise HTTPException(status_code=409, detail="The resource telemetry sample could not be stored safely.") from None


@router.delete("/telemetry-history")
def clear_engine_telemetry_history() -> dict[str, Any]:
    """Clear stored resource samples while preserving opt-in and retention settings."""
    try:
        return ResourceTelemetryStore().clear()
    except ResourceTelemetryError:
        raise HTTPException(status_code=409, detail="Resource telemetry history could not be cleared safely.") from None


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


@router.post("/benchmark")
async def engine_benchmark(
    request: NativeBenchmarkRequest,
    raw_request: Request,
) -> dict[str, Any]:
    """Run a bounded native-only benchmark with client-disconnect cancellation."""
    cancel_event = Event()
    worker = asyncio.create_task(asyncio.to_thread(
        benchmark_local,
        request.model_path,
        request.prompt,
        adapter_path=request.adapter_path,
        warmup_runs=request.warmup_runs,
        iterations=request.iterations,
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

            def consume_background_result(task: asyncio.Task[Any]) -> None:
                if not task.cancelled():
                    try:
                        task.exception()
                    except Exception:
                        pass

            worker.add_done_callback(consume_background_result)
            raise HTTPException(
                status_code=499,
                detail="Native benchmark cancelled because the client disconnected.",
            )
        await asyncio.sleep(0.05)
    try:
        report = await worker
        if request.profile_name:
            report["profile_record"] = BenchmarkProfileStore().create(
                request.profile_name, report, request.prompt, request.model_path,
                adapter_used=request.adapter_path is not None,
            )
        else:
            report["profile_record"] = None
        return report
    except (ValueError, BenchmarkProfileError):
        raise HTTPException(status_code=422, detail="Benchmark request or saved profile is invalid.") from None
    except AxiomEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("/benchmark-profiles")
def list_native_benchmark_profiles(limit: int = 100) -> dict[str, Any]:
    try:
        return BenchmarkProfileStore().list(limit)
    except BenchmarkProfileError:
        raise HTTPException(status_code=409, detail="Benchmark profile history could not be read safely.") from None


@router.get("/benchmark-profiles/compare")
def compare_native_benchmark_profiles(left_id: str, right_id: str) -> dict[str, Any]:
    try:
        return BenchmarkProfileStore().compare(left_id, right_id)
    except BenchmarkProfileError:
        raise HTTPException(status_code=422, detail="The requested benchmark profiles cannot be compared.") from None


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


class EngineFabricProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=48)
    backend_id: str = Field(pattern="^(axiom-native|ollama|llama-cpp|lm-studio|vllm|llama-cpp-python)$")
    model: str = Field(min_length=1, max_length=120)

    @field_validator("model")
    @classmethod
    def model_must_be_a_name_not_a_url(cls, value: str) -> str:
        from axiom.core.inference import validate_model_name
        return validate_model_name(value)


@router.get("/backends/profiles")
def engine_backend_profiles() -> dict[str, Any]:
    """List secret-free workspace connector profiles."""
    try:
        return EngineFabricProfileStore().list()
    except EngineFabricProfileError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None


@router.post("/backends/profiles", status_code=201)
def save_engine_backend_profile(request: EngineFabricProfileInput) -> dict[str, Any]:
    """Create or update metadata for one fixed local backend and exact model ID."""
    try:
        profile = EngineFabricProfileStore().save(name=request.name, backend_id=request.backend_id, model=request.model)
        return {"profile": profile, "credentials_persisted": False, "custom_endpoints_persisted": False}
    except EngineFabricProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.delete("/backends/profiles/{profile_id}")
def delete_engine_backend_profile(profile_id: str) -> dict[str, Any]:
    """Delete profile metadata; do not stop an external runtime."""
    try:
        deleted = EngineFabricProfileStore().delete(profile_id)
    except EngineFabricProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if deleted is None:
        raise HTTPException(status_code=404, detail="Engine Fabric profile not found.")
    return {"deleted": deleted, "credentials_persisted": False}


@router.post("/backends/profiles/{profile_id}/connect")
def connect_engine_backend_profile(profile_id: str) -> dict[str, Any]:
    """Verify local health and exact model inventory without retaining a socket."""
    try:
        profile = EngineFabricProfileStore().get(profile_id)
    except EngineFabricProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if profile is None:
        raise HTTPException(status_code=404, detail="Engine Fabric profile not found.")
    try:
        report = discover_engine_backends("balanced")
    except (ValueError, EngineFabricError):
        raise HTTPException(status_code=503, detail="Local Engine Fabric discovery failed.") from None
    backend = next((item for item in report.get("backends", []) if item.get("id") == profile["backend_id"]), None)
    if not isinstance(backend, dict) or backend.get("status") != "ready":
        raise HTTPException(status_code=409, detail="The saved local backend is not ready; no connection was established.")
    models = backend.get("models")
    if not isinstance(models, list) or profile["model"] not in models:
        raise HTTPException(status_code=409, detail="The exact saved model ID is not available on the local backend.")
    return {
        "profile_id": profile["id"],
        "backend_id": profile["backend_id"],
        "model": profile["model"],
        "status": "verified",
        "checked_at": datetime.now(UTC).isoformat(),
        "connection_scope": "one_shot_local_health_and_model_inventory_check",
        "persistent_connection": False,
        "credentials_persisted": False,
        "custom_endpoints_persisted": False,
        "processes_started": False,
    }


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


@router.get("/backends/benchmark-requirements")
def engine_backend_benchmark_requirements(backend_id: str = "ollama", model: str | None = None) -> dict[str, Any]:
    """Return declared adapter requirements and a live, generation-free preflight."""
    try:
        return inspect_engine_backend_benchmark_requirements(backend_id, model)
    except ValueError:
        raise HTTPException(status_code=422, detail="The benchmark adapter or model ID is invalid.") from None


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
    except EngineFabricBenchmarkRequirementError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except EngineFabricError:
        raise HTTPException(status_code=503, detail="The selected local backend could not complete the benchmark.") from None


class LocalEmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    model_path: str = Field(min_length=1, max_length=4096)
    texts: list[str] = Field(min_length=1, max_length=32)
    batch_size: int = Field(default=8, ge=1, le=16, strict=True)
    max_length: int = Field(default=256, ge=8, le=512, strict=True)
    chunk_size_chars: int = Field(default=2048, ge=256, le=8192, strict=True)
    chunk_overlap_chars: int = Field(default=128, ge=0, le=1024, strict=True)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|rocm|mps|xpu)$")
    normalize: bool = Field(default=True, strict=True)

    @field_validator("model_path", mode="before")
    @classmethod
    def embedding_model_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "Base model")

    @field_validator("texts")
    @classmethod
    def texts_must_be_bounded_and_nonblank(cls, value: list[str]) -> list[str]:
        from axiom.core.embeddings import MAX_EMBEDDING_TEXT_CHARS, MAX_EMBEDDING_TOTAL_CHARS
        if any(not isinstance(text, str) or not text.strip() for text in value):
            raise ValueError("Every text input must contain non-whitespace characters.")
        if any(len(text) > MAX_EMBEDDING_TEXT_CHARS for text in value):
            raise ValueError("A text input exceeds the per-input character limit.")
        if sum(len(text) for text in value) > MAX_EMBEDDING_TOTAL_CHARS:
            raise ValueError("Combined text inputs exceed the request character limit.")
        return value

    @model_validator(mode="after")
    def chunk_overlap_must_be_smaller_than_chunk_size(self) -> "LocalEmbeddingsRequest":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        return self


@router.post("/embeddings")
def engine_embeddings(request: LocalEmbeddingsRequest) -> dict[str, Any]:
    """Embed bounded local text using a trusted safetensors model; no cloud or storage."""
    try:
        return embed_local_texts(
            request.model_path, request.texts, batch_size=request.batch_size,
            max_length=request.max_length, chunk_size_chars=request.chunk_size_chars,
            chunk_overlap_chars=request.chunk_overlap_chars, device=request.device,
            normalize=request.normalize,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except EmbeddingEngineError:
        raise HTTPException(status_code=503, detail="The local embedding model could not complete this request.") from None


@router.post("/embeddings/unload")
def engine_embeddings_unload() -> dict[str, Any]:
    """Unload AXIOM's cached embedding model without changing checkpoint files."""
    return unload_local_embeddings()


class ServingProfileCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    name: str = Field(min_length=1, max_length=48, pattern=r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,47}$")
    model_path: str = Field(min_length=1, max_length=4096)
    adapter_path: str | None = Field(default=None, max_length=4096)
    device_preference: str = Field(default="auto", pattern="^(auto|cpu|cuda|rocm|mps|xpu)$")
    memory_mode: str = Field(default="auto", pattern="^(auto|full|layer_streaming|disk_streaming)$")
    precision: str = Field(default="auto", pattern="^(auto|fp32|fp16|bf16)$")
    context_length: int = Field(default=4096, ge=1, le=131072, strict=True)
    max_new_tokens: int = Field(default=256, ge=1, le=2048, strict=True)
    max_prompt_chars: int = Field(default=12000, ge=1, le=12000, strict=True)
    max_concurrent_requests: int = Field(default=1, ge=1, le=1, strict=True)
    max_queued_requests: int = Field(default=0, ge=0, le=32, strict=True)
    request_timeout_seconds: int = Field(default=300, ge=5, le=900, strict=True)

    @field_validator("model_path", mode="before")
    @classmethod
    def serving_base_model_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "Base model")

    @field_validator("adapter_path", mode="before")
    @classmethod
    def serving_adapter_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "LoRA adapter")


class ServingProfileGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=2048, strict=True)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=0, ge=0, le=100000, strict=True)
    min_p: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0, le=9223372036854775807, strict=True)
    stop_sequences: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("prompt")
    @classmethod
    def serving_prompt_must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must contain non-whitespace text.")
        return value

    @field_validator("stop_sequences")
    @classmethod
    def serving_stops_must_be_bounded(cls, value: list[str]) -> list[str]:
        unique: list[str] = []
        for sequence in value:
            if not isinstance(sequence, str) or not sequence.strip() or len(sequence) > 256:
                raise ValueError("Stop sequences must be nonblank text up to 256 characters.")
            if sequence not in unique:
                unique.append(sequence)
        return unique


@router.get("/serving-profiles")
def list_engine_serving_profiles() -> dict[str, Any]:
    try:
        return ServingProfileStore().list()
    except ServingProfileError:
        raise HTTPException(status_code=409, detail="Serving profile history could not be read safely.") from None


@router.post("/serving-profiles", status_code=201)
def create_engine_serving_profile(request: ServingProfileCreateRequest) -> dict[str, Any]:
    try:
        model_context = inspect_native_model_context(request.model_path)
        if request.context_length > model_context["effective_max_position_embeddings"]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Context length exceeds this checkpoint's supported limit of "
                    f"{model_context['effective_max_position_embeddings']} tokens."
                ),
            )
        profile = ServingProfileStore().save(request.model_dump())
        return {
            "profile": profile,
            "model_context": model_context,
            "scope": {"weights_loaded": False, "service_started": False, "network_used": False},
        }
    except HTTPException:
        raise
    except (ServingProfileError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.get("/serving-profiles/{profile_id}")
def get_engine_serving_profile(profile_id: str) -> dict[str, Any]:
    try:
        profile = ServingProfileStore().get(profile_id)
    except ServingProfileError:
        raise HTTPException(status_code=422, detail="Serving profile ID is invalid.") from None
    if profile is None:
        raise HTTPException(status_code=404, detail="Serving profile not found.")
    return {"profile": profile}


@router.delete("/serving-profiles/{profile_id}")
def delete_engine_serving_profile(profile_id: str) -> dict[str, Any]:
    try:
        deleted = ServingProfileStore().delete(profile_id)
    except ServingProfileError:
        raise HTTPException(status_code=422, detail="Serving profile ID is invalid.") from None
    if deleted is None:
        raise HTTPException(status_code=404, detail="Serving profile not found.")
    return {"deleted": deleted, "model_files_deleted": False}


@router.post("/serving-profiles/{profile_id}/readiness")
def probe_engine_serving_profile(profile_id: str) -> dict[str, Any]:
    try:
        profile = ServingProfileStore().get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Serving profile not found.")
        return readiness_probe(profile)
    except HTTPException:
        raise
    except (ServingProfileError, OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=409, detail="Serving profile readiness could not be verified safely.") from None


@router.post("/serving-profiles/{profile_id}/generate")
async def generate_with_engine_serving_profile(
    profile_id: str,
    request: ServingProfileGenerateRequest,
    raw_request: Request,
) -> dict[str, Any]:
    try:
        profile = ServingProfileStore().get(profile_id)
    except ServingProfileError:
        raise HTTPException(status_code=422, detail="Serving profile ID is invalid.") from None
    if profile is None:
        raise HTTPException(status_code=404, detail="Serving profile not found.")

    readiness = readiness_probe(profile)
    if readiness.get("ready") is not True:
        raise HTTPException(409, detail={"message": "Serving profile is not ready.", "readiness": readiness})
    try:
        budget = validate_serving_prompt(profile, request.prompt, request.max_new_tokens)
    except ServingProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    admission_cancel = Event()
    started = time.monotonic()

    async def watch_queue_disconnect() -> None:
        while not admission_cancel.is_set():
            if await raw_request.is_disconnected():
                admission_cancel.set()
                return
            await asyncio.sleep(0.05)

    disconnect_monitor = asyncio.create_task(watch_queue_disconnect())
    try:
        admission = await asyncio.to_thread(
            NATIVE_PROFILE_ADMISSION.acquire,
            queue_limit=profile["max_queued_requests"],
            timeout_seconds=profile["request_timeout_seconds"],
            cancel_event=admission_cancel,
        )
    except RequestAdmissionQueueFull as exc:
        disconnect_monitor.cancel()
        try:
            await disconnect_monitor
        except asyncio.CancelledError:
            pass
        raise HTTPException(429, detail=str(exc), headers={"Retry-After": "1"}) from None
    except RequestAdmissionTimeout as exc:
        disconnect_monitor.cancel()
        try:
            await disconnect_monitor
        except asyncio.CancelledError:
            pass
        raise HTTPException(status_code=504, detail=str(exc)) from None
    except RequestAdmissionCancelled as exc:
        disconnect_monitor.cancel()
        try:
            await disconnect_monitor
        except asyncio.CancelledError:
            pass
        raise HTTPException(status_code=499, detail=str(exc)) from None
    except (OSError, RuntimeError, ValueError) as exc:
        disconnect_monitor.cancel()
        try:
            await disconnect_monitor
        except asyncio.CancelledError:
            pass
        raise HTTPException(status_code=503, detail="Native request admission failed safely.") from exc
    else:
        disconnect_monitor.cancel()
        try:
            await disconnect_monitor
        except asyncio.CancelledError:
            pass

    if admission_cancel.is_set() or await raw_request.is_disconnected():
        NATIVE_PROFILE_ADMISSION.release()
        raise HTTPException(status_code=499, detail="Serving profile request disconnected while queued.")

    cancel_event = Event()
    worker = asyncio.create_task(asyncio.to_thread(
        generate_from_profile,
        profile,
        request.prompt,
        max_new_tokens=budget["max_new_tokens"],
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        seed=request.seed,
        stop_sequences=request.stop_sequences,
        cancel_event=cancel_event,
        prompt_budget=budget,
    ))
    release_scheduled = False
    try:
        while not worker.done():
            if await raw_request.is_disconnected():
                cancel_event.set()
                worker.add_done_callback(lambda _task: NATIVE_PROFILE_ADMISSION.release())
                release_scheduled = True
                raise HTTPException(status_code=499, detail="Serving profile generation cancelled after client disconnect.")
            if time.monotonic() - started >= profile["request_timeout_seconds"]:
                cancel_event.set()
                worker.add_done_callback(lambda _task: NATIVE_PROFILE_ADMISSION.release())
                release_scheduled = True
                raise HTTPException(status_code=504, detail="Serving profile request exceeded its configured timeout.")
            await asyncio.sleep(0.05)
        result = await worker
        return {
            "schema_version": 1,
            "profile": {"id": profile["id"], "name": profile["name"]},
            "request_budget": budget,
            "admission": {
                "queue_position": admission["queue_position"],
                "queued": admission["queue_position"] > 0,
                "waited_ms": admission["waited_ms"],
                "concurrency_limit": profile["max_concurrent_requests"],
                "queue_limit": profile["max_queued_requests"],
            },
            "result": result,
            "scope": {"local_only": True, "network_used": False, "profile_persisted": False},
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except AxiomEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    finally:
        if not release_scheduled:
            if worker.done():
                NATIVE_PROFILE_ADMISSION.release()
            else:
                cancel_event.set()
                worker.add_done_callback(lambda _task: NATIVE_PROFILE_ADMISSION.release())

class LocalRerankingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    model_path: str = Field(min_length=1, max_length=4096)
    query: str = Field(min_length=1, max_length=8192)
    documents: list[str] = Field(min_length=1, max_length=64)
    batch_size: int = Field(default=8, ge=1, le=16, strict=True)
    max_length: int = Field(default=256, ge=32, le=512, strict=True)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|rocm|mps|xpu)$")

    @field_validator("model_path", mode="before")
    @classmethod
    def reranker_model_path_must_be_trusted(cls, value: Any) -> Any:
        return _validated_path(value, "Base model")

    @field_validator("query")
    @classmethod
    def query_must_be_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The reranking query must contain non-whitespace characters.")
        return value

    @field_validator("documents")
    @classmethod
    def documents_must_be_bounded_and_nonblank(cls, value: list[str]) -> list[str]:
        from axiom.core.reranking import MAX_RERANK_TEXT_CHARS, MAX_RERANK_TOTAL_CHARS
        if any(not isinstance(text, str) or not text.strip() for text in value):
            raise ValueError("Every document must contain non-whitespace characters.")
        if any(len(text) > MAX_RERANK_TEXT_CHARS for text in value):
            raise ValueError("A document exceeds the per-document character limit.")
        return value

    @model_validator(mode="after")
    def total_text_must_be_bounded(self) -> "LocalRerankingRequest":
        from axiom.core.reranking import MAX_RERANK_TOTAL_CHARS
        if len(self.query) + sum(len(item) for item in self.documents) > MAX_RERANK_TOTAL_CHARS:
            raise ValueError("Combined query and documents exceed the request character limit.")
        return self


@router.post("/rerank")
def engine_rerank(request: LocalRerankingRequest) -> dict[str, Any]:
    """Score and sort local query/document pairs without returning source text."""
    try:
        return rerank_local_documents(
            request.model_path, request.query, request.documents,
            batch_size=request.batch_size, max_length=request.max_length, device=request.device,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RerankerError:
        raise HTTPException(status_code=503, detail="The local reranker could not complete this request.") from None


@router.post("/rerank/unload")
def engine_rerank_unload() -> dict[str, Any]:
    """Unload the cached cross-encoder without modifying checkpoint files."""
    return unload_local_reranker()
