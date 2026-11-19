"""Unified local/cloud model API for the AXIOM Control Center and CLI."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from axiom.core import llm_client

router = APIRouter(prefix="/api/llm", tags=["llm-client"])
ProviderName = Literal["axiom-local", "ollama-local", "openai", "anthropic", "gemini", "deepseek", "moonshot", "openrouter", "openai-compatible"]


class ProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: ProviderName
    api_key: SecretStr | None = None
    base_url: str | None = Field(default=None, max_length=512)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content must not be blank.")
        return value


class ChatRequest(ProviderRequest):
    model: str = Field(min_length=1, max_length=256)
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1, le=2_048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100_000, strict=True)
    min_p: float = Field(default=0.0, ge=0, le=1)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
    seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807, strict=True)
    stop_sequences: list[str] = Field(default_factory=list, max_length=16)
    consent: bool = False
    device: Literal["auto", "cpu", "cuda", "rocm", "mps", "xpu"] = "auto"
    memory_mode: Literal["auto", "full", "layer_streaming", "disk_streaming"] = "auto"
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"

    @model_validator(mode="after")
    def validate_conversation(self) -> "ChatRequest":
        combined = sum(len(message.content) for message in self.messages)
        if combined > 24_000:
            raise ValueError("Combined chat content must not exceed 24,000 characters.")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("At least one user message is required.")
        return self


def _api_call(operation: Any) -> dict[str, Any]:
    try:
        result = operation()
        if not isinstance(result, dict):
            raise llm_client.LLMClientError("The LLM client returned an unexpected result.")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except llm_client.LLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None


@router.get("/providers")
def get_providers() -> dict[str, Any]:
    return llm_client.provider_catalog()


@router.post("/models")
def get_models(request: ProviderRequest) -> dict[str, Any]:
    key = request.api_key.get_secret_value() if request.api_key is not None else None
    return _api_call(lambda: llm_client.list_models(request.provider, key, request.base_url))


@router.post("/chat")
def run_chat(request: ChatRequest) -> dict[str, Any]:
    if request.provider not in {"axiom-local", "ollama-local"} and not request.consent:
        raise HTTPException(
            status_code=400,
            detail="Confirm that the selected cloud provider will receive this conversation before sending it.",
        )
    key = request.api_key.get_secret_value() if request.api_key is not None else None
    messages = [message.model_dump() for message in request.messages]
    return _api_call(lambda: llm_client.chat(
        request.provider,
        request.model,
        messages,
        api_key=key,
        base_url=request.base_url,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        seed=request.seed,
        stop_sequences=request.stop_sequences,
        consent=request.consent,
        device=request.device,
        memory_mode=request.memory_mode,
        precision=request.precision,
    ))
