"""Unified local/cloud model API for the AXIOM Control Center and CLI."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from axiom.core import llm_client
from axiom.core.llm_profiles import LLMProfileError, LLMProfileStore
from axiom.core.secret_refs import SecretReferenceError, resolve_secret_reference

router = APIRouter(prefix="/api/llm", tags=["llm-client"])
ProviderName = Literal["axiom-local", "ollama-local", "openai", "anthropic", "gemini", "deepseek", "moonshot", "openrouter", "openai-compatible"]


class ProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: ProviderName
    api_key: SecretStr | None = None
    base_url: str | None = Field(default=None, max_length=512)
    profile_id: str | None = Field(default=None, min_length=32, max_length=32, pattern="^[0-9a-f]{32}$")




class LLMProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=48)
    provider: ProviderName
    model: str = Field(min_length=1, max_length=256)
    base_url: str | None = Field(default=None, max_length=512)
    api_key_env: str | None = Field(default=None, max_length=128)


class LLMProfileConnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr | None = None


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
    credential_source = "request_only" if key else "none"
    if request.profile_id:
        try:
            profile = LLMProfileStore().get(request.profile_id)
        except LLMProfileError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from None
        if profile is None:
            raise HTTPException(status_code=404, detail="Provider profile not found.")
        if profile["provider"] != request.provider or (profile["base_url"] or None) != (request.base_url or None):
            raise HTTPException(status_code=422, detail="Provider request does not match the selected saved profile.")
        if key is None and profile.get("api_key_env"):
            try:
                key = resolve_secret_reference(profile["api_key_env"])
            except SecretReferenceError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
            credential_source = "environment_reference"
    result = _api_call(lambda: llm_client.list_models(request.provider, key, request.base_url))
    if request.profile_id:
        result["credential_source"] = credential_source
    return result


@router.get("/profiles")
def list_profiles() -> dict[str, Any]:
    try:
        return LLMProfileStore().list()
    except LLMProfileError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None


@router.post("/profiles", status_code=201)
def save_profile(request: LLMProfileInput) -> dict[str, Any]:
    try:
        profile = LLMProfileStore().save(
            name=request.name,
            provider=request.provider,
            model=request.model,
            base_url=request.base_url,
            api_key_env=request.api_key_env,
        )
        return {"profile": profile, "credentials_persisted": False}
    except LLMProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.delete("/profiles/{profile_id}")
def delete_profile(profile_id: str) -> dict[str, Any]:
    try:
        deleted = LLMProfileStore().delete(profile_id)
    except LLMProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if deleted is None:
        raise HTTPException(status_code=404, detail="Provider profile not found.")
    return {"deleted": deleted, "credentials_persisted": False}


@router.post("/profiles/{profile_id}/connect")
def connect_profile(profile_id: str, request: LLMProfileConnectRequest) -> dict[str, Any]:
    try:
        profile = LLMProfileStore().get(profile_id)
    except LLMProfileError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    if profile is None:
        raise HTTPException(status_code=404, detail="Provider profile not found.")
    key = request.api_key.get_secret_value() if request.api_key is not None else None
    credential_source = "request_only" if key else "none"
    if key is None and profile.get("api_key_env"):
        try:
            key = resolve_secret_reference(profile["api_key_env"])
        except SecretReferenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        credential_source = "environment_reference"
    result = _api_call(lambda: llm_client.list_models(
        profile["provider"],
        key,
        profile["base_url"] or None,
    ))
    return {
        "profile_id": profile["id"],
        "provider": profile["provider"],
        "status": "verified",
        "models": result.get("models", []),
        "checked_at": datetime.now(UTC).isoformat(),
        "credential_storage": result.get("credential_storage", "not_saved" if key else "not_applicable"),
        "credential_source": credential_source,
        "connection_scope": "catalogue_check_only",
        "persistent_connection": False,
    }


@router.post("/chat")
def run_chat(request: ChatRequest) -> dict[str, Any]:
    if request.provider not in {"axiom-local", "ollama-local"} and not request.consent:
        raise HTTPException(
            status_code=400,
            detail="Confirm that the selected cloud provider will receive this conversation before sending it.",
        )
    key = request.api_key.get_secret_value() if request.api_key is not None else None
    if request.profile_id:
        try:
            profile = LLMProfileStore().get(request.profile_id)
        except LLMProfileError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from None
        if profile is None:
            raise HTTPException(status_code=404, detail="Provider profile not found.")
        if (
            profile["provider"] != request.provider
            or profile["model"] != request.model
            or (profile["base_url"] or None) != (request.base_url or None)
        ):
            raise HTTPException(status_code=422, detail="Chat request does not match the selected saved provider profile.")
        if key is None and profile.get("api_key_env"):
            try:
                key = resolve_secret_reference(profile["api_key_env"])
            except SecretReferenceError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
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
