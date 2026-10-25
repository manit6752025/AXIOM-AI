"""Local Model Studio API: catalogue search, downloads, Ollama setup, GGUF conversion."""
from __future__ import annotations
from typing import Any, Literal
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from axiom.core import model_studio as studio

router = APIRouter(prefix="/api/models/studio", tags=["model-studio"])

class ModelJobRequest(BaseModel):
    operation: Literal["pull", "pull_native", "install_gguf", "convert_gguf"]
    model_name: str = Field(default="", max_length=120)
    repo_id: str = Field(default="", max_length=192)
    filename: str = Field(default="", max_length=255)
    register_ollama: bool = True
    install_native: bool = False
    source_dir: str = Field(default="", max_length=4096)
    llama_cpp_dir: str = Field(default="", max_length=4096)
    output_path: str = Field(default="", max_length=4096)
    outtype: Literal["auto", "f16", "f32", "bf16"] = "f16"
    quantization: Literal["none", "Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"] = "Q4_K_M"

@router.get("/status")
def status() -> dict[str, Any]:
    return studio.model_studio_status()

@router.get("/search")
def search(q: str = Query(min_length=1, max_length=120), limit: int = Query(default=12, ge=1, le=20)) -> dict[str, Any]:
    try:
        return studio.search_gguf_catalog(q, limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except studio.ModelStudioError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None

@router.get("/files")
def files(repo_id: str = Query(min_length=3, max_length=192)) -> dict[str, Any]:
    try:
        return studio.list_gguf_files(repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except studio.ModelStudioError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None

class RemoteModelRequest(BaseModel):
    base_url: str = Field(min_length=1, max_length=512)
    api_key: str = Field(min_length=1, max_length=512)

class RemoteChatRequest(RemoteModelRequest):
    model: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=12000)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)

@router.post("/remote/models")
def remote_models(request: RemoteModelRequest) -> dict[str, Any]:
    try:
        return studio.list_remote_models(request.base_url, request.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except studio.ModelStudioError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None

@router.post("/remote/chat")
def remote_chat(request: RemoteChatRequest) -> dict[str, Any]:
    try:
        return studio.run_remote_chat(request.base_url, request.api_key, request.model, request.prompt,
                                      request.max_tokens, request.temperature)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except studio.ModelStudioError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None

@router.get("/jobs")
def jobs(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    try:
        return {"jobs": studio.list_jobs(limit)}
    except (ValueError, studio.ModelStudioError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

@router.get("/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    try:
        result = studio.get_job(job_id)
    except (ValueError, studio.ModelStudioError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if result is None:
        raise HTTPException(status_code=404, detail="Model Studio job was not found.")
    return result

@router.post("/jobs", status_code=202)
def start_job(request: ModelJobRequest) -> dict[str, Any]:
    try:
        return studio.start_job(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except studio.ModelStudioError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
