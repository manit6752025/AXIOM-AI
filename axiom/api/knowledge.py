"""Workspace-scoped API for safe local knowledge-base document ingestion."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from axiom.core.knowledge import KnowledgeBaseStore, KnowledgeStoreCapacityError, KnowledgeStoreError
from axiom.core.workspace_context import active_workspace_root
from axiom.core.embeddings import EmbeddingEngineError
from axiom.core.knowledge_search import KnowledgeSearchError, search_knowledge_base

router = APIRouter(prefix="/api/workspace/knowledge-bases", tags=["workspace-knowledge"])


class KnowledgeBaseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class KnowledgeDocumentIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    path: str = Field(min_length=1, max_length=4096)


def _store() -> KnowledgeBaseStore:
    return KnowledgeBaseStore(active_workspace_root())


def _translate_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KnowledgeStoreCapacityError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Knowledge base or document was not found.")
    if isinstance(exc, KnowledgeStoreError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=422, detail="Knowledge operation could not be completed safely.")


@router.get("")
def list_knowledge_bases() -> dict[str, Any]:
    """List metadata for the active workspace's knowledge bases."""
    try:
        return _store().list_bases()
    except (OSError, ValueError) as exc:
        raise _translate_store_error(exc) from None


@router.post("", status_code=status.HTTP_201_CREATED)
def create_knowledge_base(request: KnowledgeBaseCreateRequest) -> dict[str, Any]:
    """Create a private knowledge base inside the active workspace."""
    try:
        return {"schema_version": 1, "knowledge_base": _store().create_base(request.name, request.description)}
    except (OSError, ValueError) as exc:
        raise _translate_store_error(exc) from None


@router.get("/{base_id}/documents")
def list_knowledge_documents(base_id: str) -> dict[str, Any]:
    """List document metadata without returning ingested source content."""
    try:
        return _store().list_documents(base_id)
    except KnowledgeSearchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (OSError, ValueError, KeyError) as exc:
        raise _translate_store_error(exc) from None


@router.post("/{base_id}/documents", status_code=status.HTTP_201_CREATED)
def ingest_knowledge_document(base_id: str, request: KnowledgeDocumentIngestRequest) -> dict[str, Any]:
    """Ingest one bounded UTF-8 document from inside the active workspace."""
    try:
        record = _store().ingest_document(base_id, request.path)
        return {
            "schema_version": 1,
            "document": record,
            "source_text_returned": False,
            "source_text_persisted": True,
            "network_used": False,
        }
    except (OSError, ValueError, KeyError) as exc:
        raise _translate_store_error(exc) from None


@router.delete("/{base_id}/documents/{document_id}")
def remove_knowledge_document(base_id: str, document_id: str) -> dict[str, Any]:
    """Remove a document and its private extracted-text file."""
    try:
        removed = _store().remove_document(base_id, document_id)
        if removed is None:
            raise HTTPException(status_code=404, detail="Knowledge document was not found.")
        return {"schema_version": 1, "removed": removed, "source_file_modified": False}
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise _translate_store_error(exc) from None


@router.delete("/{base_id}")
def remove_knowledge_base(base_id: str) -> dict[str, Any]:
    """Remove one knowledge base and its private ingested text files."""
    try:
        removed = _store().remove_base(base_id)
        if removed is None:
            raise HTTPException(status_code=404, detail="Knowledge base was not found.")
        return {"schema_version": 1, "removed": removed, "source_files_modified": False}
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise _translate_store_error(exc) from None


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8192)
    limit: int = Field(default=8, ge=1, le=20, strict=True)
    max_documents: int = Field(default=64, ge=1, le=64, strict=True)
    chunk_size_chars: int = Field(default=1200, ge=256, le=4096, strict=True)
    chunk_overlap_chars: int = Field(default=160, ge=0, le=512, strict=True)
    formats: list[str] = Field(default_factory=list, max_length=7)
    source_prefixes: list[str] = Field(default_factory=list, max_length=16)
    document_ids: list[str] = Field(default_factory=list, max_length=64)
    embedding_model_path: str | None = Field(default=None, min_length=1, max_length=4096)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|rocm|mps|xpu)$")

    @field_validator("query")
    @classmethod
    def query_must_contain_searchable_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Search query must contain non-whitespace characters.")
        return value

    @field_validator("embedding_model_path", mode="before")
    @classmethod
    def embedding_path_must_be_trusted_when_supplied(cls, value: Any) -> Any:
        return _validated_path(value, "Base model") if value is not None else None

    @model_validator(mode="after")
    def chunk_overlap_must_be_smaller_than_chunk_size(self) -> "KnowledgeSearchRequest":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        return self


@router.post("/{base_id}/search")
def search_knowledge_documents(base_id: str, request: KnowledgeSearchRequest) -> dict[str, Any]:
    """Search a workspace knowledge base with bounded lexical or hybrid local ranking."""
    try:
        return search_knowledge_base(
            _store(),
            base_id,
            request.query,
            limit=request.limit,
            max_documents=request.max_documents,
            chunk_size_chars=request.chunk_size_chars,
            chunk_overlap_chars=request.chunk_overlap_chars,
            formats=request.formats,
            source_prefixes=request.source_prefixes,
            document_ids=request.document_ids,
            embedding_model_path=request.embedding_model_path,
            vector_weight=request.vector_weight,
            device=request.device,
        )
    except EmbeddingEngineError:
        raise HTTPException(status_code=503, detail="The local embedding model could not complete this retrieval request.") from None
    except KnowledgeSearchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (OSError, ValueError, KeyError) as exc:
        raise _translate_store_error(exc) from None
