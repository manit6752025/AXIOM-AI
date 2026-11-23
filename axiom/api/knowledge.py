"""Workspace-scoped API for safe local knowledge-base document ingestion."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from axiom.core.knowledge import KnowledgeBaseStore, KnowledgeStoreCapacityError, KnowledgeStoreError
from axiom.core.workspace_context import active_workspace_root

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
