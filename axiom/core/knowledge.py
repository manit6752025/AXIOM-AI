"""Workspace-scoped, bounded document ingestion for local knowledge bases."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import active_workspace_root
from axiom.path_safety import SafePathError, find_existing_relative_path, open_regular_file

SCHEMA_VERSION = 1
MAX_KNOWLEDGE_BASES = 64
MAX_DOCUMENTS_PER_BASE = 1_000
MAX_INDEX_BYTES = 4 * 1024 * 1024
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_DOCUMENT_BYTES = 1024 * 1024 * 1024
_ALLOWED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".jsonl"})
_BASE_ID = re.compile(r"^[0-9a-f]{32}$")
_DOCUMENT_ID = re.compile(r"^[0-9a-f]{32}$")
_ALLOWED_TEXT_CONTROLS = frozenset({9, 10, 13})


class KnowledgeStoreError(ValueError):
    """A knowledge-base operation failed its validation or storage boundary."""


class KnowledgeStoreCapacityError(KnowledgeStoreError):
    """The workspace knowledge store reached a declared capacity limit."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_name(value: Any, *, maximum: int, label: str) -> str:
    if not isinstance(value, str):
        raise KnowledgeStoreError(f"{label} must be text.")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum or any(ord(ch) < 32 for ch in normalized):
        raise KnowledgeStoreError(f"{label} must contain 1 to {maximum} printable characters.")
    return normalized


def _workspace_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 4096:
        raise KnowledgeStoreError("Document path must be a non-empty workspace-relative path.")
    if raw.startswith("/") or "\\" in raw or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise KnowledgeStoreError("Document path must be workspace-relative.")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts) or (len(parts[0]) >= 2 and parts[0][1] == ":"):
        raise KnowledgeStoreError("Document path contains an unsafe component.")
    return "/".join(parts)


class KnowledgeBaseStore:
    """Persist knowledge-base metadata and extracted text within one workspace.

    The index stores metadata only. UTF-8 source content is placed in private
    workspace .axiom/knowledge/documents files keyed by random resource IDs.
    Paths accepted from callers are workspace-relative and opened only after
    symlink-aware directory-entry validation.
    """

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        supplied = Path(workspace_root) if workspace_root is not None else active_workspace_root()
        if supplied.is_symlink():
            raise KnowledgeStoreError("Knowledge workspace root must not be a symbolic link.")
        try:
            self.workspace_root = supplied.resolve(strict=True)
        except (OSError, RuntimeError):
            raise KnowledgeStoreError("Knowledge workspace root is unavailable.") from None
        if not self.workspace_root.is_dir():
            raise KnowledgeStoreError("Knowledge workspace root must be a directory.")
        self.metadata_root = self.workspace_root / ".axiom"
        self.root = self.metadata_root / "knowledge"
        self.documents_root = self.root / "documents"
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / ".knowledge.lock"

    def _ensure_layout(self, *, create: bool) -> None:
        for path in (self.metadata_root, self.root, self.documents_root, self.index_path, self.lock_path):
            if path.is_symlink():
                raise KnowledgeStoreError("Knowledge storage paths must not be symbolic links.")
        if create:
            self.metadata_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.documents_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                self.metadata_root.chmod(0o700)
                self.root.chmod(0o700)
                self.documents_root.chmod(0o700)
            except OSError:
                pass
        for directory in (self.metadata_root, self.root, self.documents_root):
            if directory.exists() and not directory.is_dir():
                raise KnowledgeStoreError("Knowledge storage directory is not a regular directory.")

    def _read_index(self) -> list[dict[str, Any]]:
        self._ensure_layout(create=False)
        if not self.index_path.exists():
            return []
        try:
            info = self.index_path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_INDEX_BYTES:
                raise KnowledgeStoreError("Knowledge index is not a bounded regular file.")
            with open_regular_file(self.index_path) as handle:
                payload = json.loads(handle.read(MAX_INDEX_BYTES + 1).decode("utf-8"))
        except KnowledgeStoreError:
            raise
        except (OSError, UnicodeError, ValueError, RecursionError, SafePathError):
            raise KnowledgeStoreError("Knowledge index could not be read safely.") from None
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "bases"}
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or not isinstance(payload.get("bases"), list)
            or len(payload["bases"]) > MAX_KNOWLEDGE_BASES
        ):
            raise KnowledgeStoreError("Knowledge index has an unsupported schema.")
        seen_bases: set[str] = set()
        seen_names: set[str] = set()
        records = []
        for base in payload["bases"]:
            if not isinstance(base, dict) or set(base) != {
                "id", "name", "description", "created_at", "updated_at", "documents",
            }:
                raise KnowledgeStoreError("Knowledge index contains an invalid base entry.")
            base_id = base.get("id")
            name = base.get("name")
            description = base.get("description")
            docs = base.get("documents")
            if (
                not isinstance(base_id, str) or not _BASE_ID.fullmatch(base_id)
                or base_id in seen_bases
                or not isinstance(name, str) or not name.strip() or len(name) > 80
                or not isinstance(description, str) or len(description) > 500
                or not isinstance(base.get("created_at"), str) or len(base["created_at"]) > 64
                or not isinstance(base.get("updated_at"), str) or len(base["updated_at"]) > 64
                or not isinstance(docs, list) or len(docs) > MAX_DOCUMENTS_PER_BASE
            ):
                raise KnowledgeStoreError("Knowledge index contains an invalid base entry.")
            name_key = name.casefold()
            if name_key in seen_names:
                raise KnowledgeStoreError("Knowledge base names must be unique within a workspace.")
            seen_bases.add(base_id)
            seen_names.add(name_key)
            seen_docs: set[str] = set()
            normalized_docs = []
            for doc in docs:
                if not isinstance(doc, dict) or set(doc) != {
                    "id", "source_path", "file_name", "format", "content_sha256",
                    "size_bytes", "character_count", "ingested_at",
                }:
                    raise KnowledgeStoreError("Knowledge index contains an invalid document entry.")
                doc_id = doc.get("id")
                source_path = doc.get("source_path")
                file_name = doc.get("file_name")
                suffix = Path(file_name).suffix.lower() if isinstance(file_name, str) else ""
                if (
                    not isinstance(doc_id, str) or not _DOCUMENT_ID.fullmatch(doc_id) or doc_id in seen_docs
                    or not isinstance(source_path, str) or source_path != _workspace_path(source_path)
                    or not isinstance(file_name, str) or not file_name or len(file_name) > 255
                    or "/" in file_name or "\\" in file_name or suffix not in _ALLOWED_SUFFIXES
                    or doc.get("format") != suffix[1:]
                    or not isinstance(doc.get("content_sha256"), str)
                    or not re.fullmatch(r"[a-f0-9]{64}", doc["content_sha256"])
                    or isinstance(doc.get("size_bytes"), bool) or not isinstance(doc.get("size_bytes"), int)
                    or not 0 <= doc["size_bytes"] <= MAX_DOCUMENT_BYTES
                    or isinstance(doc.get("character_count"), bool) or not isinstance(doc.get("character_count"), int)
                    or not 0 <= doc["character_count"] <= MAX_DOCUMENT_BYTES
                    or not isinstance(doc.get("ingested_at"), str) or len(doc["ingested_at"]) > 64
                ):
                    raise KnowledgeStoreError("Knowledge index contains an invalid document entry.")
                seen_docs.add(doc_id)
                normalized_docs.append(doc)
            records.append({**base, "documents": normalized_docs})
        return records

    def _write_index(self, records: list[dict[str, Any]]) -> None:
        serialized = json.dumps(
            {"schema_version": SCHEMA_VERSION, "bases": records},
            indent=2, ensure_ascii=False, allow_nan=False,
        ) + "\n"
        if len(serialized.encode("utf-8")) > MAX_INDEX_BYTES:
            raise KnowledgeStoreCapacityError("Knowledge metadata exceeds the 4 MiB workspace index limit.")
        atomic_write_text(self.index_path, serialized)

    @staticmethod
    def _public_base(base: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": base["id"], "name": base["name"], "description": base["description"],
            "created_at": base["created_at"], "updated_at": base["updated_at"],
            "document_count": len(base["documents"]),
            "total_bytes": sum(item["size_bytes"] for item in base["documents"]),
        }

    def list_bases(self) -> dict[str, Any]:
        self._ensure_layout(create=True)
        with file_lock(self.lock_path):
            records = self._read_index()
        return {
            "schema_version": SCHEMA_VERSION,
            "knowledge_bases": [self._public_base(item) for item in records],
            "total": len(records),
            "limits": {"max_knowledge_bases": MAX_KNOWLEDGE_BASES, "max_documents_per_base": MAX_DOCUMENTS_PER_BASE},
        }

    def create_base(self, name: str, description: str = "") -> dict[str, Any]:
        normalized_name = _validate_name(name, maximum=80, label="Knowledge base name")
        normalized_description = _validate_name(description, maximum=500, label="Knowledge base description") if description else ""
        self._ensure_layout(create=True)
        with file_lock(self.lock_path):
            records = self._read_index()
            if len(records) >= MAX_KNOWLEDGE_BASES:
                raise KnowledgeStoreCapacityError("The workspace already has the maximum number of knowledge bases.")
            if any(item["name"].casefold() == normalized_name.casefold() for item in records):
                raise KnowledgeStoreError("A knowledge base with this name already exists.")
            now = _now()
            base = {
                "id": uuid.uuid4().hex, "name": normalized_name, "description": normalized_description,
                "created_at": now, "updated_at": now, "documents": [],
            }
            records.insert(0, base)
            self._write_index(records)
        return self._public_base(base)

    def get_base(self, base_id: str) -> dict[str, Any] | None:
        if not isinstance(base_id, str) or not _BASE_ID.fullmatch(base_id):
            raise KnowledgeStoreError("Knowledge base ID is invalid.")
        self._ensure_layout(create=True)
        with file_lock(self.lock_path):
            base = next((item for item in self._read_index() if item["id"] == base_id), None)
        return self._public_base(base) if base is not None else None

    def list_documents(self, base_id: str) -> dict[str, Any]:
        self._ensure_layout(create=True)
        with file_lock(self.lock_path):
            base = next((item for item in self._read_index() if item["id"] == base_id), None)
        if base is None:
            raise KeyError("Knowledge base not found.")
        return {
            "schema_version": SCHEMA_VERSION, "knowledge_base": self._public_base(base),
            "documents": [dict(item) for item in base["documents"]], "total": len(base["documents"]),
        }

    def ingest_document(self, base_id: str, source_path: str) -> dict[str, Any]:
        if not isinstance(base_id, str) or not _BASE_ID.fullmatch(base_id):
            raise KnowledgeStoreError("Knowledge base ID is invalid.")
        relative = _workspace_path(source_path)
        suffix = Path(relative).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise KnowledgeStoreError("Supported document formats are TXT, Markdown, RST, CSV, JSON, and JSONL.")
        try:
            source = find_existing_relative_path(self.workspace_root, relative, allow_directories=False)
            if source.stat().st_size > MAX_DOCUMENT_BYTES:
                raise KnowledgeStoreCapacityError("Document exceeds the 8 MiB ingestion limit.")
            with open_regular_file(source) as handle:
                raw = handle.read(MAX_DOCUMENT_BYTES + 1)
            if len(raw) > MAX_DOCUMENT_BYTES:
                raise KnowledgeStoreCapacityError("Document exceeds the 8 MiB ingestion limit.")
            text = raw.decode("utf-8-sig")
        except KnowledgeStoreError:
            raise
        except (SafePathError, OSError, UnicodeError):
            raise KnowledgeStoreError("Document could not be read safely as a UTF-8 workspace file.") from None
        if "\x00" in text:
            raise KnowledgeStoreError("Document contains null bytes and cannot be ingested as text.")
        if any(ord(char) < 32 and ord(char) not in _ALLOWED_TEXT_CONTROLS for char in text):
            raise KnowledgeStoreError("Document contains unsupported control characters.")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise KnowledgeStoreError("Empty documents cannot be ingested.")
        content_bytes = text.encode("utf-8")
        document = {
            "id": uuid.uuid4().hex, "source_path": relative, "file_name": source.name,
            "format": suffix[1:], "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
            "size_bytes": len(raw), "character_count": len(text), "ingested_at": _now(),
        }
        self._ensure_layout(create=True)
        destination = self.documents_root / f"{document['id']}.txt"
        with file_lock(self.lock_path):
            records = self._read_index()
            base = next((item for item in records if item["id"] == base_id), None)
            if base is None:
                raise KeyError("Knowledge base not found.")
            if len(base["documents"]) >= MAX_DOCUMENTS_PER_BASE:
                raise KnowledgeStoreCapacityError("Knowledge base reached its document limit.")
            total_bytes = sum(doc["size_bytes"] for item in records for doc in item["documents"])
            if total_bytes + len(raw) > MAX_TOTAL_DOCUMENT_BYTES:
                raise KnowledgeStoreCapacityError("Workspace knowledge documents exceed the 1 GiB storage limit.")
            if destination.exists() or destination.is_symlink():
                raise KnowledgeStoreError("Generated knowledge document ID already exists.")
            atomic_write_text(destination, text)
            base["documents"].append(document)
            base["updated_at"] = document["ingested_at"]
            try:
                self._write_index(records)
            except BaseException:
                try:
                    destination.unlink()
                except OSError:
                    pass
                raise
        return dict(document)

    def read_document(self, base_id: str, document_id: str) -> str:
        if not isinstance(document_id, str) or not _DOCUMENT_ID.fullmatch(document_id):
            raise KnowledgeStoreError("Document ID is invalid.")
        with file_lock(self.lock_path):
            base = next((item for item in self._read_index() if item["id"] == base_id), None)
            if base is None:
                raise KeyError("Knowledge base not found.")
            document = next((item for item in base["documents"] if item["id"] == document_id), None)
            if document is None:
                raise KeyError("Knowledge document not found.")
            path = self.documents_root / f"{document_id}.txt"
            if path.is_symlink() or not path.is_file():
                raise KnowledgeStoreError("Stored knowledge document is unavailable.")
            try:
                with open_regular_file(path) as handle:
                    raw = handle.read(MAX_DOCUMENT_BYTES + 1)
                if len(raw) > MAX_DOCUMENT_BYTES:
                    raise KnowledgeStoreError("Stored knowledge document exceeds its size limit.")
                text = raw.decode("utf-8")
            except KnowledgeStoreError:
                raise
            except (OSError, UnicodeError, SafePathError):
                raise KnowledgeStoreError("Stored knowledge document could not be read safely.") from None
            if hashlib.sha256(raw).hexdigest() != document["content_sha256"]:
                raise KnowledgeStoreError("Stored knowledge document failed its SHA-256 integrity check.")
            return text

    def remove_document(self, base_id: str, document_id: str) -> dict[str, Any] | None:
        if not isinstance(document_id, str) or not _DOCUMENT_ID.fullmatch(document_id):
            raise KnowledgeStoreError("Document ID is invalid.")
        self._ensure_layout(create=False)
        with file_lock(self.lock_path):
            records = self._read_index()
            base = next((item for item in records if item["id"] == base_id), None)
            if base is None:
                raise KeyError("Knowledge base not found.")
            document = next((item for item in base["documents"] if item["id"] == document_id), None)
            if document is None:
                return None
            path = self.documents_root / f"{document_id}.txt"
            if path.is_symlink():
                raise KnowledgeStoreError("Stored knowledge document is a symbolic link.")
            base["documents"] = [item for item in base["documents"] if item["id"] != document_id]
            base["updated_at"] = _now()
            self._write_index(records)
            if path.exists():
                path.unlink()
            return dict(document)

    def remove_base(self, base_id: str) -> dict[str, Any] | None:
        if not isinstance(base_id, str) or not _BASE_ID.fullmatch(base_id):
            raise KnowledgeStoreError("Knowledge base ID is invalid.")
        self._ensure_layout(create=False)
        with file_lock(self.lock_path):
            records = self._read_index()
            base = next((item for item in records if item["id"] == base_id), None)
            if base is None:
                return None
            paths = []
            for document in base["documents"]:
                path = self.documents_root / f"{document['id']}.txt"
                if path.is_symlink():
                    raise KnowledgeStoreError("Stored knowledge document is a symbolic link.")
                paths.append(path)
            records = [item for item in records if item["id"] != base_id]
            self._write_index(records)
            for path in paths:
                if path.exists():
                    path.unlink()
            return self._public_base(base)
