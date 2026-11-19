"""Durable, bounded experiment metadata for local AXIOM workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axiom.core.storage import atomic_write_text, file_lock


class ExperimentStore:
    """Store experiment plans as an atomic JSON document in ``.axiom``."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".axiom")
        self.records_file = self.root / "experiments.json"
        self.lock_file = self.root / ".experiments.lock"

    def _read(self) -> list[dict[str, Any]]:
        if self.root.is_symlink() or self.records_file.is_symlink():
            raise ValueError(f"Invalid experiment history path: {self.records_file}")
        if not self.records_file.exists():
            return []
        if not self.records_file.is_file() or self.records_file.stat().st_size > 4_000_000:
            raise ValueError(f"Invalid experiment history: {self.records_file}")

        try:
            data = json.loads(self.records_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid experiment history: {self.records_file}") from exc
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ValueError(f"Invalid experiment history: {self.records_file}")
        for record in data:
            schema_version = record.get("schema_version", 1)
            if type(schema_version) is not int or schema_version != self.SCHEMA_VERSION:
                raise ValueError(f"Unsupported experiment schema version: {schema_version}")
            record["schema_version"] = self.SCHEMA_VERSION
            record.setdefault("tags", [])
            if not isinstance(record["tags"], list) or any(not isinstance(tag, str) for tag in record["tags"]):
                raise ValueError(f"Invalid experiment metadata: {self.records_file}")
        return data

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("Experiment history limit must be between 1 and 1000.")
        return self._read()[:limit]

    def list_page(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        query: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("Experiment page limit must be between 1 and 1000.")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 1_000_000:
            raise ValueError("Experiment page offset must be between 0 and 1000000.")
        records = self._read()
        if query:
            needle = query.casefold()
            records = [
                record for record in records
                if needle in " ".join(str(record.get(field) or "") for field in ("name", "model", "dataset", "notes")).casefold()
                or any(needle in str(item).casefold() for item in record.get("tags", []))
            ]
        if tag:
            tag_needle = tag.casefold()
            records = [
                record for record in records
                if any(str(item).casefold() == tag_needle for item in record.get("tags", []))
            ]
        total = len(records)
        return {
            "experiments": records[offset:offset + limit],
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        }

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        return next(
            (record for record in self._read() if record.get("id") == experiment_id),
            None,
        )

    def add(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("Experiment records must include a string ID.")
        schema_version = record.get("schema_version", self.SCHEMA_VERSION)
        if type(schema_version) is not int or schema_version != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported experiment schema version: {schema_version}")
        record = {**record, "schema_version": self.SCHEMA_VERSION}

        if self.root.is_symlink() or self.records_file.is_symlink():
            raise ValueError(f"Invalid experiment history path: {self.records_file}")
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            records = self._read()
            if any(existing.get("id") == record["id"] for existing in records):
                raise ValueError(f"Experiment already exists: {record['id']}")
            records.insert(0, record)
            records = records[:1000]
            serialized = json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False)
            atomic_write_text(self.records_file, serialized + "\n")

    def update_metadata(
        self,
        experiment_id: str,
        *,
        name: str,
        notes: str | None,
        tags: list[str],
    ) -> dict[str, Any] | None:
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("Experiment name must contain 1 to 120 characters.")
        if notes is not None and (not isinstance(notes, str) or len(notes) > 1000):
            raise ValueError("Experiment notes cannot exceed 1000 characters.")
        if not isinstance(tags, list) or len(tags) > 20 or any(
            not isinstance(tag, str) or not tag.strip() or len(tag.strip()) > 64 for tag in tags
        ):
            raise ValueError("Experiments support at most 20 non-empty tags of 64 characters or fewer.")
        normalized_tags = [tag.strip() for tag in tags]
        if len({tag.casefold() for tag in normalized_tags}) != len(normalized_tags):
            raise ValueError("Experiment tags must be unique.")
        if self.root.is_symlink() or self.records_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid experiment history path: {self.records_file}")
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            records = self._read()
            for index, current in enumerate(records):
                if current.get("id") != experiment_id:
                    continue
                updated = {**current, "name": name.strip(), "notes": notes, "tags": normalized_tags}
                records[index] = updated
                self._write_records(records)
                return updated
        return None

    def delete(self, experiment_id: str) -> dict[str, Any] | None:
        if self.root.is_symlink() or self.records_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid experiment history path: {self.records_file}")
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            records = self._read()
            for index, record in enumerate(records):
                if record.get("id") == experiment_id:
                    removed = records.pop(index)
                    self._write_records(records)
                    return removed
        return None

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        serialized = json.dumps(records[:1000], indent=2, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) > 4_000_000:
            raise ValueError("Experiment history exceeds the 4 MB workspace limit.")
        atomic_write_text(self.records_file, serialized + "\n")
