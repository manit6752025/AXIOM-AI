"""Bounded local persistence for immutable evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axiom.core.storage import atomic_write_text, file_lock


class EvaluationStore:
    SCHEMA_VERSION = 1
    MAX_RECORDS = 1000
    MAX_BYTES = 4_000_000

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".axiom")
        self.records_file = self.root / "evaluations.json"
        self.lock_file = self.root / ".evaluations.lock"

    def _read(self) -> list[dict[str, Any]]:
        self._check_paths()
        if not self.records_file.exists():
            return []
        if not self.records_file.is_file() or self.records_file.stat().st_size > self.MAX_BYTES:
            raise ValueError(f"Invalid evaluation history: {self.records_file}")
        try:
            records = json.loads(self.records_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid evaluation history: {self.records_file}") from exc
        if not isinstance(records, list) or len(records) > self.MAX_RECORDS or any(
            not isinstance(record, dict) for record in records
        ):
            raise ValueError(f"Invalid evaluation history: {self.records_file}")
        seen_ids: set[str] = set()
        for record in records:
            run_id = record.get("id")
            if not isinstance(run_id, str) or not run_id or run_id in seen_ids:
                raise ValueError(f"Invalid or duplicate evaluation run ID in {self.records_file}")
            seen_ids.add(run_id)
            version = record.get("schema_version", self.SCHEMA_VERSION)
            if type(version) is not int or version != self.SCHEMA_VERSION:
                raise ValueError(f"Unsupported evaluation schema version: {version}")
            record["schema_version"] = self.SCHEMA_VERSION
        return records

    def list_page(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        after_run_id: str | None = None,
        baselines_only: bool = False,
    ) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Evaluation page limit must be between 1 and 1000.")
        if type(offset) is not int or not 0 <= offset <= 1_000_000:
            raise ValueError("Evaluation page offset must be between 0 and 1000000.")
        if type(baselines_only) is not bool:
            raise ValueError("Evaluation baseline filter must be a boolean.")
        if after_run_id is not None:
            if not isinstance(after_run_id, str) or not after_run_id:
                raise ValueError("Evaluation history cursor must be a non-empty run ID.")
            if offset != 0:
                raise ValueError("Evaluation history cursor cannot be combined with an offset.")
        records = self._read()
        if baselines_only:
            records = [record for record in records if record.get("baseline") is True]
        if after_run_id is not None:
            cursor_index = next(
                (index for index, record in enumerate(records) if record.get("id") == after_run_id),
                None,
            )
            if cursor_index is None:
                raise KeyError("Evaluation history cursor was not found.")
            offset = cursor_index + 1
        return {
            "runs": records[offset:offset + limit],
            "total": len(records),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(records),
        }

    def get(self, run_id: str) -> dict[str, Any] | None:
        return next((record for record in self._read() if record.get("id") == run_id), None)

    def add(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError("Evaluation reports must include a non-empty string ID.")
        version = record.get("schema_version", self.SCHEMA_VERSION)
        if type(version) is not int or version != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported evaluation schema version: {version}")
        self._check_paths()
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            self._check_paths()
            records = self._read()
            if any(existing.get("id") == record["id"] for existing in records):
                raise ValueError(f"Evaluation report already exists: {record['id']}")
            self._write([{"schema_version": self.SCHEMA_VERSION, **record}, *records])

    def delete(self, run_id: str) -> dict[str, Any] | None:
        self._check_paths()
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            self._check_paths()
            records = self._read()
            for index, record in enumerate(records):
                if record.get("id") == run_id:
                    removed = records.pop(index)
                    self._write(records)
                    return removed
        return None

    def _check_paths(self) -> None:
        if self.root.is_symlink() or self.records_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid evaluation history path: {self.records_file}")

    def _write(self, records: list[dict[str, Any]]) -> None:
        serialized = json.dumps(
            records[:self.MAX_RECORDS], indent=2, ensure_ascii=False, allow_nan=False
        )
        if len(serialized.encode("utf-8")) > self.MAX_BYTES:
            raise ValueError("Evaluation history exceeds the 4 MB workspace limit.")
        atomic_write_text(self.records_file, serialized + "\n")
