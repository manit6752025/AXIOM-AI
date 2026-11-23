"""Immutable, content-addressed JSONL dataset snapshots with verifiable lineage."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from axiom.core.storage import atomic_write_text, file_lock
from axiom.path_safety import SafePathError, open_regular_file

SCHEMA_VERSION = 1
MAX_SNAPSHOTS = 250
MAX_INDEX_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_DATASET_BYTES = 256 * 1024 * 1024
MAX_DATASET_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_SNAPSHOT_BYTES = 10 * 1024 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
_SNAPSHOT_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCK_NAME = ".dataset-snapshots.lock"
_MANIFEST_KEYS = {
    "schema_version", "snapshot_id", "created_at", "source_path", "file_name",
    "format", "file_size_bytes", "sha256", "object_name",
    "parent_snapshot_id", "parent_manifest_sha256",
}
_RECORD_KEYS = {
    "sequence", "manifest", "manifest_sha256",
    "previous_record_sha256", "record_sha256",
}


class DatasetSnapshotError(ValueError):
    """An invalid, unsafe, or inconsistent dataset snapshot operation."""


class DatasetSnapshotCapacityError(DatasetSnapshotError):
    """A configured dataset snapshot retention or storage limit was reached."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _validate_source_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise DatasetSnapshotError("Dataset snapshot source path is invalid.")
    if value.startswith("/") or "\\" in value or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise DatasetSnapshotError("Dataset snapshot source path must be workspace-relative.")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise DatasetSnapshotError("Dataset snapshot source path contains an unsafe component.")
    if not parts[-1].lower().endswith(".jsonl"):
        raise DatasetSnapshotError("Only JSONL datasets can be snapshotted.")
    return value


def _validate_file_name(value: Any) -> str:
    if (
        not isinstance(value, str) or not value or len(value) > 255
        or value in {".", ".."} or "/" in value or "\\" in value
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
        or not value.lower().endswith(".jsonl")
    ):
        raise DatasetSnapshotError("Dataset snapshot file name is invalid.")
    return value


def _validate_manifest(value: Any, previous: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise DatasetSnapshotError("Dataset snapshot manifest contains unsupported fields.")
    if type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise DatasetSnapshotError("Unsupported dataset snapshot schema version.")
    snapshot_id = value.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise DatasetSnapshotError("Dataset snapshot identifier is invalid.")
    _validate_source_path(value.get("source_path"))
    _validate_file_name(value.get("file_name"))
    if value.get("format") != "jsonl":
        raise DatasetSnapshotError("Dataset snapshot format is unsupported.")
    size = value.get("file_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= MAX_DATASET_BYTES:
        raise DatasetSnapshotError("Dataset snapshot size is outside its supported range.")
    digest = value.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise DatasetSnapshotError("Dataset snapshot content hash is invalid.")
    if value.get("object_name") != f"{digest}.jsonl":
        raise DatasetSnapshotError("Dataset snapshot object reference does not match its content hash.")
    created_at = value.get("created_at")
    if not isinstance(created_at, str) or not created_at or len(created_at) > 40:
        raise DatasetSnapshotError("Dataset snapshot timestamp is invalid.")
    try:
        moment = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        raise DatasetSnapshotError("Dataset snapshot timestamp is invalid.") from None
    if moment.tzinfo is None:
        raise DatasetSnapshotError("Dataset snapshot timestamp must include a timezone.")
    parent_id = value.get("parent_snapshot_id")
    parent_hash = value.get("parent_manifest_sha256")
    if parent_id is None:
        if parent_hash is not None:
            raise DatasetSnapshotError("A source snapshot cannot include a parent manifest hash.")
    else:
        parent = previous.get(parent_id) if isinstance(parent_id, str) else None
        if parent is None or not _SNAPSHOT_ID.fullmatch(parent_id):
            raise DatasetSnapshotError("Dataset snapshot parent must reference an earlier snapshot.")
        if parent_hash != parent["manifest_sha256"]:
            raise DatasetSnapshotError("Dataset snapshot parent manifest hash does not match its parent.")
    return value


def _index_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "records": records,
        "head_sha256": records[-1]["record_sha256"] if records else "0" * 64,
    }


class DatasetSnapshotStore:
    """Store immutable JSONL objects and an append-only, hash-chained manifest index."""

    def __init__(self, workspace_root: str | Path) -> None:
        supplied_root = Path(workspace_root)
        if supplied_root.is_symlink():
            raise DatasetSnapshotError("Dataset snapshot workspace must not be a symbolic link.")
        try:
            self.workspace_root = supplied_root.resolve(strict=True)
        except (OSError, RuntimeError):
            raise DatasetSnapshotError("Dataset snapshot workspace is unavailable.") from None
        self.metadata = self.workspace_root / ".axiom"
        self.root = self.metadata / "dataset-snapshots"
        self.objects = self.root / "objects"
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / _LOCK_NAME

    def _ensure_layout(self, *, create: bool) -> None:
        if self.workspace_root.is_symlink() or not self.workspace_root.is_dir():
            raise DatasetSnapshotError("Dataset snapshot workspace must be an existing regular directory.")
        for path in (self.metadata, self.root, self.objects, self.index_path, self.lock_path):
            if path.is_symlink():
                raise DatasetSnapshotError("Dataset snapshot storage paths must not be symbolic links.")
        if create:
            self.metadata.mkdir(mode=0o700, exist_ok=True)
            self.root.mkdir(mode=0o700, exist_ok=True)
            self.objects.mkdir(mode=0o700, exist_ok=True)
        for directory in (self.metadata, self.root, self.objects):
            if directory.exists() and not directory.is_dir():
                raise DatasetSnapshotError("Dataset snapshot storage path is not a directory.")

    def _read_index(self) -> list[dict[str, Any]]:
        self._ensure_layout(create=False)
        if not self.index_path.exists():
            return []
        try:
            info = self.index_path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_INDEX_BYTES:
                raise DatasetSnapshotError("Dataset snapshot index is not a bounded regular file.")
            with open_regular_file(self.index_path) as stream:
                raw = json.loads(stream.read(MAX_INDEX_BYTES + 1).decode("utf-8"))
        except DatasetSnapshotError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, SafePathError):
            raise DatasetSnapshotError("Dataset snapshot index could not be read safely.") from None
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "records", "head_sha256"}:
            raise DatasetSnapshotError("Dataset snapshot index is malformed.")
        if type(raw.get("schema_version")) is not int or raw["schema_version"] != SCHEMA_VERSION:
            raise DatasetSnapshotError("Unsupported dataset snapshot index schema version.")
        records = raw.get("records")
        if not isinstance(records, list) or len(records) > MAX_SNAPSHOTS:
            raise DatasetSnapshotError("Dataset snapshot index exceeds its record limit.")
        previous_hash = "0" * 64
        previous: dict[str, dict[str, Any]] = {}
        verified: list[dict[str, Any]] = []
        for sequence, record in enumerate(records, start=1):
            if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
                raise DatasetSnapshotError("Dataset snapshot index contains an invalid record.")
            manifest = _validate_manifest(record.get("manifest"), previous)
            snapshot_id = manifest["snapshot_id"]
            if snapshot_id in previous:
                raise DatasetSnapshotError("Dataset snapshot index contains a duplicate identifier.")
            if type(record.get("sequence")) is not int or record["sequence"] != sequence:
                raise DatasetSnapshotError("Dataset snapshot index sequence is invalid.")
            if record.get("previous_record_sha256") != previous_hash:
                raise DatasetSnapshotError("Dataset snapshot lineage chain is broken.")
            manifest_hash = record.get("manifest_sha256")
            if not isinstance(manifest_hash, str) or not _SHA256.fullmatch(manifest_hash) or _digest(manifest) != manifest_hash:
                raise DatasetSnapshotError("Dataset snapshot manifest integrity verification failed.")
            record_body = {key: record[key] for key in _RECORD_KEYS if key != "record_sha256"}
            record_hash = record.get("record_sha256")
            if not isinstance(record_hash, str) or not _SHA256.fullmatch(record_hash) or _digest(record_body) != record_hash:
                raise DatasetSnapshotError("Dataset snapshot record integrity verification failed.")
            previous[snapshot_id] = {"manifest": manifest, "manifest_sha256": manifest_hash, "record_sha256": record_hash}
            previous_hash = record_hash
            verified.append(record)
        expected_head = verified[-1]["record_sha256"] if verified else "0" * 64
        if raw.get("head_sha256") != expected_head:
            raise DatasetSnapshotError("Dataset snapshot index head hash is invalid.")
        return verified

    def _write_index(self, records: list[dict[str, Any]]) -> None:
        payload = json.dumps(_index_payload(records), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if len(payload.encode("utf-8")) > MAX_INDEX_BYTES:
            raise DatasetSnapshotCapacityError("Dataset snapshot metadata reached its 2 MiB limit.")
        if self.index_path.is_symlink():
            raise DatasetSnapshotError("Dataset snapshot index must not be a symbolic link.")
        atomic_write_text(self.index_path, payload)
        if os.name != "nt":
            os.chmod(self.index_path, 0o600)

    def _object_path(self, digest: str) -> Path:
        if not _SHA256.fullmatch(digest):
            raise DatasetSnapshotError("Dataset snapshot content hash is invalid.")
        return self.objects / f"{digest}.jsonl"

    def _hash_object(self, digest: str, expected_size: int) -> bool:
        target = self._object_path(digest)
        if target.is_symlink() or not target.exists():
            return False
        try:
            info = target.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != expected_size:
                return False
            with open_regular_file(target) as stream:
                hasher = hashlib.sha256()
                total = 0
                while chunk := stream.read(HASH_CHUNK_BYTES):
                    total += len(chunk)
                    if total > MAX_DATASET_BYTES:
                        return False
                    hasher.update(chunk)
        except (OSError, ValueError, SafePathError):
            return False
        return total == expected_size and hasher.hexdigest() == digest

    def create_snapshot(
        self,
        source: BinaryIO,
        *,
        source_path: str,
        file_name: str,
        max_bytes: int = DEFAULT_MAX_DATASET_BYTES,
        parent_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        """Copy one stable JSONL source into a content-addressed immutable object."""
        source_path = _validate_source_path(source_path)
        file_name = _validate_file_name(file_name)
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= MAX_DATASET_BYTES:
            raise DatasetSnapshotError(f"Snapshot byte limit must be between 1 and {MAX_DATASET_BYTES}.")
        try:
            initial = os.fstat(source.fileno())
        except (AttributeError, OSError):
            raise DatasetSnapshotError("Dataset snapshot requires an open regular file.") from None
        if not stat.S_ISREG(initial.st_mode):
            raise DatasetSnapshotError("Dataset snapshot source must be a regular file.")
        if initial.st_size > max_bytes:
            raise DatasetSnapshotCapacityError("Dataset exceeds the configured snapshot byte limit.")
        if initial.st_size > MAX_DATASET_BYTES:
            raise DatasetSnapshotCapacityError("Dataset exceeds the maximum snapshot size.")
        try:
            source.seek(0)
        except (AttributeError, OSError):
            raise DatasetSnapshotError("Dataset snapshot source could not be rewound safely.") from None

        self._ensure_layout(create=True)
        if self.lock_path.is_symlink():
            raise DatasetSnapshotError("Dataset snapshot lock path must not be a symbolic link.")
        temporary_path: Path | None = None
        try:
            with file_lock(self.lock_path):
                self._ensure_layout(create=True)
                records = self._read_index()
                if len(records) >= MAX_SNAPSHOTS:
                    raise DatasetSnapshotCapacityError(f"Dataset snapshot history is limited to {MAX_SNAPSHOTS} records.")
                previous = {
                    item["manifest"]["snapshot_id"]: {
                        "manifest": item["manifest"],
                        "manifest_sha256": item["manifest_sha256"],
                        "record_sha256": item["record_sha256"],
                    }
                    for item in records
                }
                parent = None
                if parent_snapshot_id is not None:
                    if not isinstance(parent_snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(parent_snapshot_id):
                        raise DatasetSnapshotError("Parent snapshot identifier is invalid.")
                    parent = previous.get(parent_snapshot_id)
                    if parent is None:
                        raise DatasetSnapshotError("Parent snapshot must be an earlier snapshot in this workspace.")

                descriptor, temporary_name = tempfile.mkstemp(prefix=".snapshot-", suffix=".tmp", dir=self.objects)
                temporary_path = Path(temporary_name)
                hasher = hashlib.sha256()
                total = 0
                try:
                    with os.fdopen(descriptor, "wb") as output:
                        while True:
                            chunk = source.read(min(HASH_CHUNK_BYTES, max_bytes - total + 1))
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_bytes or total > MAX_DATASET_BYTES:
                                raise DatasetSnapshotCapacityError("Dataset exceeds the configured snapshot byte limit.")
                            hasher.update(chunk)
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                except BaseException:
                    raise

                final = os.fstat(source.fileno())
                if (
                    initial.st_dev != final.st_dev
                    or initial.st_ino != final.st_ino
                    or initial.st_size != final.st_size
                    or initial.st_mtime_ns != final.st_mtime_ns
                    or total != initial.st_size
                ):
                    raise DatasetSnapshotError("Dataset changed during snapshot creation; no snapshot was committed.")

                content_hash = hasher.hexdigest()
                object_path = self._object_path(content_hash)
                if object_path.is_symlink():
                    raise DatasetSnapshotError("Dataset snapshot object path must not be a symbolic link.")

                indexed_objects = {
                    item["manifest"]["sha256"]: item["manifest"]["file_size_bytes"]
                    for item in records
                }
                if content_hash in indexed_objects and indexed_objects[content_hash] != total:
                    raise DatasetSnapshotError("Dataset snapshot index reuses a content hash with a different size.")
                if content_hash in indexed_objects or object_path.exists():
                    if not self._hash_object(content_hash, total):
                        raise DatasetSnapshotError("An existing content-addressed dataset object failed integrity verification.")
                    temporary_path.unlink(missing_ok=True)
                    temporary_path = None
                else:
                    total_existing = sum(indexed_objects.values())
                    if total_existing + total > MAX_TOTAL_SNAPSHOT_BYTES:
                        raise DatasetSnapshotCapacityError("Workspace dataset snapshot storage would exceed 10 GiB.")
                    os.replace(temporary_path, object_path)
                    temporary_path = None
                    if os.name != "nt":
                        os.chmod(object_path, 0o400)

                snapshot_id = uuid.uuid4().hex
                parent_manifest = parent["manifest"] if parent else None
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "snapshot_id": snapshot_id,
                    "created_at": _timestamp(),
                    "source_path": source_path,
                    "file_name": file_name,
                    "format": "jsonl",
                    "file_size_bytes": total,
                    "sha256": content_hash,
                    "object_name": f"{content_hash}.jsonl",
                    "parent_snapshot_id": parent_snapshot_id,
                    "parent_manifest_sha256": parent["manifest_sha256"] if parent_manifest else None,
                }
                previous_hash = records[-1]["record_sha256"] if records else "0" * 64
                body = {
                    "sequence": len(records) + 1,
                    "manifest": manifest,
                    "manifest_sha256": _digest(manifest),
                    "previous_record_sha256": previous_hash,
                }
                record = {**body, "record_sha256": _digest(body)}
                records.append(record)
                self._write_index(records)
                return {"snapshot": record, "total": len(records)}
        except DatasetSnapshotError:
            raise
        except (OSError, TimeoutError):
            raise DatasetSnapshotError("Dataset snapshot could not be stored safely.") from None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def list(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise DatasetSnapshotError("Dataset snapshot page limit must be between 1 and 100.")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= MAX_SNAPSHOTS:
            raise DatasetSnapshotError("Dataset snapshot offset is outside its supported range.")
        records = self._read_index()
        ordered = list(reversed(records))
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshots": ordered[offset:offset + limit],
            "total": len(records),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(records),
            "immutable": True,
        }

    def verify(self, snapshot_id: str) -> dict[str, Any]:
        if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
            raise DatasetSnapshotError("Dataset snapshot identifier is invalid.")
        records = self._read_index()
        record = next((item for item in records if item["manifest"]["snapshot_id"] == snapshot_id), None)
        if record is None:
            raise FileNotFoundError("Dataset snapshot was not found.")
        manifest = record["manifest"]
        valid = self._hash_object(manifest["sha256"], manifest["file_size_bytes"])
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "valid": valid,
            "expected_sha256": manifest["sha256"],
            "file_size_bytes": manifest["file_size_bytes"],
            "manifest_sha256": record["manifest_sha256"],
            "lineage_valid": True,
            "immutable": True,
        }
