from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from axiom.datasets.snapshots import (
    DatasetSnapshotCapacityError,
    DatasetSnapshotError,
    DatasetSnapshotStore,
)


def _source(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "data" / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _create(store: DatasetSnapshotStore, source: Path, **kwargs):
    with source.open("rb") as stream:
        return store.create_snapshot(
            stream,
            source_path="train.jsonl",
            file_name=source.name,
            **kwargs,
        )


def test_snapshot_stores_content_addressed_read_only_object_and_value_free_manifest(tmp_path: Path):
    content = b'{"text":"PRIVATE_SNAPSHOT_VALUE"}\n'
    source = _source(tmp_path, content)
    store = DatasetSnapshotStore(tmp_path)

    result = _create(store, source)
    record = result["snapshot"]
    manifest = record["manifest"]
    object_path = store.objects / manifest["object_name"]

    assert result["total"] == 1
    assert manifest["schema_version"] == 1
    assert manifest["file_size_bytes"] == len(content)
    assert manifest["sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest["object_name"] == manifest["sha256"] + ".jsonl"
    assert object_path.read_bytes() == content
    assert "PRIVATE_SNAPSHOT_VALUE" not in store.index_path.read_text(encoding="utf-8")
    if os.name != "nt":
        assert stat.S_IMODE(object_path.stat().st_mode) == 0o400
        assert stat.S_IMODE(store.index_path.stat().st_mode) == 0o600


def test_identical_content_is_deduplicated_and_parent_lineage_is_verifiable(tmp_path: Path):
    source = _source(tmp_path, b'{"value":1}\n')
    store = DatasetSnapshotStore(tmp_path)

    first = _create(store, source)["snapshot"]
    second = _create(store, source, parent_snapshot_id=first["manifest"]["snapshot_id"])["snapshot"]

    assert first["manifest"]["sha256"] == second["manifest"]["sha256"]
    assert first["manifest"]["object_name"] == second["manifest"]["object_name"]
    assert second["manifest"]["parent_snapshot_id"] == first["manifest"]["snapshot_id"]
    assert second["manifest"]["parent_manifest_sha256"] == first["manifest_sha256"]
    assert len(list(store.objects.iterdir())) == 1
    assert store.verify(second["manifest"]["snapshot_id"])["valid"] is True
    page = store.list()
    assert page["total"] == 2
    assert page["snapshots"][0]["manifest"]["snapshot_id"] == second["manifest"]["snapshot_id"]


def test_parent_must_be_a_valid_earlier_snapshot_in_the_same_workspace(tmp_path: Path):
    source = _source(tmp_path, b"{}\n")
    store = DatasetSnapshotStore(tmp_path)

    with pytest.raises(DatasetSnapshotError, match="earlier snapshot"):
        _create(store, source, parent_snapshot_id="a" * 32)
    with pytest.raises(DatasetSnapshotError, match="identifier"):
        _create(store, source, parent_snapshot_id="../invalid")


def test_snapshot_enforces_source_type_format_and_size_bounds(tmp_path: Path):
    source = _source(tmp_path, b"123456789")
    store = DatasetSnapshotStore(tmp_path)

    with source.open("rb") as stream:
        with pytest.raises(DatasetSnapshotCapacityError):
            store.create_snapshot(
                stream,
                source_path="train.jsonl",
                file_name="train.jsonl",
                max_bytes=4,
            )
    with source.open("rb") as stream:
        with pytest.raises(DatasetSnapshotError, match="JSONL"):
            store.create_snapshot(
                stream,
                source_path="train.csv",
                file_name="train.csv",
            )


def test_snapshot_rejects_unsafe_paths_and_file_names(tmp_path: Path):
    source = _source(tmp_path, b"{}\n")
    store = DatasetSnapshotStore(tmp_path)
    for unsafe in ("../train.jsonl", "/tmp/train.jsonl", "sub\\train.jsonl", "data//train.jsonl"):
        with source.open("rb") as stream:
            with pytest.raises(DatasetSnapshotError):
                store.create_snapshot(stream, source_path=unsafe, file_name="train.jsonl")
    with source.open("rb") as stream:
        with pytest.raises(DatasetSnapshotError):
            store.create_snapshot(stream, source_path="train.jsonl", file_name="../train.jsonl")


def test_snapshot_rejects_tampered_manifest_index_and_hash_chain(tmp_path: Path):
    source = _source(tmp_path, b"{}\n")
    store = DatasetSnapshotStore(tmp_path)
    _create(store, source)

    payload = json.loads(store.index_path.read_text(encoding="utf-8"))
    payload["records"][0]["manifest"]["source_path"] = "changed.jsonl"
    store.index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetSnapshotError, match="integrity"):
        store.list()


def test_snapshot_verification_detects_modified_object_bytes(tmp_path: Path):
    source = _source(tmp_path, b'{"value":1}\n')
    store = DatasetSnapshotStore(tmp_path)
    snapshot = _create(store, source)["snapshot"]
    object_path = store.objects / snapshot["manifest"]["object_name"]
    object_path.chmod(0o600)
    object_path.write_bytes(b'{"value":2}\n')

    verified = store.verify(snapshot["manifest"]["snapshot_id"])
    assert verified["valid"] is False
    assert verified["lineage_valid"] is True


def test_snapshot_rejects_symlinked_workspace_metadata(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (workspace / ".axiom").symlink_to(external, target_is_directory=True)
    store = DatasetSnapshotStore(workspace)

    source = _source(tmp_path, b"{}\n")
    with source.open("rb") as stream:
        with pytest.raises(DatasetSnapshotError, match="symbolic link"):
            store.create_snapshot(stream, source_path="train.jsonl", file_name="train.jsonl")


def test_snapshot_list_validates_page_bounds(tmp_path: Path):
    source = _source(tmp_path, b"{}\n")
    store = DatasetSnapshotStore(tmp_path)
    _create(store, source)

    with pytest.raises(DatasetSnapshotError):
        store.list(limit=101)
    with pytest.raises(DatasetSnapshotError):
        store.list(offset=-1)
