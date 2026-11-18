from __future__ import annotations

import io
import json
import threading

import pytest

from axiom.datasets.cleaner import clean_jsonl
from axiom.datasets.cleaning_preview import preview_jsonl_cleaning
from axiom.datasets.manifest import create_dataset_manifest
from axiom.datasets.profiler import profile_jsonl
from axiom.datasets.progress import DatasetScanCancelled


def _dataset_bytes(rows: int = 600) -> bytes:
    return b"".join(
        (json.dumps({"id": index, "text": "bounded scan"}) + "\n").encode("utf-8")
        for index in range(rows)
    )


def test_profiler_reports_row_and_byte_progress_and_can_cancel(tmp_path):
    source = tmp_path / "profile.jsonl"
    content = _dataset_bytes()
    source.write_bytes(content)
    progress = []

    report = profile_jsonl(
        source,
        on_progress=lambda rows, current, total: progress.append((rows, current, total)),
    )

    assert report["lines_scanned"] == 600
    assert progress
    assert progress[-1] == (600, len(content), len(content))

    cancel_event = threading.Event()

    def cancel_from_progress(rows, current, total):
        cancel_event.set()

    with pytest.raises(DatasetScanCancelled, match="cancelled"):
        profile_jsonl(
            source,
            on_progress=cancel_from_progress,
            cancel_event=cancel_event,
        )


def test_cleaning_preview_reports_progress_and_can_cancel_without_writes(tmp_path):
    content = _dataset_bytes()
    progress = []
    report = preview_jsonl_cleaning(
        io.BytesIO(content),
        file_name="preview.jsonl",
        file_size_bytes=len(content),
        on_progress=lambda rows, current, total: progress.append((rows, current, total)),
    )
    assert report["lines_scanned"] == 600
    assert progress[-1] == (600, len(content), len(content))

    cancel_event = threading.Event()
    with pytest.raises(DatasetScanCancelled, match="cancelled"):
        preview_jsonl_cleaning(
            io.BytesIO(content),
            file_name="cancel-preview.jsonl",
            file_size_bytes=len(content),
            on_progress=lambda *_: cancel_event.set(),
            cancel_event=cancel_event,
        )


def test_manifest_reports_chunk_progress_and_cancels_before_return():
    content = b"x" * (3 * 1024 * 1024)
    progress = []
    report = create_dataset_manifest(
        io.BytesIO(content),
        file_name="large.jsonl",
        relative_path="large.jsonl",
        file_size_bytes=len(content),
        on_progress=lambda current, total: progress.append((current, total)),
    )
    assert report["bytes_hashed"] == len(content)
    assert progress
    assert progress[-1] == (len(content), len(content))

    cancel_event = threading.Event()
    with pytest.raises(DatasetScanCancelled, match="cancelled"):
        create_dataset_manifest(
            io.BytesIO(content),
            file_name="large.jsonl",
            relative_path="large.jsonl",
            file_size_bytes=len(content),
            on_progress=lambda *_: cancel_event.set(),
            cancel_event=cancel_event,
        )


def test_cleaner_cancellation_removes_partial_temp_and_preserves_original(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "cleaned.jsonl"
    content = _dataset_bytes()
    source.write_bytes(content)
    cancel_event = threading.Event()

    with pytest.raises(DatasetScanCancelled, match="cancelled"):
        clean_jsonl(
            source,
            output,
            on_progress=lambda *_: cancel_event.set(),
            cancel_event=cancel_event,
        )

    assert source.read_bytes() == content
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]
