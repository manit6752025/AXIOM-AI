"""Shared progress and cooperative-cancellation contract for local dataset scans."""
from __future__ import annotations

from threading import Event


class DatasetScanCancelled(RuntimeError):
    """Raised when a local dataset scan receives a cancellation request."""


def check_scan_cancelled(cancel_event: Event | None) -> None:
    """Stop a scan at the next safe row/chunk boundary."""
    if cancel_event is not None and cancel_event.is_set():
        raise DatasetScanCancelled("Dataset scan was cancelled.")
