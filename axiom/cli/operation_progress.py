"""Terminal progress and cooperative Ctrl-C cancellation for local operations."""
from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import Callable, Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


ProgressUpdate = Callable[[int, int | None, str | None], None]


@contextmanager
def operation_progress(
    description: str,
    *,
    enabled: bool | None = None,
) -> Iterator[tuple[threading.Event, ProgressUpdate]]:
    """Show progress on stderr when interactive and set an event on Ctrl-C.

    The operation checks the supplied event at row/chunk/process boundaries.
    A second Ctrl-C retains the normal immediate KeyboardInterrupt behaviour.
    """
    console = Console(stderr=True)
    show_progress = console.is_terminal if enabled is None else enabled
    cancel_event = threading.Event()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
        disable=not show_progress,
    )
    task_id = progress.add_task(description, total=None)
    previous_handler = None

    def request_cancel(signum: int, frame: object) -> None:
        if cancel_event.is_set():
            raise KeyboardInterrupt
        cancel_event.set()

    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_cancel)

    progress.start()
    try:
        def update(current: int, total: int | None = None, label: str | None = None) -> None:
            values: dict[str, object] = {"completed": max(0, current)}
            if total is not None:
                values["total"] = max(0, total)
            if label:
                values["description"] = label
            progress.update(task_id, **values)

        yield cancel_event, update
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
        progress.stop()


def scan_progress_callback(update: ProgressUpdate) -> Callable[[int, int, int], None]:
    """Adapt row/byte scan metrics to the shared progress task."""
    def notify(rows: int, current_bytes: int, total_bytes: int) -> None:
        update(current_bytes, total_bytes, f"Scanning · {rows:,} rows")
    return notify


def byte_progress_callback(update: ProgressUpdate) -> Callable[[int, int | None], None]:
    """Adapt byte-transfer progress to the shared progress task."""
    def notify(current_bytes: int, total_bytes: int | None) -> None:
        update(current_bytes, total_bytes)
    return notify
