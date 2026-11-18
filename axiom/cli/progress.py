"""TTY-only progress surfaces for bounded local operations."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import signal
import threading

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from axiom.cli.errors import runtime_options

ScanProgress = Callable[[int, int, int], None]


@contextmanager
def scan_progress(
    description: str,
    total_bytes: int,
    *,
    enabled: bool = True,
) -> Iterator[ScanProgress | None]:
    """Show a transient byte/row progress bar on stderr when attached to a TTY.

    The callback is intentionally optional: JSON output, quiet mode, redirected
    streams, and tests run without a progress UI and receive identical results.
    """
    progress_console = Console(stderr=True)
    options = runtime_options()
    if not enabled or bool(options.get("quiet")) or not progress_console.is_terminal:
        yield None
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[rows]} rows"),
        console=progress_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=max(1, total_bytes), rows=0)

        def update(rows: int, completed_bytes: int, total: int | None) -> None:
            effective_total = max(1, total) if total is not None else None
            completed = max(0, completed_bytes)
            if effective_total is not None:
                completed = min(completed, effective_total)
            progress.update(
                task_id,
                total=effective_total,
                completed=completed,
                rows=rows,
            )

        yield update


@contextmanager
def cancellable_scan_progress(
    description: str,
    total_bytes: int,
    *,
    enabled: bool = True,
) -> Iterator[tuple[threading.Event, ScanProgress | None]]:
    """Yield a cooperative cancel event and a TTY-only row/byte progress callback."""
    cancel_event = threading.Event()
    previous_handler = None

    def request_cancel(signum: int, frame: object) -> None:
        if cancel_event.is_set():
            raise KeyboardInterrupt
        cancel_event.set()

    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_cancel)
    try:
        with scan_progress(description, total_bytes, enabled=enabled) as update:
            yield cancel_event, update
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)


@contextmanager
def cancellable_byte_progress(
    description: str,
    total_bytes: int,
    *,
    enabled: bool = True,
) -> Iterator[tuple[threading.Event, Callable[[int, int | None], None] | None]]:
    """Byte-transfer progress with a cancel event set by the first Ctrl-C."""
    cancel_event = threading.Event()
    previous_handler = None

    def request_cancel(signum: int, frame: object) -> None:
        if cancel_event.is_set():
            raise KeyboardInterrupt
        cancel_event.set()

    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_cancel)
    try:
        with byte_progress(description, total_bytes, enabled=enabled) as update:
            yield cancel_event, update
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)


@contextmanager
def byte_progress(
    description: str,
    total_bytes: int,
    *,
    enabled: bool = True,
) -> Iterator[Callable[[int, int | None], None] | None]:
    """Two-argument progress callback for chunked hash/download operations."""
    with scan_progress(description, total_bytes, enabled=enabled) as update:
        if update is None:
            yield None
        else:
            yield lambda completed, total: update(0, completed, total)


def report_cancelled(message: str) -> None:
    """Report Ctrl+C cancellation without contaminating stdout JSON."""
    options = runtime_options()
    if not bool(options.get("quiet")):
        Console(stderr=True).print(f"[yellow]{message}[/yellow]")
    raise SystemExit(130)
