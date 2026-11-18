"""TTY-only progress surfaces for bounded local operations."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn

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
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[rows]} rows"),
        console=progress_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=max(1, total_bytes), rows=0)

        def update(rows: int, completed_bytes: int, total: int) -> None:
            progress.update(
                task_id,
                total=max(1, total),
                completed=min(max(0, completed_bytes), max(1, total)),
                rows=rows,
            )

        yield update


@contextmanager
def byte_progress(
    description: str,
    total_bytes: int,
    *,
    enabled: bool = True,
) -> Iterator[Callable[[int, int], None] | None]:
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
