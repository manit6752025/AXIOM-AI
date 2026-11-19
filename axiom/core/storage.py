"""Small filesystem primitives used by local AXIOM registries."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(
    path: Path,
    *,
    timeout: float = 10.0,
) -> Iterator[None]:
    """Serialize updates to a local file across threads and processes.

    The lock file is deliberately retained after use.  The operating system
    releases the advisory lock if a process exits, so a crashed writer cannot
    leave a stale lock path that blocks future updates.
    """

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)

            deadline = time.monotonic() + timeout
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
                    time.sleep(0.01)

            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path: Path, content: str) -> None:
    """Replace *path* atomically after fully writing and syncing its content."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    file_descriptor: int | None = None

    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )

        handle = os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        file_descriptor = None

        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                __import__("logging").getLogger(__name__).debug(
                    "intentionally ignored exception", exc_info=True
                )
