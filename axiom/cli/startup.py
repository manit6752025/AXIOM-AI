"""Interactive-only startup output for the AXIOM command line.

The startup welcome is deliberately kept separate from command behavior.  It
can be used by the Typer entrypoint without making any command output depend
on terminal state.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.text import Text

NO_BANNER_ENV = "AXIOM_NO_BANNER"

COFFEE = "#99684E"
CREAM = "#DEC2AA"
AMBER = "#171310"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_SESSION_ENV_VARS = (
    "TERM_SESSION_ID",
    "WT_SESSION",
    "XDG_SESSION_ID",
    "KITTY_WINDOW_ID",
    "TMUX_PANE",
    "STY",
)
_MACHINE_OR_USAGE_FLAGS = frozenset(
    {
        "--help",
        "-h",
        "--json",
        "--jsonl",
        "--ndjson",
        "--machine-readable",
        "--show-completion",
        "--install-completion",
    }
)

# Five rows keep the logo legible in a narrow terminal while the solid-color
# extrusion rule gives it a small 3D sign effect without using gradients.
_LOGO_ROWS = (
    " ███   █   █  █████   ███   █   █ ",
    "█   █   █ █     █    █   █  ██ ██ ",
    "█████    █      █    █   █  █ █ █ ",
    "█   █   █ █     █    █   █  █   █ ",
    "█   █  █   █  █████   ███   █   █ ",
)


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUE_VALUES


def _is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _tty_name(stream: TextIO) -> str | None:
    try:
        return os.ttyname(stream.fileno())
    except (AttributeError, OSError, ValueError):
        return None


def _session_key(stream: TextIO, environ: Mapping[str, str]) -> str:
    session_token = next(
        (environ[name] for name in _SESSION_ENV_VARS if environ.get(name)),
        "",
    )

    try:
        session_leader = str(os.getsid(0))
    except (AttributeError, OSError):
        session_leader = ""

    tty = _tty_name(stream) or environ.get("TTY", "")
    identity = f"{session_token}|{session_leader}|{tty}"
    identity = identity or "interactive"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _default_marker_dir(environ: Mapping[str, str]) -> Path:
    runtime_dir = environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir) / "axiom"

    # tempfile.gettempdir() is user/session scoped on macOS and Windows, and
    # the uid-qualified directory below prevents a shared /tmp from becoming
    # a cross-user marker store on Unix.
    uid = str(getattr(os, "getuid", lambda: "user")())
    return Path(tempfile.gettempdir()) / f"axiom-startup-{uid}"


def _prepare_marker_dir(path: Path) -> bool:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat_result = path.stat()
    except OSError:
        return False

    if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
        return False

    if hasattr(os, "getuid") and stat_result.st_mode & 0o077:
        try:
            path.chmod(0o700)
        except OSError:
            return False

    return True


def _claim_session_marker(
    stream: TextIO,
    environ: Mapping[str, str],
    marker_dir: Path | None,
) -> bool:
    root = marker_dir or _default_marker_dir(environ)
    if not _prepare_marker_dir(root):
        return False

    marker = root / f"seen-{_session_key(stream, environ)}"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(marker, flags, 0o600)
    except FileExistsError:
        return False
    except OSError:
        return False

    try:
        os.close(descriptor)
    except OSError:
        return False
    return True


def suppress_for_arguments(arguments: Iterable[str]) -> bool:
    """Return whether startup output would pollute help or machine output."""

    for argument in arguments:
        if argument in _MACHINE_OR_USAGE_FLAGS:
            return True
        if argument.startswith("--format="):
            value = argument.partition("=")[2].strip().lower()
            if value in {"json", "jsonl", "ndjson", "raw", "machine"}:
                return True
        if argument.startswith("--output-format="):
            return True
    return False


def render_startup(stream: TextIO, *, include_next_steps: bool = False) -> None:
    """Render the AXIOM welcome to an already-approved interactive stream."""

    console = Console(
        file=stream,
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        highlight=False,
        soft_wrap=True,
    )

    for row in _LOGO_ROWS:
        console.print(Text(row, style=f"bold {CREAM}"), end="\n")

    extrusion = Text("  " + "╲" * (len(_LOGO_ROWS[0]) - 2), style=AMBER)
    console.print(extrusion)
    console.print(Text("   " + "╲" * (len(_LOGO_ROWS[0]) - 3), style=COFFEE))
    console.print(Text("  AXIOM", style=f"bold {AMBER}"))
    console.print(
        Text(
            "Know what fits before you train.",
            style=CREAM,
        )
    )
    console.print()
    console.print(
        Text("Start here: ", style=f"bold {AMBER}")
        + Text("axiom guide", style=f"bold {CREAM}")
    )

    if include_next_steps:
        console.print(Text("Next steps:", style=f"bold {AMBER}"))
        console.print(Text("  axiom --help", style=CREAM))
        console.print(Text("  axiom model inspect ./models/my-model", style=CREAM))

    console.print()


def show_startup(
    stream: TextIO | None = None,
    *,
    include_next_steps: bool = False,
    environ: Mapping[str, str] | None = None,
    marker_dir: Path | None = None,
    machine_readable: bool = False,
) -> bool:
    """Show the welcome once for the current terminal session.

    Returns ``True`` only when the welcome was rendered.  Marker failures are
    intentionally non-fatal: startup decoration must never break a command.
    """

    output = sys.stdout if stream is None else stream
    env = os.environ if environ is None else environ

    if (
        machine_readable
        or not _is_tty(output)
        or _is_truthy(env.get(NO_BANNER_ENV))
        or not _claim_session_marker(output, env, marker_dir)
    ):
        return False

    render_startup(output, include_next_steps=include_next_steps)
    return True
