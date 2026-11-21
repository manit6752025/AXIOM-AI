"""Interactive CLI welcome rendered from AXIOM's canonical README SVG."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping
from importlib.resources import files as resource_files
from typing import TextIO

from rich.color import Color
from rich.console import Console
from rich.style import Style
from rich.text import Text

NO_BANNER_ENV = "AXIOM_NO_BANNER"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_SESSION_ENV_VARS = (
    "TERM_SESSION_ID", "WT_SESSION", "XDG_SESSION_ID",
    "KITTY_WINDOW_ID", "TMUX_PANE", "STY",
)
_MACHINE_OR_USAGE_FLAGS = frozenset({
    "--help", "-h", "--json", "--jsonl", "--ndjson",
    "--machine-readable", "--show-completion", "--install-completion",
})
_LOGO_WIDTH = 24
_LOGO_HEIGHT = 24


def _read_logo_svg() -> str:
    try:
        return (
            resource_files("axiom")
            .joinpath("assets")
            .joinpath("axiom-logo.svg")
            .read_text(encoding="utf-8")
        )
    except (OSError, ModuleNotFoundError, AttributeError):
        return ""


def _rgb(value: str) -> tuple[int, int, int]:
    raw = value.lstrip("#")
    return tuple(int(raw[index:index + 2], 16) for index in (0, 2, 4))


def _logo_palette() -> dict[str, str]:
    fallback = {
        "background": "#10130f",
        "panel": "#171c14",
        "border": "#4c5730",
        "foreground": "#f2f4eb",
        "accent": "#d9f36a",
        "muted": "#aab39e",
    }
    svg = _read_logo_svg()
    colours = set(re.findall(r"#[0-9a-fA-F]{6}", svg))
    if not svg or not set(fallback.values()).issubset(colours | {fallback["border"]}):
        return fallback
    return fallback


_LOGO_PALETTE = _logo_palette()
# Kept as import-compatible aliases; values now match the actual README SVG.
AMBER = _LOGO_PALETTE["accent"]
CREAM = _LOGO_PALETTE["foreground"]
COFFEE = _LOGO_PALETTE["muted"]


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
    token = next((environ[name] for name in _SESSION_ENV_VARS if environ.get(name)), "")
    try:
        leader = str(os.getsid(0))
    except (AttributeError, OSError):
        leader = ""
    identity = f"{token}|{leader}|{_tty_name(stream) or environ.get('TTY', '')}" or "interactive"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _default_marker_dir(environ: Mapping[str, str]) -> Path:
    runtime = environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "axiom"
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


def _claim_session_marker(stream: TextIO, environ: Mapping[str, str], marker_dir: Path | None) -> bool:
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
    except (FileExistsError, OSError):
        return False
    try:
        os.close(descriptor)
    except OSError:
        return False
    return True


def suppress_for_arguments(arguments: Iterable[str]) -> bool:
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


def _line_distance(x: float, y: float, start: tuple[float, float], end: tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = dx * dx + dy * dy
    if length == 0:
        return math.hypot(x - start[0], y - start[1])
    t = max(0.0, min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / length))
    return math.hypot(x - start[0] - t * dx, y - start[1] - t * dy)


def _icon_pixel(x: float, y: float) -> str:
    p = _LOGO_PALETTE
    colour = p["panel"] if 3 <= x <= 125 and 3 <= y <= 125 else p["background"]

    # Rounded square outline from the SVG's 122px icon tile.
    if min(abs(x - 3), abs(x - 125), abs(y - 3), abs(y - 125)) < 1.2:
        colour = p["border"]

    # The SVG's white AXIOM A and crossbar.
    a_left = _line_distance(x, y, (36, 91), (57.2, 36.4))
    a_right = _line_distance(x, y, (70.8, 36.4), (92, 91))
    crossbar = _line_distance(x, y, (46.5, 68.5), (81.5, 68.5))
    if min(a_left, a_right) <= 4.5 or crossbar <= 4:
        colour = p["foreground"]

    # Rotated lime ellipse/orbit from the SVG, sampled without image libraries.
    angle = math.radians(22)
    dx, dy = x - 64, y - 64
    rotated_x = dx * math.cos(angle) - dy * math.sin(angle)
    rotated_y = dx * math.sin(angle) + dy * math.cos(angle)
    ellipse_norm = math.sqrt((rotated_x / 54) ** 2 + (rotated_y / 24) ** 2)
    if abs(ellipse_norm - 1) * 24 <= 2.25:
        colour = p["accent"]

    # Lower orbital sweep: the SVG cubic curve is sampled into line segments.
    controls = ((13.8, 83.5), (27.6, 101.9), (48.5, 112.1), (70.1, 110.6))
    previous = controls[0]
    for step in range(1, 25):
        t = step / 24
        u = 1 - t
        point = (
            u**3 * controls[0][0] + 3*u*u*t*controls[1][0] + 3*u*t*t*controls[2][0] + t**3*controls[3][0],
            u**3 * controls[0][1] + 3*u*u*t*controls[1][1] + 3*u*t*t*controls[2][1] + t**3*controls[3][1],
        )
        distance = _line_distance(x, y, previous, point)
        if distance <= 4.5:
            colour = p["panel"]
        if distance <= 2.25:
            colour = p["accent"]
        previous = point

    if math.hypot(x - 109.4, y - 42.7) <= 4.2:
        colour = p["accent"]
    return colour


def _terminal_logo_lines() -> list[Text]:
    """Rasterize the SVG mark into compact true-colour terminal half-blocks."""
    lines = []
    for row in range(_LOGO_HEIGHT // 2):
        line = Text()
        y_upper = (row * 2 + 0.5) * 128 / _LOGO_HEIGHT
        y_lower = (row * 2 + 1.5) * 128 / _LOGO_HEIGHT
        for column in range(_LOGO_WIDTH):
            x = (column + 0.5) * 128 / _LOGO_WIDTH
            upper = _icon_pixel(x, y_upper)
            lower = _icon_pixel(x, y_lower)
            line.append(
                "▀",
                style=Style(color=Color.from_rgb(*_rgb(upper)), bgcolor=Color.from_rgb(*_rgb(lower))),
            )
        if row == 3:
            line.append("  AXIOM", style=Style(color=_LOGO_PALETTE["foreground"], bold=True))
        elif row == 5:
            line.append("  LOCAL-FIRST AI ENGINEERING", style=_LOGO_PALETTE["muted"])
        elif row == 7:
            line.append("  BUILD AI. OWN AI.", style=Style(color=_LOGO_PALETTE["accent"], bold=True))
        lines.append(line)
    return lines


def render_startup(stream: TextIO, *, include_next_steps: bool = False) -> None:
    console = Console(
        file=stream, force_terminal=True, color_system="truecolor",
        no_color=False, highlight=False, soft_wrap=True,
    )
    for line in _terminal_logo_lines():
        console.print(line)
    console.print(Text("Know what fits before you train.", style=_LOGO_PALETTE["muted"]))
    console.print()
    console.print(
        Text("Start here: ", style=Style(color=_LOGO_PALETTE["foreground"], bold=True))
        + Text("axiom guide", style=Style(color=_LOGO_PALETTE["accent"], bold=True))
    )
    if include_next_steps:
        console.print(Text("Next steps:", style=Style(color=_LOGO_PALETTE["foreground"], bold=True)))
        console.print(Text("  axiom --help", style=_LOGO_PALETTE["muted"]))
        console.print(Text("  axiom model inspect ./models/my-model", style=_LOGO_PALETTE["muted"]))
    console.print()


def show_startup(
    stream: TextIO | None = None,
    *,
    include_next_steps: bool = False,
    environ: Mapping[str, str] | None = None,
    marker_dir: Path | None = None,
    machine_readable: bool = False,
) -> bool:
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
