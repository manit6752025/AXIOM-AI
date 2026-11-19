"""Timed, cross-platform keep-awake sessions.

Each session owns one child process.  The child is terminated (and, when the
default POSIX runner is used, its process group is terminated) when the
context manager exits.  Runners are injectable so platform selection and
cleanup can be tested without changing the host's power settings.
"""

from __future__ import annotations

import math
import os
import platform as platform_module
import shutil
import signal
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from .installer import SecretInputError


class AwakeUnavailableError(RuntimeError):
    """Raised when the host has no supported keep-awake primitive."""


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None:
        """Return the exit code, or None while the process is running."""

    def terminate(self) -> Any:
        """Ask the process to stop."""

    def kill(self) -> Any:
        """Force the process to stop."""

    def wait(self, timeout: float | None = None) -> Any:
        """Wait for the process to exit."""


class RunnerLike(Protocol):
    def which(self, executable: str) -> str | None:
        """Resolve an executable name to a path, or None if missing."""

    def popen(self, argv: Sequence[str]) -> ProcessLike:
        """Start a child process from an argv sequence."""


WINDOWS_KEEP_AWAKE_SCRIPT = r"""
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class AxiomPowerState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$ES_CONTINUOUS = [uint32]0x80000000
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
[void][AxiomPowerState]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
try {
    Start-Sleep -Seconds {duration}
}
finally {
    [void][AxiomPowerState]::SetThreadExecutionState($ES_CONTINUOUS)
}
""".strip()


def _normalise_platform(system: str | None) -> str:
    value = (system or platform_module.system()).strip().lower()
    if value in {"darwin", "mac", "macos", "osx"}:
        return "darwin"
    if value in {"windows", "win32", "win"}:
        return "windows"
    if value in {"linux", "gnu/linux"}:
        return "linux"
    return "other"


def _duration_seconds(value: float) -> int:
    if isinstance(value, bool):
        raise TypeError("duration must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError("duration must be a positive finite number")
    return max(1, math.ceil(number))


def _safe_reason(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    result = reason.strip()
    if len(result) > 160 or "\x00" in result or "\n" in result or "\r" in result:
        raise ValueError("reason must be a short single-line string")
    if "api_key=" in result.lower() or "token=" in result.lower():
        raise SecretInputError("keep-awake reasons cannot contain credentials")
    return result


def _runner_which(
    runner: object | None,
    executable: str,
    which: Callable[[str], str | None] | None,
) -> str | None:
    if which is not None:
        try:
            value = which(executable)
        except (OSError, TypeError):
            return None
        if isinstance(value, str):
            return value or None
        return executable if value else None
    if runner is not None and hasattr(runner, "which"):
        try:
            value = runner.which(executable)  # type: ignore[attr-defined]
        except (OSError, TypeError):
            return None
        if isinstance(value, str):
            return value or None
        return executable if value else None
    return shutil.which(executable)


@dataclass(frozen=True)
class AwakeCommand:
    """A selected platform command and any reason it could not be used."""

    platform: str
    argv: tuple[str, ...]
    executable: str | None
    available: bool
    instructions: tuple[str, ...] = ()

    @property
    def command(self) -> tuple[str, ...]:
        return self.argv


def build_awake_command(
    system: str | None = None,
    duration: float = 3600,
    *,
    which: Callable[[str], str | None] | None = None,
    runner: object | None = None,
    reason: str = "AXIOM timed session",
) -> AwakeCommand:
    """Select the fixed keep-awake command for a platform."""

    platform_name = _normalise_platform(system)
    seconds = _duration_seconds(duration)
    safe_reason = _safe_reason(reason)

    if platform_name == "darwin":
        available = _runner_which(runner, "caffeinate", which)
        return AwakeCommand(
            platform=platform_name,
            argv=("caffeinate", "-dimsu", "-t", str(seconds)),
            executable="caffeinate",
            available=bool(available) or which is None and runner is None,
            instructions=(
                "macOS provides caffeinate; keep the session scoped to the requested duration.",
            ),
        )

    if platform_name == "linux":
        executable = _runner_which(runner, "systemd-inhibit", which)
        if executable:
            return AwakeCommand(
                platform=platform_name,
                argv=(
                    "systemd-inhibit",
                    "--what=idle:sleep",
                    "--who=AXIOM",
                    f"--why={safe_reason}",
                    "--mode=block",
                    "sleep",
                    str(seconds),
                ),
                executable="systemd-inhibit",
                available=True,
                instructions=(
                    "systemd-inhibit will release its lock when the timed child exits.",
                ),
            )
        return AwakeCommand(
            platform=platform_name,
            argv=(),
            executable=None,
            available=False,
            instructions=(
                "systemd-inhibit was not found; install or enable systemd-logind before starting a keep-awake session.",
            ),
        )

    if platform_name == "windows":
        executable = None
        for candidate in ("powershell", "pwsh"):
            if _runner_which(runner, candidate, which):
                executable = candidate
                break
        if executable is None and which is None and runner is None:
            executable = "powershell"
        if executable is None:
            return AwakeCommand(
                platform=platform_name,
                argv=(),
                executable=None,
                available=False,
                instructions=(
                    "PowerShell was not found; install PowerShell and retry the session.",
                ),
            )
        script = WINDOWS_KEEP_AWAKE_SCRIPT.replace("{duration}", str(seconds))
        return AwakeCommand(
            platform=platform_name,
            argv=(
                executable,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ),
            executable=executable,
            available=True,
            instructions=(
                "The fixed PowerShell helper sets and clears SetThreadExecutionState for the timed session.",
            ),
        )

    return AwakeCommand(
        platform=platform_name,
        argv=(),
        executable=None,
        available=False,
        instructions=(
            "This operating system has no supported AXIOM keep-awake primitive.",
        ),
    )


def select_awake_command(
    system: str | None = None,
    duration: float = 3600,
    **kwargs: Any,
) -> tuple[str, ...] | None:
    """Return only the selected argv, or ``None`` when unavailable."""

    selection = build_awake_command(system, duration, **kwargs)
    return selection.argv if selection.available else None


class _DefaultRunner:
    def which(self, executable: str) -> str | None:
        return shutil.which(executable)

    def popen(self, argv: Sequence[str]) -> subprocess.Popen[Any]:
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0,
            )
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(list(argv), **kwargs)


class KeepAwakeSession:
    """Own a timed keep-awake child process for a context-managed session."""

    def __init__(
        self,
        duration: float = 3600,
        *,
        seconds: float | None = None,
        duration_seconds: float | None = None,
        system: str | None = None,
        platform_name: str | None = None,
        reason: str = "AXIOM timed session",
        runner: object | None = None,
        process_runner: object | None = None,
        which: Callable[[str], str | None] | None = None,
    ) -> None:
        if seconds is not None and duration_seconds is not None:
            raise TypeError("use only one of seconds and duration_seconds")
        if seconds is not None:
            duration = seconds
        if duration_seconds is not None:
            duration = duration_seconds
        if runner is not None and process_runner is not None:
            raise TypeError("use only one of runner and process_runner")
        self.duration = _duration_seconds(duration)
        self.system = platform_name or system
        self.reason = _safe_reason(reason)
        self.runner = runner or process_runner or _DefaultRunner()
        self.which = which
        self.selection = build_awake_command(
            self.system,
            self.duration,
            which=which,
            runner=self.runner,
            reason=self.reason,
        )
        self._process: ProcessLike | None = None

    @property
    def command(self) -> tuple[str, ...]:
        return self.selection.argv

    @property
    def process(self) -> ProcessLike | None:
        return self._process

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def plan(self) -> AwakeCommand:
        return self.selection

    def _start_process(self) -> ProcessLike:
        if not self.selection.available or not self.selection.argv:
            message = " ".join(self.selection.instructions)
            raise AwakeUnavailableError(message)
        runner = self.runner
        if hasattr(runner, "popen"):
            return runner.popen(self.selection.argv)  # type: ignore[attr-defined]
        if hasattr(runner, "start"):
            return runner.start(self.selection.argv)  # type: ignore[attr-defined]
        if callable(runner):
            return runner(self.selection.argv)
        raise TypeError("runner must expose popen/start or be callable")

    def start(self) -> KeepAwakeSession:
        if self.is_running:
            return self
        self._process = self._start_process()
        return self

    def _wait(self, process: ProcessLike, timeout: float) -> bool:
        try:
            process.wait(timeout=timeout)
            return True
        except (subprocess.TimeoutExpired, TimeoutError, TypeError):
            return False

    def _terminate(self, process: ProcessLike) -> None:
        if process.poll() is not None:
            return

        if isinstance(self.runner, _DefaultRunner) and os.name != "nt":
            pid = getattr(process, "pid", None)
            if isinstance(pid, int) and pid > 0:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    process.terminate()
            else:
                process.terminate()
        else:
            process.terminate()

        if self._wait(process, timeout=2.0):
            return
        if process.poll() is None:
            process.kill()
        self._wait(process, timeout=2.0)

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        self._terminate(process)

    close = stop

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self.stop()
        return False


AwakeSession = KeepAwakeSession
TimedAwakeSession = KeepAwakeSession


__all__ = [
    "WINDOWS_KEEP_AWAKE_SCRIPT",
    "AwakeCommand",
    "AwakeSession",
    "AwakeUnavailableError",
    "KeepAwakeSession",
    "TimedAwakeSession",
    "build_awake_command",
    "select_awake_command",
]
