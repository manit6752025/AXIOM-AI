"""Managed local Ollama process and bounded model-pull jobs.

Only processes AXIOM started and can positively identify may be stopped. The
service always binds to IPv4 loopback; model downloads require an explicit call.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.inference import runtime_status, validate_model_name

OLLAMA_HOST = "127.0.0.1:11434"
_SERVICE_START_TIMEOUT_SECONDS = 12.0
_SERVICE_STOP_TIMEOUT_SECONDS = 8.0
_MODEL_PULL_TIMEOUT_SECONDS = 1800.0
_MAX_PULL_JOBS = 20
_MAX_CONCURRENT_PULLS = 2
_MAX_PULL_LOG_LINES = 16
_SAFE_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PERCENT = re.compile(r"(?<!\d)(100|[1-9]?\d)%")

class OllamaServiceError(RuntimeError):
    """Raised when AXIOM cannot safely manage the local Ollama service."""


@dataclass
class _PullJob:
    id: str
    model: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    progress: str = "Queued"
    percent: int | None = None
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    cancel_requested: bool = False

_SERVICE_LOCK = threading.RLock()
_PULL_LOCK = threading.RLock()
_PULL_JOBS: dict[str, _PullJob] = {}


def _state_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "AXIOM" / "ollama-service.json"
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "axiom" / "ollama-service.json"


def _read_state() -> dict[str, Any] | None:
    try:
        record = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    if (
        not isinstance(record, dict)
        or isinstance(record.get("pid"), bool)
        or not isinstance(record.get("pid"), int)
        or record["pid"] <= 0
        or not isinstance(record.get("executable"), str)
        or not isinstance(record.get("identity"), str)
    ):
        return None
    return record


def _write_state(record: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _clear_state() -> None:
    try:
        _state_path().unlink(missing_ok=True)
    except OSError:
        pass


def _process_info(pid: int) -> dict[str, str] | None:
    """Read process identity without requiring third-party process libraries."""
    if sys.platform.startswith("linux"):
        try:
            root = Path("/proc") / str(pid)
            executable = os.path.realpath(root / "exe")
            raw_args = (root / "cmdline").read_bytes().split(b"\0")
            args = [os.fsdecode(part) for part in raw_args if part]
            stat = (root / "stat").read_text(encoding="utf-8")
            after_command = stat[stat.rfind(")") + 2:].split()
            # Linux proc stat field 22 (starttime), with field 3 as index zero.
            start_ticks = after_command[19]
            return {
                "executable": executable,
                "command": "\0".join(args),
                "identity": f"linux:{start_ticks}",
            }
        except (OSError, IndexError, ValueError):
            return None

    if os.name == "nt":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            return None
        # PID is parsed as an integer from local state, never supplied as user input.
        script = (
            f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}'; "
            "if ($p) { $p | Select-Object ExecutablePath,CommandLine,CreationDate | ConvertTo-Json -Compress }"
        )
        try:
            result = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=4, check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            data = json.loads(result.stdout)
            if not isinstance(data, dict) or not data.get("ExecutablePath") or not data.get("CommandLine"):
                return None
            return {
                "executable": str(data["ExecutablePath"]),
                "command": str(data["CommandLine"]),
                "identity": f"windows:{data.get('CreationDate', '')}",
            }
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None

    ps = shutil.which("ps")
    if not ps:
        return None
    try:
        result = subprocess.run(
            [ps, "-p", str(pid), "-o", "lstart=,command="],
            capture_output=True, text=True, timeout=3, check=False,
        )
        line = result.stdout.strip()
        if result.returncode != 0 or not line:
            return None
        # ps emits the process start time first, followed by the full command.
        parts = line.split(None, 5)
        if len(parts) < 6:
            return None
        start_time, command = " ".join(parts[:5]), parts[5]
        args = shlex.split(command)
        if not args:
            return None
        return {
            "executable": os.path.realpath(args[0]),
            "command": command,
            "identity": f"posix:{start_time}",
        }
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _command_has_serve(command: str) -> bool:
    if "\0" in command:
        return "serve" in command.split("\0")[1:]
    try:
        return "serve" in shlex.split(command, posix=os.name != "nt")[1:]
    except ValueError:
        return False


def _matches_state(record: dict[str, Any]) -> bool:
    info = _process_info(record["pid"])
    return bool(
        info
        and os.path.normcase(os.path.realpath(info["executable"]))
        == os.path.normcase(os.path.realpath(record["executable"]))
        and info["identity"] == record["identity"]
        and _command_has_serve(info["command"])
    )


def _managed_state() -> dict[str, Any] | None:
    record = _read_state()
    if record is None or not _matches_state(record):
        return None
    return record


def service_status() -> dict[str, Any]:
    """Combine the actual Ollama API probe with conservative process ownership."""
    report = runtime_status()
    managed = _managed_state() is not None
    online = report.get("status") == "online"
    installed = report.get("runtime_installed") is True
    report.update(
        service_managed_by_axiom=managed,
        service_can_start=bool(installed and not online and not managed),
        service_can_stop=managed,
        service_control_scope="loopback_only",
    )
    if managed:
        report["service_message"] = "AXIOM started this Ollama service and can stop it safely."
    elif online:
        report["service_message"] = "Ollama is already running independently; AXIOM will not stop it."
    elif installed:
        report["service_message"] = "Ollama is installed. AXIOM can start a loopback-only local service."
    else:
        report["service_message"] = "Install Ollama for your operating system, then refresh this status."
    return report


def start_service() -> dict[str, Any]:
    """Start Ollama only when needed and bind it explicitly to IPv4 loopback."""
    with _SERVICE_LOCK:
        report = service_status()
        if report["status"] == "online":
            return {"started": False, "already_running": True, "service": report}
        if report["service_managed_by_axiom"]:
            raise OllamaServiceError(
                "AXIOM already owns an Ollama process, but its API is not responding yet. Refresh status before retrying."
            )
        executable = shutil.which("ollama")
        if not executable:
            raise OllamaServiceError("Ollama was not found on PATH. Install Ollama before asking AXIOM to start it.")

        environment = os.environ.copy()
        environment["OLLAMA_HOST"] = OLLAMA_HOST
        creationflags = 0
        popen_options: dict[str, Any] = {"start_new_session": True}
        if os.name == "nt":
            creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            popen_options = {}
        try:
            process = subprocess.Popen(
                [executable, "serve"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                creationflags=creationflags,
                **popen_options,
            )
        except OSError:
            raise OllamaServiceError("AXIOM could not launch the Ollama service. Check the installation and permissions.") from None

        record: dict[str, Any] | None = None
        deadline = time.monotonic() + _SERVICE_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise OllamaServiceError(
                    "Ollama exited before its local API became available. It may already be managed by another service."
                )
            info = _process_info(process.pid)
            if info and os.path.normcase(os.path.realpath(info["executable"])) == os.path.normcase(os.path.realpath(executable)) and _command_has_serve(info["command"]):
                record = {
                    "pid": process.pid,
                    "executable": os.path.realpath(executable),
                    "identity": info["identity"],
                    "started_at": datetime.now(UTC).isoformat(),
                }
                try:
                    _write_state(record)
                except OSError:
                    try:
                        process.terminate()
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    except OSError:
                        pass
                    raise OllamaServiceError("AXIOM could not safely record ownership of the Ollama service; the started process was terminated.") from None
                break
            time.sleep(0.1)
        if record is None:
            try:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise OllamaServiceError("AXIOM could not verify ownership of the started Ollama process; the process was terminated and not registered for later stop.")

        while time.monotonic() < deadline:
            if process.poll() is not None:
                _clear_state()
                raise OllamaServiceError("Ollama stopped during startup. Check its local installation and system logs.")
            if runtime_status().get("status") == "online":
                return {"started": True, "already_running": False, "service": service_status()}
            time.sleep(0.2)

        if _matches_state(record):
            try:
                os.kill(process.pid, signal.SIGTERM)
            except OSError:
                pass
            cleanup_deadline = time.monotonic() + 2.0
            while time.monotonic() < cleanup_deadline and _matches_state(record):
                time.sleep(0.1)
            if _matches_state(record):
                try:
                    os.kill(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                except OSError:
                    pass
                cleanup_deadline = time.monotonic() + 2.0
                while time.monotonic() < cleanup_deadline and _matches_state(record):
                    time.sleep(0.1)
            if _matches_state(record):
                raise OllamaServiceError("Ollama did not become ready and its process is still running; AXIOM retained its ownership record so it can be stopped safely.")
        _clear_state()
        raise OllamaServiceError("Ollama did not become ready within 12 seconds. Check the local service and retry.")


def stop_service() -> dict[str, Any]:
    """Stop only a verified Ollama serve process started by AXIOM."""
    with _SERVICE_LOCK:
        record = _managed_state()
        if record is None:
            raise OllamaServiceError(
                "AXIOM will only stop an Ollama process it started and can verify. Stop independently managed services with their own service manager."
            )
        try:
            os.kill(record["pid"], signal.SIGTERM)
        except ProcessLookupError:
            _clear_state()
            return {"stopped": True, "service": service_status()}
        except OSError:
            raise OllamaServiceError("The owned Ollama service could not be signalled. Check process permissions.") from None

        deadline = time.monotonic() + _SERVICE_STOP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not _matches_state(record):
                _clear_state()
                return {"stopped": True, "service": service_status()}
            time.sleep(0.15)

        if not _matches_state(record):
            _clear_state()
            return {"stopped": True, "service": service_status()}
        try:
            os.kill(record["pid"], getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and _matches_state(record):
            time.sleep(0.1)
        if _matches_state(record):
            raise OllamaServiceError("The AXIOM-managed Ollama process did not exit; no other process was stopped.")
        _clear_state()
        return {"stopped": True, "service": service_status()}


def _sanitize_line(value: str) -> str:
    without_ansi = _ANSI_ESCAPE.sub("", value)
    safe = "".join(char for char in without_ansi if char in "\t " or ord(char) >= 32 and ord(char) != 127)
    return safe.strip()[:320]


def _job_payload(job: _PullJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "model": job.model,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "progress": job.progress,
        "percent": job.percent,
        "logs": list(job.logs),
        "error": job.error,
        "cancel_requested": job.cancel_requested,
    }


def _update_pull_job(job: _PullJob, *, progress: str | None = None, status: str | None = None,
                     error: str | None = None, log: str | None = None) -> None:
    with _PULL_LOCK:
        if progress is not None:
            job.progress = progress[:320]
            match = _PERCENT.search(progress)
            if match:
                job.percent = min(100, max(0, int(match.group(1))))
        if status is not None:
            job.status = status
        if error is not None:
            job.error = _sanitize_line(error) or "The model pull failed."
        if log:
            safe = _sanitize_line(log)
            if safe:
                job.logs.append(safe)
                del job.logs[:-_MAX_PULL_LOG_LINES]
                job.progress = safe
                match = _PERCENT.search(safe)
                if match:
                    job.percent = min(100, max(0, int(match.group(1))))
        job.updated_at = datetime.now(UTC).isoformat()


def _run_pull_job(job: _PullJob, executable: str) -> None:
    environment = os.environ.copy()
    environment["OLLAMA_HOST"] = OLLAMA_HOST
    try:
        process = subprocess.Popen(
            [executable, "pull", job.model],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
            env=environment,
        )
    except OSError:
        _update_pull_job(job, status="failed", error="AXIOM could not launch the Ollama model-download command.")
        return

    with _PULL_LOCK:
        job.process = process
        cancelled_before_start = job.cancel_requested
        job.status = "running"
        job.progress = "Connecting to the local Ollama service"
        job.updated_at = datetime.now(UTC).isoformat()
    if cancelled_before_start:
        try:
            process.terminate()
        except OSError:
            pass

    lines: queue.Queue[str] = queue.Queue(maxsize=200)

    def read_output() -> None:
        assert process.stdout is not None
        buffer: list[str] = []
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            if char in "\r\n":
                if buffer:
                    try:
                        lines.put_nowait(_sanitize_line("".join(buffer)))
                    except queue.Full:
                        pass
                    buffer.clear()
            elif len(buffer) < 1600:
                buffer.append(char)
        if buffer:
            try:
                lines.put_nowait(_sanitize_line("".join(buffer)))
            except queue.Full:
                pass

    reader = threading.Thread(target=read_output, name=f"axiom-ollama-pull-output-{job.id[:8]}", daemon=True)
    reader.start()
    started = time.monotonic()
    timed_out = False
    while process.poll() is None or not lines.empty():
        with _PULL_LOCK:
            cancel_requested = job.cancel_requested
        if cancel_requested and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        if time.monotonic() - started > _MODEL_PULL_TIMEOUT_SECONDS and process.poll() is None:
            timed_out = True
            try:
                process.terminate()
            except OSError:
                pass
        try:
            line = lines.get(timeout=0.15)
        except queue.Empty:
            if timed_out and process.poll() is not None and lines.empty():
                break
            continue
        if line:
            _update_pull_job(job, log=line)
    try:
        return_code = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = process.wait(timeout=3)
    reader.join(timeout=1)
    with _PULL_LOCK:
        cancelled = job.cancel_requested
    if cancelled:
        _update_pull_job(job, status="cancelled", progress="Download cancelled by the user")
    elif timed_out:
        _update_pull_job(job, status="failed", error="Model pull exceeded the 30-minute time limit.")
    elif return_code == 0:
        _update_pull_job(job, status="completed", progress="Model downloaded and registered locally", log="Model download complete")
        with _PULL_LOCK:
            job.percent = 100
    else:
        _update_pull_job(job, status="failed", error=f"Ollama model pull exited with status {return_code}.")


def start_model_pull(model: str) -> dict[str, Any]:
    """Queue an explicit, bounded-concurrency Ollama library pull."""
    validated = validate_model_name(model)
    if runtime_status().get("status") != "online":
        raise OllamaServiceError("Start the local Ollama service before downloading a model.")
    executable = shutil.which("ollama")
    if not executable:
        raise OllamaServiceError("Ollama's command-line executable was not found on PATH.")
    with _PULL_LOCK:
        active = sum(job.status in {"queued", "running"} for job in _PULL_JOBS.values())
        if active >= _MAX_CONCURRENT_PULLS:
            raise OllamaServiceError("Two model downloads are already active. Wait for one to finish.")
        if len(_PULL_JOBS) >= _MAX_PULL_JOBS:
            terminal = sorted(
                (job for job in _PULL_JOBS.values() if job.status not in {"queued", "running"}),
                key=lambda job: job.updated_at,
            )
            while len(_PULL_JOBS) >= _MAX_PULL_JOBS and terminal:
                _PULL_JOBS.pop(terminal.pop(0).id, None)
        if len(_PULL_JOBS) >= _MAX_PULL_JOBS:
            raise OllamaServiceError("The in-memory download history is full. Finish or clear recent model-pull jobs first.")
        job = _PullJob(id=uuid.uuid4().hex, model=validated)
        _PULL_JOBS[job.id] = job
    worker = threading.Thread(
        target=_run_pull_job, args=(job, executable),
        name=f"axiom-ollama-pull-{job.id[:8]}", daemon=True,
    )
    worker.start()
    with _PULL_LOCK:
        return _job_payload(job)


def get_model_pull(job_id: str) -> dict[str, Any]:
    if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError("Invalid model-download job ID.")
    with _PULL_LOCK:
        job = _PULL_JOBS.get(job_id)
        if job is None:
            raise KeyError("Model-download job was not found or has expired.")
        return _job_payload(job)


def cancel_model_pull(job_id: str) -> dict[str, Any]:
    if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError("Invalid model-download job ID.")
    with _PULL_LOCK:
        job = _PULL_JOBS.get(job_id)
        if job is None:
            raise KeyError("Model-download job was not found or has expired.")
        if job.status in {"completed", "failed", "cancelled"}:
            return _job_payload(job)
        job.cancel_requested = True
        process = job.process
        job.progress = "Cancellation requested"
        job.updated_at = datetime.now(UTC).isoformat()
    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    with _PULL_LOCK:
        return _job_payload(job)
