"""Bounded local LoRA fine-tuning using Transformers and PEFT.

No model downloads are permitted: the base model must be a local Transformers
model directory and all from_pretrained calls set local_files_only=True.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import random
import re
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable

from axiom.core.storage import file_lock
from axiom.path_safety import open_regular_file

_LOG = logging.getLogger(__name__)

_MAX_DATASET_BYTES = 50 * 1024 * 1024
_MAX_ROWS = 10_000
_MAX_LINE_BYTES = 1_048_576
_MAX_TEXT_CHARS = 300_000
TRAINING_MAX_DATASET_BYTES = _MAX_DATASET_BYTES
TRAINING_MAX_RECORDS = _MAX_ROWS
TRAINING_MAX_LINE_BYTES = _MAX_LINE_BYTES
TRAINING_MAX_TEXT_CHARS = _MAX_TEXT_CHARS
_MAX_MODEL_ENTRIES = 50_000
_MAX_STEPS = 1_000
_MAX_LOSS_SAMPLES = 64
_MAX_SEQUENCE_LENGTH = 2_048
_TEXT_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_PACKAGES = {"torch": "torch", "transformers": "transformers", "peft": "peft", "accelerate": "accelerate"}
_LOCAL_LOCK = threading.RLock()
AXIOM_TRAINING_SEED = 1729


class FineTuningError(RuntimeError):
    """A local fine-tuning operation could not be completed."""


class ActiveFineTuningJobError(FineTuningError):
    """Another fine-tuning job already holds the local training lock."""


class FineTuningCancelledError(FineTuningError):
    """A local fine-tuning job received a cooperative cancellation request."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _training_loss_sample_steps(max_steps: Any) -> set[int]:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= _MAX_STEPS:
        return set()
    sample_count = min(max_steps, _MAX_LOSS_SAMPLES)
    if sample_count == 1:
        return {1}
    return {
        1 + ((max_steps - 1) * index) // (sample_count - 1)
        for index in range(sample_count)
    }


def _validated_training_loss_samples(job: dict[str, Any]) -> list[dict[str, int | float]]:
    config = job.get("config")
    max_steps = config.get("max_steps") if isinstance(config, dict) else None
    completed_steps = job.get("completed_steps")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 1 <= max_steps <= _MAX_STEPS
        or isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or not 0 <= completed_steps <= max_steps
    ):
        return []
    raw_samples = job.get("loss_samples")
    if not isinstance(raw_samples, list):
        return []
    samples: list[dict[str, int | float]] = []
    previous_step = 0
    sample_steps = _training_loss_sample_steps(max_steps)
    for sample in raw_samples[:_MAX_LOSS_SAMPLES]:
        if not isinstance(sample, dict):
            continue
        step = sample.get("step")
        loss = sample.get("loss")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            continue
        try:
            loss_value = float(loss)
        except (OverflowError, TypeError, ValueError):
            continue
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step not in sample_steps
            or not previous_step < step <= min(max_steps, completed_steps)
            or not math.isfinite(loss_value)
        ):
            continue
        samples.append({"step": step, "loss": loss_value})
        previous_step = step
    return samples


def training_loss_trace_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, sanitized diagnostic trace for MCP job status."""
    samples = _validated_training_loss_samples(job)
    return {
        "available": bool(samples),
        "samples": samples,
        "sample_count": len(samples),
        "maximum_samples": _MAX_LOSS_SAMPLES,
        "sampling_strategy": "evenly_spaced_across_configured_steps",
        "scope_note": (
            "Each point is the training loss for the single selected training record at that step, "
            "so the trace can be noisy. It is not validation loss and does not establish "
            "generalization or adapter quality; use held-out evaluation to assess quality."
        ),
        "validation_loss_available": False,
        "generalization_assessed": False,
    }


def _clip(exc: BaseException, limit: int) -> str:
    return str(exc).replace("\r", " ").replace("\n", " ").strip()[:limit]


def jobs_directory() -> Path:
    configured = os.environ.get("AXIOM_FINETUNE_HOME", "").strip()
    base = Path(configured).expanduser() if configured else Path.home() / ".local" / "share" / "axiom" / "fine-tuning"
    return base.resolve() / "jobs"


def _lock_path() -> Path:
    return jobs_directory().parent / ".active-fine-tuning-job.json"


def _version_key(value: str) -> tuple[int, int, int]:
    """Normalize short dotted versions to a comparable major/minor/patch tuple."""
    parts = [int(part) for part in re.findall(r"\d+", value)[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def runtime_status() -> dict[str, Any]:
    packages: dict[str, bool] = {}
    versions: dict[str, str | None] = {}
    minimum_versions = {"torch": "2.2.0", "transformers": "5.0.0", "peft": "0.20.0", "accelerate": "1.1.0"}
    problems: list[str] = []
    for label, module in _PACKAGES.items():
        try:
            packages[label] = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            packages[label] = False
        try:
            versions[label] = importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
        if not packages[label]:
            problems.append(f"{label} is not installed")
        elif versions[label] is not None and _version_key(versions[label]) < _version_key(minimum_versions[label]):
            problems.append(f"{label} {versions[label]} is too old (need >= {minimum_versions[label]})")
    return {
        "backend": "axiom-lora-engine",
        "can_start": not problems,
        "packages": packages,
        "versions": versions,
        "minimum_versions": minimum_versions,
        "problems": problems,
        "setup_note": ("Ready for local LoRA training." if not problems else "Training setup required: " + "; ".join(problems) + ". Install or upgrade with: python -m pip install -e '.[training]'"),
        "model_loading": "local_files_only",
        "scope_note": "AXIOM runs its own bounded PyTorch training loop over a local causal language model with PEFT LoRA adapters. Package readiness does not guarantee the model fits available memory.",
    }


def require_runtime() -> None:
    status = runtime_status()
    if not status["can_start"]:
        raise FineTuningError(str(status["setup_note"]))


def _resolve_existing(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required.")
    try:
        return Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"{label} must point to an existing local path.") from None


def validate_local_model_tree(model_path: Path) -> None:
    """Reject linked model assets and cap the metadata scan before loading weights."""
    try:
        if model_path.resolve(strict=True) != model_path or model_path.is_symlink():
            raise FineTuningError("The local model directory changed or is no longer safe.")
    except (OSError, RuntimeError):
        raise FineTuningError("The local model directory changed or is no longer safe.") from None
    pending = [model_path]
    seen = 0
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    seen += 1
                    if seen > _MAX_MODEL_ENTRIES:
                        raise FineTuningError(
                            f"Model directory exceeds the {_MAX_MODEL_ENTRIES:,}-entry safety limit."
                        )
                    path = Path(entry.path)
                    is_junction = getattr(path, "is_junction", lambda: False)
                    if entry.is_symlink() or is_junction():
                        raise FineTuningError("Local model assets cannot contain symbolic links or junctions.")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
        except OSError:
            raise FineTuningError("Local model files could not be inspected safely.") from None


def validate_training_text_field(text_field: str) -> None:
    if not isinstance(text_field, str) or not _TEXT_FIELD.fullmatch(text_field):
        raise ValueError("Text field must be a simple JSON field name (letters, digits, underscore, dot, or hyphen).")


def _parse_training_line(raw_line: bytes, line_number: int, text_field: str) -> str | None:
    line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8").strip()
    if not line.strip():
        return None
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, RecursionError):
        raise ValueError(f"Dataset line {line_number} is not valid JSON.") from None
    if not isinstance(record, dict):
        raise ValueError(f"Dataset line {line_number} must contain a JSON object.")
    value = record.get(text_field)
    if value is None and isinstance(record.get("prompt"), str) and isinstance(record.get("completion"), str):
        value = record["prompt"] + "\n" + record["completion"]
    if value is None and isinstance(record.get("messages"), list):
        parts: list[str] = []
        for message in record["messages"]:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                role = message.get("role")
                parts.append((f"{role}: " if isinstance(role, str) and role else "") + message["content"])
        value = "\n".join(parts) if parts else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Dataset line {line_number} needs a non-empty string in '{text_field}', "
            "or string prompt/completion fields, or messages with content strings."
        )
    if len(value) > _MAX_TEXT_CHARS:
        raise ValueError(f"Dataset line {line_number} text exceeds the {_MAX_TEXT_CHARS:,}-character limit.")
    return value.strip()


def _training_line_byte_length(raw_line: bytes) -> int:
    # TextIOWrapper's universal-newline mode represented CRLF as one newline.
    return len(raw_line) - (1 if raw_line.endswith(b"\r\n") else 0)


def iter_training_texts(dataset_path: Path, text_field: str = "text") -> Iterator[str]:
    """Yield trainer-accepted text values without retaining the full dataset."""
    with open_regular_file(dataset_path) as stream:
        yield from iter_training_text_stream(
            stream,
            file_size_bytes=os.fstat(stream.fileno()).st_size,
            text_field=text_field,
        )


def iter_training_text_stream(
    stream: BinaryIO,
    *,
    file_size_bytes: int,
    text_field: str = "text",
) -> Iterator[str]:
    """Validate and yield accepted trainer text from an already-open stream."""
    validate_training_text_field(text_field)
    if file_size_bytes > _MAX_DATASET_BYTES:
        raise ValueError(f"Dataset exceeds the {_MAX_DATASET_BYTES // (1024 * 1024)} MiB limit.")
    accepted = 0
    actual_bytes = 0
    line_number = 0
    pending = bytearray()
    try:
        while True:
            remaining_bytes = _MAX_DATASET_BYTES - actual_bytes
            remaining_line_bytes = _MAX_LINE_BYTES - len(pending)
            read_size = min(64 * 1024, remaining_bytes + 1, remaining_line_bytes + 1)
            chunk = stream.read(max(1, read_size))
            if chunk:
                if len(chunk) > remaining_bytes:
                    raise ValueError(f"Dataset exceeds the {_MAX_DATASET_BYTES // (1024 * 1024)} MiB limit.")
                actual_bytes += len(chunk)
                pending.extend(chunk)
            at_eof = not chunk

            while pending:
                newline = pending.find(b"\n")
                carriage_return = pending.find(b"\r")
                positions = [position for position in (newline, carriage_return) if position >= 0]
                if not positions:
                    if len(pending) > _MAX_LINE_BYTES:
                        raise ValueError(f"Dataset line {line_number + 1} exceeds the 1 MiB line limit.")
                    break

                separator = min(positions)
                if pending[separator] == 13 and separator + 1 == len(pending) and not at_eof:
                    break
                end = separator + 1
                if pending[separator] == 13 and end < len(pending) and pending[end] == 10:
                    end += 1
                raw_line = bytes(pending[:end])
                del pending[:end]
                line_number += 1
                if _training_line_byte_length(raw_line) > _MAX_LINE_BYTES:
                    raise ValueError(f"Dataset line {line_number} exceeds the 1 MiB line limit.")
                value = _parse_training_line(raw_line, line_number, text_field)
                if value is not None:
                    accepted += 1
                    if accepted > _MAX_ROWS:
                        raise ValueError(f"Dataset exceeds the {_MAX_ROWS:,}-row limit.")
                    yield value
            if at_eof:
                if pending:
                    line_number += 1
                    if len(pending) > _MAX_LINE_BYTES:
                        raise ValueError(f"Dataset line {line_number} exceeds the 1 MiB line limit.")
                    value = _parse_training_line(bytes(pending), line_number, text_field)
                    if value is not None:
                        accepted += 1
                        if accepted > _MAX_ROWS:
                            raise ValueError(f"Dataset exceeds the {_MAX_ROWS:,}-row limit.")
                        yield value
                return
    except UnicodeDecodeError:
        raise ValueError("Dataset must be UTF-8 encoded JSONL.") from None


def _load_texts(dataset_path: Path, text_field: str) -> list[str]:
    texts = list(iter_training_texts(dataset_path, text_field))
    if not texts:
        raise ValueError("Dataset contains no non-empty training records.")
    return texts


def validate_config(
    config: dict[str, Any],
    *,
    workspace_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate user-supplied paths and conservative LoRA controls before creating a job."""
    if not isinstance(config, dict):
        raise ValueError("Fine-tuning configuration must be a mapping.")
    model_path = _resolve_existing(config.get("model_path", ""), "Base model directory")
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise ValueError("Base model must be an existing local Transformers model directory containing config.json.")
    dataset_path = _resolve_existing(config.get("dataset_path", ""), "Training dataset")
    if not dataset_path.is_file():
        raise ValueError("Training dataset must be a local JSONL file.")
    output_raw = config.get("output_dir", "")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise ValueError("Adapter output directory is required.")
    try:
        output_dir = Path(output_raw).expanduser().resolve(strict=False)
        parent = output_dir.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("The parent of the adapter output directory must already exist locally.") from None
    if not parent.is_dir():
        raise ValueError("The parent of the adapter output directory must be a directory.")
    if output_dir == model_path or model_path in output_dir.parents or output_dir in model_path.parents:
        raise ValueError("Adapter output and base model directories must not overlap.")
    if output_dir == dataset_path or output_dir == dataset_path.parent:
        raise ValueError("Adapter output must not overwrite the training dataset or its directory.")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError("Adapter output must be a new directory or an existing empty directory.")
    if workspace_guard is not None:
        _validate_workspace_guard({
            "model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
        }, workspace_guard)
    validate_local_model_tree(model_path)
    text_field = config.get("text_field", "text")
    validate_training_text_field(text_field)
    max_steps = config.get("max_steps", 20)
    sequence_length = config.get("sequence_length", 512)
    lora_rank = config.get("lora_rank", 8)
    learning_rate = config.get("learning_rate", 0.0001)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= _MAX_STEPS:
        raise ValueError(f"Maximum training steps must be between 1 and {_MAX_STEPS}.")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or not 32 <= sequence_length <= _MAX_SEQUENCE_LENGTH:
        raise ValueError(f"Sequence length must be between 32 and {_MAX_SEQUENCE_LENGTH} tokens.")
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or not 1 <= lora_rank <= 64:
        raise ValueError("LoRA rank must be an integer between 1 and 64.")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not 0.000001 <= float(learning_rate) <= 0.01:
        raise ValueError("Learning rate must be between 0.000001 and 0.01.")
    result = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "text_field": text_field,
        "max_steps": max_steps,
        "sequence_length": sequence_length,
        "learning_rate": float(learning_rate),
        "lora_rank": lora_rank,
    }
    _load_training_texts(
        {"dataset_path": str(dataset_path), "text_field": text_field},
        workspace_guard,
    )
    return result


def _validate_workspace_guard(config: dict[str, Any], guard: dict[str, Any] | None) -> None:
    if guard is None:
        return
    if not isinstance(guard, dict) or not isinstance(guard.get("root"), str) or not isinstance(guard.get("paths"), dict):
        raise FineTuningError("The MCP workspace training guard is invalid.")
    root = Path(guard["root"])
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise FineTuningError("The MCP workspace changed after the training request.") from None
    if resolved_root != root or not resolved_root.is_dir():
        raise FineTuningError("The MCP workspace changed after the training request.")

    from axiom.core.training_preflight import resolve_workspace_input, resolve_workspace_output

    paths = guard["paths"]
    for key, label in (("model_path", "Model path"), ("dataset_path", "Dataset path")):
        relative = paths.get(key)
        if not isinstance(relative, str):
            raise FineTuningError("The MCP workspace training paths are invalid.")
        try:
            current = resolve_workspace_input(relative, root, label)
        except (OSError, RuntimeError, ValueError):
            raise FineTuningError(f"{label} changed or is no longer safe inside the workspace.") from None
        if current != Path(config[key]):
            raise FineTuningError(f"{label} changed after the training request.")

    output_relative = paths.get("output_dir")
    if not isinstance(output_relative, str):
        raise FineTuningError("The MCP workspace training output is invalid.")
    try:
        current_output = resolve_workspace_output(output_relative, root)
    except (OSError, RuntimeError, ValueError):
        raise FineTuningError("Adapter output changed or is no longer safe inside outputs/.") from None
    if current_output != Path(config["output_dir"]):
        raise FineTuningError("Adapter output changed after the training request.")


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid fine-tuning job ID.")
    return jobs_directory() / f"{job_id}.json"


def _write_job(job: dict[str, Any]) -> None:
    folder = jobs_directory()
    folder.mkdir(parents=True, exist_ok=True)
    target = _job_path(job["id"])
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)


def _read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.is_file():
        return None
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise FineTuningError("Saved fine-tuning job state could not be read.") from None
    return job if isinstance(job, dict) else None


def _recover_if_stale(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") in {"queued", "running", "cancelling", "finalizing"} and not _pid_is_alive(job.get("runner_pid")):
        job = dict(job)
        reason = "The AXIOM process exited before this training job completed. Start a new job after checking resources."
        job.update(status="failed", error=reason, finished_at=_now(), updated_at=_now())
        job["logs"] = [*job.get("logs", []), f"Failed: {reason}"][-80:]
        _write_job(job)
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    job = _read_job(job_id)
    return _recover_if_stale(job) if job is not None else None


def cancel_finetuning_job(job_id: str) -> dict[str, Any] | None:
    """Request cancellation at the next safe training-step boundary."""
    with _LOCAL_LOCK, file_lock(jobs_directory() / ".job-state.lock"):
        job = _read_job(job_id)
        if job is None:
            return None
        job = _recover_if_stale(job)
        if job.get("status") not in {"queued", "running", "cancelling"}:
            return job
        if job.get("status") != "cancelling":
            job["cancel_requested"] = True
            job["status"] = "cancelling"
            job["updated_at"] = _now()
            job["logs"] = [*job.get("logs", []), "Cancellation requested; stopping at the next safe training boundary."][-80:]
            _write_job(job)
        return job


def list_jobs(limit: int = 20, *, workspace_root: Path | None = None) -> list[dict[str, Any]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Job list limit must be between 1 and 100.")
    root = Path(workspace_root).resolve(strict=True) if workspace_root is not None else None
    folder = jobs_directory()
    if not folder.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in folder.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and re.fullmatch(r"[0-9a-f]{32}", str(value.get("id", ""))):
                if root is None or _job_belongs_to_workspace(value, root):
                    records.append(_recover_if_stale(value))
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return records[:limit]


def _job_belongs_to_workspace(job: dict[str, Any], root: Path) -> bool:
    config = job.get("config")
    if not isinstance(config, dict):
        return False
    for key in ("model_path", "dataset_path", "output_dir"):
        raw = config.get(key)
        if not isinstance(raw, str) or not raw:
            return False
        try:
            if not Path(raw).resolve(strict=False).is_relative_to(root):
                return False
        except (OSError, RuntimeError):
            return False
    return True


def _pid_is_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _acquire_lock(
    job_id: str,
    *,
    conflict_message: str = "Another local fine-tuning job is active. Check `axiom train jobs` before starting another.",
) -> None:
    lock = _lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with _LOCAL_LOCK:
        if lock.exists():
            try:
                current = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
            if _pid_is_alive(current.get("pid")):
                raise ActiveFineTuningJobError(conflict_message)
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                raise ActiveFineTuningJobError("The fine-tuning lock could not be recovered safely.") from None
        payload = json.dumps({"job_id": job_id, "pid": os.getpid(), "created_at": _now()})
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise ActiveFineTuningJobError(conflict_message) from None
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)


def _release_lock(job_id: str) -> None:
    lock = _lock_path()
    with _LOCAL_LOCK:
        try:
            current = json.loads(lock.read_text(encoding="utf-8"))
            if current.get("job_id") == job_id and current.get("pid") == os.getpid():
                lock.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            _LOG.debug(
                "Could not release the fine-tuning lock for job %s",
                job_id,
                exc_info=True,
            )


def acquire_local_compute_job_lock(job_id: str) -> None:
    """Reserve AXIOM's shared local fine-tuning/prediction-job slot."""
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise FineTuningError("Invalid local compute job ID.")
    _acquire_lock(
        job_id,
        conflict_message="Another AXIOM local compute job is active; wait for it to finish before starting another.",
    )


def release_local_compute_job_lock(job_id: str) -> None:
    """Release the shared local training/inference job slot."""
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise FineTuningError("Invalid local compute job ID.")
    _release_lock(job_id)


def _new_job(
    config: dict[str, Any],
    *,
    workspace_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validated = validate_config(config, workspace_guard=workspace_guard)
    require_runtime()
    job_id = uuid.uuid4().hex
    _acquire_lock(job_id)
    now = _now()
    job: dict[str, Any] = {
        "schema_version": 1,
        "id": job_id,
        "runner_pid": os.getpid(),
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "config": validated,
        "cancel_requested": False,
        "completed_steps": 0,
        "loss": None,
        "loss_samples": [],
        "error": None,
        "logs": ["Job queued. The base model and dataset are local inputs; no weights will be downloaded."],
    }
    try:
        _write_job(job)
    except Exception:
        _release_lock(job_id)
        raise
    return job


def start_finetuning_job(
    config: dict[str, Any],
    *,
    workspace_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate inputs, persist a job, and run it in a single background thread."""
    job = _new_job(config, workspace_guard=workspace_guard)
    worker = threading.Thread(
        target=execute_finetuning_job,
        args=(job["id"],),
        kwargs={"workspace_guard": workspace_guard},
        name=f"axiom-lora-{job['id'][:8]}",
        daemon=True,
    )
    try:
        worker.start()
    except RuntimeError:
        _update_job(job["id"], status="failed", error="Could not start the local training worker.", finished_at=_now())
        _release_lock(job["id"])
        raise FineTuningError("Could not start the local training worker.") from None
    return get_job(job["id"]) or job


def run_finetuning_inline(config: dict[str, Any], on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run a job synchronously for CLI use, retaining the same job history format."""
    job = _new_job(config)
    execute_finetuning_job(job["id"], on_progress=on_progress)
    return get_job(job["id"]) or job


def _update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    with _LOCAL_LOCK, file_lock(jobs_directory() / ".job-state.lock"):
        job = _read_job(job_id)
        if job is None:
            raise FineTuningError("Fine-tuning job was not found.")
        if changes.get("status") == "running" and job.get("cancel_requested") is True:
            changes.pop("status")
        job.update(changes)
        job["updated_at"] = _now()
        _write_job(job)
        return job


def _begin_finalizing(job_id: str) -> None:
    """Atomically close the cancellation window before adapter files are saved."""
    with _LOCAL_LOCK, file_lock(jobs_directory() / ".job-state.lock"):
        job = _read_job(job_id)
        if job is None:
            raise FineTuningError("Fine-tuning job was not found.")
        if job.get("cancel_requested") is True or job.get("status") == "cancelling":
            raise FineTuningCancelledError("Local fine-tuning stopped at a safe training boundary.")
        job.update(status="finalizing", updated_at=_now())
        job["logs"] = [*job.get("logs", []), "Training steps finished; writing the local adapter."][-80:]
        _write_job(job)


def _load_training_texts(
    config: dict[str, Any],
    workspace_guard: dict[str, Any] | None = None,
) -> list[str]:
    if workspace_guard is None:
        return _load_texts(Path(config["dataset_path"]), config["text_field"])
    from axiom.core.mcp_paths import open_mcp_workspace_file

    root = Path(workspace_guard["root"])
    relative = workspace_guard["paths"]["dataset_path"]
    _, stream = open_mcp_workspace_file(relative, root)
    try:
        size = os.fstat(stream.fileno()).st_size
        texts = list(iter_training_text_stream(stream, file_size_bytes=size, text_field=config["text_field"]))
        if not texts:
            raise ValueError("Dataset contains no non-empty training records.")
        return texts
    finally:
        stream.close()


def _assert_output_empty(output_dir: Path) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise FineTuningError("Adapter output directory changed or is no longer a safe directory.")
    try:
        next(output_dir.iterdir())
    except StopIteration:
        return
    except OSError:
        raise FineTuningError("Adapter output directory could not be inspected safely.") from None
    raise FineTuningError("Adapter output directory is no longer empty; refusing to overwrite existing files.")


def _run_transformers(
    config: dict[str, Any],
    progress: Callable[[int, float | None], None],
    cancel_check: Callable[[], None] | None = None,
    begin_finalizing: Callable[[], None] | None = None,
    workspace_guard_check: Callable[[], None] | None = None,
    workspace_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a local causal-LM LoRA adapter with AXIOM's own bounded PyTorch loop."""
    if cancel_check:
        cancel_check()
    if workspace_guard_check:
        workspace_guard_check()
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError):
        raise FineTuningError(
            "Local LoRA dependencies could not be imported. Upgrade the optional training tools with: python -m pip install -e '.[training]'"
        ) from None

    texts = _load_training_texts(config, workspace_guard)
    if cancel_check:
        cancel_check()
    if workspace_guard_check:
        workspace_guard_check()
    model_path = config["model_path"]
    validate_local_model_tree(Path(model_path))
    output_dir = Path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FineTuningError("Adapter output directory is no longer empty; refusing to overwrite existing files.")

    try:
        if workspace_guard_check:
            workspace_guard_check()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if cancel_check:
            cancel_check()
        if workspace_guard_check:
            workspace_guard_check()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
    except FineTuningCancelledError:
        raise
    except Exception as exc:
        message = _clip(exc, 300)
        raise FineTuningError(
            "Could not load the local base model/tokenizer. Confirm model weights and tokenizer files exist "
            "and are compatible with the installed Transformers version." + (f" Detail: {message}" if message else "")
        ) from None

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise FineTuningError("Tokenizer needs an existing pad_token or eos_token to train safely.")
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    leaf_names = {
        module_name.rsplit(".", 1)[-1]
        for module_name, module in model.named_modules()
        if (isinstance(module, torch.nn.Linear) or module.__class__.__name__ == "Conv1D") and module_name
    }
    preferred = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj", "query_key_value", "Wqkv", "query", "value",
    )
    targets = [name for name in preferred if name in leaf_names]
    if not targets:
        raise FineTuningError(
            "No supported attention/MLP projection layers were found for LoRA. "
            "AXIOM supports common Llama-style q_proj/k_proj/v_proj/o_proj and several other known projection layouts."
        )

    try:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config["lora_rank"],
                lora_alpha=max(8, config["lora_rank"] * 2),
                lora_dropout=0.05,
                target_modules=targets,
                bias="none",
            ),
        )
    except Exception as exc:
        message = _clip(exc, 300)
        raise FineTuningError(
            "PEFT could not attach a LoRA adapter to this model architecture." + (f" Detail: {message}" if message else "")
        ) from None

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise FineTuningError("PEFT attached no trainable LoRA parameters; training was stopped before updating weights.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        if cancel_check:
            cancel_check()
        model.to(device)
    except FineTuningCancelledError:
        raise
    except Exception as exc:
        message = _clip(exc, 240)
        raise FineTuningError(
            f"Could not place the local model on {device}. The model may exceed available memory." + (f" Detail: {message}" if message else "")
        ) from None

    if cancel_check:
        cancel_check()
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config["learning_rate"])
    rng = random.Random(AXIOM_TRAINING_SEED)
    order: list[int] = []
    position = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, config["max_steps"] + 1):
        if cancel_check:
            cancel_check()
        if position >= len(order):
            order = list(range(len(texts)))
            rng.shuffle(order)
            position = 0
        text_index = order[position]
        position += 1
        try:
            encoded = tokenizer(
                texts[text_index],
                truncation=True,
                max_length=config["sequence_length"],
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            if not bool((labels != -100).any().item()):
                raise FineTuningError("A training record produced no unmasked tokens after tokenization.")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                raise FineTuningError(f"Training produced a non-finite loss at step {step}; the update was stopped.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except FineTuningError:
            raise
        except Exception as exc:
            message = _clip(exc, 300)
            raise FineTuningError(
                f"Local LoRA training failed at step {step} on {device}. Check available memory and model compatibility."
                + (f" Detail: {message}" if message else "")
            ) from None
        progress(step, loss_value)

    if begin_finalizing:
        begin_finalizing()
    elif cancel_check:
        cancel_check()
    if workspace_guard_check:
        workspace_guard_check()
    _assert_output_empty(output_dir)
    try:
        model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))
    except Exception as exc:
        message = _clip(exc, 300)
        raise FineTuningError("Training completed, but saving the adapter/tokenizer failed." + (f" Detail: {message}" if message else "")) from None

    adapter_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if not any(name in adapter_files for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FineTuningError("Training finished but a PEFT adapter weight file was not produced.")
    if "adapter_config.json" not in adapter_files:
        raise FineTuningError("Training finished but adapter_config.json was not produced.")
    return {
        "output_dir": str(output_dir),
        "adapter_files": adapter_files,
        "target_modules": targets,
        "dataset_rows": len(texts),
        "max_steps": config["max_steps"],
        "device": str(device),
        "engine": "axiom-lora-engine",
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
    }


def execute_finetuning_job(
    job_id: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    workspace_guard: dict[str, Any] | None = None,
) -> None:
    job = _read_job(job_id)
    if job is None:
        return
    loss_sample_steps = _training_loss_sample_steps(
        (job.get("config") or {}).get("max_steps")
        if isinstance(job.get("config"), dict)
        else None
    )

    def check_cancel() -> None:
        current = _read_job(job_id)
        if current is not None and current.get("cancel_requested") is True:
            raise FineTuningCancelledError("Local fine-tuning stopped at a safe boundary.")

    def check_workspace() -> None:
        _validate_workspace_guard(job["config"], workspace_guard)

    try:
        check_cancel()
        check_workspace()
        _update_job(job_id, status="running", started_at=_now(), error=None, logs=[*job.get("logs", []), "Loading local tokenizer/model and preparing LoRA adapters."][-80:])

        def progress(step: int, loss: float | None) -> None:
            check_cancel()
            current = _read_job(job_id)
            if current is None:
                return
            message = f"Step {max(0, step)}/{current['config']['max_steps']}" + (f" · loss {loss:.5f}" if loss is not None else "")
            changes: dict[str, Any] = {
                "completed_steps": min(max(0, step), current["config"]["max_steps"]),
                "loss": loss,
                "logs": [*current.get("logs", []), message][-80:],
            }
            if (
                step in loss_sample_steps
                and isinstance(loss, (int, float))
                and not isinstance(loss, bool)
                and math.isfinite(float(loss))
            ):
                loss_samples = _validated_training_loss_samples(current)
                sample = {"step": step, "loss": float(loss)}
                if loss_samples and loss_samples[-1]["step"] == step:
                    loss_samples[-1] = sample
                elif not loss_samples or loss_samples[-1]["step"] < step:
                    loss_samples.append(sample)
                changes["loss_samples"] = loss_samples[-_MAX_LOSS_SAMPLES:]
            updated = _update_job(job_id, **changes)
            if on_progress:
                on_progress(updated)

        result = _run_transformers(
            job["config"],
            progress,
            cancel_check=check_cancel,
            begin_finalizing=lambda: _begin_finalizing(job_id),
            workspace_guard_check=check_workspace,
            workspace_guard=workspace_guard,
        )
        current = _read_job(job_id) or job
        _update_job(
            job_id,
            status="completed",
            finished_at=_now(),
            completed_steps=current["config"]["max_steps"],
            result=result,
            error=None,
            logs=[*current.get("logs", []), "Training completed; PEFT adapter files were written to the selected output directory."][-80:],
        )
    except FineTuningCancelledError as exc:
        try:
            current = _read_job(job_id) or job
            _update_job(
                job_id,
                status="cancelled",
                cancel_requested=True,
                finished_at=_now(),
                error=None,
                logs=[*current.get("logs", []), str(exc)][-80:],
            )
        except Exception:
            _LOG.warning(
                "Could not record the cancelled state for fine-tuning job %s",
                job_id,
                exc_info=True,
            )
    except Exception as exc:
        reason = _clip(exc, 500) or type(exc).__name__
        try:
            current = _read_job(job_id) or job
            _update_job(
                job_id,
                status="failed",
                finished_at=_now(),
                error=reason,
                logs=[*current.get("logs", []), f"Failed: {reason}"][-80:],
            )
        except Exception:
            _LOG.warning(
                "Could not record the failed state for fine-tuning job %s",
                job_id,
                exc_info=True,
            )
    finally:
        _release_lock(job_id)
