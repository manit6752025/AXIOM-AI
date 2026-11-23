"""Bounded local LoRA fine-tuning using Transformers and PEFT.

No model downloads are permitted: the base model must be a local Transformers
model directory and all from_pretrained calls set local_files_only=True.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import random
import re
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable

from axiom.core.resource_ids import new_resource_id
from axiom.core.fine_tuning_checkpoints import (
    CHECKPOINT_DIRECTORY,
    checkpoint_count,
    latest_checkpoint,
    load_checkpoint_payload,
    model_tree_fingerprint,
    save_checkpoint,
)
from axiom.core.storage import file_lock
from axiom.path_safety import open_regular_file

_LOG = logging.getLogger(__name__)

_JOB_TERMINAL_STATUSES = frozenset({"completed", "cancelled", "failed"})
_JOB_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelling", "failed"}),
    "running": frozenset({"cancelling", "finalizing", "failed"}),
    "cancelling": frozenset({"cancelled", "failed"}),
    "finalizing": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
    "failed": frozenset(),
}


def _validate_job_transition(current: Any, target: str) -> None:
    """Reject lifecycle transitions that could misreport work or reopen cancellation."""
    if current == target:
        return
    allowed = _JOB_ALLOWED_TRANSITIONS.get(current) if isinstance(current, str) else None
    if allowed is None or target not in allowed:
        raise FineTuningError(f"Invalid fine-tuning job transition: {current} -> {target}.")


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
_MAX_CHECKPOINT_INTERVAL = 1_000
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


def iter_training_texts(
    dataset_path: Path,
    text_field: str = "text",
    *,
    on_bytes: Callable[[bytes], None] | None = None,
) -> Iterator[str]:
    """Yield trainer-accepted text values without retaining the full dataset."""
    with open_regular_file(dataset_path) as stream:
        yield from iter_training_text_stream(
            stream,
            file_size_bytes=os.fstat(stream.fileno()).st_size,
            text_field=text_field,
            on_bytes=on_bytes,
        )


def iter_training_text_stream(
    stream: BinaryIO,
    *,
    file_size_bytes: int,
    text_field: str = "text",
    on_bytes: Callable[[bytes], None] | None = None,
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
                if on_bytes is not None:
                    on_bytes(chunk)
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


def _load_texts(
    dataset_path: Path,
    text_field: str,
    *,
    on_bytes: Callable[[bytes], None] | None = None,
) -> list[str]:
    texts = list(iter_training_texts(dataset_path, text_field, on_bytes=on_bytes))
    if not texts:
        raise ValueError("Dataset contains no non-empty training records.")
    return texts


def _validate_resume_metadata(
    state: dict[str, Any],
    config: dict[str, Any],
    *,
    dataset_sha256: str,
    base_model_fingerprint: str,
) -> None:
    training = state.get("training")
    if not isinstance(training, dict):
        raise ValueError("Resume checkpoint has no valid training metadata.")
    expected = {
        "model_path": config["model_path"],
        "model_fingerprint": base_model_fingerprint,
        "dataset_path": config["dataset_path"],
        "dataset_sha256": dataset_sha256,
        "text_field": config["text_field"],
        "sequence_length": config["sequence_length"],
        "learning_rate": config["learning_rate"],
        "lora_rank": config["lora_rank"],
        "output_dir": config["output_dir"],
        "seed": AXIOM_TRAINING_SEED,
    }
    for key, value in expected.items():
        if training.get(key) != value:
            labels = {
                "model_path": "base-model path",
                "model_fingerprint": "base-model files",
                "dataset_path": "dataset path",
                "dataset_sha256": "dataset contents",
                "text_field": "dataset text field",
                "sequence_length": "sequence length",
                "learning_rate": "learning rate",
                "lora_rank": "LoRA rank",
                "output_dir": "output directory",
                "seed": "training seed",
            }
            raise ValueError(
                f"Resume checkpoint does not match the current {labels[key]}; "
                "use the same inputs and training settings."
            )
    completed_steps = state.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or not 1 <= completed_steps <= config["max_steps"]
    ):
        raise ValueError("Resume checkpoint step count must not exceed the requested maximum steps.")
    targets = training.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        raise ValueError("Resume checkpoint has invalid LoRA target-module metadata.")


def validate_config(
    config: dict[str, Any],
    *,
    workspace_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate local training inputs, checkpoint compatibility, and output policy."""
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
    raw_output_path = Path(output_raw).expanduser()
    if raw_output_path.is_symlink() or getattr(raw_output_path, "is_junction", lambda: False)():
        raise ValueError("Adapter output directory cannot be a symbolic link.")
    try:
        output_dir = raw_output_path.resolve(strict=False)
        parent = output_dir.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("The parent of the adapter output directory must already exist locally.") from None
    if not parent.is_dir():
        raise ValueError("The parent of the adapter output directory must be a directory.")
    if output_dir == model_path or model_path in output_dir.parents or output_dir in model_path.parents:
        raise ValueError("Adapter output and base model directories must not overlap.")
    if output_dir == dataset_path or output_dir == dataset_path.parent:
        raise ValueError("Adapter output must not overwrite the training dataset or its directory.")

    text_field = config.get("text_field", "text")
    validate_training_text_field(text_field)
    max_steps = config.get("max_steps", 20)
    sequence_length = config.get("sequence_length", 512)
    lora_rank = config.get("lora_rank", 8)
    learning_rate = config.get("learning_rate", 0.0001)
    checkpoint_interval = config.get("checkpoint_interval", 0)
    output_policy = config.get("output_policy", "empty")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= _MAX_STEPS:
        raise ValueError(f"Maximum training steps must be between 1 and {_MAX_STEPS}.")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or not 32 <= sequence_length <= _MAX_SEQUENCE_LENGTH:
        raise ValueError(f"Sequence length must be between 32 and {_MAX_SEQUENCE_LENGTH} tokens.")
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or not 1 <= lora_rank <= 64:
        raise ValueError("LoRA rank must be an integer between 1 and 64.")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not 0.000001 <= float(learning_rate) <= 0.01:
        raise ValueError("Learning rate must be between 0.000001 and 0.01.")
    if (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, int)
        or not 0 <= checkpoint_interval <= _MAX_CHECKPOINT_INTERVAL
    ):
        raise ValueError(f"Checkpoint interval must be between 0 and {_MAX_CHECKPOINT_INTERVAL} steps.")
    if not isinstance(output_policy, str) or output_policy not in {"new", "empty", "resume"}:
        raise ValueError("Output policy must be 'new', 'empty', or 'resume'.")

    resume_state: dict[str, Any] | None = None
    if output_policy == "new":
        if output_dir.exists():
            raise ValueError("The 'new' output policy requires a directory that does not already exist.")
    elif output_policy == "empty":
        if output_dir.exists():
            if output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir()):
                raise ValueError("The 'empty' output policy requires a new or existing empty directory.")
    else:
        if checkpoint_interval < 1:
            raise ValueError("Resume policy requires checkpointing to remain enabled (interval 1–1000).")
        if not output_dir.exists() or output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("Resume policy requires an existing AXIOM training output directory.")
        try:
            existing = list(output_dir.iterdir())
        except OSError:
            raise ValueError("Resume output directory could not be inspected safely.") from None
        if not existing:
            raise ValueError("Resume policy requires at least one complete checkpoint.")
        if len(existing) != 1 or existing[0].name != CHECKPOINT_DIRECTORY or existing[0].is_symlink() or not existing[0].is_dir():
            raise ValueError("Resume policy only accepts an output directory containing AXIOM checkpoints and no final adapter files.")
        try:
            _, resume_state = latest_checkpoint(output_dir)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        if resume_state.get("completed_steps", 0) > max_steps:
            raise ValueError("Maximum training steps cannot be lower than the latest checkpoint step.")

    result = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "text_field": text_field,
        "max_steps": max_steps,
        "sequence_length": sequence_length,
        "learning_rate": float(learning_rate),
        "lora_rank": lora_rank,
        "checkpoint_interval": checkpoint_interval,
        "output_policy": output_policy,
    }
    if workspace_guard is not None:
        _validate_workspace_guard(result, workspace_guard)
    validate_local_model_tree(model_path)

    _, dataset_sha256 = _load_training_texts(
        {"dataset_path": str(dataset_path), "text_field": text_field},
        workspace_guard,
        include_fingerprint=True,
    )
    if resume_state is not None:
        try:
            base_model_fingerprint = model_tree_fingerprint(model_path)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        _validate_resume_metadata(
            resume_state,
            result,
            dataset_sha256=dataset_sha256,
            base_model_fingerprint=base_model_fingerprint,
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
        current_output = resolve_workspace_output(
            output_relative,
            root,
            allow_checkpoints=config.get("output_policy") == "resume",
        )
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
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    if folder.is_symlink() or not folder.is_dir():
        raise FineTuningError("Local training job storage is not a safe directory.")
    if os.name == "posix":
        try:
            os.chmod(folder, 0o700)
            folder_mode = stat.S_IMODE(os.stat(folder, follow_symlinks=False).st_mode)
        except OSError:
            raise FineTuningError("Local training job storage permissions could not be secured.") from None
        if folder_mode & 0o077:
            raise FineTuningError("Local training job storage must be private to the current user.")
    target = _job_path(job["id"])
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(job, ensure_ascii=False, indent=2)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temp, flags, 0o600)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None
        with stream:
            stream.write(payload)
        os.replace(temp, target)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temp.unlink()
        except OSError:
            pass
        raise FineTuningError("Local training job state could not be written safely.") from None


def _read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.is_file():
        return None
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise FineTuningError("Saved fine-tuning job state could not be read.") from None
    return job if isinstance(job, dict) else None


def _public_training_job(job: dict[str, Any]) -> dict[str, Any]:
    """Return safe checkpoint and cooperative-cancellation metadata."""
    public_job = dict(job)
    result = job.get("result")
    if isinstance(result, dict) and "training_dataset_sha256" in result:
        public_result = dict(result)
        public_result.pop("training_dataset_sha256", None)
        public_job["result"] = public_result

    checkpoint = job.get("checkpoint")
    if isinstance(checkpoint, dict):
        public_job["checkpoint"] = {
            key: checkpoint.get(key)
            for key in ("enabled", "interval", "latest_step", "retained", "last_saved_at", "resume_available")
        }

    status = job.get("status")
    requested = job.get("cancel_requested") is True
    requestable = isinstance(status, str) and status in {"queued", "running"} and not requested
    if requestable:
        unavailable_reason = None
    elif status == "cancelling":
        unavailable_reason = "already_requested"
    elif status == "finalizing":
        unavailable_reason = "finalization_started"
    elif status in _JOB_TERMINAL_STATUSES:
        unavailable_reason = "job_terminal"
    else:
        unavailable_reason = "job_not_cancellable"
    public_job["cancellation"] = {
        "supported": True,
        "requested": requested,
        "requestable": requestable,
        "mode": "cooperative",
        "boundary": "before worker start or between training steps; adapter finalization is not interruptible",
        "unavailable_reason": unavailable_reason,
    }
    return public_job


def _recover_if_stale_locked(job: dict[str, Any]) -> dict[str, Any]:
    """Recover an exited job while the cross-process job-state lock is held."""
    job_id = job.get("id")
    if not isinstance(job_id, str):
        return job
    current = _read_job(job_id)
    if current is None:
        return job
    if current.get("status") in {"queued", "running", "cancelling", "finalizing"}:
        runner_state = _process_state(current.get("runner_pid"), current.get("runner_identity"))
        if runner_state == "exited":
            current = dict(current)
            reason = "The AXIOM process exited before this training job completed. Start a new job after checking resources."
            _validate_job_transition(current.get("status"), "failed")
            current.update(status="failed", error=reason, finished_at=_now(), updated_at=_now())
            logs = current.get("logs")
            current["logs"] = [*(logs if isinstance(logs, list) else []), f"Failed: {reason}"][-80:]
            _write_job(current)
        elif runner_state == "unknown":
            current = dict(current)
            current["_runner_state"] = "unknown"
    return current


def _recover_if_stale(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") not in {"queued", "running", "cancelling", "finalizing"}:
        return job
    if _process_state(job.get("runner_pid"), job.get("runner_identity")) == "alive":
        return job
    with _LOCAL_LOCK, file_lock(jobs_directory() / ".job-state.lock"):
        return _recover_if_stale_locked(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    job = _read_job(job_id)
    return _public_training_job(_recover_if_stale(job)) if job is not None else None


def read_job_record(job_id: str) -> dict[str, Any] | None:
    """Read a saved job record without stale recovery or other state mutation."""
    return _read_job(job_id)


def wait_finetuning_job(
    job_id: str,
    *,
    timeout_seconds: int | float = 30,
) -> dict[str, Any]:
    """Wait briefly for a local fine-tuning job outcome."""
    try:
        timeout = float(timeout_seconds)
    except (OverflowError, TypeError, ValueError):
        timeout = float("nan")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout)
        or not 0 <= timeout <= 30
    ):
        raise ValueError("Training job wait must be between 0 and 30 seconds.")

    started = time.monotonic()
    deadline = started + timeout
    terminal_statuses = {"completed", "cancelled", "failed"}
    active_statuses = {"queued", "running", "cancelling", "finalizing"}
    job: dict[str, Any] | None = None
    while True:
        try:
            job = get_job(job_id)
        except FineTuningError:
            outcome = "state_unavailable"
            break
        if job is None:
            outcome = "not_found"
            break
        if job.get("_runner_state") == "unknown":
            outcome = "state_unavailable"
            break
        status = job.get("status")
        if status in terminal_statuses:
            outcome = status
            break
        if status not in active_statuses:
            outcome = "state_unavailable"
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            outcome = "timed_out"
            break
        time.sleep(min(0.5, remaining))

    waited_seconds = round(max(0.0, time.monotonic() - started), 3)
    return {
        "job": job,
        "wait_outcome": outcome,
        "timed_out": outcome == "timed_out",
        "waited_seconds": waited_seconds,
    }


def cancel_finetuning_job(job_id: str) -> dict[str, Any] | None:
    """Request cancellation at the next safe training-step boundary."""
    with _LOCAL_LOCK, file_lock(jobs_directory() / ".job-state.lock"):
        job = _read_job(job_id)
        if job is None:
            return None
        job = _recover_if_stale_locked(job)
        if job.get("status") not in {"queued", "running", "cancelling"}:
            return _public_training_job(job)
        if job.get("status") != "cancelling":
            _validate_job_transition(job.get("status"), "cancelling")
            job["cancel_requested"] = True
            job["status"] = "cancelling"
            job["updated_at"] = _now()
            job["logs"] = [*job.get("logs", []), "Cancellation requested; stopping at the next safe training boundary."][-80:]
            _write_job(job)
        return _public_training_job(job)


MAX_FINE_TUNING_JOB_SCAN = 5_000


def list_jobs_page(
    limit: int = 20,
    offset: int = 0,
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Return a bounded page of newest-first, workspace-safe fine-tuning jobs."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Job list limit must be between 1 and 100.")
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= MAX_FINE_TUNING_JOB_SCAN:
        raise ValueError(f"Job list offset must be between 0 and {MAX_FINE_TUNING_JOB_SCAN}.")
    root = Path(workspace_root).resolve(strict=True) if workspace_root is not None else None
    folder = jobs_directory()
    if folder.is_symlink():
        raise FineTuningError("Fine-tuning job directory must not be a symbolic link.")
    if not folder.is_dir():
        return {"jobs": [], "total": 0, "limit": limit, "offset": offset,
                "has_more": False, "truncated": False, "scan_limit": MAX_FINE_TUNING_JOB_SCAN}

    paths = [path for path in folder.glob("*.json") if not path.is_symlink() and path.is_file()]
    truncated = len(paths) > MAX_FINE_TUNING_JOB_SCAN
    if truncated:
        def modified_ns(path: Path) -> int:
            try:
                return path.stat().st_mtime_ns
            except OSError:
                return 0
        paths.sort(key=modified_ns, reverse=True)
        paths = paths[:MAX_FINE_TUNING_JOB_SCAN]

    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            with open_regular_file(path) as stream:
                value = json.load(stream)
            if isinstance(value, dict) and re.fullmatch(r"[0-9a-f]{32}", str(value.get("id", ""))):
                if root is None or _job_belongs_to_workspace(value, root):
                    records.append(_public_training_job(_recover_if_stale(value)))
        except (OSError, json.JSONDecodeError, ValueError):
            continue

    records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    total = len(records)
    page = records[offset:offset + limit]
    return {
        "jobs": page, "total": total, "limit": limit, "offset": offset,
        "has_more": offset + len(page) < total, "truncated": truncated,
        "scan_limit": MAX_FINE_TUNING_JOB_SCAN,
    }


def list_jobs(limit: int = 20, *, workspace_root: Path | None = None) -> list[dict[str, Any]]:
    """Backward-compatible first-page view of recent local fine-tuning jobs."""
    return list_jobs_page(limit=limit, workspace_root=workspace_root)["jobs"]


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


def _windows_process_state(pid: int) -> tuple[str, str | None]:
    """Inspect Windows process liveness without using os.kill, which is unsafe there."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000 | 0x00100000, False, pid)
        if not handle:
            return ("exited", None) if ctypes.get_last_error() == 87 else ("unknown", None)
        try:
            wait_result = kernel.WaitForSingleObject(handle, 0)
            if wait_result == 0:  # WAIT_OBJECT_0: the process has exited.
                return "exited", None
            if wait_result != 258:  # WAIT_TIMEOUT: the process is still running.
                return "unknown", None
            created = wintypes.FILETIME()
            exited = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            if not kernel.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                return "unknown", None
            identity = str((created.dwHighDateTime << 32) | created.dwLowDateTime)
            return "alive", identity
        finally:
            kernel.CloseHandle(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return "unknown", None


def _process_start_identity(pid: int) -> str | None:
    """Return an OS process start token so reused PIDs are not mistaken for owners."""
    if os.name == "nt":
        state, identity = _windows_process_state(pid)
        return identity if state == "alive" else None
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file() and not proc_stat.is_symlink():
        try:
            stat_text = proc_stat.read_text(encoding="ascii")
            end_of_command = stat_text.rfind(")")
            fields = stat_text[end_of_command + 2 :].split()
            if end_of_command >= 0 and len(fields) > 19:
                return f"linux:{fields[19]}"
        except (OSError, UnicodeError, IndexError):
            return None
    try:
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
            env={"LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    token = result.stdout.strip()
    return token[:128] if token else None


def _process_state(pid: Any, identity: Any) -> str:
    """Return alive, exited, or unknown without treating query failure as exit."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return "exited"
    if os.name == "nt":
        state, observed_identity = _windows_process_state(pid)
        if state != "alive":
            return state
        if not isinstance(identity, str) or not identity:
            return "unknown"
        return "alive" if observed_identity == identity else "exited"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited"
    except PermissionError:
        pass
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return "exited"
        if exc.errno != errno.EPERM:
            return "unknown"
    if not isinstance(identity, str) or not identity:
        return "unknown"
    observed_identity = _process_start_identity(pid)
    if observed_identity is None:
        return "unknown"
    return "alive" if observed_identity == identity else "exited"


def _acquire_lock(
    job_id: str,
    *,
    conflict_message: str = "Another local fine-tuning job is active. Check `axiom train jobs` before starting another.",
) -> None:
    lock = _lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    guard = lock.with_name(f"{lock.name}.guard")
    with _LOCAL_LOCK, file_lock(guard):
        if lock.exists():
            try:
                current = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = None
            pid = current.get("pid") if isinstance(current, dict) else None
            owner_state = (
                _process_state(pid, current.get("process_identity"))
                if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
                else "unknown"
            )
            if owner_state != "exited":
                raise ActiveFineTuningJobError(conflict_message)
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                raise ActiveFineTuningJobError("The fine-tuning lock could not be recovered safely.") from None
        payload = json.dumps({
            "job_id": job_id,
            "pid": os.getpid(),
            "process_identity": _process_start_identity(os.getpid()),
            "created_at": _now(),
        })
        temporary = lock.with_name(f".{lock.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
            try:
                os.link(temporary, lock)
            except FileExistsError:
                raise ActiveFineTuningJobError(conflict_message) from None
            except OSError:
                raise ActiveFineTuningJobError("The fine-tuning lock could not be established safely.") from None
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


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
    resume_step = 0
    resume_training: dict[str, Any] = {}
    checkpoint = {
        "enabled": validated["checkpoint_interval"] > 0,
        "interval": validated["checkpoint_interval"],
        "latest_step": 0,
        "retained": 0,
        "last_saved_at": None,
        "resume_available": False,
    }
    if validated["output_policy"] == "resume":
        _, state = latest_checkpoint(Path(validated["output_dir"]))
        raw_training = state.get("training")
        resume_training = raw_training if isinstance(raw_training, dict) else {}
        resume_step = state["completed_steps"]
        checkpoint.update(
            latest_step=resume_step,
            retained=checkpoint_count(Path(validated["output_dir"])),
            last_saved_at=state.get("created_at"),
            resume_available=True,
        )
    job_id = new_resource_id()
    _acquire_lock(job_id)
    now = _now()
    queue_message = "Job queued. The base model and dataset are local inputs; no weights will be downloaded."
    if resume_step:
        queue_message = f"Job queued to resume from the latest verified checkpoint at step {resume_step}."
    job: dict[str, Any] = {
        "schema_version": 1,
        "id": job_id,
        "runner_pid": os.getpid(),
        "runner_identity": _process_start_identity(os.getpid()),
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "config": validated,
        "cancel_requested": False,
        "completed_steps": resume_step,
        "loss": float(resume_training["last_loss"]) if isinstance(resume_training.get("last_loss"), (int, float)) and not isinstance(resume_training.get("last_loss"), bool) and math.isfinite(float(resume_training["last_loss"])) else None,
        "checkpoint": checkpoint,
        "loss_samples": [],
        "error": None,
        "logs": [queue_message],
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
        requested_status = changes.get("status")
        if isinstance(requested_status, str):
            _validate_job_transition(job.get("status"), requested_status)
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
        _validate_job_transition(job.get("status"), "finalizing")
        job.update(status="finalizing", updated_at=_now())
        job["logs"] = [*job.get("logs", []), "Training steps finished; writing the local adapter."][-80:]
        _write_job(job)


def _load_training_texts(
    config: dict[str, Any],
    workspace_guard: dict[str, Any] | None = None,
    *,
    include_fingerprint: bool = False,
) -> list[str] | tuple[list[str], str]:
    digest = hashlib.sha256() if include_fingerprint else None
    on_bytes = digest.update if digest is not None else None
    if workspace_guard is None:
        texts = _load_texts(Path(config["dataset_path"]), config["text_field"], on_bytes=on_bytes)
    else:
        from axiom.core.mcp_paths import open_mcp_workspace_file

        root = Path(workspace_guard["root"])
        relative = workspace_guard["paths"]["dataset_path"]
        _, stream = open_mcp_workspace_file(relative, root)
        try:
            size = os.fstat(stream.fileno()).st_size
            texts = list(
                iter_training_text_stream(
                    stream,
                    file_size_bytes=size,
                    text_field=config["text_field"],
                    on_bytes=on_bytes,
                )
            )
            if not texts:
                raise ValueError("Dataset contains no non-empty training records.")
        finally:
            stream.close()
    if digest is not None:
        return texts, digest.hexdigest()
    return texts


def _assert_output_empty(output_dir: Path) -> None:
    if output_dir.is_symlink() or getattr(output_dir, "is_junction", lambda: False)() or not output_dir.is_dir():
        raise FineTuningError("Adapter output directory changed or is no longer a safe directory.")
    try:
        next(output_dir.iterdir())
    except StopIteration:
        return
    except OSError:
        raise FineTuningError("Adapter output directory could not be inspected safely.") from None
    raise FineTuningError("Adapter output directory is no longer empty; refusing to overwrite existing files.")


def _prepare_output_directory(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    policy = config.get("output_policy", "empty")
    if policy == "resume":
        _assert_output_ready_for_finalization(output_dir)
        return
    if policy == "new":
        try:
            output_dir.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            raise FineTuningError("The 'new' output policy path now exists; refusing to use it.") from None
        except OSError:
            raise FineTuningError("The new adapter output directory could not be created safely.") from None
        return
    if output_dir.exists():
        _assert_output_empty(output_dir)
    else:
        try:
            output_dir.mkdir(mode=0o700, exist_ok=False)
        except OSError:
            raise FineTuningError("The adapter output directory could not be created safely.") from None


def _assert_output_ready_for_finalization(
    output_dir: Path,
    *,
    require_checkpoint: bool = False,
) -> None:
    if output_dir.is_symlink() or getattr(output_dir, "is_junction", lambda: False)() or not output_dir.is_dir():
        raise FineTuningError("Adapter output directory changed or is no longer a safe directory.")
    try:
        entries = list(output_dir.iterdir())
    except OSError:
        raise FineTuningError("Adapter output directory could not be inspected safely.") from None
    if not entries:
        if require_checkpoint:
            raise FineTuningError("Resume checkpoint storage disappeared before finalization.")
        return
    if len(entries) != 1 or entries[0].name != CHECKPOINT_DIRECTORY:
        raise FineTuningError("Adapter output contains user-managed files; refusing to overwrite them.")
    try:
        retained = checkpoint_count(output_dir)
    except ValueError as exc:
        raise FineTuningError(str(exc)) from None
    if retained < 1:
        raise FineTuningError("Adapter output contains no complete checkpoint; refusing to save into an uncertain directory.")


def _run_transformers(
    config: dict[str, Any],
    progress: Callable[[int, float | None], None],
    cancel_check: Callable[[], None] | None = None,
    begin_finalizing: Callable[[], None] | None = None,
    workspace_guard_check: Callable[[], None] | None = None,
    workspace_guard: dict[str, Any] | None = None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
    on_resume: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train or resume a local causal-LM LoRA adapter with bounded AXIOM checkpoints."""
    if cancel_check:
        cancel_check()
    if workspace_guard_check:
        workspace_guard_check()
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError):
        raise FineTuningError(
            "Local LoRA dependencies could not be imported. Upgrade the optional training tools with: python -m pip install -e '.[training]'"
        ) from None

    if config["output_policy"] != "resume":
        torch.manual_seed(AXIOM_TRAINING_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(AXIOM_TRAINING_SEED)

    texts, training_dataset_sha256 = _load_training_texts(
        config,
        workspace_guard,
        include_fingerprint=True,
    )
    if cancel_check:
        cancel_check()
    if workspace_guard_check:
        workspace_guard_check()
    model_path = config["model_path"]
    validate_local_model_tree(Path(model_path))
    output_dir = Path(config["output_dir"])
    resume_checkpoint: Path | None = None
    resume_state: dict[str, Any] | None = None
    resume_payload: dict[str, Any] | None = None
    base_model_fingerprint = model_tree_fingerprint(Path(model_path))
    if config["output_policy"] == "resume":
        _assert_output_ready_for_finalization(output_dir)
        try:
            resume_checkpoint, resume_state = latest_checkpoint(output_dir)
            _validate_resume_metadata(
                resume_state,
                config,
                dataset_sha256=training_dataset_sha256,
                base_model_fingerprint=base_model_fingerprint,
            )
            loaded = load_checkpoint_payload(resume_checkpoint, torch)
            resume_state = loaded["state"]
            resume_payload = loaded["payload"]
        except ValueError as exc:
            raise FineTuningError(str(exc)) from None

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
    if resume_state is not None:
        training = resume_state.get("training", {})
        targets = training.get("target_modules")
        if not isinstance(targets, list) or not targets or any(target not in leaf_names for target in targets):
            raise FineTuningError("Resume checkpoint target modules do not match the selected base model.")
        try:
            model = PeftModel.from_pretrained(
                model, str(resume_checkpoint), is_trainable=True, local_files_only=True
            )
        except Exception as exc:
            message = _clip(exc, 300)
            raise FineTuningError(
                "PEFT could not restore the adapter from the selected local checkpoint."
                + (f" Detail: {message}" if message else "")
            ) from None
    else:
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
                "PEFT could not attach a LoRA adapter to this model architecture."
                + (f" Detail: {message}" if message else "")
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

    if resume_state is not None and resume_state.get("training", {}).get("device") != str(device):
        raise FineTuningError(
            "Resume requires the same compute device as the saved checkpoint; "
            "select the original device or start a new training run."
        )

    try:
        if cancel_check:
            cancel_check()
        model.to(device)
    except FineTuningCancelledError:
        raise
    except Exception as exc:
        message = _clip(exc, 240)
        raise FineTuningError(
            f"Could not place the local model on {device}. The model may exceed available memory."
            + (f" Detail: {message}" if message else "")
        ) from None

    if cancel_check:
        cancel_check()
    if workspace_guard_check:
        workspace_guard_check()
    _prepare_output_directory(config)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=config["learning_rate"])
    rng = random.Random(AXIOM_TRAINING_SEED)
    order: list[int] = []
    position = 0
    start_step = 0
    if resume_state is not None and resume_payload is not None:
        try:
            optimizer_state = resume_payload.get("optimizer_state_dict")
            python_rng_state = resume_payload.get("python_rng_state")
            torch_rng_state = resume_payload.get("torch_cpu_rng_state")
            saved_order = resume_payload.get("order")
            saved_position = resume_payload.get("position")
            if not isinstance(optimizer_state, dict):
                raise ValueError("optimizer state is missing")
            if not isinstance(saved_order, list) or len(saved_order) != len(texts):
                raise ValueError("dataset sampling order does not match the current dataset")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in saved_order):
                raise ValueError("dataset sampling order is invalid")
            if sorted(saved_order) != list(range(len(texts))):
                raise ValueError("dataset sampling order is not a complete permutation")
            if isinstance(saved_position, bool) or not isinstance(saved_position, int) or not 0 <= saved_position <= len(texts):
                raise ValueError("dataset sampling position is invalid")
            if torch_rng_state is None:
                raise ValueError("CPU random-number state is missing")
            optimizer.load_state_dict(optimizer_state)
            rng.setstate(python_rng_state)
            torch.set_rng_state(torch_rng_state.cpu())
            saved_cuda_states = resume_payload.get("torch_cuda_rng_states")
            if device.type == "cuda":
                if (
                    not isinstance(saved_cuda_states, list)
                    or len(saved_cuda_states) != torch.cuda.device_count()
                    or training.get("cuda_device_count") != torch.cuda.device_count()
                ):
                    raise ValueError("CUDA device count differs from the saved checkpoint")
                torch.cuda.set_rng_state_all(saved_cuda_states)
            if device.type == "mps":
                mps_rng_state = resume_payload.get("torch_mps_rng_state")
                if mps_rng_state is None:
                    raise ValueError("MPS random-number state is missing")
                torch.mps.set_rng_state(mps_rng_state)
            order = saved_order
            position = saved_position
            start_step = int(resume_state["completed_steps"])
        except Exception as exc:
            message = _clip(exc, 240)
            raise FineTuningError(
                "The local checkpoint optimizer or random-number state could not be restored safely."
                + (f" Detail: {message}" if message else "")
            ) from None
        if on_resume:
            checkpoint_summary = {
                "enabled": True,
                "interval": config["checkpoint_interval"],
                "latest_step": start_step,
                "retained": checkpoint_count(output_dir),
                "last_saved_at": resume_state.get("created_at"),
                "resume_available": True,
                "last_loss": resume_state.get("training", {}).get("last_loss"),
            }
            on_resume(checkpoint_summary)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    latest_checkpoint_summary: dict[str, Any] | None = None
    for step in range(start_step + 1, config["max_steps"] + 1):
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

        interval = config["checkpoint_interval"]
        if interval > 0 and (step % interval == 0 or step == config["max_steps"]):
            optimizer_payload = {
                "optimizer_state_dict": optimizer.state_dict(),
                "python_rng_state": rng.getstate(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "torch_mps_rng_state": torch.mps.get_rng_state() if device.type == "mps" else None,
                "order": order,
                "position": position,
            }
            training_metadata = {
                "model_path": config["model_path"],
                "model_fingerprint": base_model_fingerprint,
                "dataset_path": config["dataset_path"],
                "dataset_sha256": training_dataset_sha256,
                "text_field": config["text_field"],
                "sequence_length": config["sequence_length"],
                "learning_rate": config["learning_rate"],
                "lora_rank": config["lora_rank"],
                "output_dir": config["output_dir"],
                "seed": AXIOM_TRAINING_SEED,
                "target_modules": targets,
                "max_steps_at_save": config["max_steps"],
                "device": str(device),
                "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
                "dataset_rows": len(texts),
                "last_loss": loss_value,
            }
            try:
                latest_checkpoint_summary = save_checkpoint(
                    output_dir,
                    step,
                    model=model,
                    torch_module=torch,
                    optimizer_payload=optimizer_payload,
                    training_metadata=training_metadata,
                )
            except Exception as exc:
                message = _clip(exc, 240)
                raise FineTuningError(
                    "Could not persist a verified local training checkpoint."
                    + (f" Detail: {message}" if message else "")
                ) from None
            latest_checkpoint_summary["interval"] = interval
            if on_checkpoint:
                on_checkpoint(latest_checkpoint_summary)

        progress(step, loss_value)

    if begin_finalizing:
        begin_finalizing()
    elif cancel_check:
        cancel_check()
    if workspace_guard_check:
        workspace_guard_check()
    _assert_output_ready_for_finalization(output_dir, require_checkpoint=config["output_policy"] == "resume")
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
        "training_dataset_sha256": training_dataset_sha256,
        "max_steps": config["max_steps"],
        "device": str(device),
        "engine": "axiom-lora-engine",
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "resumed_from_step": start_step,
        "checkpoint": latest_checkpoint_summary or (resume_state and {
            "enabled": True,
            "interval": config["checkpoint_interval"],
            "latest_step": resume_state["completed_steps"],
            "retained": checkpoint_count(output_dir),
            "last_saved_at": resume_state.get("created_at"),
            "resume_available": True,
        }),
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
            # This callback runs after the optimizer step. Persist that completed
            # step before observing cancellation requested while it was in flight.
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
            check_cancel()

        def checkpoint_progress(summary: dict[str, Any]) -> None:
            current = _read_job(job_id)
            if current is None:
                return
            message = f"Saved verified checkpoint at step {summary.get('latest_step', 0)}."
            _update_job(
                job_id,
                checkpoint=summary,
                logs=[*current.get("logs", []), message][-80:],
            )

        def resume_progress(summary: dict[str, Any]) -> None:
            current = _read_job(job_id)
            if current is None:
                return
            step = summary.get("latest_step", 0)
            loss = summary.get("last_loss")
            changes: dict[str, Any] = {
                "completed_steps": step,
                "checkpoint": {
                    key: value for key, value in summary.items() if key != "last_loss"
                },
                "logs": [*current.get("logs", []), f"Resumed verified checkpoint at step {step}."][-80:],
            }
            if isinstance(loss, (int, float)) and not isinstance(loss, bool) and math.isfinite(float(loss)):
                changes["loss"] = float(loss)
            _update_job(job_id, **changes)

        result = _run_transformers(
            job["config"],
            progress,
            cancel_check=check_cancel,
            begin_finalizing=lambda: _begin_finalizing(job_id),
            workspace_guard_check=check_workspace,
            workspace_guard=workspace_guard,
            on_checkpoint=checkpoint_progress,
            on_resume=resume_progress,
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
