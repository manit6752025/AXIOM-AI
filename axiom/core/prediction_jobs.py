"""Cancellable local prediction jobs that produce evaluation-ready JSONL."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from axiom.core.engine import (
    generate_local_batch,
    inspect_native_prompt_batch,
    refresh_local_engine_asset_index,
    resolve_local_model_directory,
    runtime_status,
)
from axiom.core.fine_tuning import (
    acquire_local_compute_job_lock,
    list_jobs,
    release_local_compute_job_lock,
    validate_local_model_tree,
)
from axiom.core.mcp_paths import open_mcp_workspace_file, open_mcp_workspace_output_parent
from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.training_preflight import resolve_workspace_input
from axiom.evaluation.jsonl import DEFAULT_MAX_BYTES as _MAX_OUTPUT_BYTES
from axiom.evaluation.jsonl import DEFAULT_MAX_LINE_BYTES as _MAX_OUTPUT_LINE_BYTES

_MAX_ROWS = 100
_MAX_INPUT_BYTES = 48 * 1024 * 1024
_MAX_INPUT_LINE_BYTES = 768 * 1024
_MAX_PROMPT_CHARS = 8_000
_MAX_PREDICTION_CHARS = 16_000
_MAX_NEW_TOKENS = 64
_BATCH_SIZE = 4
_MAX_BATCH_PROMPT_CHARS = 24_000
_MAX_RETAINED_JOBS = 50
_MAX_PERSISTED_JOB_BYTES = 512 * 1024
_MAX_EVALUATION_LINKS_PER_JOB = 20
_MAX_FAILURE_EXAMPLES = 3
_MAX_FAILURE_EXAMPLE_TEXT_CHARS = 2_000
_NORMALIZED_TOKEN = re.compile(r"\S+")
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,159}$")
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_TEMP_NAME = re.compile(r"^\.axiom-prediction-(\d+)-([0-9a-f]{32})(?:-([0-9a-f]{32}))?\.tmp$")

_JOBS_LOCK = threading.RLock()
_JOBS: dict[str, dict[str, Any]] = {}
_ACTIVE_JOB_STATUSES = {"queued", "running", "cancelling", "finalizing", "publishing"}
_TERMINAL_JOB_STATUSES = {"completed", "cancelled", "failed", "interrupted"}
_SAFE_JOB_ERROR_CODES = {
    "input_size_limit",
    "input_validation_failed",
    "input_changed",
    "input_changed_or_invalid",
    "input_limit_or_changed",
    "model_validation_failed",
    "output_path_or_destination_invalid",
    "engine_load_or_generation_failed",
    "prediction_size_limit",
    "output_line_limit",
    "output_size_limit",
    "output_write_failed",
    "artifact_publish_failed",
    "prediction_failed",
    "worker_process_exited",
    "worker_start_failed",
    "worker_identity_unverified",
    "job_state_unavailable",
}
_PERSISTED_JOB_FIELDS = (
    "id",
    "status",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
    "dataset_path",
    "output_path",
    "model_path",
    "adapter_path",
    "max_new_tokens",
    "temperature",
    "seed",
    "device",
    "precision",
    "prompt_field",
    "reference_field",
    "prediction_field",
    "rows_total",
    "rows_completed",
    "prompts_truncated",
    "error_code",
    "evaluation_run_ids",
    "artifact_identity",
    "artifact_sha256",
    "pending_artifact_identity",
    "pending_artifact_sha256",
    "runner_pid",
    "runner_identity",
)


class PredictionJobCancelled(RuntimeError):
    """An inference job received a cooperative cancellation request."""


class PredictionJobFailure(RuntimeError):
    """A safe, user-visible prediction failure category."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
    )


def _safe_data_root(root: Path) -> Path:
    entry = root / "data"
    is_junction = getattr(entry, "is_junction", lambda: False)
    if entry.is_symlink() or is_junction():
        raise ValueError("The workspace data directory must not be a symbolic link.")
    try:
        data_root = entry.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("The workspace data directory is unavailable.") from None
    if not data_root.is_dir() or not data_root.is_relative_to(root):
        raise ValueError("The workspace data directory is invalid.")
    return data_root


def _validate_field_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not _FIELD_NAME.fullmatch(value):
        raise ValueError(f"{label} must be a JSON field name of at most 160 characters.")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant {value} is not supported.")


def _read_row(raw_line: bytes, line_number: int) -> dict[str, Any]:
    try:
        row = json.loads(raw_line.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError(f"Dataset line {line_number} is not valid standard UTF-8 JSON.") from None
    if not isinstance(row, dict):
        raise ValueError(f"Dataset line {line_number} must contain a JSON object.")
    pending: list[Any] = [row]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(f"Dataset line {line_number} contains an invalid Unicode string.") from None
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Dataset line {line_number} contains a non-finite number.")
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError:
                    raise ValueError(f"Dataset line {line_number} contains an invalid Unicode field name.") from None
                pending.append(item)
    return row


def _validate_row_fields(
    row: dict[str, Any],
    *,
    line_number: int,
    prompt_field: str,
    reference_field: str,
) -> str:
    prompt = row.get(prompt_field)
    reference = row.get(reference_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Dataset line {line_number} needs a non-empty string in prompt_field.")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise ValueError(f"Dataset line {line_number} prompt exceeds {_MAX_PROMPT_CHARS:,} characters.")
    if not isinstance(reference, str):
        raise ValueError(f"Dataset line {line_number} needs a string in reference_field.")
    return prompt


def _preflight_dataset(
    dataset_path: str,
    root: Path,
    *,
    prompt_field: str,
    reference_field: str,
    collect_prompts: bool = False,
) -> dict[str, Any]:
    data_root = _safe_data_root(root)
    source, source_stream = open_mcp_workspace_file(dataset_path, root)
    if source.suffix.lower() != ".jsonl" or not source.is_relative_to(data_root):
        source_stream.close()
        raise ValueError("Prediction input must be a workspace-relative JSONL file under data/.")

    digest = hashlib.sha256()
    rows = 0
    bytes_scanned = 0
    prompts: list[str] | None = [] if collect_prompts else None
    try:
        with source_stream:
            initial = os.fstat(source_stream.fileno())
            if initial.st_size > _MAX_INPUT_BYTES:
                raise ValueError("Prediction input exceeds the 48 MiB limit.")
            while True:
                raw_line = source_stream.readline(_MAX_INPUT_LINE_BYTES + 1)
                if not raw_line:
                    break
                rows += 1
                bytes_scanned += len(raw_line)
                if len(raw_line) > _MAX_INPUT_LINE_BYTES:
                    raise ValueError(f"Dataset line {rows} exceeds the bounded line size.")
                if rows > _MAX_ROWS:
                    raise ValueError(f"Prediction input exceeds the {_MAX_ROWS}-row limit.")
                if bytes_scanned > _MAX_INPUT_BYTES:
                    raise ValueError("Prediction input exceeds the 48 MiB limit.")
                digest.update(raw_line)
                if not raw_line.strip():
                    raise ValueError(f"Dataset line {rows} is empty.")
                row = _read_row(raw_line, rows)
                prompt = _validate_row_fields(
                    row,
                    line_number=rows,
                    prompt_field=prompt_field,
                    reference_field=reference_field,
                )
                if prompts is not None:
                    prompts.append(prompt)
            final = os.fstat(source_stream.fileno())
        current_path, current_stream = open_mcp_workspace_file(dataset_path, root)
        with current_stream:
            current = os.fstat(current_stream.fileno())
        if not _same_identity(initial, final) or not _same_identity(initial, current):
            raise ValueError("Dataset changed during preflight; retry against a stable file.")
        if current_path.relative_to(data_root).as_posix() != source.relative_to(data_root).as_posix():
            raise ValueError("Dataset path changed during preflight.")
    except (OSError, ValueError) as exc:
        source_stream.close()
        raise ValueError(str(exc)) from None

    if rows == 0:
        raise ValueError("Prediction input contains no JSONL rows.")
    manifest = {
        "rows": rows,
        "size_bytes": bytes_scanned,
        "sha256": digest.hexdigest(),
        "identity": (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns),
        "path": source.relative_to(root).as_posix(),
    }
    if prompts is not None:
        manifest["prompts"] = prompts
    return manifest


def preflight_prediction_job(
    root: Path,
    *,
    model_path: str,
    dataset_path: str,
    prompt_field: str,
    reference_field: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Predict native prompt clipping from a bounded JSONL snapshot without inference."""
    root = Path(root).resolve(strict=True)
    prompt_field = _validate_field_name(prompt_field, "prompt_field")
    reference_field = _validate_field_name(reference_field, "reference_field")
    if prompt_field == reference_field:
        raise ValueError("Prompt and reference fields must be different.")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= _MAX_NEW_TOKENS
    ):
        raise ValueError(f"max_new_tokens must be an integer from 1 through {_MAX_NEW_TOKENS}.")
    if not isinstance(dataset_path, str) or not dataset_path.startswith("data/"):
        raise ValueError("Prediction dataset path must be workspace-relative under data/.")

    manifest = _preflight_dataset(
        dataset_path,
        root,
        prompt_field=prompt_field,
        reference_field=reference_field,
        collect_prompts=True,
    )
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != manifest["rows"]:
        raise ValueError("Prediction preflight could not retain the bounded prompt snapshot safely.")
    token_report = inspect_native_prompt_batch(
        model_path,
        prompts,
        max_new_tokens=max_new_tokens,
    )
    base = {
        "schema_version": 1,
        "assessment_scope": "prompt_truncation_only",
        "rows_scanned": manifest["rows"],
        "max_new_tokens_requested": max_new_tokens,
        "scope": {
            "read_only": True,
            "model_weights_loaded": False,
            "model_inference_run": False,
            "network_accessed": False,
            "files_written": False,
            "dataset_values_returned": False,
            "dataset_hash_returned": False,
            "current_snapshot_only": True,
            "job_start_revalidates_input": True,
        },
    }
    if token_report.get("available") is not True:
        return {
            **base,
            "status": "inconclusive",
            "reason_code": token_report.get("reason_code", "tokenizer_or_template_unavailable"),
            "note": "AXIOM could not match the model's exact tokenizer/template behavior without loading model weights.",
        }
    prompts_truncated = token_report["prompts_truncated"]
    return {
        **base,
        "status": "truncation_expected" if prompts_truncated else "clear",
        "prompts_truncated": prompts_truncated,
        "truncated_row_indices": token_report["truncated_row_indices"],
        "prompt_token_limit": token_report["prompt_token_limit"],
        "effective_max_new_tokens": token_report["effective_max_new_tokens"],
        "prompt_token_counts": token_report["prompt_token_counts"],
        "note": "This is a tokenization snapshot; prompt clipping reported after a started job is authoritative.",
    }


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def cleanup_orphaned_prediction_temporaries(root: Path) -> int:
    """Remove hidden partial artifacts left by a prediction worker that exited."""
    removed = 0
    directory_fd: int | None = None
    try:
        data_root = _safe_data_root(root)
        descriptor_supported = (
            os.open in os.supports_dir_fd
            and os.unlink in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
        )
        if descriptor_supported:
            directory_fd = os.open(
                data_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            scan_target: int | Path = directory_fd
        else:
            scan_target = data_root
        with os.scandir(scan_target) as entries:
            for index, entry in enumerate(entries):
                if index >= 5000:
                    break
                match = _TEMP_NAME.fullmatch(entry.name)
                if match is None:
                    continue
                try:
                    if descriptor_supported:
                        metadata = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
                        if not stat.S_ISREG(metadata.st_mode):
                            continue
                    elif entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        continue
                    pid = int(match.group(1))
                    job_id = match.group(2)
                    if pid == os.getpid():
                        with _JOBS_LOCK:
                            active = _JOBS.get(job_id, {}).get("status") in {
                                "queued", "running", "cancelling", "finalizing", "publishing"
                            }
                        if active:
                            continue
                    elif _process_is_alive(pid):
                        continue
                    if descriptor_supported:
                        os.unlink(entry.name, dir_fd=directory_fd)
                    else:
                        (data_root / entry.name).unlink()
                except OSError:
                    continue
                removed += 1
    except (OSError, ValueError, RuntimeError):
        pass
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass
    return removed


def _resolve_engine_asset(path: Path, label: str) -> Path:
    try:
        return Path(resolve_local_model_directory(str(path), label))
    except ValueError:
        refresh_local_engine_asset_index()
        return Path(resolve_local_model_directory(str(path), label))


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    status = job.get("status")
    artifact_ready = status == "completed"
    summary = {
        "id": job["id"],
        "status": status,
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "dataset_path": job.get("dataset_path"),
        "output_path": job.get("output_path"),
        "artifact_ready": artifact_ready,
        "rows_total": job.get("rows_total"),
        "rows_completed": job.get("rows_completed"),
        "prompts_truncated": job.get("prompts_truncated"),
        "evaluation_run_ids": list(job.get("evaluation_run_ids", [])),
        "error_code": job.get("error_code"),
        "scope": {
            "prompts_returned": False,
            "predictions_returned": False,
            "logs_returned": False,
            "absolute_paths_returned": False,
            "output_file_published_only_after_completion": True,
            "job_state_persisted": True,
            "server_restart_clears_job_state": False,
            "generation_resumed_after_restart": False,
        },
    }
    if status == "interrupted":
        summary["recovery_note"] = (
            "The worker process ended before completion. AXIOM did not resume generation; "
            "the output file may be absent or may have been published just before shutdown. "
            "Inspect it before any path-based evaluation; this job is not eligible for job-ID evaluation."
        )
    elif status == "recovery_unverified":
        summary["error_code"] = "worker_identity_unverified"
        summary["recovery_note"] = (
            "AXIOM could not verify whether the owning process is still running. "
            "It did not resume or mark the job complete; check status again later."
        )
    elif status == "publishing":
        summary["recovery_note"] = (
            "All rows finished and AXIOM is confirming atomic artifact publication. "
            "The output is not ready for evaluation until status becomes completed."
        )
    return summary


def _job_store_paths(root: Path) -> tuple[Path, Path]:
    workspace = Path(root).resolve(strict=True)
    state_dir = workspace / ".axiom"
    state_file = state_dir / "prediction_jobs.json"
    lock_file = state_dir / ".prediction_jobs.lock"
    is_junction = getattr(state_dir, "is_junction", lambda: False)
    if state_dir.is_symlink() or is_junction():
        raise ValueError("The workspace .axiom directory must not be a symbolic link.")
    if state_dir.exists() and not state_dir.is_dir():
        raise ValueError("The workspace .axiom path must be a directory.")
    if state_file.is_symlink() or lock_file.is_symlink():
        raise ValueError("Prediction job state paths must not be symbolic links.")
    return state_file, lock_file


def _supports_job_store_dir_fd() -> bool:
    if os.name == "nt":
        return False
    supported = getattr(os, "supports_dir_fd", set())
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and all(function in supported for function in (os.open, os.rename, os.stat, os.unlink))
    )


def _open_job_store_dir(root: Path, *, create: bool = False) -> int | None:
    state_file, _lock_file = _job_store_paths(root)
    state_dir = state_file.parent
    if create:
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_file, _lock_file = _job_store_paths(root)
        state_dir = state_file.parent
    if not state_dir.exists():
        return None
    if not _supports_job_store_dir_fd():
        return None
    before = state_dir.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("The workspace .axiom path must be a regular directory.")
    descriptor = os.open(
        state_dir,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("The workspace .axiom directory changed while it was opened.")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _locked_job_store(root: Path) -> Iterator[int | None]:
    state_file, lock_file = _job_store_paths(root)
    state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_file, lock_file = _job_store_paths(root)
    directory_fd = _open_job_store_dir(root)
    if directory_fd is None:
        with file_lock(lock_file):
            _job_store_paths(root)
            yield None
        return

    lock_descriptor: int | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        lock_descriptor = os.open(lock_file.name, flags, 0o600, dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise ValueError("Prediction job lock must be a regular file.")
        import fcntl

        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        yield directory_fd
    finally:
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            except (NameError, OSError):
                pass
            os.close(lock_descriptor)
        os.close(directory_fd)


def _open_job_state_file(root: Path, directory_fd: int | None) -> int | None:
    state_file, _lock_file = _job_store_paths(root)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    name = state_file.name
    try:
        before = (
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if directory_fd is not None
            else state_file.lstat()
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Saved prediction job history must be a regular file.")
    descriptor = (
        os.open(name, flags, dir_fd=directory_fd)
        if directory_fd is not None
        else os.open(state_file, flags)
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError("Saved prediction job history changed while it was opened.")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_job_state(root: Path, directory_fd: int | None, serialized: str) -> None:
    state_file, _lock_file = _job_store_paths(root)
    if directory_fd is None:
        atomic_write_text(state_file, serialized)
        return

    temp_name = f".prediction_jobs.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(
            temp_name,
            state_file.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temp_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _valid_persisted_workspace_path(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
        or value.startswith("/")
        or "\\" in value
        or any(ord(character) < 32 or 0x7F <= ord(character) <= 0x9F for character in value)
    ):
        return False
    parts = value.split("/")
    return not any(part in {"", ".", ".."} for part in parts) and not (
        len(parts[0]) >= 2 and parts[0][1] == ":"
    )


def _valid_persisted_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 64:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return value == parsed.isoformat() and parsed.utcoffset() == UTC.utcoffset(parsed)


def _validate_persisted_job(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != set(_PERSISTED_JOB_FIELDS):
        raise ValueError("Saved prediction job state has an invalid schema.")
    if not isinstance(record.get("id"), str) or not _JOB_ID.fullmatch(record["id"]):
        raise ValueError("Saved prediction job state has an invalid ID.")
    status = record.get("status")
    if not isinstance(status, str) or status not in _ACTIVE_JOB_STATUSES | _TERMINAL_JOB_STATUSES:
        raise ValueError("Saved prediction job state has an invalid status.")
    for field in ("created_at", "updated_at"):
        if not _valid_persisted_timestamp(record.get(field)):
            raise ValueError("Saved prediction job state has an invalid timestamp.")
    for field in ("started_at", "finished_at"):
        value = record.get(field)
        if value is not None and not _valid_persisted_timestamp(value):
            raise ValueError("Saved prediction job state has an invalid timestamp.")
    for field in ("dataset_path", "output_path", "model_path"):
        value = record.get(field)
        if not _valid_persisted_workspace_path(value):
            raise ValueError("Saved prediction job state contains an invalid workspace path.")
    if (
        not record["dataset_path"].startswith("data/")
        or not re.fullmatch(r"data/[^/]+\.jsonl", record["output_path"], re.IGNORECASE)
    ):
        raise ValueError("Saved prediction job state contains an invalid data path.")
    adapter_path = record.get("adapter_path")
    if adapter_path is not None and not _valid_persisted_workspace_path(adapter_path):
        raise ValueError("Saved prediction job state contains an invalid adapter path.")
    for field, allowed in (
        ("device", {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}),
        ("precision", {"auto", "fp32", "fp16", "bf16"}),
    ):
        if not isinstance(record.get(field), str) or record[field] not in allowed:
            raise ValueError("Saved prediction job state has an invalid generation setting.")
    for field in ("prompt_field", "reference_field", "prediction_field"):
        value = record.get(field)
        if not isinstance(value, str) or not _FIELD_NAME.fullmatch(value):
            raise ValueError("Saved prediction job state has an invalid field name.")
    for field, minimum, maximum in (
        ("max_new_tokens", 1, _MAX_NEW_TOKENS),
        ("rows_total", 1, _MAX_ROWS),
        ("rows_completed", 0, _MAX_ROWS),
        ("prompts_truncated", 0, _MAX_ROWS),
    ):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError("Saved prediction job state has an invalid count.")
    if record["rows_completed"] > record["rows_total"] or record["prompts_truncated"] > record["rows_completed"]:
        raise ValueError("Saved prediction job state has inconsistent row counts.")
    temperature = record.get("temperature")
    try:
        temperature_value = float(temperature)
    except (OverflowError, TypeError, ValueError):
        temperature_value = float("nan")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature_value)
        or not 0 <= temperature_value <= 2
    ):
        raise ValueError("Saved prediction job state has an invalid temperature.")
    seed = record.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1):
        raise ValueError("Saved prediction job state has an invalid seed.")
    error_code = record.get("error_code")
    if error_code is not None and (
        not isinstance(error_code, str) or error_code not in _SAFE_JOB_ERROR_CODES
    ):
        raise ValueError("Saved prediction job state has an invalid error code.")
    links = record.get("evaluation_run_ids")
    if (
        not isinstance(links, list)
        or len(links) > _MAX_EVALUATION_LINKS_PER_JOB
        or any(not isinstance(value, str) or not _JOB_ID.fullmatch(value) for value in links)
        or len(set(links)) != len(links)
    ):
        raise ValueError("Saved prediction job state has invalid evaluation links.")
    identity = record.get("artifact_identity")
    if identity is not None and (
        not isinstance(identity, list)
        or len(identity) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in identity)
        or identity[2] > _MAX_OUTPUT_BYTES
    ):
        raise ValueError("Saved prediction job state has an invalid artifact identity.")
    digest = record.get("artifact_sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Saved prediction job state has an invalid artifact fingerprint.")
    pending_identity = record.get("pending_artifact_identity")
    if pending_identity is not None and (
        not isinstance(pending_identity, list)
        or len(pending_identity) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in pending_identity
        )
        or pending_identity[2] > _MAX_OUTPUT_BYTES
    ):
        raise ValueError("Saved prediction job state has an invalid pending artifact identity.")
    pending_digest = record.get("pending_artifact_sha256")
    if pending_digest is not None and (
        not isinstance(pending_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", pending_digest)
    ):
        raise ValueError("Saved prediction job state has an invalid pending artifact fingerprint.")
    if (pending_identity is None) != (pending_digest is None):
        raise ValueError("Saved prediction job state has incomplete pending artifact metadata.")
    if (status == "publishing") != (pending_identity is not None):
        raise ValueError("Only a publishing prediction job can contain pending artifact metadata.")
    if status == "publishing" and record["rows_completed"] != record["rows_total"]:
        raise ValueError("A publishing prediction job must contain a complete row count.")
    if record.get("status") == "completed" and (
        identity is None
        or digest is None
        or record["rows_completed"] != record["rows_total"]
        or record["finished_at"] is None
        or pending_identity is not None
    ):
        raise ValueError("A completed prediction job must identify its complete published artifact.")
    pid = record.get("runner_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or not 1 <= pid <= 2**31 - 1:
        raise ValueError("Saved prediction job state has an invalid worker identity.")
    runner_identity = record.get("runner_identity")
    if runner_identity is not None and (not isinstance(runner_identity, str) or len(runner_identity) > 128):
        raise ValueError("Saved prediction job state has an invalid worker identity.")
    return record


def _read_job_records(root: Path, *, directory_fd: int | None = None) -> list[dict[str, Any]]:
    own_directory_fd = directory_fd is None
    if own_directory_fd:
        directory_fd = _open_job_store_dir(root)
    try:
        descriptor = _open_job_state_file(root, directory_fd)
        if descriptor is None:
            return []
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw_payload = stream.read(_MAX_PERSISTED_JOB_BYTES + 1)
            if len(raw_payload) > _MAX_PERSISTED_JOB_BYTES:
                raise ValueError("Saved prediction job history is not a bounded regular file.")
            payload = json.loads(raw_payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError):
        raise ValueError("Saved prediction job history could not be read safely.") from None
    finally:
        if own_directory_fd and directory_fd is not None:
            os.close(directory_fd)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "jobs"}
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or not isinstance(payload.get("jobs"), list)
        or len(payload["jobs"]) > _MAX_RETAINED_JOBS
    ):
        raise ValueError("Saved prediction job history has an invalid schema.")
    try:
        records = [_validate_persisted_job(item) for item in payload["jobs"]]
    except (OverflowError, TypeError):
        raise ValueError("Saved prediction job history has an invalid schema.") from None
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Saved prediction job history contains duplicate IDs.")
    return records


def _persisted_job(job: dict[str, Any]) -> dict[str, Any]:
    record = {field: job.get(field) for field in _PERSISTED_JOB_FIELDS}
    for field in ("artifact_identity", "pending_artifact_identity"):
        identity = record.get(field)
        if isinstance(identity, tuple):
            record[field] = list(identity)
    return _validate_persisted_job(record)


def _persist_job_record(root: Path, job: dict[str, Any]) -> dict[str, Any]:
    with _locked_job_store(root) as directory_fd:
        records = _read_job_records(root, directory_fd=directory_fd)
        current = _persisted_job(job)
        previous = next((record for record in records if record["id"] == current["id"]), None)
        if previous is not None:
            if previous["status"] in _TERMINAL_JOB_STATUSES:
                current = dict(previous)
            current["evaluation_run_ids"] = list(dict.fromkeys(
                [*previous["evaluation_run_ids"], *current["evaluation_run_ids"]]
            ))[-_MAX_EVALUATION_LINKS_PER_JOB:]
        records = [record for record in records if record["id"] != current["id"]]
        records.append(current)
        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        while len(records) > _MAX_RETAINED_JOBS:
            removable = next(
                (index for index in range(len(records) - 1, -1, -1) if records[index]["status"] in _TERMINAL_JOB_STATUSES),
                None,
            )
            if removable is None or records[removable]["id"] == current["id"]:
                raise ValueError("Prediction job history is full; no completed job can be pruned safely.")
            records.pop(removable)
        payload = {"schema_version": 1, "jobs": records}
        serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if len(serialized.encode("utf-8")) > _MAX_PERSISTED_JOB_BYTES:
            raise ValueError("Prediction job history exceeds the 512 KiB workspace limit.")
        _write_job_state(root, directory_fd, serialized)
        return current


def _replace_persisted_job(root: Path, record: dict[str, Any]) -> None:
    with _locked_job_store(root) as directory_fd:
        records = _read_job_records(root, directory_fd=directory_fd)
        replacement = _validate_persisted_job(record)
        previous = next((item for item in records if item["id"] == replacement["id"]), None)
        if previous is not None:
            if previous["status"] in _TERMINAL_JOB_STATUSES:
                replacement = dict(previous)
            replacement["evaluation_run_ids"] = list(dict.fromkeys(
                [*previous["evaluation_run_ids"], *replacement["evaluation_run_ids"]]
            ))[-_MAX_EVALUATION_LINKS_PER_JOB:]
        records = [item for item in records if item["id"] != replacement["id"]]
        records.append(replacement)
        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        payload = {"schema_version": 1, "jobs": records[:_MAX_RETAINED_JOBS]}
        serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if len(serialized.encode("utf-8")) > _MAX_PERSISTED_JOB_BYTES:
            raise ValueError("Prediction job history exceeds the 512 KiB workspace limit.")
        _write_job_state(root, directory_fd, serialized)


def _process_start_identity(pid: int) -> str | None:
    """Return an OS process start token so reused PIDs are not mistaken for live workers."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetProcessTimes.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
            ]
            kernel.GetProcessTimes.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return None
            try:
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
                    return None
                return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
            finally:
                kernel.CloseHandle(handle)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
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
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    token = result.stdout.strip()
    return token[:128] if token else None


def _runner_state(record: dict[str, Any]) -> str:
    """Return alive, exited, or unknown without treating query failure as process exit."""
    pid = record.get("runner_pid")
    identity = record.get("runner_identity")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return "exited"
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


def _verify_pending_artifact(root: Path, record: dict[str, Any]) -> tuple[int, int, int, int] | None:
    identity = record.get("pending_artifact_identity")
    digest = record.get("pending_artifact_sha256")
    if not isinstance(identity, list) or not isinstance(digest, str):
        return None
    try:
        verified = _verify_published_output_hash(
            Path(root) / record["output_path"],
            Path(root),
            (identity[0], identity[1]),
            identity[2],
            digest,
        )
    except (OSError, RuntimeError, ValueError):
        return None
    return verified if list(verified) == identity else None


def _recover_record(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    if record["status"] not in _ACTIVE_JOB_STATUSES:
        return record
    if record["status"] == "publishing":
        verified_identity = _verify_pending_artifact(root, record)
        if verified_identity is not None:
            completed = dict(record)
            completed.update(
                status="completed",
                updated_at=_now(),
                finished_at=_now(),
                error_code=None,
                artifact_identity=list(verified_identity),
                artifact_sha256=record["pending_artifact_sha256"],
                pending_artifact_identity=None,
                pending_artifact_sha256=None,
            )
            try:
                _replace_persisted_job(root, completed)
            except (OSError, TimeoutError, ValueError):
                # The pending identity and digest remain sufficient for safe re-verification.
                pass
            return completed
    runner_state = _runner_state(record)
    if runner_state == "alive":
        return record
    if runner_state == "unknown":
        unverified = dict(record)
        unverified.update(status="recovery_unverified", error_code="worker_identity_unverified")
        return unverified
    recovered = dict(record)
    recovered.update(
        status="interrupted",
        updated_at=_now(),
        finished_at=_now(),
        error_code="worker_process_exited",
        pending_artifact_identity=None,
        pending_artifact_sha256=None,
    )
    _replace_persisted_job(root, recovered)
    return recovered


def _record_as_job(record: dict[str, Any], root: Path) -> dict[str, Any]:
    job = dict(record)
    identity = job.get("artifact_identity")
    if isinstance(identity, list):
        job["artifact_identity"] = tuple(identity)
    job["workspace_root"] = str(Path(root).resolve(strict=True))
    job["cancel_event"] = None
    return job


def _stored_job(job_id: str, root: Path) -> dict[str, Any] | None:
    for record in _read_job_records(root):
        if record["id"] == job_id:
            return _record_as_job(_recover_record(root, record), root)
    return None


def _lookup_job(job_id: str, root: Path) -> dict[str, Any] | None:
    workspace_root = str(Path(root).resolve(strict=True))
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if (
            job is not None
            and job.get("workspace_root") == workspace_root
            and job.get("cancel_event") is not None
        ):
            local_job = job
        else:
            local_job = None
    if local_job is not None:
        persisted = next(
            (record for record in _read_job_records(Path(workspace_root)) if record["id"] == job_id),
            None,
        )
        if persisted is not None:
            recovered = _recover_record(Path(workspace_root), persisted)
            if recovered.get("status") == "completed" and local_job.get("status") != "completed":
                _adopt_recovered_job(job_id, Path(workspace_root), recovered)
            with _JOBS_LOCK:
                current = _JOBS.get(job_id)
                if current is not None:
                    links = list(dict.fromkeys([
                        *persisted["evaluation_run_ids"],
                        *current["evaluation_run_ids"],
                    ]))[-_MAX_EVALUATION_LINKS_PER_JOB:]
                    current["evaluation_run_ids"] = links
                    return current
        return local_job
    return _stored_job(job_id, Path(workspace_root))


def _set_job(job_id: str, **changes: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        updated = {**job, **changes, "updated_at": _now()}
        root = Path(job["workspace_root"])
        persisted = _persist_job_record(root, updated)
        job.update(persisted)


def _mark_job_interrupted_in_memory(job_id: str, error_code: str) -> None:
    """Keep the live process fail-closed if the durable state store becomes unavailable."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(
                status="interrupted",
                updated_at=_now(),
                finished_at=_now(),
                error_code=error_code,
            )


def _adopt_recovered_job(job_id: str, root: Path, record: dict[str, Any]) -> None:
    recovered = _record_as_job(record, root)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            for field in _PERSISTED_JOB_FIELDS:
                job[field] = recovered[field]
            job["workspace_root"] = recovered["workspace_root"]


def _active_training_job_exists() -> bool:
    try:
        return any(
            job.get("status") in {"queued", "running", "cancelling", "finalizing"}
            for job in list_jobs(limit=100)
        )
    except (OSError, RuntimeError, ValueError):
        # Fail closed when AXIOM cannot determine whether training is active.
        return True


def _reserve_output_temp(
    output_path: Path,
    parent_fd: int | None,
    output_name: str,
    job_id: str,
) -> tuple[int, str, Path | None, tuple[int, int]]:
    temp_name = f".axiom-prediction-{os.getpid()}-{job_id}-{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    if parent_fd is not None:
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        return descriptor, temp_name, None, _file_identity(os.fstat(descriptor))
    temp_path = output_path.with_name(temp_name)
    descriptor = os.open(temp_path, flags, 0o600)
    return descriptor, temp_name, temp_path, _file_identity(os.fstat(descriptor))


def _remove_temp(
    parent_fd: int | None,
    temp_name: str | None,
    temp_path: Path | None,
    expected_identity: tuple[int, int] | None,
) -> None:
    if temp_name is None or expected_identity is None:
        return
    try:
        if parent_fd is not None:
            metadata = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        elif temp_path is not None:
            metadata = temp_path.lstat()
        else:
            return
        if stat.S_ISREG(metadata.st_mode) and _file_identity(metadata) == expected_identity:
            if parent_fd is not None:
                os.unlink(temp_name, dir_fd=parent_fd)
            else:
                temp_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _publish_output(
    parent_fd: int | None,
    temp_name: str,
    temp_path: Path | None,
    output_name: str,
    output_path: Path,
    expected_temp_identity: tuple[int, int],
) -> None:
    if parent_fd is not None:
        temp_stat = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(temp_stat.st_mode) or _file_identity(temp_stat) != expected_temp_identity:
            raise ValueError("Temporary prediction artifact identity changed before publication.")
        os.link(
            temp_name,
            output_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    else:
        if temp_path is None:
            raise OSError("Temporary output path is missing.")
        temp_stat = temp_path.lstat()
        if not stat.S_ISREG(temp_stat.st_mode) or _file_identity(temp_stat) != expected_temp_identity:
            raise ValueError("Temporary prediction artifact identity changed before publication.")
        os.link(temp_path, output_path, follow_symlinks=False)


def _file_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _verify_output_location(
    output_path: Path,
    root: Path,
    pinned_parent_fd: int | None,
    output_name: str,
    parent_identity: tuple[int, int],
    *,
    published_identity: tuple[int, int] | None = None,
) -> None:
    """Check that the destination parent remains the workspace data directory."""
    current_path, current_fd, current_name, available = open_mcp_workspace_output_parent(
        output_path.relative_to(root).as_posix(), root
    )
    try:
        data_root = _safe_data_root(root)
        if current_name != output_name or current_path.parent != data_root:
            raise ValueError("Prediction output directory changed during the job.")
        if (published_identity is None) != available:
            raise ValueError("Prediction destination changed during the job.")

        if pinned_parent_fd is not None:
            if current_fd is None:
                raise ValueError("Prediction output directory could not be pinned safely.")
            current_parent_identity = _file_identity(os.fstat(current_fd))
        else:
            current_parent_identity = _file_identity(current_path.parent.stat())
        if current_parent_identity != parent_identity:
            raise ValueError("Prediction output directory changed during the job.")

        if published_identity is not None:
            if current_fd is not None:
                current_output = os.stat(current_name, dir_fd=current_fd, follow_symlinks=False)
            else:
                current_output = current_path.lstat()
            if not stat.S_ISREG(current_output.st_mode) or _file_identity(current_output) != published_identity:
                raise ValueError("Prediction artifact changed during publication.")
    finally:
        if current_fd is not None:
            try:
                os.close(current_fd)
            except OSError:
                pass


def _remove_published_output_if_same(
    parent_fd: int | None,
    output_name: str,
    output_path: Path,
    identity: tuple[int, int],
) -> None:
    try:
        metadata = (
            os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
            if parent_fd is not None else output_path.lstat()
        )
        if stat.S_ISREG(metadata.st_mode) and _file_identity(metadata) == identity:
            if parent_fd is not None:
                os.unlink(output_name, dir_fd=parent_fd)
            else:
                output_path.unlink()
    except OSError:
        pass


def _verify_published_output_hash(
    output_path: Path,
    root: Path,
    expected_file_identity: tuple[int, int],
    expected_size: int,
    expected_sha256: str,
) -> tuple[int, int, int, int]:
    """Confirm the published file is the exact byte stream generated by this job."""
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or not 1 <= expected_size <= _MAX_OUTPUT_BYTES
    ):
        raise ValueError("Published prediction artifact exceeds the bounded output size.")
    relative_path = output_path.relative_to(root).as_posix()
    current_path, stream = open_mcp_workspace_file(relative_path, root)
    with stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or _file_identity(before) != expected_file_identity
            or before.st_size != expected_size
        ):
            raise ValueError("Published prediction artifact identity changed before verification.")
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Published prediction artifact became shorter during verification.")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError("Published prediction artifact grew during verification.")
        after = os.fstat(stream.fileno())
    if (
        current_path != output_path
        or not _same_identity(before, after)
        or digest.hexdigest() != expected_sha256
    ):
        raise ValueError("Published prediction artifact content does not match generated output.")
    return before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns


def _worker_failure_code(exc: Exception, stage: str) -> str:
    if isinstance(exc, PredictionJobFailure):
        return exc.code
    if stage == "model_validation":
        return "model_validation_failed"
    if stage == "dataset_validation":
        return "input_validation_failed"
    if stage == "output_prepare":
        return "output_path_or_destination_invalid"
    if stage == "dataset_read":
        return "input_changed_or_invalid"
    if stage == "dataset_verify":
        return "input_changed"
    if stage == "generation":
        return "engine_load_or_generation_failed"
    if stage == "output_write":
        return "output_write_failed"
    if stage == "artifact_publish":
        return "artifact_publish_failed"
    return "prediction_failed"


def _worker(job_id: str, config: dict[str, Any], root: Path, source_manifest: dict[str, Any]) -> None:
    cancel_event: threading.Event = config["cancel_event"]
    source_stream = None
    parent_fd: int | None = None
    parent_identity: tuple[int, int] | None = None
    output_descriptor: int | None = None
    output_stream = None
    temp_name: str | None = None
    temp_path: Path | None = None
    temp_file_identity: tuple[int, int] | None = None
    stage = "model_validation"
    try:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                changes: dict[str, Any] = {"started_at": _now()}
                if job.get("status") != "cancelling" and not cancel_event.is_set():
                    changes["status"] = "running"
                updated = {**job, **changes, "updated_at": _now()}
                _persist_job_record(root, updated)
                job.update(updated)
        if cancel_event.is_set():
            raise PredictionJobCancelled

        model_target = resolve_workspace_input(config["model_path"], root, "Model path")
        if not model_target.is_dir():
            raise ValueError("Model path must refer to a directory.")
        validate_local_model_tree(model_target)
        trusted_model = _resolve_engine_asset(model_target, "Base model")
        if trusted_model != model_target:
            raise ValueError("Model path changed during validation.")

        trusted_adapter = None
        if config["adapter_path"] is not None:
            adapter_target = resolve_workspace_input(config["adapter_path"], root, "Adapter path")
            if not adapter_target.is_dir():
                raise ValueError("Adapter path must refer to a directory.")
            validate_local_model_tree(adapter_target)
            trusted_adapter = _resolve_engine_asset(adapter_target, "LoRA adapter")
            if trusted_adapter != adapter_target:
                raise ValueError("Adapter path changed during validation.")

        stage = "dataset_validation"
        data_root = _safe_data_root(root)
        source, source_stream = open_mcp_workspace_file(config["dataset_path"], root)
        stage = "output_prepare"
        destination, parent_fd, output_name, output_available = open_mcp_workspace_output_parent(
            config["output_path"], root
        )
        if not output_available or destination.suffix.lower() != ".jsonl":
            raise ValueError("The prediction destination is no longer available.")
        parent_identity = (
            _file_identity(os.fstat(parent_fd))
            if parent_fd is not None else
            _file_identity(destination.parent.stat())
        )
        if source.relative_to(data_root).as_posix() == destination.relative_to(data_root).as_posix():
            raise ValueError("Prediction output must differ from the source dataset.")

        stage = "dataset_read"
        initial = os.fstat(source_stream.fileno())
        expected_identity = source_manifest["identity"]
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        ) != expected_identity:
            raise PredictionJobFailure(
                "input_changed",
                "Dataset changed after preflight; start a new prediction job.",
            )
        if initial.st_size > _MAX_INPUT_BYTES:
            raise PredictionJobFailure("input_size_limit", "Prediction input exceeds the bounded size limit.")

        stage = "output_prepare"
        output_descriptor, temp_name, temp_path, temp_file_identity = _reserve_output_temp(
            destination,
            parent_fd,
            output_name,
            job_id,
        )
        output_stream = os.fdopen(output_descriptor, "wb")
        output_descriptor = None
        source_digest = hashlib.sha256()
        source_bytes = 0
        rows_read = 0
        output_bytes = 0
        output_digest = hashlib.sha256()
        chunk_rows: list[dict[str, Any]] = []
        chunk_prompts: list[str] = []
        chunk_prompt_chars = 0
        stage = "dataset_read"

        def write_chunk() -> None:
            nonlocal output_bytes, chunk_rows, chunk_prompts, chunk_prompt_chars, stage
            if not chunk_rows:
                return
            if cancel_event.is_set():
                raise PredictionJobCancelled
            stage = "generation"
            generated = generate_local_batch(
                str(trusted_model),
                list(chunk_prompts),
                adapter_path=str(trusted_adapter) if trusted_adapter is not None else None,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                seed=config["seed"],
                device_preference=config["device"],
                precision=config["precision"],
                cancel_event=cancel_event,
            )
            results = generated.get("results")
            if not isinstance(results, list) or len(results) != len(chunk_rows):
                raise ValueError("Native generation returned an incomplete batch.")
            for row, result in zip(chunk_rows, results):
                if cancel_event.is_set():
                    raise PredictionJobCancelled
                if not isinstance(result, dict) or not isinstance(result.get("response"), str):
                    raise ValueError("Native generation returned an invalid prediction.")
                prediction = result["response"]
                if len(prediction) > _MAX_PREDICTION_CHARS:
                    raise PredictionJobFailure(
                        "prediction_size_limit",
                        "A generated prediction exceeded the bounded output size.",
                    )
                row[config["prediction_field"]] = prediction
                stage = "output_write"
                encoded = json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n"
                if len(encoded) > _MAX_OUTPUT_LINE_BYTES:
                    raise PredictionJobFailure(
                        "output_line_limit",
                        "A prediction output line exceeds the evaluator's 1 MiB limit.",
                    )
                output_bytes += len(encoded)
                if output_bytes > _MAX_OUTPUT_BYTES:
                    raise PredictionJobFailure(
                        "output_size_limit",
                        "Prediction output exceeds the evaluator's 64 MiB limit.",
                    )
                if output_stream.write(encoded) != len(encoded):
                    raise OSError("Prediction artifact write was incomplete.")
                output_digest.update(encoded)
                truncated = (result.get("context_management") or {}).get("token_truncation_applied") is True
                with _JOBS_LOCK:
                    job = _JOBS.get(job_id)
                    if job is not None:
                        changes = {"rows_completed": job["rows_completed"] + 1}
                        if truncated:
                            changes["prompts_truncated"] = job["prompts_truncated"] + 1
                        updated = {**job, **changes, "updated_at": _now()}
                        _persist_job_record(root, updated)
                        job.update(updated)
            chunk_rows = []
            chunk_prompts = []
            chunk_prompt_chars = 0

        while True:
            stage = "dataset_read"
            if cancel_event.is_set():
                raise PredictionJobCancelled
            raw_line = source_stream.readline(_MAX_INPUT_LINE_BYTES + 1)
            if not raw_line:
                break
            rows_read += 1
            source_bytes += len(raw_line)
            if len(raw_line) > _MAX_INPUT_LINE_BYTES or source_bytes > _MAX_INPUT_BYTES or rows_read > _MAX_ROWS:
                raise PredictionJobFailure(
                    "input_limit_or_changed",
                    "Dataset changed or exceeded the bounded prediction limits.",
                )
            source_digest.update(raw_line)
            row = _read_row(raw_line, rows_read)
            prompt = _validate_row_fields(
                row,
                line_number=rows_read,
                prompt_field=config["prompt_field"],
                reference_field=config["reference_field"],
            )
            if chunk_rows and (
                len(chunk_rows) >= _BATCH_SIZE
                or chunk_prompt_chars + len(prompt) > _MAX_BATCH_PROMPT_CHARS
            ):
                write_chunk()
            chunk_rows.append(row)
            chunk_prompts.append(prompt)
            chunk_prompt_chars += len(prompt)
            if len(chunk_rows) >= _BATCH_SIZE or chunk_prompt_chars >= _MAX_BATCH_PROMPT_CHARS:
                write_chunk()
        write_chunk()

        stage = "dataset_verify"
        if rows_read != source_manifest["rows"] or source_bytes != source_manifest["size_bytes"]:
            raise PredictionJobFailure("input_changed", "Dataset changed during prediction.")
        final = os.fstat(source_stream.fileno())
        source_stream.close()
        source_stream = None
        current_path, current_stream = open_mcp_workspace_file(config["dataset_path"], root)
        with current_stream:
            current = os.fstat(current_stream.fileno())
        if (
            not _same_identity(initial, final)
            or not _same_identity(initial, current)
            or source_digest.hexdigest() != source_manifest["sha256"]
            or current_path.relative_to(data_root).as_posix() != source.relative_to(data_root).as_posix()
        ):
            raise PredictionJobFailure(
                "input_changed",
                "Dataset changed during prediction; partial results were discarded.",
            )

        stage = "output_write"
        output_stream.flush()
        os.fsync(output_stream.fileno())
        output_stream.close()
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                raise ValueError("Prediction job state is unavailable.")
            if cancel_event.is_set() or job.get("status") == "cancelling":
                raise PredictionJobCancelled
            updated = {**job, "status": "finalizing", "updated_at": _now()}
            _persist_job_record(root, updated)
            job.update(updated)

        stage = "artifact_publish"
        if parent_identity is None:
            raise ValueError("Prediction output directory identity is unavailable.")
        _verify_output_location(destination, root, parent_fd, output_name, parent_identity)
        if parent_fd is not None:
            temp_stat = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        else:
            if temp_path is None:
                raise ValueError("Temporary prediction artifact path is unavailable.")
            temp_stat = temp_path.lstat()
        if not stat.S_ISREG(temp_stat.st_mode) or temp_stat.st_size != output_bytes:
            raise ValueError("Temporary prediction artifact changed before publication.")
        if temp_file_identity is None or _file_identity(temp_stat) != temp_file_identity:
            raise ValueError("Temporary prediction artifact identity changed before publication.")
        pending_artifact_identity = (
            temp_stat.st_dev,
            temp_stat.st_ino,
            temp_stat.st_size,
            temp_stat.st_mtime_ns,
        )
        _set_job(
            job_id,
            status="publishing",
            pending_artifact_identity=pending_artifact_identity,
            pending_artifact_sha256=output_digest.hexdigest(),
        )
        try:
            _publish_output(
                parent_fd,
                temp_name,
                temp_path,
                output_name,
                destination,
                temp_file_identity,
            )
            output_stat = (
                os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
                if parent_fd is not None else destination.lstat()
            )
            if (
                not stat.S_ISREG(output_stat.st_mode)
                or _file_identity(output_stat) != temp_file_identity
            ):
                # If the temp name was swapped just before link(), remove only the link
                # this publication created, and only while it still has that inode.
                try:
                    current_temp_stat = (
                        os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
                        if parent_fd is not None else temp_path.lstat()
                    )
                except OSError:
                    current_temp_stat = None
                if (
                    stat.S_ISREG(output_stat.st_mode)
                    and current_temp_stat is not None
                    and _file_identity(current_temp_stat) == _file_identity(output_stat)
                ):
                    _remove_published_output_if_same(
                        parent_fd,
                        output_name,
                        destination,
                        _file_identity(output_stat),
                    )
                raise ValueError("Published prediction artifact is not the generated temporary file.")
            _verify_output_location(
                destination,
                root,
                parent_fd,
                output_name,
                parent_identity,
                published_identity=temp_file_identity,
            )
            artifact_identity = _verify_published_output_hash(
                destination,
                root,
                temp_file_identity,
                output_bytes,
                output_digest.hexdigest(),
            )
            _verify_output_location(
                destination,
                root,
                parent_fd,
                output_name,
                parent_identity,
                published_identity=temp_file_identity,
            )
        except Exception:
            _remove_published_output_if_same(
                parent_fd,
                output_name,
                destination,
                temp_file_identity,
            )
            raise
        _set_job(
            job_id,
            status="completed",
            finished_at=_now(),
            error_code=None,
            artifact_identity=artifact_identity,
            artifact_sha256=output_digest.hexdigest(),
            pending_artifact_identity=None,
            pending_artifact_sha256=None,
        )
    except PredictionJobCancelled:
        try:
            _set_job(job_id, status="cancelled", finished_at=_now(), error_code=None)
        except Exception:
            _mark_job_interrupted_in_memory(job_id, "job_state_unavailable")
    except Exception as exc:
        recovered_completion = False
        if stage == "artifact_publish":
            with _JOBS_LOCK:
                current = _JOBS.get(job_id)
            if current is not None and current.get("status") == "publishing":
                try:
                    recovered = _recover_record(root, _persisted_job(current))
                    if recovered.get("status") == "completed":
                        _adopt_recovered_job(job_id, root, recovered)
                        recovered_completion = True
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
        if not recovered_completion:
            try:
                _set_job(
                    job_id,
                    status="failed",
                    finished_at=_now(),
                    error_code=_worker_failure_code(exc, stage),
                    pending_artifact_identity=None,
                    pending_artifact_sha256=None,
                )
            except Exception:
                _mark_job_interrupted_in_memory(job_id, "job_state_unavailable")
    finally:
        if output_stream is not None and not output_stream.closed:
            try:
                output_stream.close()
            except OSError:
                pass
        if source_stream is not None:
            try:
                source_stream.close()
            except OSError:
                pass
        if output_descriptor is not None:
            try:
                os.close(output_descriptor)
            except OSError:
                pass
        _remove_temp(parent_fd, temp_name, temp_path, temp_file_identity)
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        release_local_compute_job_lock(job_id)


def start_prediction_job(
    root: Path,
    *,
    model_path: str,
    adapter_path: str | None,
    dataset_path: str,
    prompt_field: str,
    reference_field: str,
    prediction_field: str,
    output_path: str,
    max_new_tokens: int,
    temperature: float,
    seed: int | None,
    device: str,
    precision: str,
) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    if not isinstance(model_path, str) or not isinstance(dataset_path, str) or not isinstance(output_path, str):
        raise ValueError("Model, dataset, and output paths must be workspace-relative strings.")
    if not output_path.startswith("data/") or output_path.count("/") != 1 or not output_path.lower().endswith(".jsonl"):
        raise ValueError("Prediction output must be a new data/NAME.jsonl file in the workspace.")
    if not dataset_path.startswith("data/"):
        raise ValueError("Prediction input must be a workspace-relative path under data/.")
    prompt_field = _validate_field_name(prompt_field, "prompt_field")
    reference_field = _validate_field_name(reference_field, "reference_field")
    prediction_field = _validate_field_name(prediction_field, "prediction_field")
    if len({prompt_field, reference_field, prediction_field}) != 3:
        raise ValueError("Prompt, reference, and prediction fields must be different.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= _MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be an integer from 1 through {_MAX_NEW_TOKENS}.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite number from 0 through 2.")
    try:
        temperature = float(temperature)
    except (OverflowError, ValueError):
        temperature = float("nan")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be a finite number from 0 through 2.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1
    ):
        raise ValueError("seed must be an integer from 0 through 2^63-1.")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}:
        raise ValueError("device must be auto, cpu, cuda, rocm, mps, or xpu.")
    if not isinstance(precision, str) or precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16.")
    if device == "mps" and precision == "bf16":
        raise ValueError("The AXIOM native engine does not support BF16 precision on Metal/MPS.")

    data_root = _safe_data_root(root)
    cleanup_orphaned_prediction_temporaries(root)
    model_target = resolve_workspace_input(model_path, root, "Model path")
    if not model_target.is_dir():
        raise ValueError("Model path must refer to a local model directory.")
    validate_local_model_tree(model_target)
    trusted_model = _resolve_engine_asset(model_target, "Base model")
    if trusted_model != model_target:
        raise ValueError("Model path changed during validation.")
    if adapter_path is not None:
        adapter_target = resolve_workspace_input(adapter_path, root, "Adapter path")
        if not adapter_target.is_dir():
            raise ValueError("Adapter path must refer to a local adapter directory.")
        validate_local_model_tree(adapter_target)
        trusted_adapter = _resolve_engine_asset(adapter_target, "LoRA adapter")
        if trusted_adapter != adapter_target:
            raise ValueError("Adapter path changed during validation.")

    if not runtime_status().get("ready"):
        raise ValueError("AXIOM Native Engine runtime is not ready in this environment.")
    if _active_training_job_exists():
        raise ValueError("A local training job is active; wait for it to finish before running prediction evaluation.")

    source_manifest = _preflight_dataset(
        dataset_path,
        root,
        prompt_field=prompt_field,
        reference_field=reference_field,
    )
    destination, output_parent_fd, output_name, output_available = open_mcp_workspace_output_parent(output_path, root)
    if output_parent_fd is not None:
        os.close(output_parent_fd)
    if not output_available:
        raise ValueError("Prediction output already exists; choose a new data/NAME.jsonl path.")
    if not destination.is_relative_to(data_root):
        raise ValueError("Prediction output must stay inside the workspace data directory.")
    if source_manifest["path"] == destination.relative_to(root).as_posix():
        raise ValueError("Prediction output must differ from the source dataset.")

    with _JOBS_LOCK:
        job_id = uuid.uuid4().hex
    acquire_local_compute_job_lock(job_id)
    worker_started = False
    try:
        with _JOBS_LOCK:
            _prune_jobs()
            if any(
                job.get("status") in {"queued", "running", "cancelling", "finalizing", "publishing"}
                for job in _JOBS.values()
            ):
                raise ValueError("Another local AXIOM compute job is active; inspect it before starting another.")
            cancel_event = threading.Event()
            config = {
                "model_path": model_path,
                "adapter_path": adapter_path,
                "dataset_path": dataset_path,
                "prompt_field": prompt_field,
                "reference_field": reference_field,
                "prediction_field": prediction_field,
                "output_path": output_path,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "seed": seed,
                "device": device,
                "precision": precision,
                "cancel_event": cancel_event,
            }
            job = {
                "id": job_id,
                "workspace_root": str(root),
                "runner_pid": os.getpid(),
                "runner_identity": _process_start_identity(os.getpid()),
                "status": "queued",
                "created_at": _now(),
                "updated_at": _now(),
                "started_at": None,
                "finished_at": None,
                "dataset_path": source_manifest["path"],
                "output_path": output_path,
                "model_path": model_path,
                "adapter_path": adapter_path,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "seed": seed,
                "device": device,
                "precision": precision,
                "prompt_field": prompt_field,
                "reference_field": reference_field,
                "prediction_field": prediction_field,
                "rows_total": source_manifest["rows"],
                "rows_completed": 0,
                "prompts_truncated": 0,
                "error_code": None,
                "evaluation_run_ids": [],
                "pending_artifact_identity": None,
                "pending_artifact_sha256": None,
                "cancel_event": cancel_event,
            }
            _JOBS[job_id] = job
            _persist_job_record(root, job)
            worker = threading.Thread(
                target=_worker,
                args=(job_id, config, root, source_manifest),
                name=f"axiom-prediction-{job_id[:8]}",
                daemon=True,
            )
            try:
                worker.start()
            except RuntimeError:
                failed = {
                    **job,
                    "status": "failed",
                    "updated_at": _now(),
                    "finished_at": _now(),
                    "error_code": "worker_start_failed",
                }
                try:
                    _persist_job_record(root, failed)
                finally:
                    _JOBS.pop(job_id, None)
                raise ValueError("AXIOM could not start the local prediction worker.") from None
            worker_started = True
            summary = _job_summary(job)
        return summary
    except Exception:
        if not worker_started:
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
                if job is not None and job.get("status") == "queued":
                    _JOBS.pop(job_id, None)
            release_local_compute_job_lock(job_id)
        raise


def _prune_jobs() -> None:
    terminal = [
        (job.get("created_at", ""), job_id)
        for job_id, job in _JOBS.items()
        if job.get("status") in _TERMINAL_JOB_STATUSES
    ]
    terminal.sort()
    while len(_JOBS) >= _MAX_RETAINED_JOBS and terminal:
        _created_at, job_id = terminal.pop(0)
        _JOBS.pop(job_id, None)
    if len(_JOBS) >= _MAX_RETAINED_JOBS:
        raise ValueError("Prediction job history is full; wait for an active job to finish and retry.")


def get_prediction_job(job_id: str, root: Path) -> dict[str, Any] | None:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        return None
    job = _lookup_job(job_id, root)
    return _job_summary(job) if job is not None else None


def list_prediction_jobs(root: Path, limit: int = 10) -> list[dict[str, Any]]:
    """List persisted prediction job summaries for one workspace, newest first."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_RETAINED_JOBS:
        raise ValueError(f"Prediction job list limit must be between 1 and {_MAX_RETAINED_JOBS}.")
    workspace = Path(root).resolve(strict=True)
    records = _read_job_records(workspace)
    jobs: dict[str, dict[str, Any]] = {}
    for record in records:
        recovered = _recover_record(workspace, record)
        with _JOBS_LOCK:
            local = _JOBS.get(recovered["id"])
            if (
                local is not None
                and local.get("workspace_root") == str(workspace)
                and local.get("cancel_event") is not None
            ):
                if recovered.get("status") == "completed" and local.get("status") != "completed":
                    _adopt_recovered_job(recovered["id"], workspace, recovered)
                    local = _JOBS.get(recovered["id"], local)
                local["evaluation_run_ids"] = list(dict.fromkeys([
                    *recovered["evaluation_run_ids"],
                    *local["evaluation_run_ids"],
                ]))[-_MAX_EVALUATION_LINKS_PER_JOB:]
                jobs[recovered["id"]] = local
                continue
        jobs[recovered["id"]] = _record_as_job(recovered, workspace)
    with _JOBS_LOCK:
        for job_id, job in _JOBS.items():
            if job.get("workspace_root") == str(workspace) and job_id not in jobs:
                jobs[job_id] = job
    ordered = sorted(
        jobs.values(),
        key=lambda job: str(job.get("created_at", "")),
        reverse=True,
    )
    return [_job_summary(job) for job in ordered[:limit]]


def get_completed_prediction_artifact(job_id: str, root: Path) -> dict[str, Any] | None:
    """Return internal artifact metadata only for a completed job in this workspace."""
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        return None
    job = _lookup_job(job_id, root)
    if job is None or job.get("status") != "completed":
        return None
    identity = job.get("artifact_identity")
    digest = job.get("artifact_sha256")
    if (
        not isinstance(identity, tuple)
        or len(identity) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in identity)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        return None
    return {
        "id": job_id,
        "dataset_path": job.get("dataset_path"),
        "output_path": job.get("output_path"),
        "model_path": job.get("model_path"),
        "adapter_path": job.get("adapter_path"),
        "max_new_tokens": job.get("max_new_tokens"),
        "temperature": job.get("temperature"),
        "seed": job.get("seed"),
        "device": job.get("device"),
        "precision": job.get("precision"),
        "prompt_field": job.get("prompt_field"),
        "prediction_field": job.get("prediction_field"),
        "reference_field": job.get("reference_field"),
        "rows_completed": job.get("rows_completed"),
        "prompts_truncated": job.get("prompts_truncated", 0),
        "evaluation_run_ids": list(job.get("evaluation_run_ids", [])),
        "artifact_identity": identity,
        "artifact_sha256": digest,
    }


def _normalized_text_equal(left: str, right: str) -> bool:
    left_tokens = _NORMALIZED_TOKEN.finditer(left)
    right_tokens = _NORMALIZED_TOKEN.finditer(right)
    while True:
        left_token = next(left_tokens, None)
        right_token = next(right_tokens, None)
        if left_token is None or right_token is None:
            return left_token is None and right_token is None
        if left_token.group().casefold() != right_token.group().casefold():
            return False


def get_prediction_job_examples(
    job_id: str,
    root: Path,
    *,
    limit: int = 3,
    include_text: bool = False,
    include_prompt: bool = False,
) -> dict[str, Any] | None:
    """Return a few mismatch examples from a hash-verified completed artifact."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_FAILURE_EXAMPLES:
        raise ValueError(f"Example limit must be between 1 and {_MAX_FAILURE_EXAMPLES}.")
    if type(include_text) is not bool or type(include_prompt) is not bool:
        raise ValueError("include_text and include_prompt must be booleans.")
    if include_prompt and not include_text:
        raise ValueError("include_prompt requires include_text=true.")
    artifact = get_completed_prediction_artifact(job_id, root)
    if artifact is None:
        return None
    identity = artifact["artifact_identity"]
    expected_sha256 = artifact["artifact_sha256"]
    expected_size = identity[2]
    if not 1 <= expected_size <= _MAX_OUTPUT_BYTES:
        raise ValueError("Completed prediction artifact exceeds the bounded output size.")

    workspace = Path(root).resolve(strict=True)
    output_path, stream = open_mcp_workspace_file(artifact["output_path"], workspace)
    examples: list[dict[str, Any]] = []
    mismatches = 0
    exact_matches = 0
    normalized_matches = 0
    bytes_scanned = 0
    digest = hashlib.sha256()
    with stream:
        initial = os.fstat(stream.fileno())
        initial_identity = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial_identity != identity
            or initial.st_size > _MAX_OUTPUT_BYTES
        ):
            raise ValueError("Completed prediction artifact identity changed before example inspection.")

        for line_number in range(1, artifact["rows_completed"] + 1):
            raw_line = stream.readline(_MAX_OUTPUT_LINE_BYTES + 1)
            if not raw_line:
                raise ValueError("Completed prediction artifact ended before all rows were read.")
            if len(raw_line) > _MAX_OUTPUT_LINE_BYTES:
                raise ValueError("Completed prediction artifact contains an oversized JSONL line.")
            bytes_scanned += len(raw_line)
            if bytes_scanned > _MAX_OUTPUT_BYTES:
                raise ValueError("Completed prediction artifact exceeds the bounded output size.")
            digest.update(raw_line)
            row = _read_row(raw_line, line_number)
            prediction = row.get(artifact["prediction_field"])
            reference = row.get(artifact["reference_field"])
            if not isinstance(prediction, str) or not isinstance(reference, str):
                raise ValueError("Completed prediction artifact has invalid evaluation fields.")

            exact_match = prediction == reference
            normalized_match = _normalized_text_equal(prediction, reference)
            exact_matches += int(exact_match)
            normalized_matches += int(normalized_match)
            if exact_match:
                continue
            mismatches += 1
            if len(examples) >= limit:
                continue

            example: dict[str, Any] = {"row_number": line_number}
            if include_text:
                example["exact_match"] = exact_match
                example["normalized_exact_match"] = normalized_match
                example["prediction_excerpt"] = prediction[:_MAX_FAILURE_EXAMPLE_TEXT_CHARS]
                example["reference_excerpt"] = reference[:_MAX_FAILURE_EXAMPLE_TEXT_CHARS]
                example["text_truncated"] = {
                    "prediction": len(prediction) > _MAX_FAILURE_EXAMPLE_TEXT_CHARS,
                    "reference": len(reference) > _MAX_FAILURE_EXAMPLE_TEXT_CHARS,
                }
                if include_prompt:
                    prompt = row.get(artifact["prompt_field"])
                    if not isinstance(prompt, str):
                        raise ValueError("Completed prediction artifact has an invalid prompt field.")
                    example["prompt_excerpt"] = prompt[:_MAX_FAILURE_EXAMPLE_TEXT_CHARS]
                    example["text_truncated"]["prompt"] = len(prompt) > _MAX_FAILURE_EXAMPLE_TEXT_CHARS
            examples.append(example)

        if stream.read(1):
            raise ValueError("Completed prediction artifact contains more rows than its job metadata.")
        final = os.fstat(stream.fileno())

    if (
        not _same_identity(initial, final)
        or bytes_scanned != initial.st_size
        or digest.hexdigest() != expected_sha256
    ):
        raise ValueError("Completed prediction artifact changed or no longer matches its published hash.")
    current_path, current_stream = open_mcp_workspace_file(artifact["output_path"], workspace)
    with current_stream:
        current = os.fstat(current_stream.fileno())
    current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if current_path != output_path or current_identity != identity:
        raise ValueError("Completed prediction artifact path changed during example inspection.")

    return {
        "job_id": artifact["id"],
        "rows_scanned": artifact["rows_completed"],
        "exact_matches": exact_matches,
        "normalized_exact_matches": normalized_matches,
        "exact_mismatch_rows": mismatches,
        "sample_limit": limit,
        "sample_strategy": "first_exact_mismatches_in_file_order",
        "mismatch_definition": "raw prediction string differs from reference string",
        "examples": examples,
        "scope": {
            "artifact_identity_and_sha256_verified": True,
            "full_source_rows_returned": False,
            "prediction_reference_text_returned": include_text,
            "prompt_text_returned": include_prompt,
            "network_accessed": False,
            "model_inference_run": False,
            "max_excerpt_chars_per_field": _MAX_FAILURE_EXAMPLE_TEXT_CHARS if include_text else 0,
        },
    }


def record_prediction_job_evaluation(job_id: str, root: Path, evaluation_run_id: str) -> bool:
    """Link a persisted aggregate evaluation to its completed prediction job."""
    if (
        not isinstance(job_id, str)
        or not _JOB_ID.fullmatch(job_id)
        or not isinstance(evaluation_run_id, str)
        or not _JOB_ID.fullmatch(evaluation_run_id)
    ):
        return False
    job = _lookup_job(job_id, root)
    if job is None or job.get("status") != "completed":
        return False
    with _JOBS_LOCK:
        current = _JOBS.get(job_id)
        if current is not None and current.get("workspace_root") == job.get("workspace_root"):
            job = current
        links = list(job.get("evaluation_run_ids", []))
        if evaluation_run_id not in links:
            links.append(evaluation_run_id)
            links = links[-_MAX_EVALUATION_LINKS_PER_JOB:]
        updated = {**job, "evaluation_run_ids": links, "updated_at": _now()}
        persisted = _persist_job_record(Path(updated["workspace_root"]), updated)
        job.update(updated)
        job["evaluation_run_ids"] = list(persisted["evaluation_run_ids"])
        return True


def cancel_prediction_job(job_id: str, root: Path) -> dict[str, Any] | None:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        return None
    job = _lookup_job(job_id, root)
    if job is None:
        return None
    workspace_root = str(Path(root).resolve(strict=True))
    with _JOBS_LOCK:
        local_job = _JOBS.get(job_id)
        if (
            local_job is not None
            and local_job.get("workspace_root") == workspace_root
            and local_job.get("cancel_event") is not None
        ):
            job = local_job
        else:
            local_job = None
        status = job.get("status")
        accepted = local_job is not None and status in {"queued", "running", "cancelling"}
        if accepted and status in {"queued", "running"}:
            updated = {**job, "status": "cancelling", "updated_at": _now()}
            _persist_job_record(Path(workspace_root), updated)
            job.update(updated)
            job["cancel_event"].set()
        return {
            "job": _job_summary(job),
            "cancellation_accepted": accepted,
            "note": (
                "Cancellation is cooperative and stops after the current model-loading or generation boundary."
                if accepted else
                "The worker is not controlled by this process, has entered finalization, or has already finished; cancellation is unavailable."
            ),
        }
