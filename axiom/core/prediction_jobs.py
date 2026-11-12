"""Cancellable local prediction jobs that produce evaluation-ready JSONL."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.engine import (
    generate_local_batch,
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
_MAX_EVALUATION_LINKS_PER_JOB = 20
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,159}$")
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_TEMP_NAME = re.compile(r"^\.axiom-prediction-(\d+)-([0-9a-f]{32})(?:-([0-9a-f]{32}))?\.tmp$")

_JOBS_LOCK = threading.RLock()
_JOBS: dict[str, dict[str, Any]] = {}


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
) -> dict[str, Any]:
    data_root = _safe_data_root(root)
    source, source_stream = open_mcp_workspace_file(dataset_path, root)
    if source.suffix.lower() != ".jsonl" or not source.is_relative_to(data_root):
        source_stream.close()
        raise ValueError("Prediction input must be a workspace-relative JSONL file under data/.")

    digest = hashlib.sha256()
    rows = 0
    bytes_scanned = 0
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
                _validate_row_fields(
                    row,
                    line_number=rows,
                    prompt_field=prompt_field,
                    reference_field=reference_field,
                )
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
    return {
        "rows": rows,
        "size_bytes": bytes_scanned,
        "sha256": digest.hexdigest(),
        "identity": (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns),
        "path": source.relative_to(root).as_posix(),
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
                                "queued", "running", "cancelling", "finalizing"
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
    return {
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
            "server_restart_clears_job_state": True,
        },
    }


def _set_job(job_id: str, **changes: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(changes)
            job["updated_at"] = _now()


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
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
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
                if job.get("status") != "cancelling" and not cancel_event.is_set():
                    job["status"] = "running"
                job["started_at"] = _now()
                job["updated_at"] = _now()
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
                        job["rows_completed"] += 1
                        if truncated:
                            job["prompts_truncated"] += 1
                        job["updated_at"] = _now()
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
            job.update(status="finalizing", updated_at=_now())

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
        )
    except PredictionJobCancelled:
        _set_job(job_id, status="cancelled", finished_at=_now(), error_code=None)
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            finished_at=_now(),
            error_code=_worker_failure_code(exc, stage),
        )
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
            if any(job.get("status") in {"queued", "running", "cancelling", "finalizing"} for job in _JOBS.values()):
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
                "cancel_event": cancel_event,
            }
            _JOBS[job_id] = job
            worker = threading.Thread(
                target=_worker,
                args=(job_id, config, root, source_manifest),
                name=f"axiom-prediction-{job_id[:8]}",
                daemon=True,
            )
            try:
                worker.start()
            except RuntimeError:
                del _JOBS[job_id]
                raise ValueError("AXIOM could not start the local prediction worker.") from None
            worker_started = True
            summary = _job_summary(job)
        return summary
    except Exception:
        if not worker_started:
            release_local_compute_job_lock(job_id)
        raise


def _prune_jobs() -> None:
    terminal = [
        (job.get("created_at", ""), job_id)
        for job_id, job in _JOBS.items()
        if job.get("status") in {"completed", "cancelled", "failed"}
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
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None or job.get("workspace_root") != str(Path(root).resolve(strict=True)):
            return None
        return _job_summary(job)


def get_completed_prediction_artifact(job_id: str, root: Path) -> dict[str, Any] | None:
    """Return internal artifact metadata only for a completed job in this workspace."""
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        return None
    workspace_root = str(Path(root).resolve(strict=True))
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if (
            job is None
            or job.get("workspace_root") != workspace_root
            or job.get("status") != "completed"
        ):
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
            "artifact_identity": identity,
            "artifact_sha256": digest,
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
    workspace_root = str(Path(root).resolve(strict=True))
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if (
            job is None
            or job.get("workspace_root") != workspace_root
            or job.get("status") != "completed"
        ):
            return False
        links = job.setdefault("evaluation_run_ids", [])
        if evaluation_run_id not in links:
            links.append(evaluation_run_id)
            del links[:-_MAX_EVALUATION_LINKS_PER_JOB]
            job["updated_at"] = _now()
        return True


def cancel_prediction_job(job_id: str, root: Path) -> dict[str, Any] | None:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        return None
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None or job.get("workspace_root") != str(Path(root).resolve(strict=True)):
            return None
        status = job.get("status")
        accepted = status in {"queued", "running", "cancelling"}
        if status in {"queued", "running"}:
            job["cancel_event"].set()
            job.update(status="cancelling", updated_at=_now())
        return {
            "job": _job_summary(job),
            "cancellation_accepted": accepted,
            "note": (
                "Cancellation is cooperative and stops after the current model-loading or generation boundary."
                if accepted else
                "The job has entered finalization or has already finished; cancellation is no longer available."
            ),
        }
