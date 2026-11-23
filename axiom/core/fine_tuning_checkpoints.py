"""Atomic, integrity-checked local fine-tuning checkpoints.

Checkpoints are kept inside an AXIOM-managed hidden folder in the selected
output directory. Only complete checkpoints are eligible for resume.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.path_safety import open_regular_file

CHECKPOINT_DIRECTORY = ".axiom-checkpoints"
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d{6})$")
_TEMP_PREFIX = ".checkpoint-"
_MAX_RETAINED_CHECKPOINTS = 3
_MAX_STATE_BYTES = 64 * 1024
_REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors", "optimizer_state.pt")


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def _safe_directory(path: Path, message: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(message)
    try:
        if path.resolve(strict=True) != path:
            raise ValueError(message)
    except (OSError, RuntimeError):
        raise ValueError(message) from None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open_regular_file(path) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_tree_fingerprint(model_dir: Path) -> str:
    """Fingerprint model file names/metadata and hash the architecture config.

    Weight contents are not re-hashed: that can require reading many GiB.
    The metadata fingerprint detects ordinary replacement/edits while keeping
    resume preflight bounded; config.json itself is content-hashed.
    """
    root = Path(model_dir)
    _safe_directory(root, "Base model directory is not safe for checkpoint validation.")
    digest = hashlib.sha256()
    entries = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            raise ValueError("Base model directory could not be fingerprinted safely.") from None
        for path in children:
            entries += 1
            if entries > 50_000:
                raise ValueError("Base model exceeds the 50,000-entry checkpoint fingerprint limit.")
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError("Base model assets cannot contain symbolic links or junctions.")
            try:
                metadata = path.lstat()
            except OSError:
                raise ValueError("Base model changed while its checkpoint fingerprint was being calculated.") from None
            relative = path.relative_to(root).as_posix()
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
                digest.update(f"D\0{relative}\0{metadata.st_mtime_ns}\0".encode("utf-8"))
            elif stat.S_ISREG(metadata.st_mode):
                digest.update(
                    f"F\0{relative}\0{metadata.st_size}\0{metadata.st_mtime_ns}\0".encode("utf-8")
                )
                if relative == "config.json":
                    digest.update(file_sha256(path).encode("ascii"))
            else:
                raise ValueError("Base model contains an unsupported filesystem entry.")
    return digest.hexdigest()


def checkpoint_root(output_dir: Path) -> Path:
    return Path(output_dir) / CHECKPOINT_DIRECTORY


def _checkpoint_directories(output_dir: Path) -> list[Path]:
    output = Path(output_dir)
    _safe_directory(output, "Adapter output directory is not a safe directory.")
    root = checkpoint_root(output)
    if root.is_symlink() or getattr(root, "is_junction", lambda: False)():
        raise ValueError("AXIOM checkpoint storage cannot be a symbolic link or junction.")
    if not root.exists():
        return []
    _safe_directory(root, "AXIOM checkpoint storage is not a safe directory.")
    checkpoints: list[tuple[int, Path]] = []
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        raise ValueError("AXIOM checkpoint storage could not be inspected.") from None
    for entry in entries:
        if entry.is_symlink() or getattr(entry, "is_junction", lambda: False)():
            raise ValueError("Checkpoint storage cannot contain symbolic links or junctions.")
        match = _CHECKPOINT_NAME.fullmatch(entry.name)
        if match and entry.is_dir():
            checkpoints.append((int(match.group(1)), entry))
        elif entry.name.startswith(_TEMP_PREFIX) and entry.is_dir():
            # An interrupted atomic write is never resumed; prior complete
            # checkpoints remain eligible and the temporary folder is retained.
            continue
        else:
            raise ValueError("Checkpoint storage contains an unknown entry; refusing to overwrite it.")
    return [path for _, path in sorted(checkpoints)]


def read_checkpoint_state(checkpoint_dir: Path) -> dict[str, Any]:
    """Validate a checkpoint's shape and file hashes before any deserialization."""
    checkpoint = Path(checkpoint_dir)
    _safe_directory(checkpoint, "Selected checkpoint is not a safe directory.")
    if not _CHECKPOINT_NAME.fullmatch(checkpoint.name):
        raise ValueError("Selected checkpoint has an invalid name.")
    required = (*_REQUIRED_FILES, "state.json")
    for name in required:
        target = checkpoint / name
        if not _regular_file(target):
            raise ValueError(f"Checkpoint is incomplete or contains an unsafe file: {name}.")
    state_path = checkpoint / "state.json"
    try:
        if state_path.stat().st_size > _MAX_STATE_BYTES:
            raise ValueError("Checkpoint state manifest is too large.")
        with open_regular_file(state_path) as stream:
            state = json.loads(stream.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("Checkpoint state manifest is missing or invalid.") from None
    version = state.get("schema_version") if isinstance(state, dict) else None
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("Checkpoint schema is unsupported.")
    step = state.get("completed_steps")
    folder_step = int(checkpoint.name.removeprefix("checkpoint-"))
    if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= 1000 or step != folder_step:
        raise ValueError("Checkpoint step count does not match its directory name.")
    files = state.get("files")
    if not isinstance(files, dict):
        raise ValueError("Checkpoint file manifest is invalid.")
    for name in _REQUIRED_FILES:
        expected = files.get(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Checkpoint file manifest is invalid for {name}.")
        if file_sha256(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint integrity check failed for {name}.")
    training = state.get("training")
    if not isinstance(training, dict):
        raise ValueError("Checkpoint training metadata is invalid.")
    return state


def latest_checkpoint(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    checkpoints = _checkpoint_directories(Path(output_dir))
    if not checkpoints:
        raise ValueError("Resume policy requires at least one complete AXIOM checkpoint.")
    selected = checkpoints[-1]
    return selected, read_checkpoint_state(selected)


def checkpoint_count(output_dir: Path) -> int:
    return len(_checkpoint_directories(Path(output_dir)))


def _fsync_checkpoint_file(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(
    output_dir: Path,
    completed_steps: int,
    *,
    model: Any,
    torch_module: Any,
    optimizer_payload: dict[str, Any],
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write a complete checkpoint and atomically publish its directory."""
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int) or not 1 <= completed_steps <= 1000:
        raise ValueError("Checkpoint step count is invalid.")
    output = Path(output_dir)
    _safe_directory(output, "Adapter output directory changed or is no longer safe.")
    root = checkpoint_root(output)
    root.mkdir(mode=0o700, exist_ok=True)
    _safe_directory(root, "AXIOM checkpoint storage is not a safe directory.")
    _checkpoint_directories(output)
    final_dir = root / f"checkpoint-{completed_steps:06d}"
    if final_dir.exists():
        raise ValueError("A checkpoint for this step already exists; refusing to overwrite it.")
    temporary = root / f"{_TEMP_PREFIX}{completed_steps:06d}-{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700)
    try:
        model.save_pretrained(str(temporary), safe_serialization=True)
        torch_module.save(optimizer_payload, str(temporary / "optimizer_state.pt"))
        hashes = {name: file_sha256(temporary / name) for name in _REQUIRED_FILES}
        for name in _REQUIRED_FILES:
            _fsync_checkpoint_file(temporary / name)
        state = {
            "schema_version": 1,
            "completed_steps": completed_steps,
            "created_at": datetime.now(UTC).isoformat(),
            "training": training_metadata,
            "files": hashes,
        }
        state_path = temporary / "state.json"
        with state_path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "posix":
            for entry in temporary.iterdir():
                if _regular_file(entry):
                    os.chmod(entry, 0o600)
            os.chmod(temporary, 0o700)
        _fsync_directory(temporary)
        temporary.rename(final_dir)
        _fsync_directory(root)
        checkpoints = _checkpoint_directories(output)
        while len(checkpoints) > _MAX_RETAINED_CHECKPOINTS:
            oldest = checkpoints.pop(0)
            _remove_checkpoint(oldest, root)
        return {
            "enabled": True,
            "latest_step": completed_steps,
            "retained": len(checkpoints),
            "last_saved_at": state["created_at"],
            "resume_available": True,
        }
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _remove_checkpoint(path: Path, root: Path) -> None:
    if path.is_symlink() or path.parent != root or not _CHECKPOINT_NAME.fullmatch(path.name):
        raise ValueError("Refusing to remove an unmanaged checkpoint path.")
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            entry = current_path / name
            if entry.is_symlink() or getattr(entry, "is_junction", lambda: False)():
                raise ValueError("Refusing to prune checkpoint storage containing a link.")
    shutil.rmtree(path)


def load_checkpoint_payload(checkpoint_dir: Path, torch_module: Any) -> dict[str, Any]:
    """Load only a manifest-verified checkpoint through PyTorch's restricted unpickler."""
    state = read_checkpoint_state(checkpoint_dir)
    payload_path = Path(checkpoint_dir) / "optimizer_state.pt"
    try:
        with open_regular_file(payload_path) as stream:
            payload = torch_module.load(stream, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Checkpoint optimizer state could not be safely loaded: {str(exc)[:180]}") from None
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint optimizer state has an invalid shape.")
    return {"state": state, "payload": payload}
