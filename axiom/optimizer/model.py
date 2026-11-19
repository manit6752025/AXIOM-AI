from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom.api.model_paths import (
    MODEL_ROOT,
    validate_model_path,
    validate_output_path,
)
from axiom.optimizer.agent_profiles import AgentProfile
from axiom.optimizer.system import SystemInfo
from axiom.models.analysis import estimate_parameters_from_config


@dataclass(frozen=True)
class ModelInfo:
    source: str
    local_path: str
    parameter_billions: float | None
    estimated_fp16_gb: float | None
    parameter_source: str
    files: int
    weights: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "local_path": self.local_path,
            "parameter_billions": (
                round(self.parameter_billions, 3)
                if self.parameter_billions is not None
                else None
            ),
            "estimated_fp16_gb": (
                round(self.estimated_fp16_gb, 2)
                if self.estimated_fp16_gb is not None
                else None
            ),
            "parameter_source": self.parameter_source,
            "files": self.files,
            "weights": self.weights,
        }


RUNTIME_KEEP_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "spiece.model",
    "tokenizer.model",
}

RUNTIME_EXTENSIONS = {
    ".safetensors",
    ".bin",
    ".gguf",
    ".json",
    ".model",
    ".txt",
}


def _trusted_files(root: Path) -> tuple[Path, ...]:
    resolved_root = root.resolve()
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        resolved_path = path.resolve(strict=True)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"model path escapes the trusted model root: {path}"
            ) from exc
        files.append(path)
    return tuple(files)


def _copy_without_following_symlinks(source: Path, destination: Path) -> None:
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with (
                os.fdopen(source_fd, "rb") as source_handle,
                os.fdopen(destination_fd, "wb") as destination_handle,
            ):
                source_fd = destination_fd = -1
                shutil.copyfileobj(source_handle, destination_handle)
        finally:
            if destination_fd >= 0:
                os.close(destination_fd)
    finally:
        if source_fd >= 0:
            os.close(source_fd)


def _config(path: Path) -> dict[str, Any]:
    trusted_path = validate_model_path(path)
    root_str = os.path.realpath(trusted_path)
    model_root_real = os.path.realpath(MODEL_ROOT)
    if root_str != model_root_real and not root_str.startswith(
        model_root_real + os.sep
    ):
        return {}
    if not os.path.isdir(root_str):
        return {}

    for name in ("config.json", "model_config.json"):
        raw = os.path.join(root_str, name)
        if os.path.islink(raw):
            raise ValueError(
                f"model configuration contains an unsupported symlink: {raw}"
            )
        cfg = os.path.realpath(raw)
        if not cfg.startswith(root_str + os.sep):
            continue
        try:
            fd = os.open(cfg, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            continue
        except OSError as exc:
            if exc.errno == getattr(os, "ELOOP", 40):
                raise ValueError(
                    f"model configuration contains an unsupported symlink: {cfg}"
                ) from exc
            continue

        try:
            with os.fdopen(fd, "r", encoding="utf-8") as config_file:
                fd = -1
                value = json.load(config_file)
            if isinstance(value, dict):
                return value
        except (OSError, ValueError):
            import logging as _axiom_logging

            _axiom_logging.getLogger(__name__).debug(
                "intentionally ignored exception", exc_info=True
            )
        finally:
            if fd >= 0:
                os.close(fd)
    return {}


def inspect_model(source: str) -> ModelInfo:
    path = validate_model_path(source)

    if not path.exists():
        return ModelInfo(
            source=source,
            local_path="",
            parameter_billions=None,
            estimated_fp16_gb=None,
            parameter_source="unknown",
            files=0,
            weights=[],
        )

    root = path if path.is_dir() else path.parent
    root = validate_model_path(os.fspath(root))
    cfg = _config(root)

    parameter_count, parameter_source = estimate_parameters_from_config(cfg)
    params_b = parameter_count / 1_000_000_000 if parameter_count else None

    weights = [
        str(p.relative_to(root))
        for p in _trusted_files(root)
        if p.is_file() and p.suffix.lower() in {".safetensors", ".bin", ".gguf"}
    ]

    count = len(_trusted_files(root))

    return ModelInfo(
        source=source,
        local_path=str(root),
        parameter_billions=params_b,
        estimated_fp16_gb=params_b * 2.0 if params_b is not None else None,
        parameter_source=parameter_source,
        files=count,
        weights=weights,
    )


def choose_quantization(
    model: ModelInfo,
    system: SystemInfo,
    profile: AgentProfile,
) -> str:
    if model.estimated_fp16_gb is None:
        return profile.preferred_quantization

    if system.gpu_available is True and system.vram_gb is not None:
        usable = system.vram_gb
    elif system.ram_gb is not None:
        usable = system.ram_gb * 0.55
    else:
        return profile.preferred_quantization

    if usable <= 0:
        return profile.preferred_quantization

    if model.estimated_fp16_gb <= usable * 0.75:
        return "int8"

    if model.estimated_fp16_gb <= usable * 1.35:
        return "int4"

    return "int4"


def create_runtime_bundle(
    source: str,
    destination: str,
) -> dict[str, Any]:
    src = validate_model_path(source)
    dst = validate_output_path(destination)

    if not src.exists():
        raise FileNotFoundError(source)

    dst_root = dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    root = src if src.is_dir() else src.parent
    root = root.resolve()

    copied: list[str] = []
    skipped: list[str] = []

    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError(
                f"runtime bundle source contains an unsupported symlink: {item}"
            )
        if not item.is_file():
            continue

        resolved_item = item.resolve(strict=True)
        try:
            resolved_item.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"runtime bundle source escapes the trusted model root: {item}"
            ) from exc

        rel = item.relative_to(root)
        name = item.name.lower()
        suffix = item.suffix.lower()

        keep = name in RUNTIME_KEEP_NAMES or suffix in {".safetensors", ".bin", ".gguf"}

        if not keep:
            skipped.append(str(rel))
            continue

        target = dst / rel
        resolved_target = target.resolve(strict=False)
        try:
            resolved_target.relative_to(dst_root)
        except ValueError as exc:
            raise ValueError(
                f"runtime bundle destination escapes the trusted output root: {target}"
            ) from exc
        if target.is_symlink():
            raise ValueError(
                f"runtime bundle destination contains an unsupported symlink: {target}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_without_following_symlinks(item, target)
        copied.append(str(rel))

    return {
        "source": str(root),
        "destination": str(dst),
        "copied": copied,
        "skipped_non_runtime": skipped,
    }


def write_runtime_profile(
    bundle: str,
    profile: AgentProfile,
    quantization: str,
    system: SystemInfo,
) -> Path:
    path = Path(validate_output_path(bundle)) / "axiom-runtime.json"
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(Path(validate_output_path(bundle)).resolve())
    except ValueError as exc:
        raise ValueError(
            "runtime profile path escapes the trusted output root"
        ) from exc
    if path.is_symlink():
        raise ValueError("runtime profile destination cannot be a symlink")

    data = {
        "agent_profile": profile.key,
        "agent_profile_number": profile.number,
        "agent_name": profile.name,
        "context_tokens": profile.context_tokens,
        "temperature": profile.temperature,
        "quantization_target": quantization,
        "system": system.to_dict(),
        "target_tokens_per_second": 10.0,
        "throughput_status": "benchmark_required",
        "note": (
            "AXIOM will only report 10 tok/s as achieved after a real "
            "benchmark confirms it on this device."
        ),
    }

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    return path
