from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from axiom.models.analysis import estimate_parameters_from_config


@dataclass
class ModelInspection:
    path: Path
    architecture: str | None
    model_type: str | None
    parameter_count: int | None
    parameter_source: str
    model_format: str
    weight_size_bytes: int
    has_config: bool
    has_tokenizer: bool
    has_safetensors: bool
    has_gguf: bool
    has_pytorch_weights: bool
    has_transformers_weights: bool
    transformers_weight_size_bytes: int
    capabilities: list[str]
    estimated_vram_gb: float | None
    status: str


def _detect_capabilities(config: dict) -> list[str]:
    model_type = str(config.get("model_type", "")).lower()
    raw_architectures = config.get("architectures", [])
    architectures_value = (
        raw_architectures if isinstance(raw_architectures, list) else []
    )
    architectures = [
        str(item).lower() for item in architectures_value if isinstance(item, str)
    ]

    markers = (
        "causallm",
        "forcausallm",
        "lmhead",
        "gpt",
        "llama",
        "qwen",
        "mistral",
    )

    capabilities = []

    if any(
        marker in model_type
        or any(marker in architecture for architecture in architectures)
        for marker in markers
    ):
        capabilities.append("Text generation")

    if config.get("vision_config") or "vision" in model_type:
        capabilities.append("Vision")

    if "audio" in model_type or "speech" in model_type:
        capabilities.append("Audio")

    return capabilities or ["Unknown"]


def inspect_model(path: str) -> ModelInspection:
    raw_path = str(path)
    if not raw_path.strip() or "\x00" in raw_path:
        raise ValueError("Model path is empty or contains an invalid character.")
    requested_path = Path(raw_path).expanduser()
    try:
        # Resolve aliases and symlinks before any child paths are constructed.
        model_path = requested_path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model path not found: {requested_path}") from None
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Invalid model path: {requested_path}") from exc

    if not model_path.is_dir():
        raise ValueError(f"Model path is not a directory: {model_path}")

    config_path = model_path / "config.json"
    if config_path.is_symlink():
        raise ValueError("Model config.json must not be a symbolic link.")
    config = {}

    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid config.json: {exc}") from exc

        if not isinstance(config, dict):
            raise ValueError("Invalid config.json: expected a JSON object.")

    # Do not follow file symlinks while collecting model metadata. In particular,
    # a link inside a workspace model must not expose metadata for an external file.
    files = [
        file
        for file in model_path.rglob("*")
        if not file.is_symlink() and file.is_file()
    ]

    safetensors = any(
        file.suffix.lower() == ".safetensors"
        or file.name.endswith(".safetensors.index.json")
        for file in files
    )

    gguf = any(file.suffix.lower() == ".gguf" for file in files)

    pytorch = any(file.suffix.lower() in {".bin", ".pt", ".pth"} for file in files)

    if gguf:
        model_format = "GGUF"
    elif safetensors:
        model_format = "Safetensors"
    elif pytorch:
        model_format = "PyTorch"
    else:
        model_format = "Unknown"

    weight_files = [
        file
        for file in files
        if file.suffix.lower()
        in {
            ".safetensors",
            ".gguf",
            ".bin",
            ".pt",
            ".pth",
        }
    ]

    weight_size = sum(file.stat().st_size for file in weight_files)
    transformers_weight_files = [
        file
        for file in weight_files
        if file.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
    ]
    transformers_weight_size = sum(file.stat().st_size for file in transformers_weight_files)

    tokenizer_files = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }

    has_tokenizer = any(file.name in tokenizer_files for file in files)

    architectures = config.get("architectures")
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else None
    )

    parameter_count, parameter_source = estimate_parameters_from_config(config)

    estimated_vram_gb = None

    if parameter_count is not None and parameter_count > 0:
        estimated_vram_gb = round(
            parameter_count * 2 * 1.2 / (1024**3),
            2,
        )

    if not config_path.is_file():
        status = "⚠ config.json not found"
    elif not weight_files:
        status = "⚠ model weights not found"
    else:
        status = "✓ Model structure detected"

    return ModelInspection(
        path=model_path,
        architecture=architecture,
        model_type=config.get("model_type"),
        parameter_count=parameter_count,
        parameter_source=parameter_source,
        model_format=model_format,
        weight_size_bytes=weight_size,
        has_config=config_path.is_file(),
        has_tokenizer=has_tokenizer,
        has_safetensors=safetensors,
        has_gguf=gguf,
        has_pytorch_weights=pytorch,
        has_transformers_weights=transformers_weight_size > 0,
        transformers_weight_size_bytes=transformers_weight_size,
        capabilities=_detect_capabilities(config),
        estimated_vram_gb=estimated_vram_gb,
        status=status,
    )
