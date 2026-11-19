"""Hardware-aware model optimization planning."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from axiom.api.model_paths import validate_model_path


@dataclass(frozen=True)
class OptimizationPlan:
    model: str
    agent_type: str
    use_case: str
    privacy: str
    latency: str
    target_tokens_per_second: float
    estimated_parameters_b: float
    estimated_memory_gb: float
    recommended_format: str
    recommended_quantization: str
    sequence_length: int
    fits_estimate: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_root() -> Path:
    configured = os.environ.get("AXIOM_MODEL_ROOT")
    if not configured:
        raise ValueError(
            "AXIOM_MODEL_ROOT must be configured for local model inspection"
        )
    # codeql[py/path-injection]
    root = Path(validate_model_path(configured)).expanduser().resolve()
    # codeql[py/path-injection]
    if not root.is_dir():
        raise ValueError("AXIOM_MODEL_ROOT is not a directory")
    return root


def _safe_model_config(model: str) -> Path | None:
    root = _model_root()
    candidate = validate_model_path(model).expanduser()
    candidate = candidate if candidate.is_absolute() else root / candidate

    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None

    # codeql[py/path-injection]
    if resolved.is_dir():
        for name in ("config.json", "model_config.json"):
            cfg = resolved / name
            # codeql[py/path-injection]
            if cfg.is_file():
                return cfg
    # codeql[py/path-injection]
    elif resolved.is_file() and resolved.suffix.lower() == ".json":
        return resolved

    return None


def _load_config(model: str) -> dict[str, Any]:
    path = _safe_model_config(model)
    if path is None:
        return {}
    try:
        # codeql[py/path-injection]
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _ram_gb() -> float:
    # codeql[py/path-injection]
    path = Path("/proc/meminfo")
    # codeql[py/path-injection]
    if not path.is_file():
        return 0.0
    try:
        # codeql[py/path-injection]
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError):
        return 0.0
    return 0.0


def plan_optimization(
    model: str,
    agent_type: str = "general-agent",
    use_case: str = "assistant",
    privacy: str = "local",
    latency: str = "low",
    target_tokens_per_second: float = 10.0,
    sequence_length: int = 2048,
    quantization: str = "auto",
) -> OptimizationPlan:
    cfg = _load_config(model)
    hidden = int(cfg.get("hidden_size") or 4096)
    layers = int(cfg.get("num_hidden_layers") or 32)
    params_b = max(1.0, hidden * hidden * layers * 12 / 1_000_000_000)

    if quantization == "auto":
        selected = "int4" if params_b <= 8 else "int4-awq"
    else:
        selected = quantization

    bits = {"fp16": 16, "bf16": 16, "int8": 8, "int4": 4, "int4-awq": 4}.get(
        selected, 4
    )
    memory = params_b * bits / 8 * 1.30 + 1.5
    ram = _ram_gb()
    fits = not (ram and ram < memory + 2)

    notes = [
        "10 tok/s is a target, not a guaranteed result.",
        "A real device benchmark is required before reporting measured throughput.",
        f"Estimated model size: {params_b:.2f}B parameters.",
    ]
    if latency == "low":
        notes.append(
            "Low-latency profile favors stronger quantization and shorter context."
        )
    if use_case == "coding":
        notes.append(
            "Coding profile may benefit from additional context when hardware permits."
        )
    if use_case == "reasoning":
        notes.append("Reasoning profile may trade throughput for model capability.")

    return OptimizationPlan(
        model=model,
        agent_type=agent_type,
        use_case=use_case,
        privacy=privacy,
        latency=latency,
        target_tokens_per_second=max(1.0, float(target_tokens_per_second)),
        estimated_parameters_b=round(params_b, 2),
        estimated_memory_gb=round(memory, 2),
        recommended_format="safetensors",
        recommended_quantization=selected,
        sequence_length=max(256, min(int(sequence_length), 32768)),
        fits_estimate=fits,
        notes=notes,
    )
