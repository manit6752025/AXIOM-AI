from __future__ import annotations

from typing import Any

from axiom.optimizer.agent_profiles import PROFILES, get_profile
from axiom.optimizer.model import choose_quantization, inspect_model
from axiom.optimizer.system import inspect_system


def optimizer_profiles() -> list[dict[str, Any]]:
    return [
        {
            "number": p.number,
            "key": p.key,
            "name": p.name,
            "description": p.description,
            "context_tokens": p.context_tokens,
            "temperature": p.temperature,
            "preferred_quantization": p.preferred_quantization,
        }
        for p in PROFILES
    ]


def build_plan(
    model: str,
    profile: int,
    target_tps: float = 10.0,
) -> dict[str, Any]:
    selected = get_profile(profile)
    system = inspect_system()
    info = inspect_model(model)
    quant = choose_quantization(info, system, selected)

    return {
        "profile": {
            "number": selected.number,
            "key": selected.key,
            "name": selected.name,
            "description": selected.description,
            "context_tokens": selected.context_tokens,
            "temperature": selected.temperature,
        },
        "model": info.to_dict(),
        "system": system.to_dict(),
        "optimization": {
            "target_tokens_per_second": target_tps,
            "recommended_quantization": quant,
            "recommended_format": "safetensors",
            "runtime_bundle": True,
            "benchmark_required": True,
        },
    }
