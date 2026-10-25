"""AXIOM Beta.6 agent/model optimization helpers."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class OptimizationPlan:
    agent_type: str
    target_tokens_per_second: float
    model_id: str | None
    quantization: str
    context_length: int
    system_memory_gb: float
    cpu: str
    notes: list[str]

    def to_dict(self):
        return asdict(self)


def detect_system() -> dict:
    memory_gb = 0.0
    try:
        pages = int(__import__("os").sysconf("SC_PHYS_PAGES"))
        page_size = int(__import__("os").sysconf("SC_PAGE_SIZE"))
        memory_gb = pages * page_size / (1024**3)
    except Exception:  # noqa: BLE001
        __import__("logging").getLogger(__name__).debug(
            "intentionally ignored exception", exc_info=True
        )
    return {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "memory_gb": round(memory_gb, 2),
    }


def recommend_quantization(memory_gb: float, model_size_b: float | None = None) -> str:
    if model_size_b is None:
        return "auto"
    if model_size_b <= 3 and memory_gb >= 8:
        return "int8"
    if model_size_b <= 8 and memory_gb >= 12:
        return "int4"
    return "int4"


def build_plan(
    *,
    agent_type: str,
    target_tokens_per_second: float = 10.0,
    model_id: str | None = None,
    model_size_b: float | None = None,
    context_length: int = 2048,
) -> OptimizationPlan:
    system = detect_system()
    quant = recommend_quantization(system["memory_gb"], model_size_b)
    notes = [
        "Target throughput is a planning target; AXIOM must benchmark the actual device.",
        "Quantization should be validated against model quality and operator support.",
    ]
    return OptimizationPlan(
        agent_type=agent_type,
        target_tokens_per_second=target_tokens_per_second,
        model_id=model_id,
        quantization=quant,
        context_length=context_length,
        system_memory_gb=system["memory_gb"],
        cpu=system["cpu"],
        notes=notes,
    )


def policy_audit(model_path: str | None = None) -> dict:
    """Report policy/safety metadata without attempting to bypass model controls."""
    result = {
        "status": "audit_only",
        "guardrail_removal": False,
        "message": "AXIOM does not remove or bypass model safety controls.",
        "model_path": model_path,
        "checks": [],
    }
    if model_path:
        p = Path(model_path)
        result["checks"].append({"path_exists": p.exists()})
        cfg = p / "config.json" if p.is_dir() else None
        if cfg and cfg.exists():
            try:
                data = json.loads(cfg.read_text())
                result["checks"].append(
                    {
                        "architecture": data.get("architectures"),
                        "model_type": data.get("model_type"),
                    }
                )
            except Exception:  # noqa: BLE001
                result["checks"].append({"config_parse": "failed"})
    return result
