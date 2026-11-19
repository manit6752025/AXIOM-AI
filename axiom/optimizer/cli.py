from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .engine import OptimizationRequest, detect_hardware, optimize_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="axiom optimize")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("hardware")
    o = sub.add_parser("model")
    o.add_argument("model", help="Hugging Face model ID or local model path")
    o.add_argument(
        "--agent-type",
        choices=["chat", "coding", "reasoning", "tool-use", "general"],
        default="general",
    )
    o.add_argument("--target-tps", type=float, default=10.0)
    o.add_argument("--context", type=int, default=4096)
    o.add_argument(
        "--quality", choices=["speed", "balanced", "quality"], default="balanced"
    )
    o.add_argument(
        "--quantization",
        choices=["auto", "fp16", "int8", "int4", "int4-cpu"],
        default="auto",
    )
    o.add_argument("--output", default="optimized-model")
    a = p.parse_args(argv)
    if a.command == "hardware":
        print(json.dumps(detect_hardware(), indent=2))
        return 0
    result = optimize_model(
        OptimizationRequest(
            model=a.model,
            agent_type=a.agent_type,
            target_tokens_per_second=a.target_tps,
            context_length=a.context,
            quality=a.quality,
            output_dir=a.output,
            quantization=a.quantization,
        )
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
