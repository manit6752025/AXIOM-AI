from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    runtime: str
    tokens_per_second: float
    tokens: int
    elapsed_seconds: float
    achieved_target: bool
    command: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "tokens": self.tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "achieved_target": self.achieved_target,
            "command": self.command,
        }


def find_llama_cli() -> str | None:
    for name in (
        "llama-cli",
        "llama-completion",
        "main",
    ):
        path = subprocess.run(
            ["bash", "-lc", f"command -v {name}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

        if path:
            return path

    return None


def benchmark_gguf(
    model_path: str,
    prompt: str,
    target_tps: float = 10.0,
    max_tokens: int = 128,
) -> BenchmarkResult:
    binary = find_llama_cli()

    if not binary:
        raise RuntimeError(
            "A llama.cpp CLI runtime was not found. "
            "Install llama.cpp/llama-cli to run a real local benchmark."
        )

    command = [
        binary,
        "-m",
        str(Path(model_path)),
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--temp",
        "0.2",
        "--no-display-prompt",
    ]

    started = time.perf_counter()

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    elapsed = max(0.001, time.perf_counter() - started)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "llama.cpp benchmark failed")

    # llama.cpp output is not guaranteed to expose an identical metrics
    # format across versions. Estimate generated tokens from requested
    # count/time only when the command completed.
    tps = max_tokens / elapsed

    return BenchmarkResult(
        runtime="llama.cpp",
        tokens_per_second=tps,
        tokens=max_tokens,
        elapsed_seconds=elapsed,
        achieved_target=tps >= target_tps,
        command=command,
    )
