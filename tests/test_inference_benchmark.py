from __future__ import annotations

import threading
import time

import pytest

from axiom.core import inference


def test_benchmark_runs_warmups_then_measured_concurrent_waves_without_returning_text(monkeypatch) -> None:
    lock = threading.Lock()
    active = 0
    peak_active = 0
    calls = 0

    def fake_run(model: str, prompt: str, *, num_predict: int, temperature: float) -> dict:
        nonlocal active, peak_active, calls
        with lock:
            calls += 1
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {
            "model": model,
            "response": "private response must not be returned",
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 3,
            "total_duration_ms": 10.0,
            "generated_at": "ignored",
        }

    monkeypatch.setattr(inference, "run_local_inference", fake_run)
    result = inference.run_local_inference_benchmark(
        "test-model",
        "private prompt must not be returned",
        num_predict=32,
        temperature=0.2,
        warmup_runs=1,
        iterations=2,
        concurrency=2,
    )

    assert calls == 5
    assert peak_active == 2
    assert result["configuration"]["measured_requests"] == 4
    assert result["measurements"]["completion_tokens_known"] == 4
    assert result["measurements"]["total_completion_tokens"] == 12
    assert result["measurements"]["completion_tokens_per_second"] > 0
    assert result["measurements"]["latency_ms"]["median"] > 0
    assert len(result["measurements"]["requests"]) == 4
    assert result["privacy"] == {
        "prompt_returned": False,
        "responses_returned": False,
        "results_persisted": False,
    }
    assert "private prompt must not be returned" not in str(result)
    assert "private response must not be returned" not in str(result)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"iterations": 0}, "iterations"),
        ({"warmup_runs": 3}, "warm-up"),
        ({"concurrency": 5}, "concurrency"),
        ({"iterations": 3, "concurrency": 4}, "may not exceed 10"),
        ({"num_predict": 513}, "token limit"),
    ],
)
def test_benchmark_rejects_unbounded_work(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        inference.run_local_inference_benchmark("test-model", "hello", **kwargs)


def test_benchmark_validates_model_prompt_and_temperature_before_generating(monkeypatch) -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("invalid benchmark input called the runtime")

    monkeypatch.setattr(inference, "run_local_inference", should_not_run)
    with pytest.raises(ValueError, match="[Mm]odel name"):
        inference.run_local_inference_benchmark("http://remote/model", "hello")
    with pytest.raises(ValueError, match="[Pp]rompt"):
        inference.run_local_inference_benchmark("test-model", "   ")
    with pytest.raises(ValueError, match="Temperature"):
        inference.run_local_inference_benchmark("test-model", "hello", temperature=3)
