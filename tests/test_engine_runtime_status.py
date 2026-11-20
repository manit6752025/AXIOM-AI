from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from axiom.core import engine


MINIMUM_VERSIONS = {
    "torch": "2.2.0",
    "tokenizers": "0.19.0",
    "safetensors": "0.4.0",
}


def _devices(*, cuda: bool = False, rocm: bool = False, mps: bool = False, xpu: bool = False, cpu: bool = True):
    return [
        {"id": "cuda", "name": "NVIDIA CUDA", "available": cuda, "reason": "cuda probe", "memory_free_gib": 4.0 if cuda else None, "memory_total_gib": 8.0 if cuda else None},
        {"id": "rocm", "name": "AMD ROCm", "available": rocm, "reason": "rocm probe", "memory_free_gib": 4.0 if rocm else None, "memory_total_gib": 8.0 if rocm else None},
        {"id": "mps", "name": "Apple Metal (MPS)", "available": mps, "reason": "mps probe", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "xpu", "name": "Intel XPU", "available": xpu, "reason": "xpu probe", "memory_free_gib": 4.0 if xpu else None, "memory_total_gib": 8.0 if xpu else None},
        {"id": "cpu", "name": "CPU", "available": cpu, "reason": "cpu probe", "memory_free_gib": None, "memory_total_gib": None},
    ]


def _install_probe(monkeypatch, *, installed=None, versions=None, devices=None):
    installed = installed or {"torch": True, "tokenizers": True, "safetensors": True, "gguf": False, "numpy": False}
    versions = versions or {**MINIMUM_VERSIONS}
    monkeypatch.setattr(
        engine.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name) if installed.get(name, False) else None,
    )

    def package_version(name: str) -> str:
        if name not in versions or versions[name] is None:
            raise PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(engine.importlib.metadata, "version", package_version)
    monkeypatch.setattr(engine, "_device_capability_rows", lambda: devices if devices is not None else _devices())


def test_runtime_status_reports_required_and_optional_dependency_readiness(monkeypatch) -> None:
    _install_probe(monkeypatch)
    monkeypatch.setattr(engine, "_LOADED_RUNTIME", None)
    monkeypatch.setattr(engine, "_LOADED_ARCHITECTURE", None)

    report = engine.runtime_status()

    assert report["ready"] is True
    assert report["minimum_versions"] == MINIMUM_VERSIONS
    assert report["packages"] == {
        "torch": True, "tokenizers": True, "safetensors": True, "gguf": False, "numpy": False,
    }
    assert report["versions"]["torch"] == "2.2.0"
    assert report["supported_model_types"] == ["llama", "mistral", "qwen2", "qwen3"]
    assert report["supported_weights"] == ["safetensors"]
    assert report["remote_code"] is False
    assert report["execution_backend"] == "pytorch-native-axiom"
    assert report["transformers_generation_used"] is False
    assert report["ollama_used"] is False
    assert report["recommended_device"] == "cpu"
    assert report["loaded"] is False


@pytest.mark.parametrize(
    ("installed", "versions", "expected_problem"),
    [
        ({"torch": False, "tokenizers": True, "safetensors": True}, MINIMUM_VERSIONS, "torch is not installed"),
        ({"torch": True, "tokenizers": False, "safetensors": True}, MINIMUM_VERSIONS, "tokenizers is not installed"),
        ({"torch": True, "tokenizers": True, "safetensors": False}, MINIMUM_VERSIONS, "safetensors is not installed"),
        ({"torch": True, "tokenizers": True, "safetensors": True}, {**MINIMUM_VERSIONS, "torch": "2.1.2"}, "torch 2.1.2 is too old"),
    ],
)
def test_runtime_status_fails_closed_for_missing_or_unsupported_dependencies(monkeypatch, installed, versions, expected_problem) -> None:
    _install_probe(monkeypatch, installed=installed, versions=versions)
    report = engine.runtime_status()

    assert report["ready"] is False
    assert any(expected_problem in problem for problem in report["problems"])


def test_runtime_status_fails_closed_when_no_device_can_initialize(monkeypatch) -> None:
    _install_probe(monkeypatch, devices=_devices(cpu=False))

    report = engine.runtime_status()

    assert report["ready"] is False
    assert any("no supported execution device" in problem.lower() for problem in report["problems"])
    assert report["recommended_device"] is None


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        (_devices(cuda=True, cpu=True), "cuda"),
        (_devices(rocm=True, cpu=True), "rocm"),
        (_devices(mps=True, cpu=True), "mps"),
        (_devices(xpu=True, cpu=True), "xpu"),
        (_devices(cpu=True), "cpu"),
    ],
)
def test_runtime_status_recommends_the_first_available_accelerator_then_cpu(monkeypatch, devices, expected) -> None:
    _install_probe(monkeypatch, devices=devices)

    report = engine.runtime_status()

    assert report["ready"] is True
    assert report["recommended_device"] == expected
    assert report["devices"] == devices


def test_runtime_status_reports_loaded_architecture_device_and_memory_mode(monkeypatch) -> None:
    _install_probe(monkeypatch, devices=_devices(mps=True, cpu=True))
    runtime = {"model": SimpleNamespace(memory_mode="disk_streaming"), "details": {"precision": "bf16"}, "device": "mps"}
    monkeypatch.setattr(engine, "_LOADED_RUNTIME", runtime)
    monkeypatch.setattr(engine, "_LOADED_ARCHITECTURE", "Qwen3ForCausalLM")

    report = engine.runtime_status()

    assert report["loaded"] is True
    assert report["architecture"] == "Qwen3ForCausalLM"
    assert report["loaded_device"] == "mps"
    assert report["loaded_memory_mode"] == "disk_streaming"
    assert report["loaded_precision"] == "bf16"
