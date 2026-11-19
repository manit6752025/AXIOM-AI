from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api import workspace
from axiom.core.hardware import HardwareInfo
from axiom.models.compatibility import estimate_model_compatibility
from axiom.models.registry import Model


def _hardware(*, ram: float = 16.0, vram: float | None = 4.0) -> HardwareInfo:
    return HardwareInfo(
        os_name="Linux",
        architecture="x86_64",
        cpu_cores=8,
        ram_gb=ram,
        gpu_name="Example NVIDIA GPU" if vram is not None else None,
        vram_gb=vram,
        cuda_available=vram is not None,
    )


def test_q4_fit_report_returns_explicit_ranges_and_distinguishes_ram_from_vram() -> None:
    model = Model(
        id="a" * 32,
        name="Qwen seven",
        source="models/qwen-seven",
        format="gguf",
        parameters="7B",
        quantization="Q4_K_M",
    )

    report = estimate_model_compatibility(model, _hardware())

    assert report["model"]["parameter_count_billions"] == 7
    assert report["quantization_bits_per_parameter"] == {"low": 4.0, "high": 5.0}
    assert report["estimated_weight_gb"] == {"low": 3.5, "high": 4.38}
    assert report["estimated_runtime_memory_gb"] is not None
    assert report["system_ram_fit"]["status"] == "likely"
    assert report["system_ram_fit"]["budget_gb"] == 11.2
    assert report["gpu_memory_fit"]["status"] == "unlikely"
    assert report["format_compatibility"]["status"] == "candidate_not_verified"
    assert any("not a benchmark" in item.lower() for item in report["warnings"])


def test_missing_vram_is_unknown_not_reported_as_no_gpu() -> None:
    model = Model(name="FP16 checkpoint", source="local/checkpoint", format="safetensors", parameters="7B", quantization="fp16")

    report = estimate_model_compatibility(model, _hardware(vram=None))

    assert report["gpu_memory_fit"]["status"] == "unknown"
    assert report["gpu_memory_fit"]["physical_gb"] is None
    assert any("does not mean a GPU is absent" in item for item in report["warnings"])
    assert report["format_compatibility"]["likely_runtimes"] == ["Transformers", "AXIOM Engine"]


def test_unknown_quantization_omits_memory_estimates_instead_of_guessing() -> None:
    model = Model(name="Unknown dtype", source="local/model", format="safetensors", parameters="8B")

    report = estimate_model_compatibility(model, _hardware())

    assert report["estimated_weight_gb"] is None
    assert report["estimated_runtime_memory_gb"] is None
    assert report["system_ram_fit"]["status"] == "unknown"
    assert any("omitted rather than guessed" in item for item in report["warnings"])


def test_parameter_metadata_without_unit_suffix_is_flagged() -> None:
    model = Model(name="Ambiguous metadata", source="local/model", format="gguf", parameters="7", quantization="Q4_K_M")

    report = estimate_model_compatibility(model, _hardware())

    assert report["model"]["parameter_count_billions"] == 7
    assert report["model"]["parameter_basis"] == "assumed_billions_without_suffix"
    assert any("confirm the registered value" in item for item in report["warnings"])


def test_compatibility_endpoint_returns_report_for_registered_model(monkeypatch) -> None:
    model = Model(
        id="b" * 32,
        name="API model",
        source="models/api-model",
        format="gguf",
        parameters="7B",
        quantization="Q4_K_M",
    )

    class StubRegistry:
        def get(self, model_id: str) -> Model | None:
            return model if model_id == model.id else None

    monkeypatch.setattr(workspace, "ModelRegistry", StubRegistry)
    monkeypatch.setattr(workspace, "detect_hardware", _hardware)
    app = FastAPI()
    app.include_router(workspace.router)
    client = TestClient(app)

    response = client.get(f"/api/workspace/models/{model.id}/compatibility")
    missing = client.get(f"/api/workspace/models/{'c' * 32}/compatibility")

    assert response.status_code == 200
    assert response.json()["report"]["model"]["id"] == model.id
    assert response.json()["report"]["gpu_memory_fit"]["status"] == "unlikely"
    assert missing.status_code == 404
