from __future__ import annotations

import json

import pytest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from axiom.api import engine as engine_api
from axiom.cli.engine import engine_app
from axiom.core.engine import estimate_native_quantization


GIB = 1024 ** 3
BASE_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 8,
    "intermediate_size": 12,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 8,
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "bos_token_id": 1,
    "tie_word_embeddings": False,
}


def _model(tmp_path: Path, name: str = "tiny-llama", **overrides: object) -> Path:
    model = tmp_path / name
    model.mkdir()
    config = {**BASE_CONFIG, **overrides}
    if config["model_type"] in {"mistral", "qwen3"}:
        config["architectures"] = {
            "mistral": "MistralForCausalLM",
            "qwen3": "Qwen3ForCausalLM",
        }[config["model_type"]]
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model


def _gib(byte_count: float) -> float:
    return round(byte_count / GIB, 9)


def _api_client() -> TestClient:
    app = FastAPI()
    app.include_router(engine_api.router)
    return TestClient(app)


def test_memory_estimator_exact_parameters_weight_formats_workspace_margin_and_range(
    tmp_path: Path, monkeypatch,
) -> None:
    model = _model(tmp_path)
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))

    report = estimate_native_quantization(
        str(model), context_length=16, batch_size=1,
        ram_budget_gib=8, vram_budget_gib=4,
    )

    # 8x8 embedding + untied 8x8 head + one 496-parameter layer + 8 norm values.
    assert report["model"]["estimated_parameter_count"] == 632
    assert report["model"]["parameter_count_method"].startswith("analytic estimate")
    assert report["assumptions"]["kv_cache_context_length"] == 16
    assert report["assumptions"]["kv_prefill_temporary_context_length"] == 0
    assert report["assumptions"]["safety_margin_percent"] == 20
    assert report["assumptions"]["uncertainty_range_percent"] == {"low": -15, "high": 25}

    # Workspace = max(batch * context * intermediate * 4,
    #                  batch * attention_heads * context**2 * 4) = 2,048 bytes.
    expected = {
        "int4": {"weight": 363.4, "kv": 256, "workspace": 2048, "total": 3200.88},
        "int8": {"weight": 663.6, "kv": 256, "workspace": 2048, "total": 3561.12},
        "bf16": {"weight": 1264, "kv": 256, "workspace": 2048, "total": 4281.6},
        "fp16": {"weight": 1264, "kv": 256, "workspace": 2048, "total": 4281.6},
        "fp32": {"weight": 2528, "kv": 512, "workspace": 2048, "total": 6105.6},
    }
    candidates = {item["format"]: item for item in report["candidates"]}
    assert set(candidates) == set(expected)
    for fmt, values in expected.items():
        item = candidates[fmt]
        assert item["estimated_weight_gib"] == _gib(values["weight"])
        assert item["estimated_kv_cache_gib"] == _gib(values["kv"])
        assert item["estimated_kv_prefill_temporary_gib"] == 0
        assert item["estimated_workspace_gib"] == _gib(values["workspace"])
        assert item["estimated_total_gib"] == _gib(values["total"])
        assert item["estimated_range_gib"] == [
            _gib(values["total"] * 0.85), _gib(values["total"] * 1.25),
        ]

    assert candidates["int4"]["native_checkpoint_supported"] is False
    assert candidates["int8"]["native_checkpoint_supported"] is False
    assert candidates["bf16"]["native_checkpoint_supported"] is True
    assert report["recommendation"]["format"] == "bf16"
    assert report["recommendation"]["status"] == "fits_budget"

    # The API and CLI must expose the same metadata-only numerical estimate.
    api_response = _api_client().post("/api/engine/quantization/estimate", json={
        "model_path": str(model), "context_length": 16, "batch_size": 1,
        "ram_budget_gib": 8, "vram_budget_gib": 4,
    })
    assert api_response.status_code == 200, api_response.text
    api_report = api_response.json()
    assert api_report["model"]["estimated_parameter_count"] == 632
    assert api_report["candidates"][2]["estimated_total_gib"] == _gib(4281.6)

    cli = CliRunner().invoke(engine_app, [
        "estimate", "--model-path", str(model), "--context-length", "16",
        "--batch-size", "1", "--ram-budget-gib", "8", "--vram-budget-gib", "4", "--json",
    ])
    assert cli.exit_code == 0, cli.output
    cli_report = json.loads(cli.stdout)
    assert cli_report["model"]["estimated_parameter_count"] == 632
    assert cli_report["candidates"][2]["estimated_total_gib"] == _gib(4281.6)

    # This fixture deliberately contains config.json only: estimate paths must
    # not require model weights or load any checkpoint tensor.
    assert sorted(path.name for path in model.iterdir()) == ["config.json"]


def test_memory_estimator_mistral_sliding_cache_and_prefill_temporary_formula(tmp_path: Path, monkeypatch) -> None:
    model = _model(
        tmp_path, "tiny-mistral", model_type="mistral", sliding_window=3,
    )

    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))

    report = estimate_native_quantization(
        str(model), context_length=24, batch_size=2,
    )

    assert report["model"]["estimated_parameter_count"] == 632
    assert report["assumptions"]["kv_cache_context_length"] == 3
    assert report["assumptions"]["kv_prefill_temporary_context_length"] == 24

    # Two-byte KV elements: retained cache = 2 * 1 layer * 1 KV head *
    # 4 head-dim * 3 tokens * batch 2 * 2 bytes = 96 bytes.
    # Active-layer full-context prefill temporary = 768 bytes.
    # Workspace = max(2 * 24 * 12 * 4, 2 * 2 * 24**2 * 4) = 9,216 bytes.
    bf16 = next(item for item in report["candidates"] if item["format"] == "bf16")
    assert bf16["estimated_weight_gib"] == _gib(1264)
    assert bf16["estimated_kv_cache_gib"] == _gib(96)
    assert bf16["estimated_kv_prefill_temporary_gib"] == _gib(768)
    assert bf16["estimated_workspace_gib"] == _gib(9216)
    assert bf16["estimated_total_gib"] == _gib(13612.8)
    assert bf16["estimated_range_gib"] == [_gib(13612.8 * 0.85), _gib(13612.8 * 1.25)]


def test_memory_estimator_accounts_for_tied_embeddings_and_optional_projection_biases(tmp_path: Path, monkeypatch) -> None:
    model = _model(
        tmp_path, "biased-tied-llama",
        tie_word_embeddings=True, attention_bias=True, mlp_bias=True,
    )

    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))

    report = estimate_native_quantization(str(model), context_length=16, batch_size=1)

    # Tying removes the 64-value output head; attention bias adds 24 values
    # and MLP bias adds 32 values to the base 632-parameter estimate.
    assert report["model"]["estimated_parameter_count"] == 624



def test_memory_estimator_api_rejects_boolean_context_batch_and_budgets(tmp_path: Path, monkeypatch) -> None:
    model = _model(tmp_path)
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))
    base = {"model_path": str(model), "ram_budget_gib": 8}
    invalid_payloads = (
        {**base, "context_length": True},
        {**base, "batch_size": True},
        {**base, "ram_budget_gib": True},
        {"model_path": str(model), "vram_budget_gib": False},
    )
    for payload in invalid_payloads:
        response = _api_client().post("/api/engine/quantization/estimate", json=payload)
        assert response.status_code == 422, (payload, response.text)


def test_memory_estimator_reports_no_native_candidate_when_every_candidate_exceeds_budget(
    tmp_path: Path, monkeypatch,
) -> None:
    model = _model(
        tmp_path,
        "large-llama",
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32000,
        max_position_embeddings=8192,
    )
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))
    report = estimate_native_quantization(
        str(model),
        context_length=4096,
        batch_size=1,
        ram_budget_gib=0.1,
        vram_budget_gib=0.1,
    )
    assert report["recommendation"]["format"] is None
    assert report["recommendation"]["status"] == "no_native_candidate_fits"
    native = [item for item in report["candidates"] if item["native_checkpoint_supported"]]
    assert native
    assert all(item["fits_ram_budget"] is False for item in native)
    assert all(item["fits_vram_budget"] is False for item in native)



def test_memory_estimator_counts_qwen3_query_and_key_norm_parameters(tmp_path: Path, monkeypatch) -> None:
    model = _model(tmp_path, "tiny-qwen3", model_type="qwen3")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))

    report = estimate_native_quantization(
        str(model),
        context_length=16,
        batch_size=1,
        ram_budget_gib=8,
    )

    # Qwen3 adds q_norm and k_norm vectors of head_dim=8/2=4 per layer.
    # The tiny Llama-shaped fixture therefore gains 2 * 4 = 8 parameters.
    assert report["model"]["estimated_parameter_count"] == 640
    assert report["assumptions"]["qk_norm_parameters"] == 8
    bf16 = next(item for item in report["candidates"] if item["format"] == "bf16")
    assert bf16["estimated_weight_gib"] == _gib(1280)


