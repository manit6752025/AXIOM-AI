from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom.core import benchmark_profiles
from axiom.core.benchmark_profiles import (
    BenchmarkProfileError,
    BenchmarkProfileStore,
)


def _report(elapsed: float = 100.0) -> dict:
    metric_keys = (
        "total_wall_time_ms", "measured_wall_time_ms", "median_elapsed_ms",
        "p95_elapsed_ms", "median_prefill_ms", "median_decode_ms",
        "median_time_to_first_token_ms", "median_generation_tokens_per_second",
        "completion_tokens", "prompt_tokens", "aggregate_tokens_per_second",
    )
    metrics = {key: elapsed for key in metric_keys}
    metrics["median_elapsed_ms"] = elapsed
    metrics["p95_elapsed_ms"] = elapsed + 10
    metrics["median_generation_tokens_per_second"] = 20
    metrics["aggregate_tokens_per_second"] = 19
    return {
        "schema_version": 1,
        "engine": "axiom-native-engine",
        "model": "qwen-test",
        "architecture": "qwen3",
        "model_type": "qwen3",
        "device": "cpu",
        "precision": "fp32",
        "memory_mode": "full",
        "checkpoint_format": "safetensors",
        "warmup_runs_requested": 1,
        "warmup_runs_completed": 1,
        "iterations_requested": 2,
        "iterations_completed": 2,
        "max_new_tokens": 32,
        "sampling": {
            "temperature": 0.0,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "seed": 42,
        },
        "metrics": metrics,
        "runs": [
            {
                "iteration": index,
                "elapsed_ms": elapsed + index,
                "prefill_ms": 20.0,
                "decode_ms": 30.0,
                "time_to_first_token_ms": 20.0,
                "generation_tokens_per_second": 20.0,
                "prompt_tokens": 16,
                "completion_tokens": 32,
                "stop_reason": "max_new_tokens",
            }
            for index in (1, 2)
        ],
        "memory_stats": {"layer_weight_bytes_staged": 0, "disk_layer_weight_loads": 0},
        "generated_at": "2026-11-21T00:00:00+00:00",
    }


def _hardware(gpu_name="Test GPU"):
    return {
        "os_name": "Linux",
        "os_release": "6.12",
        "architecture": "x86_64",
        "cpu_cores": 8,
        "ram_bytes": 16 * 1024 ** 3,
        "gpu_name": gpu_name,
        "vram_bytes": 12 * 1024 ** 3 if gpu_name else None,
        "cuda_available": bool(gpu_name),
    }


def _patch_environment(monkeypatch: pytest.MonkeyPatch, model_root: Path, gpu_name="Test GPU"):
    from axiom.core import engine

    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(model_root))
    engine.refresh_local_engine_asset_index()
    monkeypatch.setattr(benchmark_profiles, "_hardware_record", lambda: _hardware(gpu_name))
    monkeypatch.setattr(benchmark_profiles, "_software_record", lambda: {
        "axiom_version": "0.3.0-test",
        "python_version": "3.11.9",
        "torch_version": "2.6.0",
    })


def test_profile_store_records_reproducible_provenance_without_prompt_or_model_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_environment(monkeypatch, tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    store = BenchmarkProfileStore(tmp_path / ".axiom")

    record = store.create(
        "Qwen CPU baseline",
        _report(),
        "A private representative prompt that must never be saved",
        model,
    )
    result = store.list()

    assert result["total"] == 1
    assert result["profiles"][0]["id"] == record["id"]
    assert record["workload"]["config"]["model"] == "qwen-test"
    assert record["workload"]["metrics"]["median_elapsed_ms"] == 100
    assert record["hardware"]["gpu_name"] == "Test GPU"
    assert record["software"]["torch_version"] == "2.6.0"
    assert record["model_config_sha256"] == benchmark_profiles._model_config_fingerprint(model)
    raw = (tmp_path / ".axiom" / "benchmark-profiles.json").read_text(encoding="utf-8")
    assert "A private representative prompt" not in raw
    assert str(model) not in raw
    assert record["prompt_sha256"]
    assert record["fingerprint"]
    assert result["privacy"] == {
        "prompt_content_included": False,
        "model_paths_included": False,
        "credentials_included": False,
    }
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / ".axiom" / "benchmark-profiles.json").stat().st_mode) == 0o600


def test_profiles_compare_metrics_only_when_hardware_software_model_and_workload_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_environment(monkeypatch, tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    from axiom.core import engine

    roots = engine._allowed_model_roots()
    matching_roots = [
        str(root) for root in roots if engine._path_is_within_root(model, root)
    ]
    try:
        resolved_model = engine.resolve_local_model_directory(str(model), "Base model")
    except ValueError as exc:
        raise AssertionError(
            f"Windows model resolution failed: model={str(model)!r}; "
            f"roots={tuple(map(str, roots))!r}; matching_roots={matching_roots!r}; "
            f"normalised_model={engine._normalise_lexical_model_path(model)!r}; "
            f"error={exc}"
        ) from exc
    assert resolved_model, (
        f"model resolver returned no directory: model={str(model)!r}; "
        f"roots={tuple(map(str, roots))!r}; matching_roots={matching_roots!r}"
    )
    store = BenchmarkProfileStore(tmp_path / ".axiom")
    first = store.create("Baseline A", _report(100), "same prompt", model)
    second = store.create("Baseline B", _report(90), "same prompt", model)

    same = store.compare(first["id"], second["id"])
    assert same["comparable"] is True
    assert all(same["match"].values())
    assert same["metrics"]["median_elapsed_ms"] == {"left": 100.0, "right": 90.0, "delta": -10.0}

    _patch_environment(monkeypatch, tmp_path, gpu_name="Different GPU")
    third = store.create("Different hardware", _report(80), "same prompt", model)
    different = store.compare(first["id"], third["id"])
    assert different["comparable"] is False
    assert different["match"]["hardware"] is False
    assert different["metrics"]["median_elapsed_ms"]["delta"] == -20.0


def test_profile_store_rejects_duplicate_names_and_invalid_profile_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_environment(monkeypatch, tmp_path)
    store = BenchmarkProfileStore(tmp_path / ".axiom")
    store.create("First", _report(), "prompt", tmp_path)
    with pytest.raises(BenchmarkProfileError, match="already exists"):
        store.create("first", _report(), "prompt", tmp_path)
    with pytest.raises(BenchmarkProfileError, match="safe ASCII"):
        store.create("../escape", _report(), "prompt", tmp_path)
    with pytest.raises(BenchmarkProfileError, match="prompts"):
        store.create("Second", _report(), "   ", tmp_path)
    invalid_engine_report = _report()
    invalid_engine_report["engine"] = "other"
    with pytest.raises(BenchmarkProfileError, match="Only AXIOM Native Engine"):
        store.create("Third", invalid_engine_report, "prompt", tmp_path)


def test_profile_store_rejects_tampering_unknown_fields_and_future_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_environment(monkeypatch, tmp_path)
    store = BenchmarkProfileStore(tmp_path / ".axiom")
    record = store.create("Trusted", _report(), "prompt", tmp_path)
    path = tmp_path / ".axiom" / "benchmark-profiles.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["profiles"][0]["name"] = "tampered"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(BenchmarkProfileError, match="fingerprint"):
        store.list()

    path.write_text(json.dumps({"schema_version": 99, "profiles": []}), encoding="utf-8")
    with pytest.raises(BenchmarkProfileError, match="unsupported schema"):
        store.list()


def test_profile_store_rejects_symlinked_store_paths(tmp_path: Path):
    root = tmp_path / ".axiom"
    root.mkdir()
    real = tmp_path / "real.json"
    real.write_text('{"schema_version":1,"profiles":[]}', encoding="utf-8")
    (root / "benchmark-profiles.json").symlink_to(real)
    with pytest.raises(BenchmarkProfileError, match="symbolic links"):
        BenchmarkProfileStore(root).list()


def test_profile_comparison_requires_two_distinct_existing_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_environment(monkeypatch, tmp_path)
    store = BenchmarkProfileStore(tmp_path / ".axiom")
    first = store.create("One", _report(), "prompt", tmp_path)
    with pytest.raises(BenchmarkProfileError, match="Two distinct"):
        store.compare(first["id"], first["id"])
    with pytest.raises(BenchmarkProfileError, match="not found"):
        store.compare(first["id"], "f" * 32)


def test_profile_store_rejects_invalid_sampling_and_untrusted_model_directory_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_environment(monkeypatch, tmp_path)
    store = BenchmarkProfileStore(tmp_path / ".axiom")
    bad = _report()
    bad["sampling"]["top_p"] = 1.5
    with pytest.raises(BenchmarkProfileError, match="sampling"):
        store.create("Invalid sampling", bad, "prompt", tmp_path)

    model = tmp_path / "model"
    model.mkdir()
    link = tmp_path / "model-link"
    try:
        link.symlink_to(model, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not permitted in this test environment.")
    from axiom.core import engine

    with os.scandir(tmp_path) as entries:
        link_entry = next(entry for entry in entries if entry.name == link.name)
    metadata = link_entry.stat(follow_symlinks=False)
    diagnostic = (
        f"mode={oct(metadata.st_mode)} "
        f"attributes={getattr(metadata, 'st_file_attributes', None)!r} "
        f"reparse_tag={getattr(metadata, 'st_reparse_tag', None)!r} "
        f"islink={link_entry.is_symlink()!r}"
    )
    assert engine._entry_is_link_or_reparse_point(link_entry), diagnostic
    with pytest.raises(BenchmarkProfileError, match="symbolic link"):
        store.create("Symlinked model", _report(), "prompt", link)


def test_missing_model_configuration_fingerprint_prevents_controlled_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_environment(monkeypatch, tmp_path)
    store = BenchmarkProfileStore(tmp_path / ".axiom")
    model = tmp_path / "model"
    model.mkdir()
    first = store.create("No metadata A", _report(100), "same prompt", model)
    second = store.create("No metadata B", _report(90), "same prompt", model)
    comparison = store.compare(first["id"], second["id"])
    assert comparison["match"]["model_metadata"] is False
    assert comparison["comparable"] is False
    assert comparison["metrics"]["median_elapsed_ms"]["delta"] == -10.0



def test_model_config_fingerprint_refuses_untrusted_model_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = tmp_path / "outside-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    monkeypatch.setattr(benchmark_profiles, "_trusted_model_directory", lambda _value: None)
    assert benchmark_profiles._model_config_fingerprint(model) is None


def test_model_config_fingerprint_opens_fixed_metadata_without_following_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "model"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"model_type":"qwen3"}', encoding="utf-8")
    try:
        (root / "config.json").symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks unavailable: {exc}")
    monkeypatch.setattr(benchmark_profiles, "_trusted_model_directory", lambda _value: root)
    with pytest.raises(BenchmarkProfileError, match="symbolic links or junctions"):
        benchmark_profiles._model_config_fingerprint(root)


def test_profile_store_rejects_model_directory_with_symlinked_ancestor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_environment(monkeypatch, tmp_path)
    store = BenchmarkProfileStore(tmp_path / ".axiom")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    model = real_parent / "model"
    model.mkdir()
    link_parent = tmp_path / "linked-parent"
    try:
        link_parent.symlink_to(real_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not permitted in this test environment.")
    from axiom.core import engine

    linked_model_path = link_parent / "model"
    with os.scandir(tmp_path) as entries:
        link_entry = next(entry for entry in entries if entry.name == link_parent.name)
    metadata = link_entry.stat(follow_symlinks=False)
    diagnostic = (
        f"mode={oct(metadata.st_mode)} "
        f"attributes={getattr(metadata, 'st_file_attributes', None)!r} "
        f"reparse_tag={getattr(metadata, 'st_reparse_tag', None)!r} "
        f"islink={link_entry.is_symlink()!r}"
    )
    assert engine._entry_is_link_or_reparse_point(link_entry), diagnostic
    with pytest.raises(BenchmarkProfileError, match="symbolic links or junctions"):
        store.create("Linked ancestor", _report(), "prompt", linked_model_path)
