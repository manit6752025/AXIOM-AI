from __future__ import annotations

import json

import pytest

from axiom.core import engine
from axiom.core.engine import (
    _inspect_model_config,
    _resolve_model_directory,
    validate_generation,
)


def test_validate_generation_accepts_bounded_local_prompt() -> None:
    assert validate_generation("hello", 32, 0.7) == ("hello", 32, 0.7)
    assert validate_generation("hello", 1, 0) == ("hello", 1, 0.0)


@pytest.mark.parametrize(
    ("prompt", "tokens", "temperature"),
    [
        (" ", 1, 0.7),
        ("x" * 12_001, 1, 0.7),
        ("hello", 0, 0.7),
        ("hello", 2_049, 0.7),
        ("hello", 32, -0.1),
        ("hello", 32, 2.1),
        ("hello", True, 0.7),
    ],
)
def test_validate_generation_rejects_invalid_bounds(prompt, tokens, temperature) -> None:
    with pytest.raises(ValueError):
        validate_generation(prompt, tokens, temperature)


def test_model_config_is_read_only_and_reports_architecture(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    inspected = _inspect_model_config(tmp_path)
    assert inspected["model_type"] == "llama"
    assert inspected["architecture"] == "LlamaForCausalLM"


def test_model_config_rejects_missing_model_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 8}), encoding="utf-8")
    with pytest.raises(ValueError, match="model_type"):
        _inspect_model_config(tmp_path)


def test_model_directory_must_exist_and_be_a_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(tmp_path))
    file_path = tmp_path / "model.txt"
    file_path.write_text("not a model", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint directory|trusted model root"):
        _resolve_model_directory(str(file_path), "Base model")


def test_model_directory_accepts_only_configured_roots(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "model-a"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    assert _resolve_model_directory(str(model), "Base model") == model.resolve()


def test_model_directory_accepts_relative_paths_under_trusted_root(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    model = root / "model-a"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    assert _resolve_model_directory("models/model-a", "Base model") == model.resolve()


def test_model_directory_rejects_parent_traversal_components(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    with pytest.raises(ValueError, match="parent-directory traversal"):
        _resolve_model_directory("models/../private", "Base model")


def test_model_directory_rejects_existing_path_outside_configured_roots(tmp_path, monkeypatch) -> None:
    root = tmp_path / "trusted-models"
    root.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    with pytest.raises(ValueError, match="allowed model directory|trusted model root|checkpoint directory"):
        _resolve_model_directory(str(outside), "Base model")


def test_model_directory_rejects_symlink_escape_from_configured_root(tmp_path, monkeypatch) -> None:
    root = tmp_path / "trusted-models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = root / "linked-model"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable on this platform: {exc}")
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(root))

    with pytest.raises(ValueError, match="symlink|trusted model root|checkpoint directory"):
        _resolve_model_directory(str(escape), "Base model")



def test_unload_releases_cached_model_state_and_reports_the_change(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_LOADED_RUNTIME", {
        "key": ("/allowed/models/model-a", None),
        "model": object(),
        "tokenizer": object(),
        "torch": None,
        "device": "cpu",
    })
    monkeypatch.setattr(engine, "_LOADED_ARCHITECTURE", "LlamaForCausalLM")

    result = engine.unload_local_model()

    assert result["unloaded"] is True
    assert result["loaded"] is False
    assert engine._LOADED_RUNTIME is None
    assert engine._LOADED_ARCHITECTURE is None
    assert "may retain" in result["scope_note"]


def test_unload_is_safe_when_no_model_is_loaded(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_LOADED_RUNTIME", None)
    monkeypatch.setattr(engine, "_LOADED_ARCHITECTURE", None)

    result = engine.unload_local_model()

    assert result["unloaded"] is False
    assert result["loaded"] is False
