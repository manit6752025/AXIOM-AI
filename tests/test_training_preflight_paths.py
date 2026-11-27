from __future__ import annotations

from pathlib import Path

import pytest

from axiom.core.training_preflight import resolve_workspace_input


def test_training_preflight_resolves_existing_files_and_directories_from_workspace_entries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    model = workspace / "models" / "tiny"
    data = workspace / "data"
    model.mkdir(parents=True)
    data.mkdir()
    dataset = data / "train.jsonl"
    dataset.write_text('{"text":"hello"}', encoding="utf-8")

    assert resolve_workspace_input("models/tiny", workspace, "Model path") == model
    assert resolve_workspace_input("data/train.jsonl", workspace, "Dataset path") == dataset


@pytest.mark.parametrize("raw", ["../outside", "models/../outside", "/etc/passwd", "models\\\\tiny", "models//tiny"])
def test_training_preflight_rejects_unsafe_workspace_path_components(tmp_path: Path, raw: str) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "models").mkdir(parents=True)
    with pytest.raises(ValueError, match="workspace-relative|safe workspace-relative"):
        resolve_workspace_input(raw, workspace, "Model path")


def test_training_preflight_rejects_symlinked_final_entry_and_ancestor(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    models.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    try:
        (models / "linked-model").symlink_to(external, target_is_directory=True)
        (workspace / "linked-models").symlink_to(models, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlink creation is unavailable in this environment.")

    with pytest.raises(ValueError, match="symbolic links or junctions"):
        resolve_workspace_input("models/linked-model", workspace, "Model path")
    with pytest.raises(ValueError, match="symbolic links or junctions"):
        resolve_workspace_input("linked-models", workspace, "Model path")


def test_training_preflight_rejects_symlinked_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "workspace-alias"
    try:
        alias.symlink_to(workspace, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlink creation is unavailable in this environment.")

    with pytest.raises(ValueError, match="workspace root changed"):
        resolve_workspace_input("models/tiny", alias, "Model path")
