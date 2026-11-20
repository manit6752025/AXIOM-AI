from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom.core import fine_tuning_checkpoints as checkpoints


class FakeModel:
    def save_pretrained(self, destination: str, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        folder = Path(destination)
        (folder / "adapter_config.json").write_text('{"r":4}', encoding="utf-8")
        (folder / "adapter_model.safetensors").write_bytes(b"adapter-weights")


class FakeTorch:
    @staticmethod
    def save(payload, destination: str) -> None:
        Path(destination).write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def save(output: Path, step: int):
    return checkpoints.save_checkpoint(
        output,
        step,
        model=FakeModel(),
        torch_module=FakeTorch(),
        optimizer_payload={"optimizer_state_dict": {}, "order": [0], "position": 1},
        training_metadata={"dataset_sha256": "a" * 64, "last_loss": 0.25},
    )


def test_checkpoint_is_published_atomically_and_verified(tmp_path):
    output = tmp_path / "adapter"
    output.mkdir()

    summary = save(output, 2)

    assert summary == {
        "enabled": True,
        "latest_step": 2,
        "retained": 1,
        "last_saved_at": summary["last_saved_at"],
        "resume_available": True,
    }
    checkpoint, state = checkpoints.latest_checkpoint(output)
    assert checkpoint.name == "checkpoint-000002"
    assert state["completed_steps"] == 2
    assert state["training"]["last_loss"] == 0.25


def test_checkpoint_manifest_detects_corrupted_adapter_weights(tmp_path):
    output = tmp_path / "adapter"
    output.mkdir()
    save(output, 1)
    checkpoint, _ = checkpoints.latest_checkpoint(output)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="integrity check failed"):
        checkpoints.latest_checkpoint(output)


def test_checkpoint_rotation_retains_only_three_complete_checkpoints(tmp_path):
    output = tmp_path / "adapter"
    output.mkdir()
    for step in range(1, 5):
        summary = save(output, step)

    checkpoint_root = output / checkpoints.CHECKPOINT_DIRECTORY
    assert [path.name for path in sorted(checkpoint_root.iterdir()) if path.is_dir()] == [
        "checkpoint-000002",
        "checkpoint-000003",
        "checkpoint-000004",
    ]
    assert summary["retained"] == 3
    assert checkpoints.latest_checkpoint(output)[1]["completed_steps"] == 4


def test_interrupted_temporary_checkpoint_is_never_selected_for_resume(tmp_path):
    output = tmp_path / "adapter"
    output.mkdir()
    save(output, 3)
    (output / checkpoints.CHECKPOINT_DIRECTORY / ".checkpoint-000004-abandoned.tmp").mkdir()

    selected, state = checkpoints.latest_checkpoint(output)
    assert selected.name == "checkpoint-000003"
    assert state["completed_steps"] == 3


def test_unknown_checkpoint_files_fail_closed(tmp_path):
    output = tmp_path / "adapter"
    output.mkdir()
    save(output, 1)
    (output / checkpoints.CHECKPOINT_DIRECTORY / "notes.txt").write_text("unknown", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown entry"):
        checkpoints.latest_checkpoint(output)
