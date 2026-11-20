from __future__ import annotations

import json

import pytest

from axiom.core import fine_tuning


def make_inputs(tmp_path, dataset_text=None):
    model = tmp_path / "base-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(dataset_text if dataset_text is not None else '{"text":"hello local model"}\n', encoding="utf-8")
    return {"model_path": str(model), "dataset_path": str(dataset), "output_dir": str(tmp_path / "adapter"), "text_field": "text", "max_steps": 3, "sequence_length": 64, "learning_rate": 0.0001, "lora_rank": 4}


def test_validate_config_accepts_local_model_and_jsonl(tmp_path):
    inputs = make_inputs(tmp_path)
    result = fine_tuning.validate_config(inputs)
    assert result["model_path"] == str((tmp_path / "base-model").resolve())
    assert result["max_steps"] == 3
    assert result["lora_rank"] == 4
    assert result["output_dir"].endswith("/adapter")


def test_dataset_reader_supports_text_prompt_completion_and_messages(tmp_path):
    path = tmp_path / "multi.jsonl"
    path.write_text('\n'.join([
        json.dumps({"text": "plain"}),
        json.dumps({"prompt": "question", "completion": "answer"}),
        json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}),
    ]) + '\n', encoding="utf-8")
    texts = fine_tuning._load_texts(path, "text")
    assert texts == ["plain", "question\nanswer", "user: hi\nassistant: hello"]


@pytest.mark.parametrize(("key", "value", "message"), [
    ("max_steps", 1001, "Maximum training steps"),
    ("sequence_length", 16, "Sequence length"),
    ("lora_rank", 65, "LoRA rank"),
    ("learning_rate", 0.5, "Learning rate"),
    ("text_field", "bad field", "Text field"),
])
def test_training_bounds_are_validated_before_launch(tmp_path, key, value, message):
    config = make_inputs(tmp_path)
    config[key] = value
    with pytest.raises(ValueError, match=message):
        fine_tuning.validate_config(config)


def test_remote_model_ids_and_missing_model_directories_are_not_accepted(tmp_path):
    config = make_inputs(tmp_path)
    config["model_path"] = "org/example-model"
    with pytest.raises(ValueError, match="existing local path"):
        fine_tuning.validate_config(config)


def test_dataset_invalid_json_and_non_objects_are_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        fine_tuning._load_texts(path, "text")
    path.write_text('[1,2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        fine_tuning._load_texts(path, "text")


def test_runtime_status_only_reports_optional_package_presence(monkeypatch):
    monkeypatch.setattr(fine_tuning.importlib.util, "find_spec", lambda name: object() if name in {"torch", "peft"} else None)
    result = fine_tuning.runtime_status()
    assert result["packages"] == {"torch": True, "transformers": False, "peft": True, "accelerate": False}
    assert result["can_start"] is False
    assert result["model_loading"] == "local_files_only"




def test_cancel_job_is_idempotent_and_does_not_interrupt_finalization(tmp_path, monkeypatch):
    import os

    jobs = tmp_path / "jobs"
    jobs.mkdir(mode=0o700)
    monkeypatch.setattr(fine_tuning, "jobs_directory", lambda: jobs)
    job_id = "c" * 32
    runner_pid = os.getpid()
    runner_identity = fine_tuning._process_start_identity(runner_pid)
    running_job = {
        "schema_version": 1,
        "id": job_id,
        "runner_pid": runner_pid,
        "runner_identity": runner_identity,
        "status": "running",
        "cancel_requested": False,
        "logs": ["Training started."],
    }
    job_path = jobs / f"{job_id}.json"
    job_path.write_text(json.dumps(running_job), encoding="utf-8")
    job_path.chmod(0o600)

    first = fine_tuning.cancel_finetuning_job(job_id)
    assert first is not None
    assert first["status"] == "cancelling"
    assert first["cancel_requested"] is True
    assert first["cancellation"]["requested"] is True
    assert first["cancellation"]["requestable"] is False

    second = fine_tuning.cancel_finetuning_job(job_id)
    assert second is not None
    assert second["status"] == "cancelling"
    assert second["logs"] == first["logs"]

    finalizing_id = "d" * 32
    finalizing_job = {
        **running_job,
        "id": finalizing_id,
        "status": "finalizing",
        "cancel_requested": False,
    }
    finalizing_path = jobs / f"{finalizing_id}.json"
    finalizing_path.write_text(json.dumps(finalizing_job), encoding="utf-8")
    finalizing_path.chmod(0o600)
    finalizing = fine_tuning.cancel_finetuning_job(finalizing_id)
    assert finalizing is not None
    assert finalizing["status"] == "finalizing"
    assert finalizing["cancel_requested"] is False
    assert finalizing["cancellation"]["requestable"] is False
    assert finalizing["cancellation"]["unavailable_reason"] == "finalization_started"


def test_job_list_limit_is_bounded():
    with pytest.raises(ValueError, match="between 1 and 100"):
        fine_tuning.list_jobs(101)

@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("queued", "running"),
        ("queued", "cancelling"),
        ("queued", "failed"),
        ("running", "cancelling"),
        ("running", "finalizing"),
        ("running", "failed"),
        ("cancelling", "cancelled"),
        ("cancelling", "failed"),
        ("finalizing", "completed"),
        ("finalizing", "failed"),
        ("completed", "completed"),
        ("cancelled", "cancelled"),
        ("failed", "failed"),
    ],
)
def test_job_lifecycle_accepts_only_documented_forward_transitions(current, target):
    fine_tuning._validate_job_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("queued", "completed"),
        ("running", "completed"),
        ("cancelling", "running"),
        ("cancelling", "finalizing"),
        ("finalizing", "cancelling"),
        ("completed", "running"),
        ("cancelled", "completed"),
        ("failed", "cancelling"),
        ("unknown", "running"),
    ],
)
def test_job_lifecycle_rejects_invalid_or_terminal_transitions(current, target):
    with pytest.raises(fine_tuning.FineTuningError, match="Invalid fine-tuning job transition"):
        fine_tuning._validate_job_transition(current, target)


@pytest.mark.parametrize(
    ("status", "requested", "requestable", "reason"),
    [
        ("queued", False, True, None),
        ("running", False, True, None),
        ("cancelling", True, False, "already_requested"),
        ("finalizing", False, False, "finalization_started"),
        ("completed", False, False, "job_terminal"),
        ("cancelled", True, False, "job_terminal"),
        ("failed", False, False, "job_terminal"),
    ],
)
def test_public_job_reports_explicit_cancellation_contract(status, requested, requestable, reason):
    public = fine_tuning._public_training_job({
        "id": "a" * 32,
        "status": status,
        "cancel_requested": requested,
        "config": {"model_path": "/local/model"},
    })
    assert public["cancellation"] == {
        "supported": True,
        "requested": requested,
        "requestable": requestable,
        "mode": "cooperative",
        "boundary": "before worker start or between training steps; adapter finalization is not interruptible",
        "unavailable_reason": reason,
    }
