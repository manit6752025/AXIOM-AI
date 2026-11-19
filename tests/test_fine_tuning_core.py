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


def test_job_list_limit_is_bounded():
    with pytest.raises(ValueError, match="between 1 and 100"):
        fine_tuning.list_jobs(101)
