from __future__ import annotations

import json

import pytest

import axiom.core.fine_tuning as fine_tuning


def test_training_version_comparison_normalizes_short_versions() -> None:
    assert fine_tuning._version_key("5.2") == (5, 2, 0)
    assert fine_tuning._version_key("5.20.1+cpu") == (5, 20, 1)
    assert fine_tuning._version_key("0.20.0") < fine_tuning._version_key("0.21.0")


def test_require_runtime_rejects_incompatible_packages(monkeypatch) -> None:
    monkeypatch.setattr(
        fine_tuning,
        "runtime_status",
        lambda: {"can_start": False, "setup_note": "PEFT is too old."},
    )
    with pytest.raises(fine_tuning.FineTuningError, match="PEFT is too old"):
        fine_tuning.require_runtime()


def test_load_texts_accepts_text_prompt_completion_and_messages(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"text": " plain text "}),
                json.dumps({"prompt": "question", "completion": "answer"}),
                json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    assert fine_tuning._load_texts(dataset, "text") == [
        "plain text",
        "question\nanswer",
        "user: hi\nassistant: hello",
    ]


def test_load_texts_rejects_invalid_jsonl(tmp_path) -> None:
    dataset = tmp_path / "broken.jsonl"
    dataset.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        fine_tuning._load_texts(dataset, "text")
