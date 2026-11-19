from __future__ import annotations

import json

import pytest

from axiom.core import inference


class FakeResponse:
    def __init__(self, payload):
        self.raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit=-1):
        return self.raw


def test_runtime_status_checks_only_local_tags_endpoint(monkeypatch):
    observed = []

    def fake_open(request, timeout):
        observed.append((request.full_url, request.method, timeout))
        return FakeResponse({"models": [{"name": "qwen2.5:0.5b", "size": 100}]})

    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    monkeypatch.setattr(inference.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)
    report = inference.runtime_status()
    assert report["status"] == "online"
    assert report["models"] == ["qwen2.5:0.5b"]
    assert report["runtime_installed"] is True
    assert observed[0][0] == "http://127.0.0.1:11434/api/tags"
    assert observed[0][1] == "GET"


def test_runtime_status_reports_offline_without_exposing_local_paths(monkeypatch):
    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("/private/path")))
    monkeypatch.setattr(inference.shutil, "which", lambda _name: None)
    report = inference.runtime_status()
    assert report["status"] == "offline"
    assert report["models"] == []
    assert "/private/path" not in str(report)
    assert "127.0.0.1:11434" in report["error"]


def test_generation_uses_loopback_and_bounds_options(monkeypatch):
    observed = {}

    def fake_open(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": "Hello from a local model.", "done": True, "eval_count": 7, "total_duration": 12000000})

    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    result = inference.run_local_inference("qwen2.5:0.5b", "Say hello", num_predict=32, temperature=0.2)
    assert observed["url"] == "http://127.0.0.1:11434/api/generate"
    assert observed["payload"]["model"] == "qwen2.5:0.5b"
    assert observed["payload"]["prompt"] == "Say hello"
    assert observed["payload"]["stream"] is False
    assert observed["payload"]["options"] == {"num_predict": 32, "temperature": 0.2}
    assert observed["timeout"] <= 300
    assert result["response"] == "Hello from a local model."
    assert result["eval_count"] == 7
    assert result["total_duration_ms"] == 12


@pytest.mark.parametrize("model", ["", "http://example.com", "../model", "bad model", "x" * 121])
def test_invalid_model_names_are_rejected_before_any_request(monkeypatch, model):
    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: pytest.fail("network request must not run"))
    with pytest.raises(ValueError):
        inference.run_local_inference(model, "Hello")


def test_prompt_and_generation_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: pytest.fail("network request must not run"))
    with pytest.raises(ValueError, match="12,000-character"):
        inference.run_local_inference("qwen2.5:0.5b", "x" * 12_001)
    with pytest.raises(ValueError, match="between 1 and 2048"):
        inference.run_local_inference("qwen2.5:0.5b", "Hello", num_predict=2049)
    with pytest.raises(ValueError, match="Temperature"):
        inference.run_local_inference("qwen2.5:0.5b", "Hello", temperature=2.1)


def test_missing_model_error_is_actionable(monkeypatch):
    from urllib.error import HTTPError
    from io import BytesIO

    def fake_open(_request, _timeout):
        raise HTTPError("http://127.0.0.1:11434/api/generate", 404, "Not Found", {}, BytesIO(b'{"error":"model not found; pull it first"}'))

    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    with pytest.raises(inference.InferenceRuntimeError, match="model not found"):
        inference.run_local_inference("qwen2.5:0.5b", "Hello")


def test_chat_uses_loopback_ollama_api_with_bounded_conversation(monkeypatch):
    observed = {}

    def fake_open(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({
            "message": {"role": "assistant", "content": "Ollama chat answer"},
            "done": True,
            "prompt_eval_count": 8,
            "eval_count": 5,
            "total_duration": 2_500_000,
            "done_reason": "stop",
        })

    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Follow up"},
    ]
    result = inference.run_local_chat("qwen2.5:0.5b", messages, num_predict=24, temperature=0.1)

    assert observed["url"] == "http://127.0.0.1:11434/api/chat"
    assert observed["payload"]["stream"] is False
    assert observed["payload"]["model"] == "qwen2.5:0.5b"
    assert observed["payload"]["messages"] == messages
    assert observed["payload"]["options"] == {"num_predict": 24, "temperature": 0.1, "top_p": 0.9, "repeat_penalty": 1.0}
    assert observed["timeout"] == inference._GENERATION_TIMEOUT_SECONDS
    assert result["response"] == "Ollama chat answer"
    assert result["prompt_eval_count"] == 8
    assert result["eval_count"] == 5
    assert result["done_reason"] == "stop"
    assert result["execution"] == "local"


def test_chat_rejects_invalid_messages_before_loopback(monkeypatch):
    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: pytest.fail("network request must not run"))
    with pytest.raises(ValueError, match="At least one user message"):
        inference.run_local_chat("qwen2.5:0.5b", [{"role": "assistant", "content": "answer"}])
    with pytest.raises(ValueError, match="Combined chat content"):
        inference.run_local_chat("qwen2.5:0.5b", [
            {"role": "user", "content": "x" * 12_000},
            {"role": "assistant", "content": "y" * 12_000},
            {"role": "user", "content": "z"},
        ])


def test_advanced_sampling_options_are_forwarded_as_bounded_ollama_options(monkeypatch):
    observed = {}

    def fake_open(request, timeout):
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": "Configured", "done": True, "eval_count": 2, "total_duration": 1_000_000})

    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    inference.run_local_inference(
        "qwen2.5:0.5b", "Say hello",
        top_p=0.82, top_k=40, min_p=0.15, repetition_penalty=1.15,
        seed=17, stop_sequences=["END", "STOP", "END"], num_ctx=8192,
    )

    assert observed["payload"]["options"] == {
        "num_predict": 256,
        "temperature": 0.7,
        "top_p": 0.82,
        "top_k": 40,
        "min_p": 0.15,
        "repeat_penalty": 1.15,
        "seed": 17,
        "stop": ["END", "STOP"],
        "num_ctx": 8192,
    }


@pytest.mark.parametrize("options", [
    {"top_p": 0}, {"top_p": 1.1}, {"top_k": -1}, {"top_k": 100001},
    {"min_p": -0.01}, {"min_p": 1.01}, {"min_p": float("nan")}, {"min_p": True},
    {"repetition_penalty": 0.09}, {"repetition_penalty": 5.1},
    {"seed": -1}, {"seed": True}, {"num_ctx": 511}, {"num_ctx": 131073},
    {"stop_sequences": [""]}, {"stop_sequences": ["x" * 257]},
    {"stop_sequences": ["X"] * 17},
])
def test_invalid_sampling_options_are_rejected_before_any_request(monkeypatch, options):
    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: pytest.fail("invalid sampling options must not reach Ollama"))
    with pytest.raises(ValueError):
        inference.run_local_inference("qwen2.5:0.5b", "Hello", **options)
