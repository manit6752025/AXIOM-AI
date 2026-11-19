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




class FakeStreamResponse:
    def __init__(self, records: list[dict]):
        self.lines = [json.dumps(record).encode("utf-8") + b"\n" for record in records]
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True
        return False

    def readline(self, limit=-1):
        if not self.lines:
            return b""
        line = self.lines.pop(0)
        return line if limit < 0 else line[:limit]

    def close(self):
        self.closed = True


def test_model_inspection_returns_metadata_without_model_templates(monkeypatch):
    replies = iter([
        FakeResponse({
            "details": {
                "format": "gguf", "family": "qwen2", "families": ["qwen2"],
                "parameter_size": "0.5B", "quantization_level": "Q4_K_M",
            },
            "model_info": {"qwen2.context_length": 32768, "general.parameter_count": 494032768},
            "capabilities": ["completion", "tools"],
            "template": "SECRET TEMPLATE CONTENT",
            "license": "SECRET LICENSE CONTENT",
            "system": "SECRET SYSTEM CONTENT",
        }),
        FakeResponse({"models": [{"name": "qwen2.5:0.5b", "size": 401000000, "modified_at": "2026-10-24T00:00:00Z"}]}),
    ])
    observed = []

    def fake_open(request, timeout):
        observed.append((request.full_url, request.method, json.loads(request.data.decode("utf-8")) if request.data else None))
        return next(replies)

    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    report = inference.inspect_local_model("qwen2.5:0.5b")

    assert report["installed"] is True
    assert report["format"] == "gguf"
    assert report["family"] == "qwen2"
    assert report["parameter_size"] == "0.5B"
    assert report["quantization_level"] == "Q4_K_M"
    assert report["context_length"] == 32768
    assert report["parameter_count"] == 494032768
    assert report["size_bytes"] == 401000000
    assert report["capabilities"] == ["completion", "tools"]
    assert report["template_configured"] and report["license_present"] and report["system_prompt_configured"]
    assert "SECRET TEMPLATE CONTENT" not in str(report)
    assert "SECRET LICENSE CONTENT" not in str(report)
    assert "SECRET SYSTEM CONTENT" not in str(report)
    assert observed[0][0] == "http://127.0.0.1:11434/api/show"
    assert observed[0][2] == {"model": "qwen2.5:0.5b", "verbose": False}
    assert observed[1][0] == "http://127.0.0.1:11434/api/tags"


def test_unload_releases_memory_without_deleting_downloaded_model(monkeypatch):
    observed = {}

    def fake_open(request, timeout):
        observed["url"] = request.full_url
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"model": "qwen2.5:0.5b", "done": True, "response": ""})

    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    result = inference.unload_local_model("qwen2.5:0.5b")

    assert observed["url"] == "http://127.0.0.1:11434/api/generate"
    assert observed["payload"] == {"model": "qwen2.5:0.5b", "prompt": "", "stream": False, "keep_alive": 0}
    assert result["unloaded"] is True
    assert result["model_files_deleted"] is False


@pytest.mark.parametrize("model", ["", "http://example.com/model", "../model"])
def test_model_operations_reject_nonlocal_model_references(monkeypatch, model):
    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: pytest.fail("invalid references must not reach Ollama"))
    with pytest.raises(ValueError):
        inference.inspect_local_model(model)
    with pytest.raises(ValueError):
        inference.unload_local_model(model)


def test_running_model_inventory_reports_loaded_memory_without_raw_content(monkeypatch):
    observed = {}
    payload = {
        "models": [{
            "name": "qwen2.5:0.5b", "size": 800000000, "size_vram": 300000000,
            "expires_at": "2026-10-24T10:00:00Z", "digest": "abcdef1234567890",
            "details": {"format": "gguf", "family": "qwen2", "parameter_size": "0.5B", "quantization_level": "Q4_K_M"},
            "template": "PRIVATE TEMPLATE MUST NOT BE RETURNED",
        }],
    }
    def fake_open(request, timeout):
        observed["url"] = request.full_url
        return FakeResponse(payload)
    monkeypatch.setattr(inference, "_open_loopback", fake_open)

    report = inference.list_running_local_models()

    assert observed["url"] == "http://127.0.0.1:11434/api/ps"
    assert report["count"] == 1
    model = report["models"][0]
    assert model["name"] == "qwen2.5:0.5b"
    assert model["size_bytes"] == 800000000
    assert model["size_vram_bytes"] == 300000000
    assert model["size_ram_bytes"] == 500000000
    assert model["memory_placement"] == "mixed"
    assert model["quantization_level"] == "Q4_K_M"
    assert "PRIVATE TEMPLATE MUST NOT BE RETURNED" not in str(report)


def test_running_model_inventory_rejects_invalid_runtime_shape(monkeypatch):
    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: FakeResponse({"models": {}}))
    with pytest.raises(inference.InferenceRuntimeError, match="invalid loaded-model inventory"):
        inference.list_running_local_models()


def test_inspect_missing_model_returns_actionable_local_pull_error(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    def fake_open(request, _timeout):
        raise HTTPError(request.full_url, 404, "Not Found", {}, BytesIO(b'{"error":"model not found"}'))
    monkeypatch.setattr(inference, "_open_loopback", fake_open)

    with pytest.raises(inference.InferenceRuntimeError, match="model not found"):
        inference.inspect_local_model("qwen2.5:0.5b")


def test_stream_local_inference_yields_token_events_and_bounded_completion(monkeypatch):
    response = FakeStreamResponse([
        {"response": "Hello ", "done": False},
        {"response": "local", "done": False},
        {
            "response": "", "done": True, "done_reason": "stop",
            "prompt_eval_count": 4, "eval_count": 2, "total_duration": 5_000_000,
        },
    ])
    observed = {}

    def fake_open(request, timeout):
        observed["url"] = request.full_url
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        observed["timeout"] = timeout
        return response

    registered = []
    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    events = list(inference.stream_local_inference(
        "qwen2.5:0.5b", "Say hello", num_predict=32, temperature=0.2,
        on_response=registered.append,
    ))

    assert observed["url"] == "http://127.0.0.1:11434/api/generate"
    assert observed["payload"]["stream"] is True
    assert observed["payload"]["options"] == {"num_predict": 32, "temperature": 0.2}
    assert observed["timeout"] == inference._GENERATION_TIMEOUT_SECONDS
    assert [event for event in events if event["type"] == "token"] == [
        {"type": "token", "token": "Hello "},
        {"type": "token", "token": "local"},
    ]
    completed = events[-1]
    assert completed["type"] == "complete"
    assert completed["result"]["response"] == "Hello local"
    assert completed["result"]["eval_count"] == 2
    assert completed["result"]["prompt_eval_count"] == 4
    assert completed["result"]["done_reason"] == "stop"
    assert registered == [response]
    assert response.closed is True


def test_stream_local_inference_rejects_malformed_events(monkeypatch):
    class MalformedResponse(FakeStreamResponse):
        def __init__(self):
            self.lines = [b"not-json\n"]
            self.closed = False

    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: MalformedResponse())
    with pytest.raises(inference.InferenceRuntimeError, match="invalid streaming event"):
        list(inference.stream_local_inference("qwen2.5:0.5b", "Hello"))


def test_stream_local_inference_stops_when_cancel_is_already_requested(monkeypatch):
    from threading import Event

    event = Event()
    event.set()
    monkeypatch.setattr(inference, "_open_loopback", lambda *_args, **_kwargs: pytest.fail("cancelled stream must not open Ollama"))
    assert list(inference.stream_local_inference("qwen2.5:0.5b", "Hello", cancel_event=event)) == []


def test_remove_local_model_uses_argument_list_and_verifies_result(monkeypatch):
    import subprocess

    reports = iter([
        {"status": "online", "models": ["qwen2.5:0.5b"]},
        {"status": "online", "models": []},
    ])
    observed = {}

    class Completed:
        returncode = 0
        stdout = "success"
        stderr = ""

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(inference, "runtime_status", lambda: next(reports))
    monkeypatch.setattr(inference.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = inference.remove_local_model("qwen2.5:0.5b")

    assert observed["args"] == ["/usr/bin/ollama", "rm", "qwen2.5:0.5b"]
    assert observed["kwargs"]["shell"] if "shell" in observed["kwargs"] else True
    assert observed["kwargs"]["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert observed["kwargs"]["timeout"] == 120
    assert result["removed"] is True
    assert result["model_files_deleted"] is True
    assert result["remaining_models"] == []


def test_remove_local_model_refuses_uninstalled_tags_before_subprocess(monkeypatch):
    monkeypatch.setattr(inference, "runtime_status", lambda: {"status": "online", "models": []})
    monkeypatch.setattr(inference.shutil, "which", lambda _name: "/usr/bin/ollama")
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("must not remove a non-installed tag"))

    with pytest.raises(ValueError, match="not installed"):
        inference.remove_local_model("qwen2.5:0.5b")


def test_remove_local_model_refuses_model_names_that_are_urls(monkeypatch):
    monkeypatch.setattr(inference, "runtime_status", lambda: pytest.fail("invalid model name must be rejected first"))
    with pytest.raises(ValueError):
        inference.remove_local_model("https://example.com/model")
