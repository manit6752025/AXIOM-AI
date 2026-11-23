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


def test_structured_generation_passes_validated_schema_to_ollama_format(monkeypatch):
    observed = {}

    def fake_open(request, timeout):
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({
            "response": '{"name":"AXIOM","count":3}',
            "done": True,
            "eval_count": 8,
            "total_duration": 10_000_000,
        })

    monkeypatch.setattr(inference, "_open_loopback", fake_open)
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name", "count"],
    }
    result = inference.run_local_inference(
        "qwen2.5:0.5b", "Return a product object.", num_predict=128,
        response_format=schema,
    )

    assert observed["payload"]["format"]["type"] == "object"
    assert observed["payload"]["format"]["additionalProperties"] is False
    assert observed["payload"]["format"]["required"] == ["name", "count"]
    assert "$schema" not in observed["payload"]["format"]
    assert observed["payload"]["prompt"].startswith("Return a product object.")
    assert "Return only one JSON value" in observed["payload"]["prompt"]
    assert result["response"] == '{"name":"AXIOM","count":3}'


def test_invalid_structured_schema_is_rejected_before_loopback(monkeypatch):
    monkeypatch.setattr(
        inference, "_open_loopback",
        lambda *_args, **_kwargs: pytest.fail("invalid schema must not reach Ollama"),
    )
    with pytest.raises(ValueError, match="Unsupported JSON Schema keyword"):
        inference.run_local_inference(
            "qwen2.5:0.5b", "Return JSON.",
            response_format={"type": "object", "$ref": "https://example.invalid/schema"},
        )



def _vision_test_png(width: int = 8, height: int = 4) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")


def test_local_vision_checks_model_capability_and_sends_only_bounded_workspace_image(tmp_path, monkeypatch):
    import base64

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    image = _vision_test_png()
    (root / "data" / "sample.png").write_bytes(image)
    observed = []

    def fake_request(path, *, payload=None, timeout):
        observed.append((path, payload, timeout))
        if path == "/api/show":
            return {"capabilities": ["completion", "vision"]}
        return {
            "response": "A small blue square is visible.",
            "done": True,
            "eval_count": 12,
            "total_duration": 1_500_000_000,
        }

    monkeypatch.setattr(inference, "_request_json", fake_request)
    report = inference.run_local_vision(
        "llava:latest", "data/sample.png", "Describe the visible object.",
        workspace_root=root, max_new_tokens=96, timeout_seconds=30,
    )

    assert [item[0] for item in observed] == ["/api/show", "/api/generate"]
    assert observed[0][1] == {"model": "llava:latest", "verbose": False}
    payload = observed[1][1]
    assert payload["model"] == "llava:latest"
    assert payload["prompt"] == "Describe the visible object."
    assert payload["stream"] is False
    assert payload["options"] == {"num_predict": 96}
    assert payload["images"] == [base64.b64encode(image).decode("ascii")]
    assert observed[1][2] == 30
    assert report["response"] == "A small blue square is visible."
    assert report["image"] == {"format": "png", "width": 8, "height": 4, "size_bytes": len(image)}
    assert report["generation"]["total_duration_ms"] == 1500.0
    assert report["privacy"]["network_used"] is False
    assert report["privacy"]["image_persisted_by_axiom"] is False
    assert image.hex() not in json.dumps(report)


def test_local_vision_refuses_models_without_explicit_vision_capability(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    (root / "data" / "sample.png").write_bytes(_vision_test_png())
    calls = []

    def fake_request(path, *, payload=None, timeout):
        calls.append(path)
        return {"capabilities": ["completion"]}

    monkeypatch.setattr(inference, "_request_json", fake_request)
    with pytest.raises(inference.InferenceRuntimeError, match="did not explicitly report the vision capability"):
        inference.run_local_vision("qwen2.5:0.5b", "data/sample.png", "Describe it.", workspace_root=root)
    assert calls == ["/api/show"]


@pytest.mark.parametrize("image_path", ["../secret.png", "/etc/passwd", "models/sample.png", "data/../secret.png"])
def test_local_vision_rejects_non_workspace_image_paths_before_runtime(tmp_path, monkeypatch, image_path):
    monkeypatch.setattr(inference, "_request_json", lambda *_args, **_kwargs: pytest.fail("runtime must not be contacted"))
    with pytest.raises(ValueError, match="under data"):
        inference.run_local_vision("llava:latest", image_path, "Describe it.", workspace_root=tmp_path)


def test_local_vision_rejects_oversized_or_malformed_images_before_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    (root / "data" / "invalid.png").write_bytes(b"not a PNG image")
    (root / "data" / "large.png").write_bytes(_vision_test_png(10_000, 10_000))
    monkeypatch.setattr(inference, "_request_json", lambda *_args, **_kwargs: pytest.fail("runtime must not be contacted"))
    with pytest.raises(ValueError, match="signature"):
        inference.run_local_vision("llava:latest", "data/invalid.png", "Describe it.", workspace_root=root)
    with pytest.raises(ValueError, match="40-megapixel"):
        inference.run_local_vision("llava:latest", "data/large.png", "Describe it.", workspace_root=root)


def test_local_vision_rejects_invalid_limits_before_reading_file(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "open_mcp_workspace_file", lambda *_args: pytest.fail("image must not be opened"))
    with pytest.raises(ValueError, match="prompt"):
        inference.run_local_vision("llava:latest", "data/image.png", " ", workspace_root=tmp_path)
    with pytest.raises(ValueError, match="tokens"):
        inference.run_local_vision("llava:latest", "data/image.png", "Describe it.", workspace_root=tmp_path, max_new_tokens=1025)
    with pytest.raises(ValueError, match="timeout"):
        inference.run_local_vision("llava:latest", "data/image.png", "Describe it.", workspace_root=tmp_path, timeout_seconds=181)
