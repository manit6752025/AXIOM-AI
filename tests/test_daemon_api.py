import json
from http.client import HTTPConnection

import pytest

import axiom.daemon as daemon_module

serve_in_thread = daemon_module.serve_in_thread


def fetch(server, path, headers=None):
    host, port = server.server_address[:2]
    connection = HTTPConnection(host, port, timeout=3)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        return response.status, response.getheader("Content-Type", ""), body
    finally:
        connection.close()


def get_json(server, path, headers=None):
    status, _, body = fetch(server, path, headers)
    return status, json.loads(body)


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXIOM_MODEL_ROOT", str(tmp_path / "models"))
    model = tmp_path / "models" / "tiny"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 2,
            }
        ),
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(
        '{"text": "a"}\n{"text": "b"}\n{"text": "a"}\n', encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def server(workspace):
    srv, thread = serve_in_thread()
    try:
        yield srv
    finally:
        srv.shutdown()
        thread.join(timeout=2)
        srv.server_close()


def test_file_picker_lists_models_and_datasets(server):
    status, models = get_json(server, "/api/files?kind=models")
    assert status == 200
    assert {"path": "models/tiny"} in models["items"]

    status, datasets = get_json(server, "/api/files?kind=datasets")
    assert status == 200
    assert {"path": "data/train.jsonl"} in datasets["items"]

    status, payload = get_json(server, "/api/files?kind=other")
    assert status == 422


def test_model_inspect_reads_workspace_model(server):
    status, payload = get_json(server, "/api/model/inspect?path=models/tiny")
    assert status == 200
    assert payload["inspection"]["model_type"] == "llama"
    assert payload["inspection"]["has_config"] is True


def test_dataset_inspect_reports_duplicates(server):
    status, payload = get_json(server, "/api/dataset/inspect?path=data/train.jsonl")
    assert status == 200
    assert payload["inspection"]["samples"] == 3
    assert payload["inspection"]["duplicates"] == 1


def test_paths_outside_the_workspace_are_rejected(server):
    for path in ("/etc", "../outside", "models/../../etc"):
        status, payload = get_json(server, f"/api/model/inspect?path={path}")
        assert status == 400
        assert payload["error"] == "invalid_path"

    status, payload = get_json(server, "/api/dataset/inspect?path=/etc/passwd")
    assert status == 400


def test_missing_path_returns_not_found(server):
    status, payload = get_json(server, "/api/model/inspect?path=models/missing")
    assert status == 404


def test_training_plan_validates_input(server):
    status, payload = get_json(server, "/api/train/plan?params=7&method=qlora")
    assert status == 200
    assert "fits_hardware" in payload["plan"]

    for query in ("params=abc", "params=nan", "params=-1", "params=7&method=bad"):
        status, _ = get_json(server, f"/api/train/plan?{query}")
        assert status in (400, 422)


def test_ai_plan_and_headroom_and_registry(server):
    status, payload = get_json(server, "/api/ai/plan?model=models/tiny")
    assert status == 200
    assert "recommended_quantization" in payload["plan"]

    status, payload = get_json(server, "/api/headroom")
    assert status == 200
    assert payload["headroom"]["id"] == "headroom"

    status, payload = get_json(server, "/api/models")
    assert status == 200
    assert payload["models"] == []


def test_non_loopback_host_header_is_refused(server):
    status, payload = get_json(server, "/health", {"Host": "evil.example"})
    assert status == 403


def test_web_ui_is_served_from_ui_path(server, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "assets").mkdir(parents=True)
    (bundle / "index.html").write_text("<html>AXIOM UI</html>", encoding="utf-8")
    (bundle / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setattr(daemon_module, "WEBUI_DIR", bundle)

    status, kind, body = fetch(server, "/ui/")
    assert status == 200 and "text/html" in kind and b"AXIOM UI" in body

    status, kind, _ = fetch(server, "/ui/assets/app.js")
    assert status == 200 and "javascript" in kind

    status, _, _ = fetch(server, "/ui/../secret.txt")
    assert status in (403, 404)

    status, _, _ = fetch(server, "/ui/assets/missing.js")
    assert status == 404


def test_web_ui_missing_bundle_explains_how_to_build(server, tmp_path, monkeypatch):
    monkeypatch.setattr(daemon_module, "WEBUI_DIR", tmp_path / "nothing")
    status, payload = get_json(server, "/ui/")
    assert status == 404
    assert payload["error"] == "web_ui_missing"
