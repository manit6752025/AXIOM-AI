from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from axiom.api.server import create_app
from axiom.cli.api import api_app
from axiom.core.api_examples import build_api_examples, render_api_examples_markdown


def _openapi():
    return create_app().openapi()


def test_generated_examples_cover_the_live_openapi_operation_set_and_are_deterministic() -> None:
    schema = _openapi()
    report = build_api_examples(schema)
    expected = {
        (path, method.upper())
        for path, path_item in schema["paths"].items()
        for method in ("get", "post", "put", "patch", "delete", "head", "options")
        if isinstance(path_item.get(method), dict)
    }
    actual = {(item["path"], item["method"]) for item in report["operations"]}

    assert report["schema_version"] == 1
    assert report["api_contract_version"] == 1
    assert report["source_sha256"] == build_api_examples(schema)["source_sha256"]
    assert report["operation_count"] == len(actual)
    assert actual == expected
    assert report["truncated"] is False
    assert report["privacy"] == {
        "credential_values_included": False,
        "environment_values_included": False,
        "schema_defaults_replayed": False,
        "requests_executed": False,
    }


def test_secret_fields_are_redacted_but_environment_reference_names_are_typed() -> None:
    report = build_api_examples(_openapi())
    chat = next(item for item in report["operations"] if item["path"] == "/api/llm/chat" and item["method"] == "POST")
    assert chat["request"] is not None
    assert "api_key" not in chat["request"]["example"]
    assert "$.body.api_key" in chat["request"]["redacted_fields"]
    assert chat["request"]["schema"]["properties"]["api_key"]["x-axiom-redacted"] is True
    assert "api_key: string" in chat["request"]["typescript_type"]

    save_profile = next(item for item in report["operations"] if item["path"] == "/api/llm/profiles" and item["method"] == "POST")
    env_reference = save_profile["request"]["schema"]["properties"]["api_key_env"]
    assert "string" in json.dumps(env_reference)
    assert env_reference.get("x-axiom-redacted") is not True
    assert "$.body.api_key_env" not in save_profile["request"]["redacted_fields"]

    serialized = json.dumps(report, ensure_ascii=False)
    assert "never-return-this" not in serialized
    assert "request-only-sentinel" not in serialized


def test_markdown_examples_are_typed_and_explicitly_non_executing() -> None:
    report = build_api_examples(_openapi())
    rendered = render_api_examples_markdown(report)

    assert "# Generated AXIOM API examples" in rendered
    assert "TypeScript contract" in rendered
    assert "cURL template" in rendered
    assert "not executed" in rendered
    assert "provide-secret-securely" not in rendered or "api_key" in rendered
    assert "never-return-this" not in rendered


def test_api_examples_endpoint_returns_generated_contract_without_dispatching_operations(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        response = client.get("/api/contract/examples")

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["operation_count"] > 0
    assert report["privacy"]["requests_executed"] is False
    assert any(item["path"] == "/api/llm/chat" for item in report["operations"])


def test_cli_emits_json_and_markdown_from_the_contract(monkeypatch) -> None:
    from axiom.api import server

    minimal = {
        "openapi": "3.1.0",
        "info": {"version": "test"},
        "components": {"schemas": {}},
        "paths": {
            "/api/health": {
                "get": {
                    "operationId": "getHealth",
                    "summary": "Health",
                    "responses": {"200": {"description": "OK", "content": {"application/json": {"schema": {"type": "object", "properties": {"status": {"type": "string"}}}}}}},
                },
            },
        },
    }
    monkeypatch.setattr(server, "create_app", lambda: SimpleNamespace(openapi=lambda: minimal))
    runner = CliRunner()

    json_result = runner.invoke(api_app, ["examples", "--format", "json"])
    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.stdout)["operations"][0]["path"] == "/api/health"

    markdown_result = runner.invoke(api_app, ["examples", "--format", "markdown"])
    assert markdown_result.exit_code == 0, markdown_result.output
    assert "GET /api/health" in markdown_result.stdout



def test_response_examples_omit_sensitive_schema_properties() -> None:
    report = build_api_examples({
        "openapi": "3.1.0",
        "info": {"version": "test"},
        "components": {"schemas": {}},
        "paths": {
            "/api/test": {
                "get": {
                    "operationId": "testResponseRedaction",
                    "responses": {
                        "200": {
                            "description": "Synthetic response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["id", "access_token"],
                                        "properties": {
                                            "id": {"type": "string"},
                                            "access_token": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    })
    operation = report["operations"][0]
    assert operation["response"]["example"] == {"id": "example"}
    assert "$.response.access_token" in operation["response"]["redacted_fields"]
