from __future__ import annotations

import asyncio
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, Self

import pytest

from axiom.core.hardware import HardwareInfo, estimate_model_fit
from axiom.core.integrations import (
    AgentIntegration,
    IntegrationRegistry,
    IntegrationType,
)
from axiom.core.project import create_project
from axiom.core.tool_permissions import TOOL_DESCRIPTIONS, ToolPermissionStore
from axiom.datasets.cleaner import clean_jsonl
from axiom.datasets.inspector import inspect_dataset
from axiom.mcp import server as mcp_module
from axiom.mcp.server import mcp
from axiom.models.inspector import inspect_model
from axiom.models.registry import Model, ModelRegistry
from axiom.runtime import supercompress
from axiom.runtime.supercompress import CompressionResult, compress_context
from axiom.training.planner import create_training_plan


def _call_tool(name: str, arguments: dict) -> tuple[dict, object]:
    result = asyncio.run(mcp.call_tool(name, arguments))
    assert not result.is_error, result
    if result.structured_content is not None:
        return result.structured_content.get(
            "result", result.structured_content
        ), result
    assert result.content
    return json.loads(result.content[0].text), result


def test_mcp_schema_and_all_handlers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tools = asyncio.run(mcp.list_tools())
    tool_names = {tool.name for tool in tools}
    assert tool_names == {
        "axiom_dataset_clean",
        "axiom_dataset_clean_preview",
        "axiom_dataset_inspect",
        "axiom_engine_generate",
        "axiom_engine_preflight",
        "axiom_evaluation_compare",
        "axiom_evaluation_compare_examples",
        "axiom_evaluation_examples",
        "axiom_evaluation_history",
        "axiom_evaluation_preflight",
        "axiom_evaluation_run",
        "axiom_info",
        "axiom_model_info",
        "axiom_model_list",
        "axiom_prediction_job_cancel",
        "axiom_prediction_job_compare",
        "axiom_prediction_job_compare_examples",
        "axiom_prediction_job_evaluate",
        "axiom_prediction_job_examples",
        "axiom_prediction_job_preflight",
        "axiom_prediction_job_validation_preflight",
        "axiom_prediction_job_start",
        "axiom_prediction_job_status",
        "axiom_prediction_job_wait",
        "axiom_supercompress",
        "axiom_supercompress_status",
        "axiom_system_info",
        "axiom_training_eval_overlap_preflight",
        "axiom_training_job_cancel",
        "axiom_training_job_eval_overlap_preflight",
        "axiom_training_job_prediction_compare",
        "axiom_training_job_start",
        "axiom_training_job_status",
        "axiom_training_job_validation_preflight",
        "axiom_training_job_wait",
        "axiom_training_plan",
        "axiom_training_preflight",
        "axiom_workspace_assets",
        "axiom_workspace_context",
        "axiom_workflow_save",
    }

    monkeypatch.chdir(tmp_path)
    permission_store = ToolPermissionStore(tmp_path / "mcp-permissions")
    permission_store.save_profile("default", list(TOOL_DESCRIPTIONS))
    monkeypatch.setattr(mcp_module, "ToolPermissionStore", lambda: permission_store)
    info, _ = _call_tool("axiom_info", {})
    assert info["name"] == "AXIOM"

    models, _ = _call_tool("axiom_model_list", {})
    assert models == []
    if (tmp_path / ".axiom").exists():
        raise AssertionError(None)

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DemoForCausalLM"],
                "model_type": "demo",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "vocab_size": 32,
                "intermediate_size": 16,
            }
        ),
        encoding="utf-8",
    )
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_path / "weights.safetensors").write_bytes(b"weights")

    model_info, _ = _call_tool(
        "axiom_model_info",
        {"path": str(model_path)},
    )
    assert model_info["format"] == "Safetensors"
    assert model_info["has_tokenizer"] is True
    assert model_info["capabilities"] == ["Text generation"]

    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        '{"prompt":"a","answer":"b"}\n{"prompt":"a","answer":"b"}\nnot json\n[]\n\n',
        encoding="utf-8",
    )
    inspected, _ = _call_tool(
        "axiom_dataset_inspect",
        {"path": "records.jsonl"},
    )
    assert inspected["profile"]["valid_rows"] == 2
    assert inspected["profile"]["invalid_rows"] == 2
    assert inspected["profile"]["duplicate_rows"] == 1
    assert inspected["scope"]["dataset_values_returned"] is False
    assert inspected["scope"]["absolute_paths_returned"] is False

    cleaned_path = tmp_path / "cleaned.jsonl"
    cleaned, _ = _call_tool(
        "axiom_dataset_clean",
        {"path": "records.jsonl", "output": "cleaned.jsonl"},
    )
    assert cleaned["kept"] == 1
    if not (dataset.read_text(encoding="utf-8").count("not json") == 1):
        raise AssertionError(None)
    if not (
        cleaned_path.read_text(encoding="utf-8") == ('{"prompt": "a", "answer": "b"}\n')
    ):
        raise AssertionError(None)

    hardware = HardwareInfo(
        os_name="test",
        architecture="test",
        cpu_cores=4,
        ram_gb=8,
        gpu_name="Test GPU",
        vram_gb=24,
        cuda_available=True,
    )
    monkeypatch.setattr(mcp_module, "detect_hardware", lambda: hardware)

    system_info, _ = _call_tool("axiom_system_info", {})
    assert system_info["gpu"] == "Test GPU"

    plan, _ = _call_tool(
        "axiom_training_plan",
        {"parameters_billions": 3, "method": " LORA "},
    )
    assert plan["method"] == "lora"
    assert plan["precision"] == "FP16"

    monkeypatch.setenv("SUPERCOMPRESS_API_KEY", "secret")
    monkeypatch.setenv(
        "SUPERCOMPRESS_API_BASE",
        "https://user:password@example.test:443/base?token=do-not-leak",
    )
    status, _ = _call_tool("axiom_supercompress_status", {})
    assert status == {
        "configured": True,
        "endpoint": "https://example.test:443/base",
    }

    expected_result = CompressionResult(
        compressed_text="short",
        original_tokens=10,
        kept_tokens=5,
        tokens_saved=5,
        savings_pct=50.0,
        compression_risk="low",
        policy_name="default",
        mode="neural",
    )
    monkeypatch.setattr(
        mcp_module,
        "compress_context",
        lambda *args, **kwargs: expected_result,
    )
    compressed, _ = _call_tool(
        "axiom_supercompress",
        {"context": "long", "query": "question"},
    )
    assert compressed["compressed_text"] == "short"
    assert compressed["savings_pct"] == 50.0


def test_dataset_cleaning_never_truncates_source(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    original = '{"a": 1}\n{"a": 1}\n'
    source.write_text(original, encoding="utf-8")

    with pytest.raises(OSError, match="must differ"):
        clean_jsonl(source, source)

    if not (source.read_text(encoding="utf-8") == original):
        raise AssertionError(None)

    output = tmp_path / "nested" / "clean.jsonl"
    result = clean_jsonl(source, output)
    assert result.kept == 1
    if not (output.exists()):
        raise AssertionError(None)
    if not (source.read_text(encoding="utf-8") == original):
        raise AssertionError(None)
    if list(output.parent.glob(".clean.jsonl.*.tmp")):
        raise AssertionError(None)


def test_dataset_cleaning_keeps_valid_objects_with_different_fields(tmp_path: Path):
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        '{"prompt":"hello"}\n{"messages":[{"role":"user"}]}\n',
        encoding="utf-8",
    )
    output = tmp_path / "cleaned.jsonl"

    result = clean_jsonl(source, output)

    assert result.kept == 2
    assert result.removed_non_objects == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        '{"prompt": "hello"}',
        '{"messages": [{"role": "user"}]}',
    ]


def test_dataset_inspection_reports_invalid_line_details(tmp_path: Path):
    source = tmp_path / "invalid.jsonl"
    source.write_text('{"ok": 1}\nnot json\n[1, 2]\n', encoding="utf-8")

    result = inspect_dataset(str(source))

    assert result.valid == 1
    assert result.invalid == 2
    assert result.errors == [
        (2, "Expecting value: line 1 column 1 (char 0)"),
        (3, "record is not a JSON object"),
    ]


def test_model_inspection_rejects_non_object_config(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        inspect_model(str(model_path))

    (model_path / "config.json").write_text(
        json.dumps({"architectures": None, "model_type": "unknown"}),
        encoding="utf-8",
    )
    result = inspect_model(str(model_path))
    assert result.capabilities == ["Unknown"]


def test_model_inspection_does_not_treat_weight_bytes_as_parameter_count(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"hidden_size": 128, "num_hidden_layers": 4, "vocab_size": 1000}),
        encoding="utf-8",
    )
    (model_path / "weights.bin").write_bytes(b"x" * 4096)

    result = inspect_model(str(model_path))

    assert result.weight_size_bytes == 4096
    assert result.parameter_count is None
    assert result.parameter_source == "unknown"
    assert result.estimated_vram_gb is None


def test_model_analysis_accounts_for_grouped_query_attention():
    from axiom.models.analysis import analyze_config

    config = {
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "vocab_size": 1000,
        "intermediate_size": 352,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "model_type": "llama",
        "tie_word_embeddings": True,
    }

    result = analyze_config("local/demo", config, [])

    assert result.parameter_count == 833_664
    assert result.parameter_source == "architecture estimate"


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True])
def test_numeric_boundaries_reject_invalid_values(value: object):
    hardware = HardwareInfo(
        os_name="test",
        architecture="test",
        cpu_cores=1,
        ram_gb=1,
        gpu_name="test",
        vram_gb=8,
        cuda_available=True,
    )

    with pytest.raises(ValueError):
        create_training_plan(value, hardware)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        estimate_model_fit(value, hardware)  # type: ignore[arg-type]


def test_project_name_cannot_escape_parent(tmp_path: Path):
    for unsafe_name in ("../outside", "/tmp/outside", "..\\outside", ""):
        with pytest.raises(ValueError):
            create_project(unsafe_name, tmp_path)

    project = create_project("safe-project", tmp_path)
    assert project == tmp_path / "safe-project"
    if not ((project / "axiom.yaml").is_file()):
        raise AssertionError(None)


def test_model_registry_is_atomic_and_concurrent(tmp_path: Path):
    root = tmp_path / "registry"

    def add(index: int) -> None:
        ModelRegistry(root).add(Model(name=f"model-{index}", source=f"source-{index}"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add, range(24)))

    registry = ModelRegistry(root, create=False)
    if not (
        {model.name for model in registry.list()}
        == {f"model-{index}" for index in range(24)}
    ):
        raise AssertionError(None)
    if not (json.loads((root / "models.json").read_text(encoding="utf-8"))):
        raise AssertionError(None)


def test_registries_do_not_create_state_on_read_and_reject_corruption(
    tmp_path: Path,
):
    model_root = tmp_path / "models"
    integration_root = tmp_path / "integrations"

    if not (ModelRegistry(model_root, create=False).list() == []):
        raise AssertionError(None)
    if not (IntegrationRegistry(integration_root, create=False).list() == []):
        raise AssertionError(None)
    if model_root.exists():
        raise AssertionError(None)
    if integration_root.exists():
        raise AssertionError(None)

    model_root.mkdir()
    (model_root / "models.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON array"):
        ModelRegistry(model_root, create=False).list()

    integration_root.mkdir()
    (integration_root / "integrations.json").write_text(
        '{"bad": true}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected a JSON array"):
        IntegrationRegistry(integration_root, create=False).list()


def test_integration_registry_is_atomic_and_case_insensitive(tmp_path: Path):
    root = tmp_path / "integrations"
    registry = IntegrationRegistry(root)
    registry.add(
        AgentIntegration(
            name="Local MCP",
            type=IntegrationType.MCP,
            command="axiom-mcp",
        )
    )

    with pytest.raises(ValueError, match="already exists"):
        IntegrationRegistry(root).add(
            AgentIntegration(
                name="local mcp",
                type=IntegrationType.MCP,
            )
        )

    if not (registry.remove("LOCAL MCP") is True):
        raise AssertionError(None)
    if not (registry.list() == []):
        raise AssertionError(None)
    if not (registry.remove("missing") is False):
        raise AssertionError(None)


class _ProviderHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, dict, str | None]]] = []
    redirect = False

    def do_POST(self) -> None:
        if self.redirect:
            self.send_response(302)
            self.send_header("Location", "/target")
            self.end_headers()
            return

        body_size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(body_size).decode("utf-8"))
        self.requests.append(
            (
                self.path,
                payload,
                self.headers.get("X-API-Key"),
            )
        )
        response = json.dumps(
            {
                "compressed_text": "compressed",
                "original_tokens": 10,
                "kept_tokens": 4,
                "tokens_saved": 6,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def test_provider_http_boundary_and_redirect_protection(
    monkeypatch: pytest.MonkeyPatch,
):
    _ProviderHandler.requests = []
    _ProviderHandler.redirect = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("SUPERCOMPRESS_API_KEY", "api-secret")
        monkeypatch.setenv(
            "SUPERCOMPRESS_API_BASE",
            f"http://127.0.0.1:{server.server_port}",
        )
        result = compress_context("context", "query")
        assert result.compressed_text == "compressed"
        assert result.savings_pct == 60.0
        assert _ProviderHandler.requests == [
            (
                "/api/v1/compress",
                {
                    "context": "context",
                    "query": "query",
                    "budget_ratio": 0.35,
                    "ccr": False,
                },
                "api-secret",
            )
        ]

        _ProviderHandler.redirect = True
        with pytest.raises(RuntimeError, match="HTTP 302"):
            compress_context("context", "query")
        if not (len(_ProviderHandler.requests) == 1):
            raise AssertionError(None)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_provider_rejects_invalid_configuration_and_payload(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SUPERCOMPRESS_API_KEY", "key")
    monkeypatch.setenv("SUPERCOMPRESS_API_BASE", "file:///private/secret")

    with pytest.raises(RuntimeError, match=r"HTTP\(S\) URL"):
        compress_context("context", "query")

    monkeypatch.setenv("SUPERCOMPRESS_API_BASE", "https://example.test")
    with pytest.raises(ValueError, match="budget_ratio"):
        compress_context("context", "query", budget_ratio=math.inf)

    with pytest.raises(ValueError, match="budget_ratio"):
        compress_context("context", "query", budget_ratio=0)


def test_provider_malformed_response_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeResponse:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return b"not-json"

    monkeypatch.setenv("SUPERCOMPRESS_API_KEY", "key")
    monkeypatch.setenv("SUPERCOMPRESS_API_BASE", "https://example.test")
    monkeypatch.setattr(
        supercompress,
        "_open_request",
        lambda request, *, timeout: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="invalid JSON"):
        compress_context("context", "query")
