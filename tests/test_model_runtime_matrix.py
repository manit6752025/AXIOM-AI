from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import axiom.cli.main as cli_main
from axiom.api import workspace
from axiom.models import runtime_matrix


def _reset_cache(monkeypatch) -> None:
    monkeypatch.setattr(runtime_matrix, "_CACHE_VALUE", None)
    monkeypatch.setattr(runtime_matrix, "_CACHE_CREATED_AT", 0.0)


def test_runtime_matrix_reports_local_versions_and_candidate_format_evidence(monkeypatch) -> None:
    _reset_cache(monkeypatch)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_which(command: str) -> str | None:
        return f"/mock/bin/{command}" if command in {"ollama", "llama-cli"} else None

    def fake_run(args, **kwargs):
        calls.append((args, kwargs["env"]))
        if args[0].endswith("/ollama"):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="ollama version is 0.5.0 token=private-secret /private/build/ollama\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="version: 1234 (abcdef) /private/build/llama.cpp\n",
            stderr="",
        )

    monkeypatch.setattr(runtime_matrix.shutil, "which", fake_which)
    monkeypatch.setattr(runtime_matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime_matrix.importlib.metadata, "version", lambda name: "4.55.0")

    result = runtime_matrix.probe_local_runtime_matrix(force_refresh=True)

    by_id = {item["id"]: item for item in result["runtimes"]}
    assert by_id["ollama"]["availability"] == "installed"
    assert by_id["ollama"]["version"].startswith("ollama version is 0.5.0")
    assert "private-secret" not in by_id["ollama"]["version"]
    assert "/private/build" not in by_id["ollama"]["version"]
    assert by_id["llama.cpp"]["version"].startswith("version: 1234")
    assert by_id["transformers"]["version"] == "4.55.0"
    assert result["network_access"] is False
    assert result["model_load_tested"] is False

    formats = {item["format"]: item["candidates"] for item in result["formats"]}
    assert {item["runtime_id"] for item in formats["GGUF"]} == {"ollama", "llama.cpp"}
    assert [item["runtime_id"] for item in formats["Safetensors"]] == ["transformers"]
    assert [item["runtime_id"] for item in formats["PyTorch"]] == ["transformers"]
    assert all(item["status"] == "candidate_not_verified" for candidates in formats.values() for item in candidates)

    assert len(calls) == 2
    for args, env in calls:
        assert args[-1] == "--version"
        assert set(env) == {"PATH", "LANG", "LC_ALL"}
        assert env["LANG"] == "C"


def test_runtime_matrix_distinguishes_missing_and_unavailable_version_probes(monkeypatch) -> None:
    _reset_cache(monkeypatch)
    monkeypatch.setattr(runtime_matrix.shutil, "which", lambda _command: None)
    monkeypatch.setattr(
        runtime_matrix.importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(runtime_matrix.importlib.metadata.PackageNotFoundError()),
    )

    result = runtime_matrix.probe_local_runtime_matrix(force_refresh=True)

    assert {item["availability"] for item in result["runtimes"]} == {"not_detected"}
    assert all(item["version"] is None for item in result["runtimes"])
    assert "not interpreted as proof" in " ".join(result["limitations"])


def test_runtime_matrix_times_out_without_claiming_runtime_is_unsupported(monkeypatch) -> None:
    _reset_cache(monkeypatch)
    monkeypatch.setattr(
        runtime_matrix.shutil,
        "which",
        lambda command: "/mock/bin/ollama" if command == "ollama" else None,
    )

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ollama --version", timeout=2.0)

    monkeypatch.setattr(runtime_matrix.subprocess, "run", timeout)
    monkeypatch.setattr(
        runtime_matrix.importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(runtime_matrix.importlib.metadata.PackageNotFoundError()),
    )

    result = runtime_matrix.probe_local_runtime_matrix(force_refresh=True)
    ollama = next(item for item in result["runtimes"] if item["id"] == "ollama")
    assert ollama["availability"] == "version_probe_timed_out"
    assert ollama["version"] is None
    assert any(item["status"] == "candidate_not_verified" for item in result["formats"][0]["candidates"])


def test_runtime_matrix_cache_is_bounded_and_force_refresh_reprobes(monkeypatch) -> None:
    _reset_cache(monkeypatch)
    calls = 0

    def fake_which(_command: str) -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(runtime_matrix.shutil, "which", fake_which)
    monkeypatch.setattr(
        runtime_matrix.importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(runtime_matrix.importlib.metadata.PackageNotFoundError()),
    )

    first = runtime_matrix.probe_local_runtime_matrix()
    second = runtime_matrix.probe_local_runtime_matrix()
    assert calls == 0  # No command is present on PATH.
    assert first == second

    runtime_matrix.probe_local_runtime_matrix(force_refresh=True)
    assert calls > 0


def test_runtime_matrix_api_endpoint_supports_explicit_cache_refresh(monkeypatch) -> None:
    calls: list[bool] = []
    matrix = {
        "schema_version": 1,
        "network_access": False,
        "model_load_tested": False,
        "runtimes": [],
        "formats": [],
        "scope_note": "Local version evidence only.",
    }

    def fake_probe(*, force_refresh: bool = False):
        calls.append(force_refresh)
        return matrix

    monkeypatch.setattr(workspace, "probe_local_runtime_matrix", fake_probe)
    app = FastAPI()
    app.include_router(workspace.router)
    client = TestClient(app)

    first = client.get("/api/workspace/models/runtime-matrix")
    second = client.get("/api/workspace/models/runtime-matrix", params={"refresh": "true"})

    assert first.status_code == second.status_code == 200
    assert first.json()["matrix"] == matrix
    assert calls == [False, True]


def test_model_runtimes_cli_json_exposes_version_matrix(monkeypatch) -> None:
    expected = {
        "schema_version": 1,
        "network_access": False,
        "model_load_tested": False,
        "formats": [{"format": "GGUF", "candidates": []}],
    }
    calls: list[bool] = []

    def fake_probe(*, force_refresh: bool = False):
        calls.append(force_refresh)
        return expected

    monkeypatch.setattr(cli_main, "probe_local_runtime_matrix", fake_probe)
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["model", "runtimes", "--refresh", "--json"])

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout) == expected
    assert calls == [True]
