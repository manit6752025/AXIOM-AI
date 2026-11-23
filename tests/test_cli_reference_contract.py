from __future__ import annotations

from pathlib import Path

from typer.main import get_command

from axiom.cli.main import app


ROOT = Path(__file__).resolve().parents[1]


def test_cli_reference_lists_every_registered_root_command():
    reference = (ROOT / "docs" / "wiki" / "CLI-Reference.md").read_text(encoding="utf-8")
    commands = get_command(app).commands

    missing = [
        f"axiom {name}"
        for name in sorted(commands)
        if f"`axiom {name}`" not in reference
    ]

    assert missing == [], f"Unlisted registered CLI commands: {missing}"


def test_cli_reference_api_examples_match_typed_api_client():
    reference = (ROOT / "docs" / "wiki" / "CLI-Reference.md").read_text(encoding="utf-8")
    api_client = (ROOT / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    endpoints = [
        "/api/health",
        "/api/diagnostics/workspace",
        "/api/diagnostics/roadmap",
        "/api/workspace/models",
        "/api/workspace/datasets",
        "/api/workspace/experiments",
        "/api/workspace/dataset-cleaning-preview",
        "/api/workspace/dataset-quality",
        "/api/workflows",
        "/api/fine-tuning/jobs",
        "/api/evaluation/runs",
        "/api/inference/generate",
        "/api/inference/status",
        "/api/engine/status",
        "/api/engine/telemetry",
        "/api/llm/providers",
        "/api/llm/models",
        "/api/llm/chat",
        "/api/optimizer/plan",
        "/api/agent-profiles",
        "/api/security/tool-permissions",
        "/api/diagnostics/runtime-connectors",
    ]

    for endpoint in endpoints:
        assert endpoint in api_client, f"Typed client no longer exposes {endpoint}"
        assert endpoint in reference, f"CLI reference omits API surface {endpoint}"


def test_cli_reference_gui_names_match_control_center_navigation():
    reference = (ROOT / "docs" / "wiki" / "CLI-Reference.md").read_text(encoding="utf-8")
    app_source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    pages = [
        "Models",
        "Datasets",
        "Training",
        "Evaluation",
        "Runtime",
        "API Playground",
        "Agent Profiles",
        "Roadmap",
        "Settings",
    ]

    for page in pages:
        assert f'label: "{page}"' in app_source, f"Control Center navigation is missing {page}"
        assert page in reference, f"CLI reference omits Control Center surface {page}"
