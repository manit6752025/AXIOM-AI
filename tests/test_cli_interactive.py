from __future__ import annotations

import json

from typer.testing import CliRunner

from axiom.cli.main import app


runner = CliRunner()


def test_root_exposes_non_interactive_and_no_input_aliases():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--non-interactive" in result.output
    assert "--no-input" in result.output


def test_non_interactive_mode_rejects_the_interactive_model_studio_wizard():
    result = runner.invoke(app, [
        "--no-banner", "--non-interactive", "model", "studio", "wizard",
    ])
    assert result.exit_code == 1
    assert "interactive-only" in result.output
    assert "pull-native" in result.output


def test_non_interactive_cloud_cli_uses_configured_api_key_environment_variable(monkeypatch):
    from axiom.core import llm_client

    monkeypatch.setenv("AXIOM_TEST_PROVIDER_KEY", "test-secret")
    observed = {}

    def fake_chat(provider, model, messages, **kwargs):
        observed.update(provider=provider, model=model, api_key=kwargs.get("api_key"))
        return {
            "response": "ok",
            "execution": "remote",
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    result = runner.invoke(app, [
        "--no-banner", "--non-interactive", "llm", "chat",
        "test-model", "hello", "--provider", "openai",
        "--api-key-env", "AXIOM_TEST_PROVIDER_KEY", "--send-to-provider",
    ])
    assert result.exit_code == 0, result.output
    assert observed == {
        "provider": "openai",
        "model": "test-model",
        "api_key": "test-secret",
    }


def test_non_interactive_cloud_cli_fails_with_env_guidance_instead_of_prompting(monkeypatch):
    monkeypatch.delenv("AXIOM_MISSING_PROVIDER_KEY", raising=False)
    result = runner.invoke(app, [
        "--no-banner", "--json-errors", "--non-interactive", "llm", "models",
        "--provider", "openai", "--api-key-env", "AXIOM_MISSING_PROVIDER_KEY",
    ])
    assert result.exit_code == 1
    payload = json.loads(result.stderr or result.stdout)
    assert payload["schema_version"] == 1
    assert payload["error"]["code"] == "interactive_input_required"
    assert "AXIOM_MISSING_PROVIDER_KEY" in payload["error"]["message"]


def test_non_interactive_remote_prompt_requires_explicit_consent():
    result = runner.invoke(app, [
        "--no-banner", "--non-interactive", "model", "studio", "remote-run",
        "https://provider.example/v1", "model", "send nothing",
    ])
    assert result.exit_code == 1
    assert "--send-to-provider" in result.output


def test_non_interactive_destructive_confirmation_defaults_to_cancel():
    result = runner.invoke(app, [
        "--no-banner", "--non-interactive", "evaluation", "delete", "missing-run",
    ])
    assert result.exit_code == 0, result.output
    assert "Cancelled." in result.output
