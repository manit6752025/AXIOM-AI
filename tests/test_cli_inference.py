from __future__ import annotations

from typer.main import get_command
from typer.testing import CliRunner

from axiom.cli.main import app


def test_infer_status_json_is_a_registered_cli_command(monkeypatch):
    fake = {
        "backend": "ollama", "endpoint": "http://127.0.0.1:11434", "runtime_installed": True,
        "status": "online", "models": ["qwen2.5:0.5b"], "error": None,
        "checked_at": "2026-10-21T00:00:00+00:00",
    }
    monkeypatch.setattr("axiom.cli.inference.service_status", lambda: fake)
    result = CliRunner().invoke(app, ["--no-banner", "infer", "status", "--json"])
    assert result.exit_code == 0, result.output
    assert '"status": "online"' in result.output
    assert "qwen2.5:0.5b" in result.output


def test_infer_service_commands_are_registered():
    runner = CliRunner()
    result = runner.invoke(app, ["--no-banner", "infer", "service", "--help"])
    assert result.exit_code == 0, result.output
    assert "start" in result.output and "stop" in result.output and "status" in result.output


def test_infer_pull_command_is_registered(plain):
    result = CliRunner().invoke(app, ["--no-banner", "infer", "pull", "--help"])
    assert result.exit_code == 0, plain(result.output)
    pull_command = get_command(app).commands["infer"].commands["pull"]
    searchable = " ".join([str(pull_command.help or ""), plain(result.output)])
    assert "Ollama model tag" in searchable


def test_infer_model_operation_commands_are_registered():
    runner = CliRunner()
    inspect = runner.invoke(app, ["--no-banner", "infer", "inspect", "--help"])
    unload = runner.invoke(app, ["--no-banner", "infer", "unload", "--help"])
    assert inspect.exit_code == 0, inspect.output
    assert unload.exit_code == 0, unload.output
    assert "metadata" in inspect.output.lower()
    assert "without deleting" in unload.output.lower()


def test_infer_loaded_model_inventory_command_is_registered():
    result = CliRunner().invoke(app, ["--no-banner", "infer", "ps", "--help"])
    assert result.exit_code == 0, result.output
    assert "loaded" in result.output.lower() and "memory" in result.output.lower()


def test_infer_run_help_exposes_advanced_sampling_options(plain):
    result = CliRunner().invoke(
        app,
        ["--no-banner", "infer", "run", "--help"],
        env={"COLUMNS": "140", "TERMINAL_WIDTH": "140"},
        terminal_width=140,
    )
    assert result.exit_code == 0, plain(result.output)
    run_command = get_command(app).commands["infer"].commands["run"]
    option_flags = {
        option
        for parameter in run_command.params
        for option in (*getattr(parameter, "opts", ()), *getattr(parameter, "secondary_opts", ()))
    }
    for option in ("--top-p", "--top-k", "--min-p", "--repeat-penalty", "--seed", "--num-ctx", "--stop"):
        assert option in option_flags
    for option in ("--top-p", "--top-k", "--min-p", "--repeat-penalty", "--seed", "--num-ctx", "--stop"):
        assert option in plain(result.output)


def test_infer_run_forwards_advanced_sampling_options_to_local_runtime(monkeypatch):
    observed = {}

    def fake_run(model, prompt, **options):
        observed.update(model=model, prompt=prompt, **options)
        return {"response": "local test response", "execution": "local"}

    monkeypatch.setattr("axiom.cli.inference.run_local_inference", fake_run)
    result = CliRunner().invoke(app, [
        "--no-banner", "infer", "run", "qwen2.5:0.5b",
        "--prompt", "Keep this local.", "--top-p", "0.82", "--top-k", "32", "--min-p", "0.2",
        "--repeat-penalty", "1.15", "--seed", "17", "--num-ctx", "4096",
        "--stop", "END_OF_TEXT", "--json",
    ])
    assert result.exit_code == 0, result.output
    assert observed["model"] == "qwen2.5:0.5b"
    assert observed["prompt"] == "Keep this local."
    assert observed["top_p"] == 0.82
    assert observed["top_k"] == 32
    assert observed["min_p"] == 0.2
    assert observed["repetition_penalty"] == 1.15
    assert observed["seed"] == 17
    assert observed["num_ctx"] == 4096
    assert observed["stop_sequences"] == ["END_OF_TEXT"]


def test_infer_run_supports_json_schema_constrained_local_output(monkeypatch, tmp_path):
    schema_file = tmp_path / "response.schema.json"
    schema_file.write_text(
        '{"type":"object","properties":{"answer":{"type":"string"},"confidence":{"type":"number","minimum":0,"maximum":1}},"required":["answer","confidence"]}',
        encoding="utf-8",
    )
    observed = {}

    def fake_run(model, prompt, **options):
        observed.update(model=model, prompt=prompt, **options)
        return {
            "model": model, "response": '{"answer":"local only","confidence":0.9}',
            "done": True, "prompt_eval_count": 4, "eval_count": 8,
            "total_duration_ms": 20, "generated_at": "2026-11-21T00:00:00Z",
        }

    monkeypatch.setattr("axiom.cli.inference.run_local_inference", fake_run)
    result = CliRunner().invoke(app, [
        "--no-banner", "infer", "run", "qwen2.5:0.5b",
        "--prompt", "Return a structured answer.",
        "--schema-file", str(schema_file), "--json",
    ])
    assert result.exit_code == 0, result.output
    assert observed["response_format"]["type"] == "object"
    assert observed["response_format"]["additionalProperties"] is False
    payload = __import__("json").loads(result.output)
    assert payload["structured_output"] == {"answer": "local only", "confidence": 0.9}
    assert payload["structured_output_validated"] is True


def test_infer_run_rejects_remote_reference_schemas_before_generation(monkeypatch, tmp_path):
    schema_file = tmp_path / "unsafe.schema.json"
    schema_file.write_text(
        '{"type":"object","$ref":"https://example.invalid/schema"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "axiom.cli.inference.run_local_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runtime must not run")),
    )
    result = CliRunner().invoke(app, [
        "--no-banner", "infer", "run", "qwen2.5:0.5b",
        "--prompt", "Return JSON.", "--schema-file", str(schema_file), "--json",
    ])
    assert result.exit_code != 0
    assert "Unsupported JSON Schema keyword" in result.output


def test_infer_run_rejects_symlinked_schema_files(monkeypatch, tmp_path):
    target = tmp_path / "schema.json"
    target.write_text('{"type":"string"}', encoding="utf-8")
    linked = tmp_path / "linked-schema.json"
    linked.symlink_to(target)
    monkeypatch.setattr(
        "axiom.cli.inference.run_local_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runtime must not run")),
    )
    result = CliRunner().invoke(app, [
        "--no-banner", "infer", "run", "qwen2.5:0.5b",
        "--prompt", "Return JSON.", "--schema-file", str(linked), "--json",
    ])
    assert result.exit_code != 0
