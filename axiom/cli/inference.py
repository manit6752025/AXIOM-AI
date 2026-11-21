"""CLI commands for actual local model inference through Ollama."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time

import typer

from axiom.cli.errors import emit_error

from axiom.cli.interactive import confirm as safe_confirm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axiom.core.inference import InferenceRuntimeError, inspect_local_model, list_running_local_models, remove_local_model, run_local_inference, unload_local_model
from axiom.core.json_schema import MAX_SCHEMA_BYTES, JsonSchemaError, parse_and_validate_json_output, validate_json_schema
from axiom.core.ollama_service import (
    OllamaServiceError, cancel_model_pull, get_model_pull, service_status,
    start_model_pull, start_service, stop_service,
)

infer_app = typer.Typer(
    help="Manage and run local Ollama inference.",
    no_args_is_help=True,
    # Click 8.5 added first-class argument help. Use Click's renderer for this
    # sub-app until Typer's Rich renderer handles that new positional section.
    rich_markup_mode=None,
)
service_app = typer.Typer(help="Manage the loopback Ollama service.", no_args_is_help=True)
infer_app.add_typer(service_app, name="service")
console = Console()


def _fail(message: str) -> None:
    emit_error(message, code="inference_error")


def _load_json_schema_file(path: Path) -> dict:
    """Read a small regular schema file without following symlinks where supported."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if path.is_symlink():
        raise JsonSchemaError("JSON Schema path may not be a symlink.")
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise JsonSchemaError("JSON Schema path must be a regular file.")
        if info.st_size > MAX_SCHEMA_BYTES:
            raise JsonSchemaError(f"JSON Schema file exceeds the {MAX_SCHEMA_BYTES}-byte limit.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(MAX_SCHEMA_BYTES + 1)
        if len(raw) > MAX_SCHEMA_BYTES:
            raise JsonSchemaError(f"JSON Schema file exceeds the {MAX_SCHEMA_BYTES}-byte limit.")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise JsonSchemaError("JSON Schema file must contain valid UTF-8 JSON.") from None
        return validate_json_schema(parsed)
    finally:
        os.close(descriptor)


@infer_app.command("status")
def infer_status(json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON.")) -> None:
    """Check whether the local Ollama API responds and list loaded model names."""
    report = service_status()
    if json_output:
        console.print_json(json.dumps(report, ensure_ascii=False))
        return
    table = Table(title="Local inference runtime")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Backend", "Ollama")
    table.add_row("Endpoint", report["endpoint"])
    table.add_row("Executable on PATH", "yes" if report["runtime_installed"] else "no")
    table.add_row("API status", str(report["status"]).upper())
    table.add_row("Local models", str(len(report["models"])))
    console.print(table)
    for name in report["models"]:
        console.print(f"  • {name}")
    if report["error"]:
        console.print(f"[yellow]{report['error']}[/yellow]")
        raise typer.Exit(code=1)


@service_app.command("status")
def infer_service_status(json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON.")) -> None:
    """Report whether Ollama is online and whether AXIOM owns its service process."""
    report = service_status()
    if json_output:
        console.print_json(json.dumps(report, ensure_ascii=False))
        return
    table = Table(title="Ollama local service")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("API status", str(report["status"]).upper())
    table.add_row("Endpoint", str(report["endpoint"]))
    table.add_row("Executable", "detected" if report["runtime_installed"] else "not found")
    table.add_row("Managed by AXIOM", "yes" if report["service_managed_by_axiom"] else "no")
    table.add_row("AXIOM can start", "yes" if report["service_can_start"] else "no")
    table.add_row("AXIOM can stop", "yes" if report["service_can_stop"] else "no")
    console.print(table)
    console.print(report["service_message"])
    if report["status"] != "online":
        raise typer.Exit(code=1)


@service_app.command("start")
def infer_service_start() -> None:
    """Start Ollama as a detached, loopback-only AXIOM-managed service."""
    try:
        result = start_service()
    except OllamaServiceError as exc:
        _fail(str(exc))
    console.print("Ollama is online at 127.0.0.1:11434." if result["service"]["status"] == "online" else "Ollama start requested.")


@service_app.command("stop")
def infer_service_stop() -> None:
    """Stop only the Ollama process started and verified by AXIOM."""
    try:
        stop_service()
    except OllamaServiceError as exc:
        _fail(str(exc))
    console.print("AXIOM-managed Ollama service stopped.")


@infer_app.command("pull")
def infer_pull(
    model: str = typer.Argument(..., help="Ollama model tag to download after explicitly invoking this command."),
    json_output: bool = typer.Option(False, "--json", help="Print the final job report as JSON."),
) -> None:
    """Download a local Ollama model tag and print progress; this requires network access."""
    try:
        job = start_model_pull(model)
    except (ValueError, OllamaServiceError) as exc:
        _fail(str(exc))
    model_name = job["model"]
    console.print(f"Pulling [bold]{model_name}[/bold] from the Ollama library…")
    shown = 0
    try:
        while job["status"] in {"queued", "running"}:
            for line in job["logs"][shown:]:
                console.print(line)
            shown = len(job["logs"])
            time.sleep(0.35)
            job = get_model_pull(job["id"])
    except KeyboardInterrupt:
        try:
            cancel_model_pull(job["id"])
        finally:
            raise typer.Exit(code=130)
    if json_output:
        console.print_json(json.dumps(job, ensure_ascii=False))
    elif job["status"] == "completed":
        console.print(f"[green]Downloaded {model_name}.[/green]")
    else:
        pull_status = job["status"]
        _fail(job.get("error") or f"Model pull {pull_status}.")


@infer_app.command("ps")
def infer_ps(json_output: bool = typer.Option(False, "--json", help="Print the loaded-model inventory as JSON.")) -> None:
    """List models currently loaded by Ollama and memory sizes reported by the runtime."""
    try:
        report = list_running_local_models()
    except InferenceRuntimeError as exc:
        _fail(str(exc))
    if json_output:
        console.print_json(json.dumps(report, ensure_ascii=False))
        return
    table = Table(title="Ollama loaded models · point-in-time")
    table.add_column("Model", style="cyan")
    table.add_column("Placement")
    table.add_column("Loaded size")
    table.add_column("VRAM")
    table.add_column("Family / quantization")
    for item in report["models"]:
        def gib(value: int | None) -> str:
            return "unknown" if value is None else f"{value / (1024**3):.2f} GiB"
        table.add_row(item["name"], item["memory_placement"], gib(item["size_bytes"]), gib(item["size_vram_bytes"]), " · ".join(value for value in (item["family"], item["quantization_level"]) if value) or "unknown")
    console.print(table)
    console.print(report["scope_note"])


@infer_app.command("inspect")
def infer_inspect(
    model: str = typer.Argument(..., help="Exact model tag already installed in local Ollama."),
    json_output: bool = typer.Option(False, "--json", help="Print model metadata as JSON."),
) -> None:
    """Inspect local model family, format, parameter size, quantization and context metadata."""
    try:
        report = inspect_local_model(model)
    except (ValueError, InferenceRuntimeError) as exc:
        _fail(str(exc))
    if json_output:
        console.print_json(json.dumps(report, ensure_ascii=False))
        return
    model_name = report["model"]
    table = Table(title=f"Local model · {model_name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, label in (("format", "Format"), ("family", "Family"), ("parameter_size", "Parameter size"), ("quantization_level", "Quantization"), ("context_length", "Context length"), ("size_bytes", "Disk size (bytes)"), ("capabilities", "Capabilities")):
        value = report.get(key)
        table.add_row(label, str(value) if value not in (None, []) else "Not reported")
    console.print(table)
    console.print(report["scope_note"])


@infer_app.command("remove")
def infer_remove(
    model: str = typer.Argument(..., help="Exact installed Ollama model tag to remove from local storage."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm permanent removal without prompting."),
    json_output: bool = typer.Option(False, "--json", help="Print the removal report as JSON."),
) -> None:
    """Remove an installed Ollama model tag. This deletes local model weights for that tag."""
    if not yes and not safe_confirm(f"Permanently remove the downloaded Ollama model tag {model!r}?", default=False):
        console.print("Model removal cancelled.")
        raise typer.Exit(code=0)
    try:
        report = remove_local_model(model)
    except (ValueError, InferenceRuntimeError) as exc:
        _fail(str(exc))
    if json_output:
        console.print_json(json.dumps(report, ensure_ascii=False))
        return
    console.print(report["scope_note"] if report["removed"] else "Ollama did not confirm model removal.")
    if not report["removed"]:
        raise typer.Exit(code=1)


@infer_app.command("unload")
def infer_unload(model: str = typer.Argument(..., help="Installed local model to unload from memory without deleting its files.")) -> None:
    """Ask Ollama to unload a model from RAM/VRAM without deleting its download."""
    try:
        result = unload_local_model(model)
    except (ValueError, InferenceRuntimeError) as exc:
        _fail(str(exc))
    console.print(result["scope_note"])


@infer_app.command("models")
def infer_models(json_output: bool = typer.Option(False, "--json", help="Print model names as JSON.")) -> None:
    """List models registered with the local Ollama service."""
    report = service_status()
    if report["status"] != "online":
        _fail(report["error"] or "The local Ollama runtime is offline.")
    if json_output:
        console.print_json(json.dumps({"models": report["models"]}, ensure_ascii=False))
        return
    if not report["models"]:
        console.print("No local Ollama models found. Pull one, for example: ollama pull qwen2.5:0.5b")
        return
    for name in report["models"]:
        console.print(name)


@infer_app.command(
    "run",
    epilog="Advanced sampling options: --top-p, --top-k, --min-p, --repeat-penalty, --seed, --num-ctx, and repeatable --stop.",
)
def infer_run(
    model: str = typer.Argument(..., help="Name of an already-downloaded local Ollama model."),
    prompt: str | None = typer.Option(None, "--prompt", help="Prompt text (up to 12,000 characters)."),
    prompt_file: Path | None = typer.Option(None, "--prompt-file", exists=True, dir_okay=False, readable=True, help="Read the prompt from a local UTF-8 text file."),
    num_predict: int = typer.Option(256, "--num-predict", min=1, max=2_048, help="Maximum generated tokens (1–2048)."),
    temperature: float = typer.Option(0.7, min=0, max=2, help="Sampling temperature (0–2)."),
    top_p: float | None = typer.Option(None, "--top-p", min=0.001, max=1, help="Nucleus sampling threshold (0–1)."),
    top_k: int | None = typer.Option(None, "--top-k", min=0, max=100_000, help="Candidate token limit (0 uses runtime default)."),
    min_p: float | None = typer.Option(None, "--min-p", min=0, max=1, help="Relative-probability cutoff (0 disables min-p)."),
    repetition_penalty: float | None = typer.Option(None, "--repeat-penalty", min=0.1, max=5, help="Repetition penalty (0.1–5)."),
    seed: int | None = typer.Option(None, "--seed", min=0, help="Optional deterministic seed."),
    num_ctx: int | None = typer.Option(None, "--num-ctx", min=512, max=131_072, help="Optional context window (512–131072 tokens)."),
    stop_sequences: list[str] | None = typer.Option(None, "--stop", help="Optional stop sequence; repeat the flag for multiple values."),
    schema_file: Path | None = typer.Option(None, "--schema-file", exists=True, dir_okay=False, readable=True, help="Apply a local, bounded JSON Schema to constrain and validate output."),
    json_output: bool = typer.Option(False, "--json", help="Print the complete result as JSON."),
) -> None:
    """Run real text generation on the local Ollama runtime; no cloud API is used."""
    if (prompt is None) == (prompt_file is None):
        _fail("Supply exactly one of --prompt or --prompt-file.")
    if prompt_file is not None:
        try:
            if prompt_file.stat().st_size > 48_000:
                _fail("Prompt file is too large; maximum UTF-8 file size is 48 KB.")
            prompt = prompt_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            _fail(f"Could not read prompt file ({type(exc).__name__}).")
    assert prompt is not None
    try:
        sampling: dict[str, object] = {}
        if top_p is not None: sampling["top_p"] = top_p
        if top_k is not None: sampling["top_k"] = top_k
        if min_p is not None: sampling["min_p"] = min_p
        if repetition_penalty is not None: sampling["repetition_penalty"] = repetition_penalty
        if seed is not None: sampling["seed"] = seed
        if num_ctx is not None: sampling["num_ctx"] = num_ctx
        if stop_sequences: sampling["stop_sequences"] = stop_sequences
        response_schema = _load_json_schema_file(schema_file) if schema_file is not None else None
        if response_schema is not None:
            if num_predict > 512:
                _fail("Structured output is limited to 512 generated tokens; reduce --num-predict.")
            sampling["response_format"] = response_schema
        result = run_local_inference(model, prompt, num_predict=num_predict, temperature=temperature, **sampling)
        if response_schema is not None:
            result["structured_output"] = parse_and_validate_json_output(result["response"], response_schema)
            result["structured_output_validated"] = True
    except (ValueError, InferenceRuntimeError) as exc:
        _fail(str(exc))
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return
    metrics = []
    if result["eval_count"] is not None:
        metrics.append(f"{result['eval_count']} generated tokens")
    if result["total_duration_ms"] is not None:
        metrics.append(f"{result['total_duration_ms']} ms total runtime")
    console.print(Panel(result["response"] or "(The model returned an empty response.)", title=f"{result['model']} · local inference"))
    if metrics:
        console.print(" · ".join(metrics))
