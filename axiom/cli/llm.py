"""CLI commands for the unified local/cloud LLM client."""
from __future__ import annotations

import json
import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axiom.cli.interactive import prompt as safe_prompt
from axiom.core import llm_client

console = Console()
llm_app = typer.Typer(help="Use the AXIOM local engine or a supported cloud LLM provider.", no_args_is_help=True)


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1)


def _credential(provider: str, env_name: str) -> str | None:
    if provider in {"axiom-local", "ollama-local"}:
        return None
    if not env_name or len(env_name) > 128:
        _fail("API-key environment variable name must be between 1 and 128 characters.")
    value = os.environ.get(env_name, "")
    if value.strip():
        return value
    return safe_prompt(f"{provider} API key (used once, never saved by AXIOM)", hide_input=True, non_interactive_hint=f"Set {env_name} or pass --api-key-env <ENVIRONMENT_VARIABLE>.")


@llm_app.command("providers")
def providers(json_output: bool = typer.Option(False, "--json", help="Print machine-readable provider metadata.")) -> None:
    """Show available local and cloud provider adapters without making network calls."""
    result = llm_client.provider_catalog()
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    table = Table(title="AXIOM unified LLM providers")
    table.add_column("Provider")
    table.add_column("Type")
    table.add_column("API key")
    table.add_column("Privacy")
    for item in result["providers"]:
        table.add_row(
            item["id"],
            item["kind"],
            "not required" if not item["requires_api_key"] else "request only",
            item["privacy"],
        )
    console.print(table)


@llm_app.command("models")
def models(
    provider: str = typer.Option("axiom-local", "--provider", help="axiom-local, ollama-local, openai, anthropic, gemini, deepseek, moonshot, openrouter, or openai-compatible."),
    base_url: str = typer.Option("", "--base-url", help="Required only for openai-compatible."),
    api_key_env: str = typer.Option("AXIOM_LLM_API_KEY", "--api-key-env", help="Environment variable to read the API key from."),
    json_output: bool = typer.Option(False, "--json", help="Print model IDs as JSON."),
) -> None:
    """List compatible local models or fetch a selected provider's model catalogue."""
    key = _credential(provider, api_key_env)
    try:
        result = llm_client.list_models(provider, key, base_url or None)
    except (ValueError, llm_client.LLMClientError) as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    table = Table(title=f"Models · {provider}")
    table.add_column("Model ID")
    for name in result.get("models", []):
        table.add_row(str(name))
    console.print(table)
    console.print(result.get("note", "Local inventory only; no model weights were downloaded."))


@llm_app.command("chat")
def chat(
    model: str = typer.Argument(..., help="Exact model ID from axiom llm models."),
    prompt: str = typer.Argument(..., help="Text sent to the selected model."),
    provider: str = typer.Option("axiom-local", "--provider", help="Select local or cloud provider."),
    system: str = typer.Option("", "--system", help="Optional system instruction."),
    base_url: str = typer.Option("", "--base-url", help="Required only for openai-compatible."),
    api_key_env: str = typer.Option("AXIOM_LLM_API_KEY", "--api-key-env", help="Environment variable to read the API key from."),
    max_tokens: int = typer.Option(256, "--max-tokens", min=1, max=2048),
    temperature: float = typer.Option(0.7, "--temperature", min=0, max=2),
    top_p: float = typer.Option(0.9, "--top-p", min=0.000001, max=1, help="Local nucleus sampling threshold."),
    top_k: int = typer.Option(0, "--top-k", min=0, max=100_000, help="Local candidate token limit; 0 disables the limit."),
    min_p: float = typer.Option(0.0, "--min-p", min=0, max=1, help="Local relative-probability cutoff; 0 disables min-p."),
    repetition_penalty: float = typer.Option(1.0, "--repeat-penalty", min=0.1, max=5, help="Local repetition penalty; 1.0 disables it."),
    seed: int | None = typer.Option(None, "--seed", min=0, max=9_223_372_036_854_775_807, help="Optional local random seed."),
    stop_sequences: list[str] = typer.Option([], "--stop", help="Local stop text; repeat for multiple stop sequences."),
    send_to_provider: bool = typer.Option(False, "--send-to-provider", help="Required confirmation before a cloud provider receives the conversation."),
    device: str = typer.Option("auto", "--device", help="Local device: auto, cpu, cuda, rocm, mps, xpu."),
    memory_mode: str = typer.Option("auto", "--memory-mode", help="Local memory mode: auto, full, layer_streaming, disk_streaming."),
    json_output: bool = typer.Option(False, "--json", help="Print the completion and usage metadata as JSON."),
) -> None:
    """Generate a response locally or through an explicitly selected cloud provider."""
    if not prompt.strip() or len(prompt) > 12_000:
        _fail("Prompt must contain 1–12,000 characters.")
    remote = provider not in {"axiom-local", "ollama-local"}
    if remote and not send_to_provider:
        _fail("Cloud use requires --send-to-provider because your prompt will be sent to the selected provider.")
    key = _credential(provider, api_key_env)
    messages = ([{"role": "system", "content": system}] if system.strip() else []) + [{"role": "user", "content": prompt}]
    try:
        result = llm_client.chat(
            provider,
            model,
            messages,
            api_key=key,
            base_url=base_url or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            consent=send_to_provider,
            device=device,
            memory_mode=memory_mode,
        )
    except (ValueError, llm_client.LLMClientError) as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    console.print(Panel(result["response"], title=f"{provider} · {model}"))
    usage = f"Prompt tokens: {result.get('prompt_tokens', 'unknown')} · Completion tokens: {result.get('completion_tokens', 'unknown')}"
    console.print(usage)
    console.print("Execution: local" if result["execution"] == "local" else "Execution: remote; credentials were not saved.")
