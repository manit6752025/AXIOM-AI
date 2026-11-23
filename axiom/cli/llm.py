"""CLI commands for the unified local/cloud LLM client."""
from __future__ import annotations

import json

import typer

from axiom.cli.errors import emit_error
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axiom.cli.interactive import prompt as safe_prompt
from axiom.core import llm_client
from axiom.core.llm_profiles import LLMProfileError, LLMProfileStore
from axiom.core.secret_refs import SecretReferenceError, normalize_secret_reference, resolve_secret_reference

console = Console()
llm_app = typer.Typer(help="Use the AXIOM local engine or a supported cloud LLM provider.", no_args_is_help=True)


def _fail(message: str) -> None:
    emit_error(message, code="llm_error")


def _credential(provider: str, env_name: str | None) -> str | None:
    if provider in {"axiom-local", "ollama-local"}:
        return None
    reference = env_name or "AXIOM_LLM_API_KEY"
    try:
        normalized_reference = normalize_secret_reference(reference)
    except SecretReferenceError:
        _fail("API-key environment variable reference must be a valid variable name.")
        return None
    try:
        return resolve_secret_reference(normalized_reference)
    except SecretReferenceError:
        return safe_prompt(
            f"{provider} API key (used once, never saved by AXIOM)",
            hide_input=True,
            non_interactive_hint=f"Set {reference} or pass --api-key-env <ENVIRONMENT_VARIABLE>.",
        )


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


profile_app = typer.Typer(help="Manage workspace-local provider profiles without saving credentials.", no_args_is_help=True)
llm_app.add_typer(profile_app, name="profile")


@profile_app.command("list")
def profile_list(
    json_output: bool = typer.Option(False, "--json", help="Print profile metadata as JSON."),
) -> None:
    """List workspace-local provider profiles; no credentials are included."""
    try:
        result = LLMProfileStore().list()
    except LLMProfileError as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    table = Table(title="Saved AXIOM LLM provider profiles")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Credential reference")
    table.add_column("Profile ID")
    for item in result["profiles"]:
        table.add_row(item["name"], item["provider"], item["model"], item.get("api_key_env") or "request-only", item["id"])
    console.print(table)
    console.print("Only environment-variable names are stored; credential values are resolved for the outgoing request and never displayed.")


@profile_app.command("save")
def profile_save(
    name: str = typer.Argument(..., help="Unique profile name (1–48 safe characters)."),
    provider: str = typer.Option("axiom-local", "--provider", help="Provider ID from axiom llm providers."),
    model: str = typer.Option(..., "--model", help="Exact provider model ID."),
    base_url: str = typer.Option("", "--base-url", help="Public HTTPS /v1 endpoint for openai-compatible only."),
    api_key_env: str = typer.Option("", "--api-key-env", help="Optional environment-variable name holding the cloud API key; only the name is stored."),
    json_output: bool = typer.Option(False, "--json", help="Print the saved profile as JSON."),
) -> None:
    """Create or update a workspace profile without storing provider credentials."""
    try:
        profile = LLMProfileStore().save(
            name=name,
            provider=provider,
            model=model,
            base_url=base_url or None,
            api_key_env=api_key_env or None,
        )
    except LLMProfileError as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps({"profile": profile, "credentials_persisted": False}, ensure_ascii=False, indent=2))
        return
    console.print(f"Saved provider profile [cyan]{profile['name']}[/cyan] ({profile['provider']} · {profile['model']}).")
    console.print("API keys, prompts and conversation history were not stored.")


@profile_app.command("delete")
def profile_delete(
    profile_id: str = typer.Argument(..., help="Exact profile ID from axiom llm profile list."),
    yes: bool = typer.Option(False, "--yes", help="Confirm deletion without an interactive prompt."),
    json_output: bool = typer.Option(False, "--json", help="Print the deletion result as JSON."),
) -> None:
    """Delete one saved provider profile; provider-side data is not affected."""
    if not yes and not typer.confirm("Delete this local provider profile?"):
        raise typer.Exit(0)
    try:
        deleted = LLMProfileStore().delete(profile_id)
    except LLMProfileError as exc:
        _fail(str(exc))
        return
    if deleted is None:
        _fail("Provider profile not found.")
        return
    if json_output:
        typer.echo(json.dumps({"deleted": deleted, "credentials_persisted": False}, ensure_ascii=False, indent=2))
        return
    console.print(f"Deleted provider profile [cyan]{deleted['name']}[/cyan].")


@profile_app.command("connect")
def profile_connect(
    profile_id: str = typer.Argument(..., help="Exact profile ID from axiom llm profile list."),
    api_key_env: str = typer.Option("", "--api-key-env", help="Use this variable instead of the saved reference; defaults to AXIOM_LLM_API_KEY."),
    json_output: bool = typer.Option(False, "--json", help="Print the catalogue verification result as JSON."),
) -> None:
    """Verify a provider/model catalogue using a request-only key or saved environment reference."""
    try:
        profile = LLMProfileStore().get(profile_id)
    except LLMProfileError as exc:
        _fail(str(exc))
        return
    if profile is None:
        _fail("Provider profile not found.")
        return
    reference = profile.get("api_key_env") or api_key_env or "AXIOM_LLM_API_KEY"
    key = _credential(profile["provider"], reference)
    try:
        result = llm_client.list_models(profile["provider"], key, profile["base_url"] or None)
    except (ValueError, llm_client.LLMClientError) as exc:
        _fail(str(exc))
        return
    verification = {
        "profile_id": profile["id"],
        "provider": profile["provider"],
        "status": "verified",
        "models": result.get("models", []),
        "connection_scope": "catalogue_check_only",
        "persistent_connection": False,
        "credential_storage": result.get("credential_storage", "not_saved" if key else "not_applicable"),
    }
    if json_output:
        typer.echo(json.dumps(verification, ensure_ascii=False, indent=2))
        return
    console.print(f"Verified [green]{profile['provider']}[/green] provider catalogue.")
    console.print(f"{len(verification['models'])} model ID(s) returned. No connection or credential was retained.")


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
