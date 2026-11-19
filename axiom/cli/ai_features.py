from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axiom.api.huggingface import search_models
from axiom.api.optimization import plan_optimization
from axiom.api.policy_audit import audit_model

ai_app = typer.Typer(
    help="AI optimization, Hugging Face discovery, and model auditing."
)

console = Console()


@ai_app.command("plan")
def plan(
    model: str = typer.Option(..., "--model"),
    agent_type: str = typer.Option("general-agent", "--type"),
    use_case: str = typer.Option("assistant", "--use-case"),
    privacy: str = typer.Option("local", "--privacy"),
    latency: str = typer.Option("low", "--latency"),
    target_tps: float = typer.Option(10.0, "--target-tps"),
    quantization: str = typer.Option("auto", "--quantization"),
):
    """Build a hardware-aware model optimization plan."""
    result = plan_optimization(
        model=model,
        agent_type=agent_type,
        use_case=use_case,
        privacy=privacy,
        latency=latency,
        target_tokens_per_second=target_tps,
        quantization=quantization,
    )

    table = Table(title="AXIOM Agent Model Plan")
    table.add_column("Setting")
    table.add_column("Value")

    rows = [
        ("Model", result.model),
        ("Agent", result.agent_type),
        ("Use case", result.use_case),
        ("Privacy", result.privacy),
        ("Latency", result.latency),
        ("Target", f"{result.target_tokens_per_second:g} tok/s"),
        ("Parameters", f"{result.estimated_parameters_b:.2f}B"),
        ("Memory", f"{result.estimated_memory_gb:.2f} GB"),
        ("Format", result.recommended_format),
        ("Quantization", result.recommended_quantization),
        ("Sequence", str(result.sequence_length)),
        ("Memory fit", "PASS" if result.fits_estimate else "TIGHT"),
    ]

    for key, value in rows:
        table.add_row(key, value)

    console.print(table)

    for note in result.notes:
        console.print(f"• {note}")


@ai_app.command("hf-search")
def hf_search(
    query: str,
    limit: int = typer.Option(10, "--limit"),
):
    """Search Hugging Face models."""
    try:
        results = search_models(query, limit)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(str(exc))

    table = Table(title=f"Hugging Face: {query}")
    table.add_column("Model")
    table.add_column("Downloads")
    table.add_column("Likes")
    table.add_column("Task")

    for row in results:
        table.add_row(
            str(row.get("id") or ""),
            str(row.get("downloads") or "-"),
            str(row.get("likes") or "-"),
            str(row.get("pipeline_tag") or "-"),
        )

    console.print(table)


@ai_app.command("policy-audit")
def policy_audit(model_path: str):
    """Audit model policy/safety indicators without modifying it."""
    try:
        result = audit_model(model_path)
    except FileNotFoundError:
        raise typer.BadParameter(f"Model path does not exist: {model_path}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    console.print(
        Panel(
            json.dumps(result, indent=2),
            title="AXIOM Model Policy Audit",
        )
    )

    console.print(
        "[yellow]Audit only:[/yellow] "
        "AXIOM does not remove or bypass model safety controls."
    )
