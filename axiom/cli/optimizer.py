from __future__ import annotations

from pathlib import Path

import typer

from axiom.cli.interactive import confirm as safe_confirm, prompt as safe_prompt
from rich.console import Console
from rich.table import Table

from axiom.optimizer.agent_profiles import get_profile, menu
from axiom.optimizer.model import (
    choose_quantization,
    create_runtime_bundle,
    inspect_model,
    write_runtime_profile,
)
from axiom.optimizer.system import inspect_system

optimizer_app = typer.Typer(
    help="Optimize a large model for a specific agent workload and device."
)

console = Console()


@optimizer_app.command("run")
def run(
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Local model path or Hugging Face model ID.",
    ),
    profile: int = typer.Option(
        0,
        "--profile",
        "-p",
        min=0,
        max=8,
        help="Agent profile 1-8. 0 opens the questionnaire.",
    ),
    target_tps: float = typer.Option(
        10.0,
        "--target-tps",
        min=1.0,
        max=1000.0,
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Optimized runtime bundle directory.",
    ),
):
    """Questionnaire → analyze → optimize → bundle → benchmark."""

    if profile == 0:
        console.print("\n[bold cyan]AXIOM AGENT OPTIMIZER[/bold cyan]\n")
        console.print(menu())
        profile = safe_prompt(
            "\nChoose agent profile (1-8)",
            type=int,
            non_interactive_hint="Supply --profile 1 through 8 to select an agent profile.",
        )

    selected = get_profile(profile)

    if not model:
        model = safe_prompt("\nHugging Face model ID or local model path", non_interactive_hint="Supply --model with a local path or Hugging Face repository ID.")

    if not output:
        safe_name = model.replace("/", "__").replace("\\", "__")
        output = str(Path(".axiom") / "optimized" / safe_name / selected.key)

    if (
        safe_confirm(
            f"\nTarget {target_tps:g} tok/s. Keep this target?",
            default=True,
        )
        is False
    ):
        target_tps = safe_prompt(
            "Target tokens/sec",
            type=float,
            default=10.0,
        )

    system = inspect_system()

    console.print("\n[bold]SYSTEM[/bold]")
    table = Table()
    table.add_column("Property")
    table.add_column("Value")

    for key, value in system.to_dict().items():
        table.add_row(str(key), str(value))

    console.print(table)

    # HF IDs are downloaded through the existing HF dependency.
    if "/" in model and not Path(model).exists():
        try:
            from huggingface_hub import snapshot_download

            console.print(f"\n[cyan]Downloading model snapshot:[/cyan] {model}")

            local = snapshot_download(
                repo_id=model,
                local_dir=str(Path(".axiom") / "models" / model.replace("/", "__")),
            )
            model = str(local)

            console.print(f"[green]✓ Model available at {model}[/green]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] Could not download Hugging Face model: {exc}")
            raise typer.Exit(code=1) from exc

    info = inspect_model(model)

    console.print("\n[bold]MODEL[/bold]")
    model_table = Table()
    model_table.add_column("Property")
    model_table.add_column("Value")

    for key, value in info.to_dict().items():
        if key == "weights":
            value = ", ".join(value) if value else "-"
        elif value is None:
            value = "Unknown"
        model_table.add_row(key, str(value))

    console.print(model_table)

    quant = choose_quantization(info, system, selected)

    console.print(f"\n[bold green]Recommended quantization:[/bold green] {quant}")

    _ = create_runtime_bundle(model, output)

    write_runtime_profile(
        output,
        selected,
        quant,
        system,
    )

    typer.echo()
    typer.echo(f"Runtime bundle: {output}")
    typer.echo(
        "Non-runtime repository artifacts were omitted; model weights and "
        "inference-critical configuration were retained."
    )

    if Path(output).suffix.lower() == ".gguf":
        try:
            from axiom.optimizer.runner import benchmark_gguf

            result = benchmark_gguf(
                output,
                f"Act as a {selected.name}. Respond concisely.",
                target_tps,
            )

            console.print(
                f"\nMeasured throughput: {result.tokens_per_second:.2f} tok/s"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[yellow]Benchmark not run:[/yellow] {exc}")
    else:
        console.print(
            "\n[yellow]10 tok/s is a target.[/yellow] "
            "Install a supported local runtime such as llama.cpp and "
            "benchmark the resulting model before claiming measured 10 tok/s."
        )

    console.print(
        f"\n[bold cyan]AXIOM optimizer complete for profile "
        f"{selected.number}: {selected.name}[/bold cyan]"
    )


@optimizer_app.command("profiles")
def profiles():
    """List the 8 AXIOM agent optimization profiles."""
    console.print(menu())
