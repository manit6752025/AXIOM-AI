"""CLI commands for AXIOM's direct local model engine."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from axiom.core.engine import AxiomEngineError, benchmark_local, discover_local_engine_assets, estimate_native_quantization, generate_local, generate_local_batch, generate_local_stream, runtime_status, runtime_telemetry
from axiom.core import model_studio
from axiom.core.engine_fabric import EngineFabricError, discover_engine_backends, run_engine_backend_benchmark, run_engine_backend_chat, run_engine_fabric_chat

engine_app = typer.Typer(help="Run causal language models directly through AXIOM's local engine.", no_args_is_help=True)
console = Console()


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1)


@engine_app.command("discover")
def engine_discover(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Discover native-compatible local models and LoRA adapters below trusted roots."""
    report = discover_local_engine_assets()
    if json_output:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        return

    models = report["models"]
    adapters = report["adapters"]
    if models:
        table = Table(title=f"Native models ({len(models)})")
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Architecture")
        table.add_column("Checkpoint")
        table.add_column("Path")
        for model in models:
            table.add_row(
                model["name"], model["model_type"], model["architecture"],
                model["weight_format"], model["path"],
            )
        console.print(table)
    else:
        console.print("[yellow]No compatible native models found under trusted roots.[/yellow]")

    if adapters:
        table = Table(title=f"LoRA adapters ({len(adapters)})")
        table.add_column("Name", style="cyan")
        table.add_column("Format")
        table.add_column("Path")
        for adapter in adapters:
            table.add_row(adapter["name"], adapter["weight_format"], adapter["path"])
        console.print(table)
    elif not models:
        console.print("No native-compatible LoRA adapters found.")

    if report["truncated"]:
        console.print("[yellow]The inventory reached its result cap. Narrow AXIOM_MODEL_ROOTS to inspect a smaller set.[/yellow]")
    console.print(report["scope_note"])


@engine_app.command("status")
def engine_status(json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON.")) -> None:
    """Check AXIOM Engine dependencies and loaded-model state."""
    report = runtime_status()
    if json_output:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        table = Table(title="AXIOM Engine")
        table.add_column("Dependency / setting", style="cyan")
        table.add_column("Value")
        table.add_row("Ready", "yes" if report["ready"] else "no")
        table.add_row("Model loaded", "yes" if report["loaded"] else "no")
        table.add_row("Architecture", str(report["architecture"] or "not loaded"))
        for name, installed in report["packages"].items():
            table.add_row(name, f"{'available' if installed else 'not installed'} ({report['versions'].get(name) or 'unknown version'})")
        table.add_row("Recommended device", str(report.get("recommended_device") or "none available"))
        for device in report.get("devices", []):
            value = "available" if device.get("available") else "unavailable"
            if device.get("memory_total_gib") is not None:
                value += f" ({device.get('memory_free_gib') or 0} GiB free / {device['memory_total_gib']} GiB total)"
            table.add_row(f"Device: {device.get('name', device.get('id', 'unknown'))}", value)
        table.add_row("Native model pull", "public Hugging Face safetensors via `axiom engine pull`")
        table.add_row("Remote model code", "disabled")
        console.print(table)
        if not report["ready"]:
            console.print(f"[yellow]{report['setup_note']}[/yellow]")
    if not report["ready"]:
        raise typer.Exit(code=1)


@engine_app.command("estimate")
def engine_estimate(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Existing local checkpoint directory under an AXIOM trusted model root.",
    ),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        min=1,
        max=131072,
        help="Context tokens used for the estimate (1–131072).",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        min=1,
        max=16,
        help="Concurrent sequences included in the estimate (1–16).",
    ),
    ram_budget_gib: float | None = typer.Option(
        None,
        "--ram-budget-gib",
        min=0.1,
        max=2048,
        help="Available system-memory budget in GiB.",
    ),
    vram_budget_gib: float | None = typer.Option(
        None,
        "--vram-budget-gib",
        min=0.1,
        max=2048,
        help="Available accelerator-memory budget in GiB.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the complete estimate as JSON."),
) -> None:
    """Compare bounded model-memory estimates with RAM/VRAM budgets; no weights are loaded."""
    if ram_budget_gib is None and vram_budget_gib is None:
        _fail("Supply at least one memory budget: --ram-budget-gib and/or --vram-budget-gib.")
    try:
        report = estimate_native_quantization(
            str(model_path),
            context_length=context_length,
            batch_size=batch_size,
            ram_budget_gib=ram_budget_gib,
            vram_budget_gib=vram_budget_gib,
        )
    except (ValueError, AxiomEngineError) as exc:
        _fail(str(exc))
    if json_output:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        return

    table = Table(title=f"Estimated model memory · {report['model']['name']}")
    table.add_column("Format", style="cyan")
    table.add_column("Weights")
    table.add_column("KV cache")
    table.add_column("Prefill temp")
    table.add_column("Workspace")
    table.add_column("Total")
    table.add_column("Estimated range")
    table.add_column("RAM budget")
    table.add_column("VRAM budget")
    table.add_column("Native format")
    for candidate in report["candidates"]:
        low, high = candidate["estimated_range_gib"]

        def budget_result(value: bool | None) -> str:
            return "not set" if value is None else "fits" if value else "over budget"

        table.add_row(
            candidate["format"],
            f"{candidate['estimated_weight_gib']:.6f} GiB",
            f"{candidate['estimated_kv_cache_gib']:.6f} GiB",
            f"{candidate['estimated_kv_prefill_temporary_gib']:.6f} GiB",
            f"{candidate['estimated_workspace_gib']:.6f} GiB",
            f"{candidate['estimated_total_gib']:.6f} GiB",
            f"{low:.6f}–{high:.6f} GiB",
            budget_result(candidate["fits_ram_budget"]),
            budget_result(candidate["fits_vram_budget"]),
            "yes" if candidate["native_checkpoint_supported"] else "no",
        )
    console.print(table)
    recommendation = report["recommendation"]
    console.print(Panel(
        f"Recommended: {recommendation['format'] or 'none'} · {recommendation['status']}\n"
        f"{recommendation['note']}",
        title="AXIOM planning recommendation",
    ))
    console.print(report["scope_note"])


@engine_app.command("telemetry")
def engine_telemetry(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable resource telemetry."),
) -> None:
    """Inspect local CPU, RAM, device allocator and trusted model-storage resources."""
    report = runtime_telemetry()
    if json_output:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        return
    table = Table(title="AXIOM Engine resource telemetry")
    table.add_column("Metric", style="cyan")
    table.add_column("Current sample")
    table.add_row("Sampled at", report["sampled_at"])
    table.add_row("CPU cores", str(report["cpu"].get("logical_cores") or "not exposed"))
    table.add_row("System CPU", str(report["cpu"].get("system_percent")) + "%" if report["cpu"].get("system_percent") is not None else "not exposed")
    table.add_row("Process CPU", str(report["cpu"].get("process_percent")) + "%" if report["cpu"].get("process_percent") is not None else "not exposed")
    table.add_row("System RAM total", str(report["memory"].get("total_bytes")) + " bytes" if report["memory"].get("total_bytes") is not None else "not exposed")
    table.add_row("System RAM available", str(report["memory"].get("available_bytes")) + " bytes" if report["memory"].get("available_bytes") is not None else "not exposed")
    table.add_row("Process RSS", str(report["process"].get("rss_bytes")) + " bytes" if report["process"].get("rss_bytes") is not None else "not exposed")
    table.add_row("Model loaded", "yes" if report["loaded_runtime"]["loaded"] else "no")
    table.add_row("Active device", str(report["loaded_runtime"].get("device") or "none"))
    console.print(table)
    device_table = Table(title="Device memory")
    device_table.add_column("Device", style="cyan")
    device_table.add_column("Available")
    device_table.add_column("Free / total")
    for item in report["devices"]:
        free = item.get("memory_free_gib")
        total = item.get("memory_total_gib")
        memory_text = f"{free} / {total} GiB" if free is not None and total is not None else "not exposed"
        device_table.add_row(str(item.get("name") or item.get("id")), "yes" if item.get("available") else "no", memory_text)
    console.print(device_table)
    if report["disks"]:
        disk_table = Table(title="Trusted model storage (paths hidden)")
        disk_table.add_column("Root")
        disk_table.add_column("Free bytes")
        disk_table.add_column("Total bytes")
        for disk in report["disks"]:
            disk_table.add_row(disk["label"], str(disk["free_bytes"]), str(disk["total_bytes"]))
        console.print(disk_table)
    console.print(report["scope_note"])


@engine_app.command("pull")
def engine_pull(
    repo_id: str = typer.Argument(..., help="Public Hugging Face repository ID, e.g. Qwen/Qwen2.5-0.5B-Instruct."),
    json_output: bool = typer.Option(False, "--json", help="Print the install result as machine-readable JSON."),
) -> None:
    """Pull compatible public Llama/Mistral/Qwen2/Qwen3 safetensors directly for AXIOM Native Engine."""
    try:
        result = model_studio.pull_native_model(
            repo_id,
            on_log=(lambda message: console.print(message)) if not json_output else None,
            on_progress=(lambda current, total: console.print(
                "\rDownloaded " + f"{current:,}" + (f" / {total:,} bytes" if total else " bytes"),
                end="",
            )) if not json_output else None,
        )
    except (ValueError, model_studio.ModelStudioError) as exc:
        _fail(str(exc))
    if json_output:
        console.print()
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    console.print("\n[green]Installed for AXIOM Native Engine:[/green] " + result["path"])
    console.print("[dim]Run it with:[/dim] axiom engine run --model-path " + repr(result["path"]) + " --prompt \"Hello.\"")

@engine_app.command("benchmark")
def engine_benchmark(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Existing local checkpoint directory under an AXIOM trusted model root.",
    ),
    prompt: str = typer.Option(..., "--prompt", help="Local benchmark prompt. It is not returned in benchmark output."),
    adapter_path: Path | None = typer.Option(None, "--adapter-path", exists=True, file_okay=False, dir_okay=True, help="Optional local native-compatible LoRA adapter directory."),
    warmup_runs: int = typer.Option(1, "--warmup-runs", min=0, max=2, help="Excluded warm-up generations (0–2)."),
    iterations: int = typer.Option(3, "--iterations", min=1, max=5, help="Measured generations (1–5)."),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", min=1, max=128, help="Per-run output cap (1–128 tokens)."),
    temperature: float = typer.Option(0.0, "--temperature", min=0, max=2, help="Sampling temperature."),
    top_p: float = typer.Option(0.9, "--top-p", min=0.000001, max=1, help="Nucleus sampling threshold (0, 1]."),
    top_k: int = typer.Option(0, "--top-k", min=0, max=100_000, help="Candidate token limit; 0 disables top-k."),
    min_p: float = typer.Option(0.0, "--min-p", min=0, max=1, help="Relative probability cutoff."),
    repetition_penalty: float = typer.Option(1.0, "--repeat-penalty", min=0.1, max=5, help="Penalty for previously seen tokens."),
    seed: int | None = typer.Option(None, "--seed", min=0, max=9_223_372_036_854_775_807, help="Optional reproducible sampling seed."),
    stop_sequences: list[str] = typer.Option([], "--stop-sequence", help="Exact stop string; repeat for multiple strings."),
    device: str = typer.Option("auto", "--device", help="Execution backend: auto, cpu, cuda, rocm, mps, or xpu."),
    memory_mode: str = typer.Option("auto", "--memory-mode", help="Memory strategy: auto, full, layer_streaming, or disk_streaming."),
    precision: str = typer.Option("auto", "--precision", help="Compute dtype: auto, fp32, fp16, or bf16."),
    json_output: bool = typer.Option(False, "--json", help="Print benchmark metrics as machine-readable JSON."),
) -> None:
    """Measure bounded AXIOM-native inference locally; generated text is discarded."""
    try:
        report = benchmark_local(
            str(model_path),
            prompt,
            adapter_path=str(adapter_path) if adapter_path is not None else None,
            warmup_runs=warmup_runs,
            iterations=iterations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            device_preference=device,
            memory_mode=memory_mode,
            precision=precision,
        )
    except (ValueError, AxiomEngineError) as exc:
        _fail(str(exc))
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
        return

    metrics = report["metrics"]
    console.print(Panel.fit(
        f"Model: {report['model']} · {report.get('architecture') or 'unknown architecture'}\n"
        f"Device: {report.get('device') or 'unknown'} · Precision: {report.get('precision') or 'unknown'} · "
        f"Memory mode: {report.get('memory_mode') or memory_mode}\n"
        f"Measured runs: {report['iterations_completed']} · Excluded warm-ups: {report['warmup_runs_completed']}\n"
        f"Median latency: {metrics['median_elapsed_ms']} ms · p95 latency: {metrics['p95_elapsed_ms']} ms\n"
        f"Median prefill: {metrics['median_prefill_ms']} ms · median decode: {metrics['median_decode_ms']} ms\n"
        f"Median generation throughput: {metrics['median_generation_tokens_per_second']} tokens/s · "
        f"Aggregate throughput: {metrics['aggregate_tokens_per_second']} tokens/s",
        title="AXIOM NATIVE ENGINE · BENCHMARK",
    ))
    table = Table(title="Measured iterations (no generated text)")
    table.add_column("Run", style="cyan")
    table.add_column("Elapsed")
    table.add_column("Prefill")
    table.add_column("Decode")
    table.add_column("Tokens")
    table.add_column("Tokens/s")
    for item in report["runs"]:
        table.add_row(
            str(item["iteration"]),
            f"{item['elapsed_ms']:.2f} ms",
            f"{item['prefill_ms']:.2f} ms",
            f"{item['decode_ms']:.2f} ms",
            str(item["completion_tokens"]),
            f"{item['generation_tokens_per_second']:.2f}",
        )
    console.print(table)
    console.print(report["scope_note"])


@engine_app.command("run")
def engine_run(
    model_path: Path = typer.Option(..., "--model-path", exists=True, file_okay=False, dir_okay=True, help="Existing local Llama/Mistral/Qwen2 safetensors model directory."),
    prompt: str | None = typer.Option(None, "--prompt", help="Prompt text; mutually exclusive with --prompt-file."),
    prompt_file: Path | None = typer.Option(None, "--prompt-file", exists=True, file_okay=True, dir_okay=False, help="Read the prompt from a local UTF-8 text file."),
    adapter_path: Path | None = typer.Option(None, "--adapter-path", exists=True, file_okay=False, dir_okay=True, help="Optional local native-compatible LoRA adapter directory."),
    device: str = typer.Option("auto", "--device", help="Execution backend: auto, cpu, cuda, rocm, mps, or xpu."),
    memory_mode: str = typer.Option("auto", "--memory-mode", help="Memory policy: auto (fit to observed resources), full, layer_streaming (CPU-resident layers), or disk_streaming (read one safetensors layer at a time; lowest RAM/VRAM, slowest)."),
    precision: str = typer.Option("auto", "--precision", help="Weight/compute dtype: auto (checkpoint dtype), fp32, fp16, or bf16."),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens", min=1, max=2048, help="Maximum generated tokens (1–2048)."),
    temperature: float = typer.Option(0.7, "--temperature", min=0, max=2, help="Sampling temperature; 0 disables sampling."),
    top_p: float = typer.Option(0.9, "--top-p", min=0.000001, max=1, help="Nucleus sampling threshold (0, 1]."),
    top_k: int = typer.Option(0, "--top-k", min=0, max=100_000, help="Candidate token limit; 0 disables top-k."),
    min_p: float = typer.Option(0.0, "--min-p", min=0, max=1, help="Relative probability cutoff; 0 disables min-p."),
    repetition_penalty: float = typer.Option(1.0, "--repeat-penalty", min=0.1, max=5, help="Penalty for tokens already present in the prompt or completion."),
    seed: int | None = typer.Option(None, "--seed", min=0, max=9_223_372_036_854_775_807, help="Optional non-negative seed for reproducible sampling."),
    stop_sequences: list[str] = typer.Option([], "--stop-sequence", help="Stop when this exact text is generated; repeat the option for multiple stop strings."),
    json_output: bool = typer.Option(False, "--json", help="Print response and metrics as JSON."),
    stream_output: bool = typer.Option(False, "--stream", help="Show decoded output updates while the model generates."),
) -> None:
    """Generate text locally without requiring Ollama or downloading model weights."""
    if (prompt is None) == (prompt_file is None):
        _fail("Provide exactly one of --prompt or --prompt-file.")
    if prompt_file is not None:
        try:
            with prompt_file.open("r", encoding="utf-8") as handle:
                prompt_value = handle.read(12_001)
        except (OSError, UnicodeError):
            _fail("Prompt file could not be read as UTF-8 text.")
        if len(prompt_value) > 12_000:
            _fail("Prompt file exceeds AXIOM Engine's 12,000-character limit.")
    else:
        prompt_value = prompt or ""
    if json_output and stream_output:
        _fail("--stream cannot be combined with --json; JSON mode emits one complete result.")
    try:
        adapter_value = str(adapter_path) if adapter_path is not None else None
        if stream_output:
            result = None
            events = generate_local_stream(
                str(model_path),
                prompt_value,
                adapter_path=adapter_value,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_sequences=stop_sequences,
                device_preference=device,
                memory_mode=memory_mode,
                precision=precision,
            )
            try:
                with Live(console=console, refresh_per_second=12, transient=False) as live:
                    for event in events:
                        if event.get("type") == "token":
                            visible = event.get("response") or "(Waiting for printable output…)"
                            live.update(Panel(Text(str(visible), overflow="fold"), title="AXIOM ENGINE · LIVE OUTPUT"))
                        elif event.get("type") == "complete" and isinstance(event.get("result"), dict):
                            result = event["result"]
                            visible = result.get("response") or "(The model returned an empty response.)"
                            live.update(Panel(Text(str(visible), overflow="fold"), title="AXIOM ENGINE · COMPLETE"))
            finally:
                close = getattr(events, "close", None)
                if callable(close):
                    close()
            if result is None:
                _fail("Native generation ended before a completion event was produced.")
        else:
            result = generate_local(
                str(model_path),
                prompt_value,
                adapter_path=adapter_value,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_sequences=stop_sequences,
                device_preference=device,
                memory_mode=memory_mode,
                precision=precision,
            )
    except (ValueError, AxiomEngineError) as exc:
        _fail(str(exc))
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    device_selection = result.get("device_selection") if isinstance(result.get("device_selection"), dict) else {}
    actual_device = device_selection.get("selected", result.get("device", "unknown"))
    device_reason = device_selection.get("reason", "")
    console.print(Panel.fit(
        f"Engine: AXIOM Native Engine\nModel: {result['model']}\nArchitecture: {result['architecture']}\n"
        f"Execution device: {actual_device}\nMemory mode: {result.get('memory_mode', memory_mode)}\n"
        f"Compute precision: {result.get('precision', 'unknown')} (requested {result.get('precision_selection', {}).get('requested', precision) if isinstance(result.get('precision_selection'), dict) else precision})\n"
        f"{device_reason + chr(10) if device_reason else ''}"
        f"Generated tokens: {result['completion_tokens']}\nFinish reason: {result.get('stop_reason', 'unknown')}\nElapsed: {result['elapsed_ms']} ms",
        title="AXIOM ENGINE",
    ))
    if not stream_output:
        console.print(result["response"], markup=False)


@engine_app.command("batch")
def engine_batch(
    model_path: Path = typer.Option(..., "--model-path", exists=True, file_okay=False, dir_okay=True, help="Trusted local safetensors or GGUF model directory."),
    prompt: list[str] = typer.Option([], "--prompt", help="Prompt text; repeat this option for each batch item."),
    prompts_file: Path | None = typer.Option(None, "--prompts-file", exists=True, file_okay=True, dir_okay=False, readable=True, help="UTF-8 file with one prompt per line; blank lines are ignored."),
    adapter_path: Path | None = typer.Option(None, "--adapter-path", exists=True, file_okay=False, dir_okay=True, help="Optional local native-compatible LoRA adapter directory."),
    device: str = typer.Option("auto", "--device", help="Execution backend: auto, cpu, cuda, rocm, mps, or xpu."),
    memory_mode: str = typer.Option("auto", "--memory-mode", help="Memory policy: auto, full, layer_streaming, or disk_streaming."),
    precision: str = typer.Option("auto", "--precision", help="Compute dtype: auto, fp32, fp16, or bf16."),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens", min=1, max=2048),
    temperature: float = typer.Option(0.7, "--temperature", min=0, max=2),
    top_p: float = typer.Option(0.9, "--top-p", min=0.000001, max=1),
    top_k: int = typer.Option(0, "--top-k", min=0, max=100_000),
    min_p: float = typer.Option(0.0, "--min-p", min=0, max=1),
    repetition_penalty: float = typer.Option(1.0, "--repeat-penalty", min=0.1, max=5),
    seed: int | None = typer.Option(None, "--seed", min=0, max=9_223_372_036_854_775_807),
    stop_sequences: list[str] = typer.Option([], "--stop-sequence", help="Stop at exact text; repeat the option for multiple stops."),
    json_output: bool = typer.Option(False, "--json", help="Print results and batch metrics as JSON."),
) -> None:
    """Generate independent prompt responses through AXIOM's native batched decoder."""
    if bool(prompt) == (prompts_file is not None):
        _fail("Provide repeated --prompt options or --prompts-file, but not both.")
    if prompts_file is not None:
        try:
            with prompts_file.open("r", encoding="utf-8") as handle:
                prompt_text = handle.read(24_001)
        except (OSError, UnicodeError):
            _fail("Prompts file could not be read as UTF-8 text.")
        if len(prompt_text) > 24_000:
            _fail("Prompts file exceeds the combined 24,000-character batch limit.")
        prompts = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    else:
        prompts = [value.strip() for value in prompt if value.strip()]
    if not prompts:
        _fail("Batch generation requires at least one non-blank prompt.")
    if len(prompts) > 128:
        _fail("Batch generation is limited to 128 prompts per request.")
    if any(len(value) > 12_000 for value in prompts):
        _fail("Each batch prompt is limited to 12,000 characters.")
    if sum(len(value) for value in prompts) > 24_000:
        _fail("Combined batch prompts exceed the 24,000-character limit.")
    try:
        result = generate_local_batch(
            str(model_path),
            prompts,
            adapter_path=str(adapter_path) if adapter_path is not None else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            device_preference=device,
            memory_mode=memory_mode,
            precision=precision,
        )
    except (ValueError, AxiomEngineError) as exc:
        _fail(str(exc))
    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return

    console.print(Panel.fit(
        f"Model: {result.get('model', 'local-model')}\n"
        f"Architecture: {result.get('architecture', 'unknown')}\n"
        f"Execution: {result.get('execution_backend', 'pytorch-native-axiom')} · {result.get('device', 'unknown')}\n"
        f"Batch size: {result.get('batch_size', len(result.get('results', [])))}\n"
        f"Prompt tokens: {result.get('prompt_tokens', 0)} · Completion tokens: {result.get('completion_tokens', 0)}\n"
        f"Wall time: {result.get('elapsed_ms', 0)} ms",
        title="AXIOM ENGINE · NATIVE BATCH",
    ))
    for item in result.get("results", []):
        console.print(Panel(
            Text(str(item.get("response", "")), overflow="fold"),
            title=f"Prompt {item.get('index', '?')} · {item.get('stop_reason', 'unknown')}",
        ))


__all__ = ["engine_app"]


@engine_app.command("backends")
def engine_backends_cli(
    profile: str = typer.Option("balanced", help="balanced, low-memory, throughput, easy-setup, portable, minimal-daemons"),
    json_output: bool = typer.Option(False, "--json", help="Print the complete backend matrix as JSON."),
) -> None:
    """Compare the native AXIOM engine with installed local inference runtimes."""
    try:
        report = discover_engine_backends(profile)
    except ValueError as exc:
        _fail(str(exc))
    if json_output:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        return
    table = Table(title="AXIOM Engine Fabric")
    table.add_column("Engine", style="cyan")
    table.add_column("Status")
    table.add_column("Models")
    table.add_column("Best fit")
    for backend in report["backends"]:
        table.add_row(
            backend["name"],
            backend["status"],
            str(backend["model_count"]),
            ", ".join(backend["best_for"]),
        )
    console.print(table)
    recommendation = report["recommendation"]
    console.print(Panel(
        f"{recommendation['name']} ({recommendation['status']})\n{recommendation['reason']}",
        title=f"Recommended profile: {profile}",
    ))
    console.print(report["scope_note"])


@engine_app.command("route-chat")
def engine_route_chat(
    backend: str = typer.Option(..., "--backend", help="axiom-native, ollama, llama-cpp, lm-studio, vllm, llama-cpp-python"),
    model: str = typer.Option(..., "--model", help="Exact model ID reported by the selected local service."),
    prompt: str = typer.Option(..., "--prompt", help="Prompt text, at most 12,000 characters."),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens", min=1, max=2048),
    temperature: float = typer.Option(0.7, min=0, max=2),

    top_p: float = typer.Option(0.9, "--top-p", min=0.000001, max=1),
    top_k: int = typer.Option(0, "--top-k", min=0, max=100_000),
    min_p: float = typer.Option(0.0, "--min-p", min=0, max=1, help="Relative-probability cutoff; supported by AXIOM Native Engine and Ollama."),
    repetition_penalty: float = typer.Option(1.0, "--repeat-penalty", min=0.1, max=5),
    seed: int | None = typer.Option(None, "--seed", min=0, max=9_223_372_036_854_775_807),
    stop_sequences: list[str] = typer.Option([], "--stop-sequence", help="Stop on exact text; repeat the option for multiple stops."),
    json_output: bool = typer.Option(False, "--json", help="Print the response and runtime metadata as JSON."),
) -> None:
    """Run one bounded prompt through a selected local compatible backend."""
    try:
        result = run_engine_backend_chat(
            backend, model, [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty,
            seed=seed, stop_sequences=stop_sequences,
        )
    except (ValueError, EngineFabricError) as exc:
        _fail(str(exc))
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    console.print(Panel(result["response"] or "(The model returned an empty response.)", title=f"{result['backend_name']} · {result['model']}"))
    console.print(f"{result['eval_count'] if result['eval_count'] is not None else 'Unknown'} output tokens · {result['total_duration_ms']} ms · local only")


@engine_app.command("auto-chat")
def engine_auto_chat(
    model: str = typer.Option(..., "--model", help="Exact model ID advertised by a local service."),
    prompt: str = typer.Option(..., "--prompt", help="Prompt text, at most 12,000 characters."),
    profile: str = typer.Option("balanced", "--profile", help="balanced, low-memory, throughput, easy-setup, portable, minimal-daemons"),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens", min=1, max=2048),
    temperature: float = typer.Option(0.7, min=0, max=2),

    top_p: float = typer.Option(0.9, "--top-p", min=0.000001, max=1),
    top_k: int = typer.Option(0, "--top-k", min=0, max=100_000),
    min_p: float = typer.Option(0.0, "--min-p", min=0, max=1, help="Relative-probability cutoff; supported by AXIOM Native Engine and Ollama."),
    repetition_penalty: float = typer.Option(1.0, "--repeat-penalty", min=0.1, max=5),
    seed: int | None = typer.Option(None, "--seed", min=0, max=9_223_372_036_854_775_807),
    stop_sequences: list[str] = typer.Option([], "--stop-sequence", help="Stop on exact text; repeat the option for multiple stops."),
    json_output: bool = typer.Option(False, "--json", help="Print response, selected backend and routing attempts as JSON."),
) -> None:
    """Auto-route to an exact-ID match across ready local inference engines."""
    if not model.strip():
        _fail("Model ID must not be blank.")
    if not prompt.strip() or len(prompt) > 12_000:
        _fail("Prompt must contain non-whitespace text and be at most 12,000 characters.")
    try:
        result = run_engine_fabric_chat(
            profile, model.strip(), [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty,
            seed=seed, stop_sequences=stop_sequences,
        )
    except (ValueError, EngineFabricError) as exc:
        _fail(str(exc))
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    routing = result.get("routing", {})
    console.print(Panel(
        result["response"] or "(The model returned an empty response.)",
        title=f"{result['backend_name']} · {result['model']} · auto-routed",
    ))
    console.print(
        f"{result['eval_count'] if result['eval_count'] is not None else 'Unknown'} output tokens · "
        f"{result['total_duration_ms']} ms · local only"
    )
    for attempt in routing.get("attempts", []):
        status = "selected" if attempt.get("outcome") == "selected" else "failed"
        reason = f" — {attempt['reason']}" if attempt.get("reason") else ""
        console.print(f"{attempt.get('backend_name', attempt.get('backend_id', 'backend'))}: {status}{reason}")
    if routing.get("scope_note"):
        console.print(routing["scope_note"])


@engine_app.command("route-benchmark")
def engine_route_benchmark(
    backend: str = typer.Option(..., "--backend", help="axiom-native, ollama, llama-cpp, lm-studio, vllm, llama-cpp-python"),
    model: str = typer.Option(..., "--model", help="Exact model ID reported by the selected local service."),
    prompt: str = typer.Option(..., "--prompt", help="Short representative prompt, at most 12,000 characters."),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", min=1, max=256),
    warmup_runs: int = typer.Option(1, "--warmup-runs", min=0, max=1),
    iterations: int = typer.Option(3, "--iterations", min=1, max=5),
    temperature: float = typer.Option(0.2, min=0, max=2),

    top_p: float = typer.Option(0.9, "--top-p", min=0.000001, max=1),
    top_k: int = typer.Option(0, "--top-k", min=0, max=100_000),
    min_p: float = typer.Option(0.0, "--min-p", min=0, max=1, help="Relative-probability cutoff; supported by AXIOM Native Engine and Ollama."),
    repetition_penalty: float = typer.Option(1.0, "--repeat-penalty", min=0.1, max=5),
    seed: int | None = typer.Option(None, "--seed", min=0, max=9_223_372_036_854_775_807),
    stop_sequences: list[str] = typer.Option([], "--stop-sequence", help="Stop on exact text; repeat the option for multiple stops."),
    json_output: bool = typer.Option(False, "--json", help="Print measurements without generated text."),
) -> None:
    """Measure a few real local requests without retaining prompt or response text."""
    try:
        result = run_engine_backend_benchmark(
            backend, model, prompt, max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs, iterations=iterations, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty,
            seed=seed, stop_sequences=stop_sequences,
        )
    except (ValueError, EngineFabricError) as exc:
        _fail(str(exc))
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    measurements = result["measurements"]
    table = Table(title=f"Engine Fabric benchmark · {result['backend_name']}")
    table.add_column("Measurement", style="cyan")
    table.add_column("Value")
    table.add_row("Model", result["model"])
    table.add_row("Measured requests", str(result["configuration"]["measured_requests"]))
    table.add_row("Median latency", f"{measurements['latency_ms']['median']} ms")
    table.add_row("P95 latency", f"{measurements['latency_ms']['p95']} ms")
    table.add_row("Wall time", f"{measurements['wall_time_ms']} ms")
    throughput = measurements["completion_tokens_per_second"]
    table.add_row("Completion throughput", f"{throughput} tokens/s" if throughput is not None else "not reported by backend")
    console.print(table)
    console.print(result["scope_note"])
