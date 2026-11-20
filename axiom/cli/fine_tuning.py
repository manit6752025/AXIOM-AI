"""Commands for actual local LoRA fine-tuning and persisted job inspection."""
from __future__ import annotations

import json
from typing import Any

import typer

from axiom.cli.errors import emit_error
from rich.console import Console
from rich.table import Table

from axiom.core.fine_tuning import (
    ActiveFineTuningJobError,
    FineTuningError,
    cancel_finetuning_job,
    get_job,
    list_jobs,
    list_jobs_page,
    run_finetuning_inline,
)

console = Console()


def _fail(message: str) -> None:
    emit_error(message, code="fine_tuning_error")


def register(train_app: typer.Typer) -> None:
    @train_app.command("run")
    def train_run(
        model_path: str = typer.Option(..., "--model-path", help="Existing local Transformers model directory."),
        dataset_path: str = typer.Option(..., "--dataset", help="Local JSONL file with training text."),
        output_dir: str = typer.Option(..., "--output-dir", help="New or empty local adapter output directory."),
        text_field: str = typer.Option("text", "--text-field", help="JSON string field to train on; defaults to text."),
        max_steps: int = typer.Option(20, min=1, max=1000, help="Maximum optimization steps (1–1000)."),
        sequence_length: int = typer.Option(512, min=32, max=2048, help="Maximum tokens per example (32–2048)."),
        learning_rate: float = typer.Option(0.0001, min=0.000001, max=0.01, help="LoRA learning rate."),
        lora_rank: int = typer.Option(8, min=1, max=64, help="LoRA adapter rank (1–64)."),
        json_output: bool = typer.Option(False, "--json", help="Print the final job record as JSON."),
    ) -> None:
        """Actually train a local LoRA adapter. This may use substantial CPU/GPU memory."""
        config = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "output_dir": output_dir,
            "text_field": text_field,
            "max_steps": max_steps,
            "sequence_length": sequence_length,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
        }
        def progress(job: dict[str, Any]) -> None:
            if json_output:
                return
            steps = job.get("completed_steps", 0)
            loss = job.get("loss")
            console.print(f"[cyan]Step {steps}/{job.get('config', {}).get('max_steps', max_steps)}[/cyan]" + (f" · loss {loss:.5f}" if isinstance(loss, (int, float)) else ""))
        try:
            job = run_finetuning_inline(config, on_progress=progress)
        except (ValueError, FineTuningError, ActiveFineTuningJobError) as exc:
            _fail(str(exc))
        if json_output:
            console.print_json(json.dumps(job, ensure_ascii=False))
            if job.get("status") != "completed":
                raise typer.Exit(code=1)
        elif job.get("status") == "completed":
            result = job.get("result", {})
            console.print(f"[green]Fine-tuning completed.[/green] Adapter: {result.get('output_dir', output_dir)}")
            console.print(f"Job ID: {job.get('id')} · rows: {result.get('dataset_rows')} · steps: {job.get('completed_steps')}")
        else:
            _fail(str(job.get("error") or "Fine-tuning job did not complete."))

    @train_app.command("jobs")
    def train_jobs(
        limit: int = typer.Option(20, min=1, max=100, help="Maximum saved jobs to list."),
        offset: int = typer.Option(0, min=0, max=5000, help="Number of matching jobs to skip."),
        json_output: bool = typer.Option(False, "--json", help="Print page metadata and jobs as JSON."),
    ) -> None:
        """List a bounded page of recent local fine-tuning jobs."""
        try:
            page = list_jobs_page(limit=limit, offset=offset)
            jobs = page["jobs"]
        except (ValueError, FineTuningError) as exc:
            _fail(str(exc))
        if json_output:
            console.print_json(json.dumps(page, ensure_ascii=False))
            return
        if not jobs:
            console.print("No local fine-tuning jobs on this page. Use `axiom train run --help` to start one.")
            return
        table = Table(title=f"Local fine-tuning jobs · {offset + 1}–{offset + len(jobs)} of {page['total']}")
        for column in ("ID", "Status", "Steps", "Model", "Updated"):
            table.add_column(column)
        for job in jobs:
            config = job.get("config", {})
            table.add_row(
                str(job.get("id", ""))[:12], str(job.get("status", "unknown")).upper(),
                f"{job.get('completed_steps', 0)}/{config.get('max_steps', '?')}",
                str(config.get("model_path", ""))[-32:], str(job.get("updated_at", "")),
            )
        console.print(table)
        if page["truncated"]:
            console.print(f"History scan capped at {page['scan_limit']:,} newest job records.")
        if page["has_more"]:
            next_offset = offset + len(jobs)
            console.print(f"Next page: axiom train jobs --limit {limit} --offset {next_offset}")

    @train_app.command("cancel")
    def train_cancel(
        job_id: str = typer.Argument(..., help="Full ID of the local fine-tuning job."),
        json_output: bool = typer.Option(False, "--json", help="Print the updated job and cancellation contract as JSON."),
    ) -> None:
        """Request cooperative cancellation before adapter finalization."""
        try:
            job = cancel_finetuning_job(job_id)
        except (ValueError, FineTuningError) as exc:
            _fail(str(exc))
        if job is None:
            _fail("Fine-tuning job was not found.")
        if json_output:
            console.print_json(json.dumps(job, ensure_ascii=False))
            return
        status = job.get("status")
        if status == "cancelling":
            console.print("Cancellation requested; the worker will stop at its next safe training boundary.")
        else:
            contract = job.get("cancellation")
            reason = (
                contract.get("unavailable_reason")
                if isinstance(contract, dict)
                else None
            ) or "not_cancellable"
            console.print(f"Cancellation was not started: job status is {status} ({reason}).")

    @train_app.command("status")
    def train_status(
        job_id: str | None = typer.Argument(None, help="A full job ID; omit to show the recent jobs."),
        json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    ) -> None:
        """Inspect one saved job or list recent jobs."""
        try:
            job = get_job(job_id) if job_id else None
            jobs = list_jobs(20) if not job_id else []
        except (ValueError, FineTuningError) as exc:
            _fail(str(exc))
        if job_id and job is None:
            _fail("Fine-tuning job was not found.")
        if json_output:
            console.print_json(json.dumps(job if job_id else {"jobs": jobs}, ensure_ascii=False))
        elif job_id and job:
            console.print_json(json.dumps(job, ensure_ascii=False))
        else:
            console.print("No job ID supplied; showing recent jobs. Use `axiom train jobs` for a table.")
            console.print_json(json.dumps({"jobs": jobs}, ensure_ascii=False))
