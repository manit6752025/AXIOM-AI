"""Dataset profile CLI command."""
import json
from pathlib import Path
import typer
from axiom.cli.errors import emit_error
from axiom.cli.progress import byte_progress, report_cancelled, scan_progress
from rich.console import Console
from rich.table import Table
from axiom.datasets.profiler import profile_jsonl
from axiom.datasets.manifest import create_dataset_manifest
from axiom.datasets.cleaning_preview import preview_jsonl_cleaning
console=Console()
def register(dataset_app: typer.Typer) -> None:
 @dataset_app.command("profile")
 def dataset_profile(
  path: Path=typer.Argument(...,exists=True,dir_okay=False,readable=True),
  max_rows: int=typer.Option(100000,min=1),
  max_mb: int=typer.Option(64,min=1),
  duplicate_key: str=typer.Option("",help="Comma-separated field names for duplicate-key analysis."),
  duplicate_rate_threshold: float=typer.Option(None,min=0,max=1,help="Warn when duplicate-key rate exceeds this fraction."),
  text_fields: str=typer.Option("",help="Comma-separated text fields for aggregate token-length bins."),
  sensitive_rule: str=typer.Option("",help="Comma-separated additional sensitive field-name substrings."),
  ignore_sensitive_field: str=typer.Option("",help="Comma-separated exact field names to exclude from privacy flags."),
  scan_pii: bool=typer.Option(False,"--scan-pii",help="Explicitly scan local string values for common PII patterns; matched values are never shown."),
 ) -> None:
  """Profile bounded JSONL structure and privacy-safe diagnostics."""
  if path.suffix.lower()!=".jsonl": emit_error("Only .jsonl files are supported.", code="unsupported_dataset_format")
  duplicate_fields=[field.strip() for field in duplicate_key.split(",") if field.strip()]
  selected_text_fields=[field.strip() for field in text_fields.split(",") if field.strip()]
  selected_rules=[field.strip() for field in sensitive_rule.split(",") if field.strip()]
  excluded_fields=[field.strip() for field in ignore_sensitive_field.split(",") if field.strip()]
  try:
   with scan_progress("Profiling JSONL",path.stat().st_size) as progress:
    report=profile_jsonl(
     path,max_rows=max_rows,max_bytes=max_mb*1024*1024,
     duplicate_key_fields=duplicate_fields,duplicate_rate_threshold=duplicate_rate_threshold,
     text_fields=selected_text_fields,sensitive_field_rules=selected_rules,
     ignored_sensitive_fields=excluded_fields,scan_pii=scan_pii,on_progress=progress,
    )
  except KeyboardInterrupt:
   report_cancelled("Dataset profile cancelled; the source file was not changed.")
  except (OSError,ValueError) as exc:
   emit_error(str(exc), code="dataset_profile_failed")
  table=Table(title="AXIOM DATASET PROFILE"); table.add_column("Metric"); table.add_column("Value",justify="right")
  for key in ("valid_rows","invalid_rows","empty_rows","duplicate_rows","unique_fields_tracked","bytes_scanned"): table.add_row(key.replace("_"," ").title(),str(report[key]))
  table.add_row("UTF-8 errors",str(report["diagnostics"]["encoding_errors"]))
  table.add_row("Malformed JSON",str(report["diagnostics"]["malformed_json_rows"]))
  table.add_row("Newline style",report["diagnostics"]["newline_style"])
  if report["pii_scan"]["opted_in"]:
   pii=report["pii_scan"]["matches_by_type"]
   table.add_row("PII pattern matches",f"email {pii['email']} · phone-like {pii['phone_like']} · payment card {pii['payment_card']}")
  if duplicate_fields:
   summary=report["duplicate_key_summary"]
   table.add_row("Duplicate-key rows",f"{summary['duplicate_rows']} / {summary['eligible_rows']} ({summary['rate']:.2%})")
   if summary["threshold_exceeded"]: console.print("[yellow]Warning:[/yellow] Configured duplicate-key rate threshold exceeded.")
  console.print(table)
  if report["pii_scan"]["opted_in"] and sum(report["pii_scan"]["matches_by_type"].values()):
   fields=sorted({name for names in report["pii_scan"]["fields_by_type"].values() for name in names})
   console.print("[yellow]Local PII review:[/yellow] pattern matches in fields: "+", ".join(fields)+". Matched values are never displayed.")
  for field, stats in report["text_token_lengths"].items(): console.print(f"{field} whitespace-token counts: {stats['histogram'] or 'no string values'}")
  if report["possibly_sensitive_fields"]: console.print("[yellow]Privacy review:[/yellow] "+", ".join(report["possibly_sensitive_fields"]))
  for warning in report["warnings"]: console.print(f"[yellow]Warning:[/yellow] {warning}")

 @dataset_app.command("manifest")
 def dataset_manifest(
  path: Path=typer.Argument(...,exists=True,dir_okay=False,readable=True),
  max_mb: int=typer.Option(256,min=1,max=1024,help="Maximum bytes to hash in MiB; larger files receive a partial digest."),
  parent_manifest_sha256: str=typer.Option(None,"--parent-manifest-sha256",help="Optional SHA-256 of the source manifest to link this file in a lineage chain."),
 ) -> None:
  """Print a bounded SHA-256 provenance manifest without dataset values."""
  if path.suffix.lower()!=".jsonl": emit_error("Only .jsonl files are supported.", code="unsupported_dataset_format")
  try:
   source=path.expanduser().resolve(strict=True)
   with source.open("rb") as stream:
    size=source.stat().st_size
    with byte_progress("Hashing dataset manifest",min(size,max_mb*1024*1024)) as progress:
     manifest=create_dataset_manifest(
      stream,file_name=source.name,relative_path=source.name,file_size_bytes=size,
      max_bytes=max_mb*1024*1024,parent_manifest_sha256=parent_manifest_sha256,
      on_progress=progress,
     )
  except KeyboardInterrupt:
   report_cancelled("Dataset manifest cancelled; no files were changed.")
  except (OSError,ValueError) as exc:
   emit_error(str(exc), code="dataset_manifest_failed")
  console.print_json(json.dumps(manifest,ensure_ascii=False))

 @dataset_app.command("preview-clean")
 def dataset_preview_clean(
  path: Path=typer.Argument(...,exists=True,dir_okay=False,readable=True),
  max_rows: int=typer.Option(100000,min=1),
  max_mb: int=typer.Option(64,min=1,max=256),
 ) -> None:
  """Preview JSONL cleaning counts without writing or changing a file."""
  if path.suffix.lower()!=".jsonl": emit_error("Only .jsonl files are supported.", code="unsupported_dataset_format")
  try:
   source=path.expanduser().resolve(strict=True)
   with source.open("rb") as stream:
    size=source.stat().st_size
    with scan_progress("Previewing JSONL cleaning",size) as progress:
     preview=preview_jsonl_cleaning(
      stream,file_name=source.name,file_size_bytes=size,max_bytes=max_mb*1024*1024,
      max_rows=max_rows,on_progress=progress,
     )
  except KeyboardInterrupt:
   report_cancelled("Dataset cleaning preview cancelled; no files were changed.")
  except (OSError,ValueError) as exc:
   emit_error(str(exc), code="dataset_clean_preview_failed")
  console.print_json(json.dumps(preview,ensure_ascii=False))
