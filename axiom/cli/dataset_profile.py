"""Dataset profile CLI command."""
import json
from pathlib import Path
import typer
from axiom.cli.errors import emit_error
from axiom.cli.progress import cancellable_byte_progress, cancellable_scan_progress, report_cancelled
from rich.console import Console
from rich.table import Table
from axiom.datasets.profiler import profile_jsonl
from axiom.datasets.progress import DatasetScanCancelled
from axiom.datasets.manifest import create_dataset_manifest
from axiom.datasets.cleaning_preview import preview_jsonl_cleaning
from axiom.core.workspace_context import active_workspace_root
from axiom.datasets.snapshots import DatasetSnapshotCapacityError, DatasetSnapshotError, DatasetSnapshotStore
from axiom.path_safety import SafePathError, find_existing_relative_path, open_regular_file
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
  json_output: bool=typer.Option(False,"--json",help="Emit a versioned privacy-safe profile report as JSON."),
 ) -> None:
  """Profile bounded JSONL structure and privacy-safe diagnostics."""
  if path.suffix.lower()!=".jsonl": emit_error("Only .jsonl files are supported.", code="unsupported_dataset_format")
  duplicate_fields=[field.strip() for field in duplicate_key.split(",") if field.strip()]
  selected_text_fields=[field.strip() for field in text_fields.split(",") if field.strip()]
  selected_rules=[field.strip() for field in sensitive_rule.split(",") if field.strip()]
  excluded_fields=[field.strip() for field in ignore_sensitive_field.split(",") if field.strip()]
  try:
   with cancellable_scan_progress("Profiling JSONL",path.stat().st_size) as (cancel_event,progress):
    report=profile_jsonl(
     path,max_rows=max_rows,max_bytes=max_mb*1024*1024,
     duplicate_key_fields=duplicate_fields,duplicate_rate_threshold=duplicate_rate_threshold,
     text_fields=selected_text_fields,sensitive_field_rules=selected_rules,
     ignored_sensitive_fields=excluded_fields,scan_pii=scan_pii,
     on_progress=progress,cancel_event=cancel_event,
    )
  except (DatasetScanCancelled,KeyboardInterrupt):
   report_cancelled("Dataset profile cancelled; the source file was not changed.")
  except (OSError,ValueError) as exc:
   emit_error(str(exc), code="dataset_profile_failed")
  if json_output:
   typer.echo(json.dumps({
    "schema_version": 1,
    "profile": report,
    "scope_note": "Aggregate dataset diagnostics only; source row values and matched PII values are never returned.",
   },ensure_ascii=False,indent=2,sort_keys=True))
   return
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
    with cancellable_byte_progress("Hashing dataset manifest",min(size,max_mb*1024*1024)) as (cancel_event,progress):
     manifest=create_dataset_manifest(
      stream,file_name=source.name,relative_path=source.name,file_size_bytes=size,
      max_bytes=max_mb*1024*1024,parent_manifest_sha256=parent_manifest_sha256,
      on_progress=progress,cancel_event=cancel_event,
     )
  except (DatasetScanCancelled,KeyboardInterrupt):
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
    with cancellable_scan_progress("Previewing JSONL cleaning",size) as (cancel_event,progress):
     preview=preview_jsonl_cleaning(
      stream,file_name=source.name,file_size_bytes=size,max_bytes=max_mb*1024*1024,
      max_rows=max_rows,on_progress=progress,cancel_event=cancel_event,
     )
  except (DatasetScanCancelled,KeyboardInterrupt):
   report_cancelled("Dataset cleaning preview cancelled; no files were changed.")
  except (OSError,ValueError) as exc:
   emit_error(str(exc), code="dataset_clean_preview_failed")
  console.print_json(json.dumps(preview,ensure_ascii=False))


 @dataset_app.command("snapshot")
 def dataset_snapshot(
  path: Path=typer.Argument(...,exists=True,dir_okay=False,readable=True),
  max_mb: int=typer.Option(256,min=1,max=1024,help="Maximum snapshot size in MiB."),
  parent_snapshot_id: str=typer.Option(None,"--parent-snapshot-id",help="Optional earlier snapshot ID to link in immutable lineage."),
  json_output: bool=typer.Option(False,"--json",help="Emit the complete snapshot record as JSON."),
 ) -> None:
  """Create an immutable, content-addressed snapshot of a workspace data/ JSONL file."""
  workspace=active_workspace_root()
  data_entry=workspace / "data"
  if data_entry.is_symlink() or not data_entry.is_dir():
   emit_error("Workspace data directory is unavailable or unsafe.", code="dataset_snapshot_source_unavailable")
  try:
   data_root=data_entry.resolve(strict=True)
   supplied=path.expanduser()
   if supplied.is_absolute():
    relative=supplied.relative_to(data_root).as_posix()
   else:
    parts=supplied.parts
    if parts and parts[0] == "data":
     parts=parts[1:]
    relative=Path(*parts).as_posix()
   candidate=find_existing_relative_path(data_root,relative,allow_directories=False)
   with open_regular_file(candidate) as stream:
    result=DatasetSnapshotStore(workspace).create_snapshot(
     stream,
     source_path=candidate.relative_to(data_root).as_posix(),
     file_name=candidate.name,
     max_bytes=max_mb*1024*1024,
     parent_snapshot_id=parent_snapshot_id,
    )
  except DatasetSnapshotCapacityError as exc:
   emit_error(str(exc), code="dataset_snapshot_capacity")
  except (OSError,ValueError,SafePathError) as exc:
   emit_error(str(exc), code="dataset_snapshot_failed")
  if json_output:
   typer.echo(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))
  else:
   manifest=result["snapshot"]["manifest"]
   console.print(f"[green]Snapshot created[/green] · {manifest['snapshot_id']}")
   console.print(f"SHA-256: {manifest['sha256']}")
   console.print(f"Size: {manifest['file_size_bytes']} bytes · Object: {manifest['object_name']}")
   if manifest["parent_snapshot_id"]:
    console.print(f"Parent: {manifest['parent_snapshot_id']}")


 @dataset_app.command("snapshots")
 def dataset_snapshots(
  limit: int=typer.Option(50,min=1,max=100),
  offset: int=typer.Option(0,min=0,max=250),
  json_output: bool=typer.Option(False,"--json",help="Emit the snapshot history as JSON."),
 ) -> None:
  """List immutable dataset snapshots and their parent lineage."""
  try:
   report=DatasetSnapshotStore(active_workspace_root()).list(limit=limit,offset=offset)
  except DatasetSnapshotError as exc:
   emit_error(str(exc),code="dataset_snapshot_history_failed")
  if json_output:
   typer.echo(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
   return
  table=Table(title="AXIOM DATASET SNAPSHOTS")
  table.add_column("Snapshot ID")
  table.add_column("Created")
  table.add_column("SHA-256")
  table.add_column("Size",justify="right")
  table.add_column("Parent")
  for record in report["snapshots"]:
   manifest=record["manifest"]
   table.add_row(
    manifest["snapshot_id"],
    manifest["created_at"],
    manifest["sha256"],
    str(manifest["file_size_bytes"]),
    manifest["parent_snapshot_id"] or "source",
   )
  console.print(table)
  console.print(f"{report['total']} immutable snapshot(s) · page offset {offset}")


 @dataset_app.command("snapshot-verify")
 def dataset_snapshot_verify(
  snapshot_id: str=typer.Argument(...,help="32-character immutable snapshot ID."),
  json_output: bool=typer.Option(False,"--json",help="Emit machine-readable verification results."),
 ) -> None:
  """Verify snapshot object content and its manifest lineage chain."""
  try:
   report=DatasetSnapshotStore(active_workspace_root()).verify(snapshot_id)
  except FileNotFoundError:
   emit_error("Dataset snapshot was not found.",code="dataset_snapshot_not_found")
  except DatasetSnapshotError as exc:
   emit_error(str(exc),code="dataset_snapshot_verification_failed")
  if json_output:
   typer.echo(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
  else:
   marker="[green]VALID[/green]" if report["valid"] and report["lineage_valid"] else "[red]INVALID[/red]"
   console.print(f"{marker} · {report['snapshot_id']}")
   console.print(f"SHA-256: {report['expected_sha256']} · {report['file_size_bytes']} bytes")
