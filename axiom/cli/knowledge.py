"""Public CLI for workspace-scoped knowledge bases and safe document ingestion."""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from axiom.cli.errors import emit_error
from axiom.cli.interactive import confirm as safe_confirm
from axiom.core.knowledge import KnowledgeBaseStore, KnowledgeStoreCapacityError, KnowledgeStoreError

console = Console()
knowledge_app = typer.Typer(help="Manage local workspace knowledge bases and documents.", no_args_is_help=True)
base_app = typer.Typer(help="Create, list, and remove workspace knowledge bases.", no_args_is_help=True)
knowledge_app.add_typer(base_app, name="base")


def _store() -> KnowledgeBaseStore:
    return KnowledgeBaseStore()


def _fail(message: str, code: str = "knowledge_operation_failed") -> None:
    emit_error(message, code=code)


def register(root_app: typer.Typer) -> None:
    """Register the knowledge command group on the public AXIOM CLI."""

    @base_app.command("create")
    def create_base(
        name: str = typer.Argument(..., help="Unique knowledge base name within this workspace."),
        description: str = typer.Option("", "--description", help="Optional description (up to 500 characters)."),
        json_output: bool = typer.Option(False, "--json", help="Print a versioned JSON result."),
    ) -> None:
        """Create an empty knowledge base in the active workspace."""
        try:
            result = _store().create_base(name, description)
        except (OSError, ValueError) as exc:
            _fail(str(exc))
        payload = {"schema_version": 1, "knowledge_base": result}
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return
        console.print(f"[green]Created knowledge base[/green] {result['name']} ({result['id']}).")

    @base_app.command("list")
    def list_bases(
        json_output: bool = typer.Option(False, "--json", help="Print a versioned JSON result."),
    ) -> None:
        """List knowledge bases belonging to the active workspace."""
        try:
            result = _store().list_bases()
        except (OSError, ValueError) as exc:
            _fail(str(exc))
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            return
        table = Table(title="Workspace knowledge bases")
        table.add_column("Name")
        table.add_column("ID", style="dim")
        table.add_column("Documents", justify="right")
        table.add_column("Stored bytes", justify="right")
        for item in result["knowledge_bases"]:
            table.add_row(item["name"], item["id"], str(item["document_count"]), f"{item['total_bytes']:,}")
        console.print(table if result["knowledge_bases"] else "No knowledge bases in this workspace.")

    @base_app.command("remove")
    def remove_base(
        base_id: str,
        yes: bool = typer.Option(False, "--yes", help="Remove the base without an interactive prompt."),
        json_output: bool = typer.Option(False, "--json", help="Print a versioned JSON result."),
    ) -> None:
        """Remove a knowledge base and only its AXIOM-managed ingested text."""
        if not yes and not safe_confirm(f"Remove knowledge base {base_id} and its ingested copies?", default=False):
            console.print("Cancelled; no knowledge files were changed.")
            return
        try:
            removed = _store().remove_base(base_id)
        except (OSError, ValueError, KeyError) as exc:
            _fail(str(exc))
        if removed is None:
            _fail("Knowledge base not found.", code="knowledge_base_not_found")
        payload = {"schema_version": 1, "removed": removed, "source_files_modified": False}
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        else:
            console.print(f"[yellow]Removed[/yellow] {removed['name']}; original workspace files were left unchanged.")

    @knowledge_app.command("ingest")
    def ingest_document(
        base_id: str,
        path: str = typer.Argument(..., help="Workspace-relative UTF-8 TXT, Markdown, RST, CSV, JSON, or JSONL file."),
        json_output: bool = typer.Option(False, "--json", help="Print a versioned JSON result."),
    ) -> None:
        """Copy one safe, bounded workspace document into a knowledge base."""
        try:
            document = _store().ingest_document(base_id, path)
        except (OSError, ValueError, KeyError) as exc:
            _fail(str(exc))
        payload = {
            "schema_version": 1,
            "document": document,
            "source_text_returned": False,
            "source_text_persisted": True,
            "network_used": False,
        }
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return
        console.print(f"[green]Ingested[/green] {document['file_name']} · {document['character_count']:,} characters · SHA-256 {document['content_sha256']}")

    @knowledge_app.command("documents")
    def list_documents(
        base_id: str,
        json_output: bool = typer.Option(False, "--json", help="Print a versioned JSON result."),
    ) -> None:
        """List document metadata without printing ingested text."""
        try:
            result = _store().list_documents(base_id)
        except (OSError, ValueError, KeyError) as exc:
            _fail(str(exc))
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            return
        table = Table(title=f"Knowledge documents · {result['knowledge_base']['name']}")
        table.add_column("File")
        table.add_column("ID", style="dim")
        table.add_column("Characters", justify="right")
        table.add_column("Format")
        for document in result["documents"]:
            table.add_row(document["file_name"], document["id"], f"{document['character_count']:,}", document["format"])
        console.print(table if result["documents"] else "This knowledge base has no documents.")

    @knowledge_app.command("remove-document")
    def remove_document(
        base_id: str,
        document_id: str,
        yes: bool = typer.Option(False, "--yes", help="Remove the ingested copy without an interactive prompt."),
        json_output: bool = typer.Option(False, "--json", help="Print a versioned JSON result."),
    ) -> None:
        """Remove an ingested copy without changing its original source file."""
        if not yes and not safe_confirm(f"Remove ingested document {document_id}?", default=False):
            console.print("Cancelled; no knowledge files were changed.")
            return
        try:
            removed = _store().remove_document(base_id, document_id)
        except (OSError, ValueError, KeyError) as exc:
            _fail(str(exc))
        if removed is None:
            _fail("Knowledge document not found.", code="knowledge_document_not_found")
        payload = {"schema_version": 1, "removed": removed, "source_file_modified": False}
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        else:
            console.print(f"[yellow]Removed ingested copy[/yellow] {removed['file_name']}; source file was left unchanged.")

    root_app.add_typer(knowledge_app, name="knowledge")
