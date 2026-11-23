"""CLI for validating declarative plugin manifests and explicit grants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from axiom.cli.errors import emit_error
from axiom.core.plugin_manifests import (
    MAX_MANIFEST_BYTES,
    PluginManifestError,
    PluginManifestStore,
    validate_plugin_manifest,
)
from axiom.core.security_events import SecurityEventError, record_security_event
from axiom.path_safety import open_regular_file

console = Console()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Manifest JSON contains duplicate object keys.")
        result[key] = value
    return result


def _read_manifest(path: str) -> dict[str, Any]:
    target = Path(path).expanduser()
    try:
        info = target.lstat()
        if target.is_symlink() or not target.is_file() or info.st_size > MAX_MANIFEST_BYTES:
            raise PluginManifestError("Manifest path must be a bounded, regular, non-symlink file.")
        with open_regular_file(target) as stream:
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
    except PluginManifestError:
        raise
    except (OSError, ValueError):
        raise PluginManifestError("Manifest file could not be opened safely.") from None
    if len(raw) > MAX_MANIFEST_BYTES:
        raise PluginManifestError("Manifest file exceeds the 64 KiB limit.")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise PluginManifestError("Manifest file is not valid duplicate-free UTF-8 JSON.") from None
    return document


def _record_plugin_event(event_type: str) -> None:
    try:
        record_security_event(event_type)
    except SecurityEventError:
        emit_error(
            "The plugin policy change may have taken effect, but its security event could not be recorded. Inspect the registry and event journal before retrying.",
            code="security_event_record_failed",
            exit_code=2,
        )


def register(app: typer.Typer) -> None:
    """Register the public axiom plugin command group."""
    command = typer.Typer(
        help="Validate declarative plugin manifests and manage default-deny scoped grants.",
        no_args_is_help=True,
    )
    manifest_commands = typer.Typer(
        help="Validate or register a declarative plugin manifest JSON file.",
        no_args_is_help=True,
    )
    command.add_typer(manifest_commands, name="manifest")
    app.add_typer(command, name="plugin")

    @command.command("list")
    def list_plugins(
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable plugin metadata."),
    ) -> None:
        """List registered plugins, manifest fingerprints, and permission state."""
        try:
            result = PluginManifestStore().list()
        except (OSError, PluginManifestError) as exc:
            emit_error(str(exc), code="plugin_registry_error")
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return
        table = Table(title="AXIOM DECLARATIVE PLUGINS")
        table.add_column("Plugin", style="cyan")
        table.add_column("Version")
        table.add_column("Manifest SHA-256")
        table.add_column("Permission")
        table.add_column("Tools")
        for record in result["manifests"]:
            manifest = record["manifest"]
            table.add_row(
                manifest["id"],
                manifest["version"],
                record["sha256"][:16] + "…",
                record["permission_status"].replace("_", " ").upper(),
                str(len(manifest["tools"])),
            )
        console.print(table)
        console.print(result["execution_note"])

    @manifest_commands.command("validate")
    def validate(
        path: str = typer.Argument(..., help="UTF-8 JSON manifest file."),
        json_output: bool = typer.Option(False, "--json", help="Emit the normalized manifest and SHA-256."),
    ) -> None:
        """Validate a plugin manifest without registering or granting it."""
        try:
            normalized, digest = validate_plugin_manifest(_read_manifest(path))
        except (OSError, PluginManifestError) as exc:
            emit_error(str(exc), code="plugin_manifest_invalid")
        result = {"valid": True, "manifest": normalized, "sha256": digest, "permission_status": "default_deny"}
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            console.print(
                f"[green]Valid manifest[/green] {normalized['id']} {normalized['version']} · "
                f"SHA-256 {digest} · {len(normalized['tools'])} tool(s) · permissions DENY"
            )

    @manifest_commands.command("register")
    def register_manifest(
        path: str = typer.Argument(..., help="UTF-8 JSON manifest file."),
        json_output: bool = typer.Option(False, "--json", help="Emit the registered manifest record."),
    ) -> None:
        """Register a new manifest or a strictly newer version; grants remain denied/stale."""
        try:
            result = PluginManifestStore().register(_read_manifest(path))
        except (OSError, PluginManifestError) as exc:
            emit_error(str(exc), code="plugin_manifest_registration_failed")
        _record_plugin_event("plugin.manifest.registered")
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            console.print(
                f"[green]Registered[/green] {result['manifest']['id']} "
                f"{result['manifest']['version']} · SHA-256 {result['sha256']}"
            )
            console.print("Plugin permissions remain denied until explicitly granted.")

    @command.command("grant")
    def grant_plugin(
        plugin_id: str = typer.Argument(..., help="Registered reverse-DNS plugin ID."),
        manifest_sha256: str = typer.Option(..., "--manifest-sha256", help="Exact current manifest hash from plugin list."),
        tool: list[str] = typer.Option(
            ...,
            "--tool",
            help="Repeat as tool-name:scope[,scope...] (for example search:read).",
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit the scoped grant record."),
    ) -> None:
        """Grant only selected tools and scopes for the exact registered manifest hash."""
        grants = []
        for entry in tool:
            name, separator, raw_scopes = entry.partition(":")
            scopes = raw_scopes.split(",") if separator else []
            if not name or not separator or not scopes or any(not scope for scope in scopes):
                emit_error(
                    "Each --tool must use tool-name:scope[,scope...] syntax.",
                    code="invalid_plugin_grant",
                )
            grants.append({"name": name, "scopes": scopes})
        try:
            result = PluginManifestStore().grant(
                plugin_id,
                expected_sha256=manifest_sha256,
                tools=grants,
            )
        except (OSError, PluginManifestError) as exc:
            emit_error(str(exc), code="plugin_grant_failed")
        _record_plugin_event("plugin.permissions.granted")
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            console.print(
                f"[green]Granted[/green] {plugin_id}: "
                + ", ".join(f"{item['name']} ({', '.join(item['scopes'])})" for item in result["tools"])
            )
            console.print("Only this manifest hash and these exact scopes are approved.")

    @command.command("revoke")
    def revoke_plugin(
        plugin_id: str = typer.Argument(..., help="Registered reverse-DNS plugin ID."),
    ) -> None:
        """Remove all scoped grants for a plugin without deleting its manifest."""
        try:
            revoked = PluginManifestStore().revoke(plugin_id)
        except (OSError, PluginManifestError) as exc:
            emit_error(str(exc), code="plugin_revoke_failed")
        if revoked:
            _record_plugin_event("plugin.permissions.revoked")
            console.print(f"[yellow]Revoked[/yellow] all grants for {plugin_id}.")
        else:
            console.print(f"No active grant exists for {plugin_id}.")
