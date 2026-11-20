"""Read-only security posture checks for local AXIOM workspaces and APIs.

The audit reports configuration presence and permissions, never credential
values. It does not probe the network or enumerate operating-system listeners;
network findings describe AXIOM's built-in launch contracts and an optional
operator-supplied daemon bind profile.
"""
from __future__ import annotations

import hmac
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from axiom.core.diagnostics import discover_workspace_root
from axiom.core.tool_permissions import ToolPermissionStore

_SCHEMA_VERSION = 1
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _check(
    check_id: str,
    title: str,
    status: str,
    severity: str,
    detail: str,
    remediation: str | None = None,
) -> dict[str, str | None]:
    return {
        "id": check_id,
        "title": title,
        "status": status,
        "severity": severity,
        "detail": detail,
        "remediation": remediation,
    }


def _is_loopback_origin(hostname: str) -> bool:
    normalized = hostname.lower().rstrip(".")
    if normalized in _LOOPBACK_NAMES:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _cors_origin_status(origin: str) -> tuple[str, str]:
    """Return (status, reason) without echoing a configured origin."""
    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
        # Accessing .port validates malformed/out-of-range port syntax.
        _ = parsed.port
    except ValueError:
        return "fail", "An allowed browser origin has invalid URL syntax."
    if (
        not hostname
        or parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return "fail", "An allowed browser origin is not a strict HTTP(S) origin."
    if parsed.scheme == "http" and not _is_loopback_origin(hostname):
        return "fail", "A non-loopback browser origin uses unencrypted HTTP."
    if not _is_loopback_origin(hostname):
        return "warn", "A remote HTTPS browser origin is allowed; CORS is not an authentication boundary."
    return "pass", "All configured browser origins are loopback-only."


def audit_security_posture(
    workspace: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    daemon_host: str = "127.0.0.1",
    allow_network: bool = False,
) -> dict:
    """Audit local configuration, API credentials, MCP grants, CORS, and bind intent.

    `daemon_host` and `allow_network` describe the target daemon profile being
    evaluated. They do not claim to discover or change a running listener.
    """
    env = os.environ if environ is None else environ
    root = discover_workspace_root(workspace, environ=env)
    checks: list[dict[str, str | None]] = []

    config = root / "axiom.yaml"
    if config.is_symlink():
        checks.append(_check(
            "security.config.symlink", "Project configuration path", "fail", "error",
            "axiom.yaml is a symbolic link and could redirect configuration outside the workspace.",
            "Replace it with a regular file after verifying the intended configuration.",
        ))
    elif config.is_file():
        try:
            mode = config.stat().st_mode
        except OSError:
            checks.append(_check(
                "security.config.permissions", "Project configuration permissions", "warn", "warning",
                "Configuration file permissions could not be inspected.",
                "Review the file owner and access bits manually.",
            ))
        else:
            if mode & 0o022:
                checks.append(_check(
                    "security.config.permissions", "Project configuration permissions", "fail", "error",
                    "axiom.yaml is writable by group or other local users.",
                    "Remove group/other write permission and restrict file ownership to trusted users.",
                ))
            elif mode & 0o044:
                checks.append(_check(
                    "security.config.permissions", "Project configuration permissions", "warn", "warning",
                    "axiom.yaml can be read by group or other local users; configuration may contain local paths or provider settings.",
                    "Use restrictive read permissions when configuration contains sensitive workspace metadata.",
                ))
            else:
                checks.append(_check(
                    "security.config.permissions", "Project configuration permissions", "pass", "info",
                    "axiom.yaml is not group/other readable or writable.",
                ))
    else:
        checks.append(_check(
            "security.config.permissions", "Project configuration permissions", "info", "info",
            "No regular axiom.yaml file is present in the selected workspace.",
        ))

    metadata_dir = root / ".axiom"
    policy_file = metadata_dir / "tool_permissions.json"
    if metadata_dir.is_symlink() or policy_file.is_symlink():
        checks.append(_check(
            "security.mcp.policy_path", "MCP permission policy path", "fail", "error",
            "The .axiom metadata directory or MCP policy file is a symbolic link.",
            "Remove the unexpected link and restore a workspace-local metadata directory.",
        ))
    elif not metadata_dir.exists():
        checks.append(_check(
            "security.mcp.policy_path", "MCP permission policy path", "pass", "info",
            "No MCP policy file exists; AXIOM's implicit default profile denies every tool.",
        ))
    elif not metadata_dir.is_dir():
        checks.append(_check(
            "security.mcp.policy_path", "MCP permission policy path", "fail", "error",
            "The .axiom metadata entry is not a directory.",
            "Replace the conflicting entry with a restricted local directory.",
        ))
    else:
        try:
            directory_mode = metadata_dir.stat().st_mode
            policy_mode = policy_file.stat().st_mode if policy_file.exists() else None
        except OSError:
            checks.append(_check(
                "security.mcp.policy_permissions", "MCP policy permissions", "warn", "warning",
                "MCP policy permissions could not be inspected.",
                "Review .axiom ownership and permissions manually.",
            ))
        else:
            if os.name == "posix" and directory_mode & 0o022:
                checks.append(_check(
                    "security.mcp.policy_permissions", "MCP policy permissions", "fail", "error",
                    "The .axiom directory is writable by group or other local users.",
                    "Restrict the .axiom directory to its owning user.",
                ))
            elif policy_mode is not None and stat.S_ISREG(policy_mode):
                if os.name == "posix" and policy_mode & 0o077:
                    checks.append(_check(
                        "security.mcp.policy_permissions", "MCP policy permissions", "fail", "error",
                        "tool_permissions.json is accessible to group or other local users.",
                        "Restrict tool_permissions.json to the owning user (for example, mode 0600).",
                    ))
                else:
                    checks.append(_check(
                        "security.mcp.policy_permissions", "MCP policy permissions", "pass", "info",
                        "The MCP policy is a regular file with restricted local permissions.",
                    ))
            elif policy_mode is None:
                checks.append(_check(
                    "security.mcp.policy_permissions", "MCP policy permissions", "pass", "info",
                    "No persisted MCP policy exists; all tools remain denied by default.",
                ))
            else:
                checks.append(_check(
                    "security.mcp.policy_permissions", "MCP policy permissions", "fail", "error",
                    "tool_permissions.json is not a regular file.",
                    "Replace it with a valid regular policy file or remove it to restore default-deny behavior.",
                ))

    try:
        policy = ToolPermissionStore(metadata_dir).list()
        active_name = policy["active_profile"]
        active_profile = next(
            profile for profile in policy["profiles"] if profile["name"] == active_name
        )
        granted = len(active_profile["allowed_tools"])
        checks.append(_check(
            "security.mcp.default_deny", "MCP active permission profile", "pass" if granted == 0 else "warn",
            "info" if granted == 0 else "warning",
            "The active MCP profile grants no tools; all tool calls are denied."
            if granted == 0 else
            f"The active MCP profile contains {granted} explicit tool grant(s); each grant should be reviewed.",
            None if granted == 0 else "Review active grants in the Control Center MCP permissions page and revoke unused tools.",
        ))
    except (OSError, ValueError, KeyError, StopIteration):
        checks.append(_check(
            "security.mcp.default_deny", "MCP active permission profile", "fail", "error",
            "The MCP permission policy is invalid or could not be read safely.",
            "Run the MCP permission diagnostics and restore a validated default-deny policy.",
        ))

    read_token = env.get("AXIOM_API_READ_TOKEN")
    write_token = env.get("AXIOM_API_WRITE_TOKEN")
    if read_token is None and write_token is None:
        scopes_enabled = False
        checks.append(_check(
            "security.api.scopes", "API read/write token scopes", "warn", "warning",
            "API token-scope enforcement is disabled; keep the built-in API on loopback.",
            "Configure distinct high-entropy AXIOM_API_READ_TOKEN and AXIOM_API_WRITE_TOKEN values before exposing API routes.",
        ))
    elif not read_token or not write_token:
        scopes_enabled = False
        checks.append(_check(
            "security.api.scopes", "API read/write token scopes", "fail", "error",
            "Only one API scope token is configured; AXIOM requires the read and write pair.",
            "Configure both scope tokens with distinct 32–256 character values.",
        ))
    elif not _TOKEN_PATTERN.fullmatch(read_token) or not _TOKEN_PATTERN.fullmatch(write_token):
        scopes_enabled = False
        checks.append(_check(
            "security.api.scopes", "API read/write token scopes", "fail", "error",
            "One or both API scope tokens do not meet the required length and character policy.",
            "Use two independently generated 32–256 character URL-safe tokens.",
        ))
    elif hmac.compare_digest(read_token, write_token):
        scopes_enabled = False
        checks.append(_check(
            "security.api.scopes", "API read/write token scopes", "fail", "error",
            "The API read and write scope tokens are identical.",
            "Generate independent read and write tokens.",
        ))
    else:
        scopes_enabled = True
        checks.append(_check(
            "security.api.scopes", "API read/write token scopes", "pass", "info",
            "Distinct, policy-compliant API read and write token scopes are configured; token values are withheld.",
        ))

    openai_token = env.get("AXIOM_OPENAI_API_TOKEN", "")
    if openai_token:
        checks.append(_check(
            "security.api.openai_token", "OpenAI-compatible API token", "pass", "info",
            "A separate bearer token is configured for the OpenAI-compatible API; its value is withheld.",
        ))
    else:
        checks.append(_check(
            "security.api.openai_token", "OpenAI-compatible API token", "warn" if not scopes_enabled else "pass",
            "warning" if not scopes_enabled else "info",
            "No separate OpenAI-compatible bearer token is configured."
            if not scopes_enabled else
            "No separate OpenAI-compatible bearer token is configured; the API read/write scope middleware remains enabled.",
            "Configure AXIOM_OPENAI_API_TOKEN when the OpenAI-compatible endpoint needs its own credential boundary."
            if not scopes_enabled else None,
        ))

    cors_value = env.get("AXIOM_CORS_ORIGINS")
    origins = tuple(item.strip() for item in cors_value.split(",")) if cors_value is not None else _DEFAULT_CORS_ORIGINS
    if len(origins) > 32 or any(not item for item in origins):
        checks.append(_check(
            "security.cors.origins", "Browser CORS origins", "fail", "error",
            "The CORS origin list is empty or exceeds the 32-origin audit limit.",
            "Configure a bounded comma-separated list of exact origins.",
        ))
    elif any(item == "*" for item in origins):
        checks.append(_check(
            "security.cors.origins", "Browser CORS origins", "fail", "error",
            "A wildcard browser origin is configured.",
            "Replace wildcard CORS with exact trusted origins; CORS is not authentication.",
        ))
    else:
        origin_results = [_cors_origin_status(item) for item in origins]
        if any(status == "fail" for status, _reason in origin_results):
            checks.append(_check(
                "security.cors.origins", "Browser CORS origins", "fail", "error",
                "At least one allowed browser origin is malformed or uses unencrypted remote HTTP.",
                "Use exact loopback development origins or HTTPS origins that you explicitly trust.",
            ))
        elif any(status == "warn" for status, _reason in origin_results) and not scopes_enabled:
            checks.append(_check(
                "security.cors.origins", "Browser CORS origins", "warn", "warning",
                "Remote browser origins are allowed while API token-scope enforcement is disabled.",
                "Restrict CORS to loopback or enable API read/write token scopes.",
            ))
        elif any(status == "warn" for status, _reason in origin_results):
            checks.append(_check(
                "security.cors.origins", "Browser CORS origins", "warn", "warning",
                "Remote HTTPS browser origins are allowed; CORS does not replace authentication.",
                "Keep the allowed origin list limited to trusted applications.",
            ))
        else:
            checks.append(_check(
                "security.cors.origins", "Browser CORS origins", "pass", "info",
                f"All {len(origins)} configured browser origins are strict loopback HTTP(S) origins.",
            ))

    normalized_host = daemon_host.strip()
    is_loopback = normalized_host.lower() in _LOOPBACK_NAMES
    if not is_loopback:
        try:
            import ipaddress

            is_loopback = ipaddress.ip_address(normalized_host).is_loopback
        except ValueError:
            is_loopback = False
    if is_loopback:
        checks.append(_check(
            "security.network.daemon_bind", "Daemon network binding", "pass", "info",
            "The audited daemon profile binds to loopback and is not intended to accept remote connections.",
        ))
    elif not allow_network:
        checks.append(_check(
            "security.network.daemon_bind", "Daemon network binding", "fail", "error",
            "The requested non-loopback daemon host would be rejected without --allow-network.",
            "Use 127.0.0.1/localhost or explicitly review the remote-exposure risks before opting in.",
        ))
    else:
        checks.append(_check(
            "security.network.daemon_bind", "Daemon network binding", "fail", "error",
            "The requested daemon profile exposes unauthenticated HTTP endpoints beyond loopback.",
            "Do not expose the lightweight daemon directly; keep it on loopback and use an authenticated, encrypted tunnel for remote access.",
        ))

    checks.append(_check(
        "security.network.gui_bind", "Control Center network binding", "pass", "info",
        "The built-in axiom gui launcher binds its FastAPI listener to 127.0.0.1.",
    ))
    checks.append(_check(
        "security.network.listener_visibility", "Live listener visibility", "info", "info",
        "This audit does not scan operating-system listeners or inspect services started outside AXIOM's built-in launchers.",
        "Review host firewall rules and externally managed ASGI/server deployments separately.",
    ))

    failed = sum(item["status"] == "fail" for item in checks)
    warnings = sum(item["status"] == "warn" for item in checks)
    passed = sum(item["status"] == "pass" for item in checks)
    informational = sum(item["status"] == "info" for item in checks)
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "fail" if failed else "warn" if warnings else "pass",
        "workspace": {"name": root.name or "workspace"},
        "summary": {
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "informational": informational,
            "api_scope_tokens_enabled": scopes_enabled,
        },
        "checks": checks,
    }
