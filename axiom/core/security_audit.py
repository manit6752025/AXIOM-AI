"""Read-only security posture checks for local AXIOM workspaces and APIs.

The audit reports configuration presence and permissions, never credential
values. It does not probe the network or enumerate operating-system listeners;
network findings describe AXIOM's built-in launch contracts and an optional
operator-supplied daemon bind profile.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.metadata
import importlib.util
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from axiom.core.diagnostics import (
    _MINIMUM_VERSIONS,
    _REQUIRED_MODULES,
    _version_at_least,
    discover_workspace_root,
)
from axiom.core.tool_permissions import ToolPermissionStore
from axiom.path_safety import SafePathError, open_regular_file

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


def _package_integrity_check(
    root: Path,
    module_name: str,
    distribution_name: str,
) -> dict[str, str | None]:
    """Verify a bounded subset of installed package files against RECORD hashes."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return _check(
            f"security.package.{module_name}",
            f"Installed package: {distribution_name}",
            "fail",
            "error",
            f"{distribution_name} distribution metadata is not installed in the active Python environment.",
            f"Install {distribution_name}>={_MINIMUM_VERSIONS[distribution_name]} in the active environment.",
        )

    metadata_name = str(distribution.metadata.get("Name", "")).strip()
    def canonical_name(value: str) -> str:
        return re.sub(r"[-_.]+", "-", value).lower()
    version = str(distribution.version)
    problems: list[str] = []
    if not metadata_name or canonical_name(metadata_name) != canonical_name(distribution_name):
        problems.append("distribution metadata name does not match the required package")

    minimum = _MINIMUM_VERSIONS[distribution_name]
    if not _version_at_least(version, minimum):
        problems.append(f"installed version is below the supported minimum {minimum}")

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is None:
        problems.append("the runtime module cannot be resolved by the active interpreter")
    else:
        module_locations: list[str] = []
        if spec.origin and spec.origin not in {"built-in", "frozen"}:
            module_locations.append(spec.origin)
        if spec.submodule_search_locations:
            module_locations.extend(str(item) for item in spec.submodule_search_locations)
        for location in module_locations:
            try:
                resolved_location = Path(location).resolve(strict=False)
                if resolved_location == root or root in resolved_location.parents:
                    problems.append("the runtime module resolves from the workspace instead of the installed package")
                    break
            except (OSError, RuntimeError):
                problems.append("the runtime module origin could not be resolved safely")
                break

    try:
        package_root = Path(distribution.locate_file("")).resolve(strict=True)
        if not package_root.is_dir():
            raise OSError("package installation root is not a directory")
    except (OSError, RuntimeError):
        return _check(
            f"security.package.{module_name}",
            f"Installed package: {distribution_name}",
            "fail",
            "error",
            f"{distribution_name} installation location could not be resolved safely.",
            "Reinstall the package in the active Python environment.",
        )

    try:
        files = distribution.files
    except (OSError, ValueError):
        files = None
    if files is None:
        problems.append("the installation does not expose a RECORD file inventory")
        files = ()

    verified = 0
    hashed_files = 0
    skipped_external = 0
    bounded = True
    verified_bytes = 0
    for entry_index, package_file in enumerate(files):
        if entry_index >= 512 or hashed_files >= 64 or verified_bytes >= 2 * 1024 * 1024:
            bounded = False
            break
        relative = Path(str(package_file))
        if relative.is_absolute() or ".." in relative.parts:
            # Install schemes may legitimately place console scripts outside site-packages.
            skipped_external += 1
            continue
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            problems.append("the installed RECORD contains an unsafe relative path")
            continue
        expected_hash = getattr(package_file, "hash", None)
        if expected_hash is None:
            continue
        hashed_files += 1
        algorithm = str(getattr(expected_hash, "mode", "")).lower()
        expected_digest = str(getattr(expected_hash, "value", "")).rstrip("=")
        if algorithm not in {"sha256", "sha384", "sha512"} or not re.fullmatch(r"[A-Za-z0-9_-]+", expected_digest):
            problems.append("the installed RECORD contains a weak or malformed file digest")
            continue

        target = package_root
        unsafe_component = False
        for part in relative.parts:
            target = target / part
            if target.is_symlink():
                unsafe_component = True
                break
        if unsafe_component:
            problems.append("an installed package file or parent directory is a symbolic link")
            continue
        try:
            resolved_target = target.resolve(strict=True)
            resolved_target.relative_to(package_root)
            with open_regular_file(resolved_target) as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    problems.append("an installed package RECORD entry is not a regular file")
                    continue
                recorded_size = getattr(package_file, "size", None)
                if recorded_size is not None and int(recorded_size) != info.st_size:
                    problems.append("an installed package file size does not match RECORD")
                    continue
                if info.st_size > 2 * 1024 * 1024 or verified_bytes + info.st_size > 2 * 1024 * 1024:
                    bounded = False
                    break
                hasher = hashlib.new(algorithm)
                actual_size = 0
                while chunk := stream.read(1024 * 1024):
                    actual_size += len(chunk)
                    verified_bytes += len(chunk)
                    hasher.update(chunk)
                digest = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode("ascii")
                if actual_size != info.st_size or not hmac.compare_digest(digest, expected_digest):
                    problems.append("an installed package file does not match its RECORD hash")
                    continue
                verified += 1
        except FileNotFoundError:
            problems.append("an installed package RECORD entry is missing")
        except (OSError, RuntimeError, ValueError, SafePathError):
            problems.append("an installed package RECORD entry could not be opened safely")

    if hashed_files == 0:
        problems.append("no hashed installation files were available to verify")
    if skipped_external:
        problems.append(f"{skipped_external} install-scheme file(s) outside the package directory were not verified")
    if bounded:
        if verified != hashed_files:
            problems.append("not every hashed installation file could be verified")
    else:
        problems.append("the 512-entry, 64-file, or 2 MiB per-distribution inspection limit was reached")

    if problems:
        status = "fail" if any(
            phrase in problem
            for problem in problems
            for phrase in (
                "does not match",
                "symbolic link",
                "unsafe relative path",
                "weak or malformed",
                "is missing",
                "not a regular file",
                "name does not match",
                "below the supported minimum",
                "resolves from the workspace",
                "cannot be resolved",
                "could not be opened safely",
            )
        ) else "warn"
        detail = f"{distribution_name} {version}: " + "; ".join(dict.fromkeys(problems)) + f". Verified {verified} hashed file(s)."
        remediation = "Reinstall the package from a trusted source and review local import paths."
    else:
        detail = f"{distribution_name} {version} meets the minimum {minimum}; verified {verified} installed file hash(es) against RECORD."
        remediation = None
        status = "pass"
    return _check(
        f"security.package.{module_name}",
        f"Installed package: {distribution_name}",
        status,
        "error" if status == "fail" else "warning" if status == "warn" else "info",
        detail,
        remediation,
    )


def _audit_installed_packages(workspace: Path) -> list[dict[str, str | None]]:
    """Audit required runtime packages without importing or executing their code."""
    return [
        _package_integrity_check(workspace, module_name, distribution_name)
        for module_name, distribution_name in _REQUIRED_MODULES.items()
    ]


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
            if os.name == "posix" and mode & 0o022:
                checks.append(_check(
                    "security.config.permissions", "Project configuration permissions", "fail", "error",
                    "axiom.yaml is writable by group or other local users.",
                    "Remove group/other write permission and restrict file ownership to trusted users.",
                ))
            elif os.name == "posix" and mode & 0o044:
                checks.append(_check(
                    "security.config.permissions", "Project configuration permissions", "warn", "warning",
                    "axiom.yaml can be read by group or other local users; configuration may contain local paths or provider settings.",
                    "Use restrictive read permissions when configuration contains sensitive workspace metadata.",
                ))
            else:
                checks.append(_check(
                    "security.config.permissions", "Project configuration permissions", "pass", "info",
                    "No unsafe POSIX group/other configuration permissions were detected.",
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
                "Remote browser origins are allowed while API token-scope enforcement is disabled; CORS is not an authentication boundary.",
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
        "security.network.gui_bind", "Built-in Control Center bind contract", "info", "info",
        "The axiom gui CLI launcher binds its FastAPI socket to 127.0.0.1; this does not assert the bind of externally started ASGI servers.",
    ))
    checks.append(_check(
        "security.network.listener_visibility", "Live listener visibility", "info", "info",
        "This audit does not scan operating-system listeners or inspect services started outside AXIOM's built-in launchers.",
        "Review host firewall rules and externally managed ASGI/server deployments separately.",
    ))

    checks.extend(_audit_installed_packages(root))

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
