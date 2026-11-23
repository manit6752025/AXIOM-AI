"""Bounded SPDX inventory and explicitly network-authorized dependency audit."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from urllib.parse import quote
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Any, Iterable

MAX_DISTRIBUTIONS = 4096
MAX_RELATIONSHIPS = 20_000
MAX_AUDIT_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_AUDIT_STDERR_BYTES = 1 * 1024 * 1024
MAX_SCAN_SECONDS = 90
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]{0,199}$")
_DEPENDENCY_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]{0,199})")


class SupplyChainError(ValueError):
    """An invalid, incomplete, or unsafe software supply-chain operation."""


def _utc_timestamp(value: datetime | None = None) -> str:
    moment = value or datetime.now(UTC)
    if moment.tzinfo is None:
        raise SupplyChainError("Supply-chain report timestamps must include a timezone.")
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _package_id(name: str, version: str) -> str:
    digest = hashlib.sha256(f"{_canonical_name(name)}@{version}".encode("utf-8")).hexdigest()[:20]
    return f"SPDXRef-Package-{digest}"


def _distribution_values(distribution: Any) -> tuple[str, str, list[str]]:
    try:
        metadata = distribution.metadata
        name = str(metadata.get("Name", "")).strip()
        version = str(distribution.version).strip()
        requirements = list(distribution.requires or ())
    except Exception as exc:
        raise SupplyChainError("Installed distribution metadata could not be read safely.") from exc
    if not _NAME_PATTERN.fullmatch(name) or not _VERSION_PATTERN.fullmatch(version):
        raise SupplyChainError("An installed distribution has an invalid or unsupported name/version.")
    if len(requirements) > 4096 or any(not isinstance(item, str) or len(item) > 2048 for item in requirements):
        raise SupplyChainError(f"Dependency metadata for {name} exceeds its supported bounds.")
    return name, version, requirements


def generate_spdx_sbom(
    *,
    distributions: Iterable[Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an SPDX 2.3 JSON SBOM from installed package metadata, without running packages."""
    try:
        found = list(distributions if distributions is not None else importlib.metadata.distributions())
    except Exception as exc:
        raise SupplyChainError("Installed package inventory could not be enumerated safely.") from exc
    if len(found) > MAX_DISTRIBUTIONS:
        raise SupplyChainError(f"Installed package inventory exceeds {MAX_DISTRIBUTIONS} distributions.")

    packages_by_name: dict[str, dict[str, Any]] = {}
    requirements_by_name: dict[str, list[str]] = {}
    for distribution in found:
        name, version, requirements = _distribution_values(distribution)
        canonical = _canonical_name(name)
        if canonical in packages_by_name:
            existing = packages_by_name[canonical]
            if existing["versionInfo"] != version:
                raise SupplyChainError(f"Multiple installed versions of {name} cannot be represented safely.")
            continue
        package_id = _package_id(name, version)
        packages_by_name[canonical] = {
            "name": name,
            "SPDXID": package_id,
            "versionInfo": version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "primaryPackagePurpose": "LIBRARY",
            "externalRefs": [{
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:pypi/{_canonical_name(name)}@{quote(version, safe='.-_+')}",
            }],
        }
        requirements_by_name[canonical] = requirements

    packages = [
        packages_by_name[key]
        for key in sorted(packages_by_name)
    ]
    relationships: list[dict[str, str]] = []
    for package in packages:
        relationships.append({
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": package["SPDXID"],
        })
    for canonical in sorted(requirements_by_name):
        package = packages_by_name[canonical]
        for requirement in requirements_by_name[canonical]:
            match = _DEPENDENCY_NAME_PATTERN.match(requirement)
            if not match:
                continue
            dependency = packages_by_name.get(_canonical_name(match.group(1)))
            if dependency is None:
                continue
            relationships.append({
                "spdxElementId": package["SPDXID"],
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": dependency["SPDXID"],
            })
            if len(relationships) > MAX_RELATIONSHIPS:
                raise SupplyChainError(f"Dependency relationship inventory exceeds {MAX_RELATIONSHIPS} entries.")

    created = _utc_timestamp(generated_at)
    content_fingerprint = hashlib.sha256(
        json.dumps(packages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    namespace = f"https://spdx.org/spdxdocs/axiom-{content_fingerprint}-{created.replace(':', '-')}"
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "AXIOM installed Python environment",
        "documentNamespace": namespace,
        "creationInfo": {
            "creators": ["Tool: AXIOM-AI"],
            "created": created,
            "comment": "Metadata-only inventory of installed Python distributions. No package code was executed.",
        },
        "documentDescribes": [package["SPDXID"] for package in packages],
        "packages": packages,
        "relationships": relationships,
        "annotations": [{
            "annotationDate": created,
            "annotationType": "OTHER",
            "annotator": "Tool: AXIOM-AI",
            "comment": f"Inventory contains {len(packages)} installed distributions. Package file contents and host paths were not collected.",
        }],
        "axiom": {
            "schema_version": 1,
            "package_count": len(packages),
            "relationship_count": len(relationships),
            "content_sha256": content_fingerprint,
            "scope": "active Python environment; installed distribution metadata only",
            "file_contents_collected": False,
            "host_paths_collected": False,
            "network_access_performed": False,
        },
    }


def write_report_new(path: str | Path, content: str) -> Path:
    """Create a new local report with restrictive permissions; never overwrite or follow symlinks."""
    if not isinstance(content, str) or len(content.encode("utf-8")) > 32 * 1024 * 1024:
        raise SupplyChainError("Report content exceeds the 32 MiB export limit.")
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = Path(os.path.abspath(target))
    if target.name in {"", ".", ".."}:
        raise SupplyChainError("Report output must name a file.")
    current = Path(target.anchor)
    for component in target.parent.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise SupplyChainError("Report output parent paths must not contain symbolic links.")
        if not current.is_dir():
            raise SupplyChainError("Report output parent must be an existing directory.")
    if target.is_symlink() or target.exists():
        raise SupplyChainError("Report output already exists; choose a new path.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(str(target), flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            os.chmod(target, 0o600)
    except FileExistsError:
        raise SupplyChainError("Report output already exists; choose a new path.") from None
    except OSError:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise SupplyChainError("Report output could not be written safely.") from None
    return target


def _parse_pip_audit_output(payload: Any, *, generated_at: str, scanner_version: str | None, exit_code: int) -> dict[str, Any]:
    if isinstance(payload, dict):
        dependencies = payload.get("dependencies")
    else:
        dependencies = payload
    if not isinstance(dependencies, list) or len(dependencies) > MAX_DISTRIBUTIONS:
        raise SupplyChainError("Dependency scanner output has an unsupported JSON schema.")

    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise SupplyChainError("Dependency scanner output contains an invalid dependency record.")
        name = dependency.get("name")
        version = dependency.get("version")
        vulnerabilities = dependency.get("vulns", dependency.get("vulnerabilities", []))
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise SupplyChainError("Dependency scanner output contains an invalid package name.")
        if not isinstance(version, str) or not _VERSION_PATTERN.fullmatch(version):
            raise SupplyChainError("Dependency scanner output contains an invalid package version.")
        if not isinstance(vulnerabilities, list) or len(vulnerabilities) > 1000:
            raise SupplyChainError("Dependency scanner output contains an invalid vulnerability list.")
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict):
                raise SupplyChainError("Dependency scanner output contains an invalid vulnerability record.")
            identifier = vulnerability.get("id")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", identifier):
                raise SupplyChainError("Dependency scanner output contains an invalid vulnerability identifier.")
            aliases = vulnerability.get("aliases", [])
            fixes = vulnerability.get("fix_versions", [])
            if not isinstance(aliases, list) or not isinstance(fixes, list):
                raise SupplyChainError("Dependency scanner output contains malformed fix/alias metadata.")
            safe_aliases = sorted({item[:128] for item in aliases if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", item)})[:20]
            safe_fixes = sorted({item[:200] for item in fixes if isinstance(item, str) and _VERSION_PATTERN.fullmatch(item)})[:20]
            key = (_canonical_name(name), version, identifier)
            if key in seen:
                continue
            seen.add(key)
            findings.append({
                "id": identifier,
                "package": name,
                "installed_version": version,
                "aliases": safe_aliases,
                "fix_versions": safe_fixes,
            })

    findings.sort(key=lambda item: (item["package"].casefold(), item["id"]))
    vulnerable_packages = {(item["package"], item["installed_version"]) for item in findings}
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "scanner": {"name": "pip-audit", "version": scanner_version},
        "status": "complete",
        "scan_scope": "active Python environment; installed distributions",
        "network_accessed": True,
        "dependency_count": len(dependencies),
        "vulnerable_dependency_count": len(vulnerable_packages),
        "vulnerability_count": len(findings),
        "findings": findings,
        "exit_code": exit_code,
        "privacy": {
            "credentials_included": False,
            "environment_values_included": False,
            "package_file_contents_included": False,
            "network_identifiers_included": False,
        },
        "limitations": [
            "The report reflects the scanner's advisory sources and timestamp; it is not a guarantee of exploitability or absence of unknown vulnerabilities.",
            "Only the active Python environment was scanned; OS packages, browser dependencies and optional model runtimes are outside this scope.",
        ],
    }


def _consume_bounded_pipe(stream: Any, maximum_bytes: int, result: dict[str, Any]) -> None:
    """Drain a child pipe without retaining more than its declared byte budget."""
    captured = bytearray()
    total = 0
    try:
        while True:
            block = stream.read(64 * 1024)
            if not block:
                break
            total += len(block)
            remaining = maximum_bytes - len(captured)
            if remaining > 0:
                captured.extend(block[:remaining])
            if total > maximum_bytes:
                result["overflow"] = True
    except OSError:
        result["read_error"] = True
    result["content"] = bytes(captured)


def _scanner_failure(generated_at: str, scanner_version: str | None, message: str, *, network_accessed: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "scanner": {"name": "pip-audit", "version": scanner_version},
        "status": "error",
        "network_accessed": network_accessed,
        "dependency_count": 0,
        "vulnerable_dependency_count": 0,
        "vulnerability_count": 0,
        "findings": [],
        "message": message,
    }


def audit_dependency_vulnerabilities(
    *,
    allow_network: bool,
    timeout_seconds: int = MAX_SCAN_SECONDS,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run pip-audit only on explicit operator request; the scanner may contact advisory services."""
    if type(allow_network) is not bool or not allow_network:
        raise SupplyChainError("A dependency vulnerability scan requires explicit network authorization.")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 300:
        raise SupplyChainError("Dependency scan timeout must be an integer between 1 and 300 seconds.")
    generated_at = _utc_timestamp()
    try:
        spec = importlib.util.find_spec("pip_audit")
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is None:
        return {
            "schema_version": 1,
            "generated_at": generated_at,
            "scanner": {"name": "pip-audit", "version": None},
            "status": "unavailable",
            "network_accessed": False,
            "dependency_count": 0,
            "vulnerable_dependency_count": 0,
            "vulnerability_count": 0,
            "findings": [],
            "message": "pip-audit is not installed in the active Python environment; no vulnerability conclusion was produced.",
            "privacy": {"credentials_included": False, "environment_values_included": False},
        }

    try:
        scanner_version = importlib.metadata.version("pip-audit")
    except importlib.metadata.PackageNotFoundError:
        scanner_version = None
    executable = python_executable or sys.executable
    if not isinstance(executable, str) or not executable or len(executable) > 4096:
        raise SupplyChainError("Python scanner executable is invalid.")

    # Do not pass arbitrary environment variables (notably PIP_INDEX_URL, private index URLs, or tokens)
    # into the network scanner. Standard OS/temp/certificate paths are retained.
    allowed_environment = (
        "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
    )
    environment = {key: os.environ[key] for key in allowed_environment if key in os.environ}
    # Do not let pip configuration in the invoking workspace or user profile
    # inject private package-index URLs or credentials into the audit process.
    environment["PIP_CONFIG_FILE"] = os.devnull
    args = [executable, "-m", "pip_audit", "--local", "--format", "json", "--progress-spinner", "off"]
    scan_cwd = Path(sys.prefix)
    if not scan_cwd.is_dir():
        raise SupplyChainError("The active Python environment directory is unavailable.")
    try:
        completed = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            cwd=str(scan_cwd),
        )
        if completed.stdout is None or completed.stderr is None:
            completed.kill()
            completed.wait()
            raise SupplyChainError("Dependency scanner output streams could not be bounded.")
        stdout_capture: dict[str, Any] = {}
        stderr_capture: dict[str, Any] = {}
        stdout_reader = Thread(
            target=_consume_bounded_pipe,
            args=(completed.stdout, MAX_AUDIT_OUTPUT_BYTES, stdout_capture),
            daemon=True,
        )
        stderr_reader = Thread(
            target=_consume_bounded_pipe,
            args=(completed.stderr, MAX_AUDIT_STDERR_BYTES, stderr_capture),
            daemon=True,
        )
        stdout_reader.start()
        stderr_reader.start()
        try:
            return_code = completed.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            completed.kill()
            completed.wait()
            stdout_reader.join(timeout=5)
            stderr_reader.join(timeout=5)
            return _scanner_failure(
                generated_at,
                scanner_version,
                "Dependency scanner timed out; no complete vulnerability conclusion was produced.",
                network_accessed=True,
            )
        stdout_reader.join(timeout=5)
        stderr_reader.join(timeout=5)
        if stdout_reader.is_alive() or stderr_reader.is_alive():
            completed.kill()
            raise SupplyChainError("Dependency scanner output streams did not close within the allowed time.")
        if stdout_capture.get("read_error") or stderr_capture.get("read_error"):
            raise SupplyChainError("Dependency scanner output could not be read safely.")
        if stdout_capture.get("overflow"):
            return _scanner_failure(
                generated_at,
                scanner_version,
                "Dependency scanner JSON output exceeded the 8 MiB limit; no vulnerability conclusion was produced.",
                network_accessed=True,
            )
        if stderr_capture.get("overflow"):
            return _scanner_failure(
                generated_at,
                scanner_version,
                "Dependency scanner diagnostic output exceeded the 1 MiB limit; no vulnerability conclusion was produced.",
                network_accessed=True,
            )
        raw = stdout_capture.get("content", b"")
    except SupplyChainError:
        raise
    except OSError:
        return _scanner_failure(
            generated_at,
            scanner_version,
            "Dependency scanner could not be started or its output could not be captured safely.",
            network_accessed=False,
        )
    if return_code not in {0, 1}:
        return {
            "schema_version": 1,
            "generated_at": generated_at,
            "scanner": {"name": "pip-audit", "version": scanner_version},
            "status": "error",
            "network_accessed": True,
            "dependency_count": 0,
            "vulnerable_dependency_count": 0,
            "vulnerability_count": 0,
            "findings": [],
            "message": "Dependency scanner failed; no complete vulnerability conclusion was produced.",
            "exit_code": return_code,
        }
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise SupplyChainError("Dependency scanner returned invalid JSON; no vulnerability conclusion was produced.") from None
    return _parse_pip_audit_output(
        payload, generated_at=generated_at, scanner_version=scanner_version, exit_code=return_code
    )
