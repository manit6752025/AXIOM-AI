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
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Iterable
from urllib.parse import quote

MAX_DISTRIBUTIONS = 4096
MAX_RELATIONSHIPS = 20_000
MAX_TOTAL_REQUIREMENTS = 50_000
MAX_AUDIT_FINDINGS = 20_000
MAX_AUDIT_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_AUDIT_STDERR_BYTES = 1 * 1024 * 1024
MAX_SCAN_SECONDS = 90
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]{0,199}$")
_AUDIT_LOCK = Lock()
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
        raw_requirements = distribution.requires or ()
        requirements: list[str] = []
        for item in raw_requirements:
            if len(requirements) >= 4096:
                raise SupplyChainError(f"Dependency metadata for {name} exceeds its supported bounds.")
            if not isinstance(item, str) or len(item) > 2048:
                raise SupplyChainError(f"Dependency metadata for {name} exceeds its supported bounds.")
            requirements.append(item)
    except SupplyChainError:
        raise
    except Exception as exc:
        raise SupplyChainError("Installed distribution metadata could not be read safely.") from exc
    if not _NAME_PATTERN.fullmatch(name) or not _VERSION_PATTERN.fullmatch(version):
        raise SupplyChainError("An installed distribution has an invalid or unsupported name/version.")
    return name, version, requirements


def generate_spdx_sbom(
    *,
    distributions: Iterable[Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an SPDX 2.3 JSON SBOM from installed package metadata, without running packages."""
    try:
        source = distributions if distributions is not None else importlib.metadata.distributions()
        found: list[Any] = []
        for distribution in source:
            if len(found) >= MAX_DISTRIBUTIONS:
                raise SupplyChainError(f"Installed package inventory exceeds {MAX_DISTRIBUTIONS} distributions.")
            found.append(distribution)
    except SupplyChainError:
        raise
    except Exception as exc:
        raise SupplyChainError("Installed package inventory could not be enumerated safely.") from exc

    packages_by_name: dict[str, dict[str, Any]] = {}
    requirements_by_name: dict[str, list[str]] = {}
    total_requirements = 0
    for distribution in found:
        name, version, requirements = _distribution_values(distribution)
        total_requirements += len(requirements)
        if total_requirements > MAX_TOTAL_REQUIREMENTS:
            raise SupplyChainError(
                f"Installed dependency metadata exceeds {MAX_TOTAL_REQUIREMENTS} requirement entries."
            )
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
        dependency_ids: set[str] = set()
        for requirement in sorted(requirements_by_name[canonical], key=str.casefold):
            match = _DEPENDENCY_NAME_PATTERN.match(requirement)
            if not match:
                continue
            dependency = packages_by_name.get(_canonical_name(match.group(1)))
            if dependency is not None:
                dependency_ids.add(dependency["SPDXID"])
        for dependency_id in sorted(dependency_ids):
            relationships.append({
                "spdxElementId": package["SPDXID"],
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": dependency_id,
            })
            if len(relationships) > MAX_RELATIONSHIPS:
                raise SupplyChainError(f"Dependency relationship inventory exceeds {MAX_RELATIONSHIPS} entries.")

    created = _utc_timestamp(generated_at)
    content_fingerprint = hashlib.sha256(
        json.dumps(
            {"packages": packages, "relationships": relationships},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
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
            "comment": (
                f"AXIOM inventory: schema_version=1; package_count={len(packages)}; "
                f"relationship_count={len(relationships)}; content_sha256={content_fingerprint}; "
                "scope=active Python environment, installed distribution metadata only; "
                "file_contents_collected=false; host_paths_collected=false; "
                "network_access_performed=false; package_code_executed=false."
            ),
        }],
    }


def write_report_new(path: str | Path, content: str) -> Path:
    """Create a private report atomically without overwriting or following symlinks."""
    if not isinstance(content, str) or len(content.encode("utf-8")) > 32 * 1024 * 1024:
        raise SupplyChainError("Report content exceeds the 32 MiB export limit.")
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = Path(os.path.abspath(target))
    if target.name in {"", ".", ".."}:
        raise SupplyChainError("Report output must name a file.")

    # Keep a human-readable rejection for pre-existing symlinks, then use
    # directory descriptors on POSIX so replacing a parent with a symlink
    # between validation and creation cannot redirect the output.
    current = Path(target.anchor)
    for component in target.parent.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise SupplyChainError("Report output parent paths must not contain symbolic links.")
        if not current.is_dir():
            raise SupplyChainError("Report output parent must be an existing directory.")
    if target.is_symlink() or target.exists():
        raise SupplyChainError("Report output already exists; choose a new path.")

    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags |= getattr(os, "O_BINARY", 0)
    directory_fd: int | None = None
    descriptor: int | None = None
    created = False
    supports_directory_descriptors = (
        os.open in getattr(os, "supports_dir_fd", set())
        and os.unlink in getattr(os, "supports_dir_fd", set())
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )
    try:
        if supports_directory_descriptors:
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            directory_fd = os.open(target.anchor, directory_flags)
            for component in target.parent.parts[1:]:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            descriptor = os.open(target.name, file_flags, 0o600, dir_fd=directory_fd)
        else:
            descriptor = os.open(str(target), file_flags, 0o600)
        created = True
        if os.name != "nt" and hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        descriptor = None
        with stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if directory_fd is not None:
            os.fsync(directory_fd)
    except FileExistsError:
        raise SupplyChainError("Report output already exists; choose a new path.") from None
    except SupplyChainError:
        if created:
            try:
                if directory_fd is not None:
                    os.unlink(target.name, dir_fd=directory_fd)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    except OSError:
        if created:
            try:
                if directory_fd is not None:
                    os.unlink(target.name, dir_fd=directory_fd)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                pass
        raise SupplyChainError("Report output could not be written safely.") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass
    return target

def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SupplyChainError("Dependency scanner JSON contains duplicate object keys.")
        result[key] = value
    return result


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
            if len(findings) > MAX_AUDIT_FINDINGS:
                raise SupplyChainError(
                    f"Dependency audit finding inventory exceeds {MAX_AUDIT_FINDINGS} entries."
                )

    if (exit_code == 0 and findings) or (exit_code == 1 and not findings):
        raise SupplyChainError(
            "Dependency scanner exit status conflicts with its vulnerability inventory; no complete conclusion was produced."
        )
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
    except (OSError, ValueError):
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


def _audit_dependency_vulnerabilities_unlocked(
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
    if (
        not isinstance(executable, str)
        or not executable
        or len(executable) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in executable)
    ):
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
    args = [executable, "-m", "pip_audit", "--local", "--skip-editable", "--format", "json", "--progress-spinner", "off"]
    scan_cwd = Path(sys.prefix)
    if not scan_cwd.is_dir():
        raise SupplyChainError("The active Python environment directory is unavailable.")
    completed: subprocess.Popen[bytes] | None = None
    stdout_reader: Thread | None = None
    stderr_reader: Thread | None = None
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
        if completed is not None:
            try:
                completed.kill()
                completed.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if stdout_reader is not None:
            stdout_reader.join(timeout=5)
        if stderr_reader is not None:
            stderr_reader.join(timeout=5)
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
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise SupplyChainError("Dependency scanner returned invalid or ambiguous JSON; no vulnerability conclusion was produced.") from None
    return _parse_pip_audit_output(
        payload, generated_at=generated_at, scanner_version=scanner_version, exit_code=return_code
    )
def audit_dependency_vulnerabilities(
    *,
    allow_network: bool,
    timeout_seconds: int = MAX_SCAN_SECONDS,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run at most one bounded advisory scan in this AXIOM process at a time."""
    if type(allow_network) is not bool or not allow_network:
        raise SupplyChainError("A dependency vulnerability scan requires explicit network authorization.")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 300:
        raise SupplyChainError("Dependency scan timeout must be an integer between 1 and 300 seconds.")
    if not _AUDIT_LOCK.acquire(blocking=False):
        return _scanner_failure(
            _utc_timestamp(),
            None,
            "Another dependency scan is already active in this AXIOM process; no second scanner was started.",
            network_accessed=False,
        )
    try:
        return _audit_dependency_vulnerabilities_unlocked(
            allow_network=allow_network,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
    finally:
        _AUDIT_LOCK.release()

