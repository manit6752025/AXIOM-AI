"""Safe AXIOM release discovery and Python-wheel self-updates.

Only the application distribution is replaced. User workspaces, models, browser
storage and AXIOM configuration live outside the wheel and are never opened here.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

GITHUB_RELEASES_URL = "https://api.github.com/repos/manit6752025/AXIOM-AI/releases?per_page=30"
PYPI_JSON_URL = "https://pypi.org/pypi/axiom-all/{version}/json"
PACKAGE_NAME = "axiom-all"
MAX_WHEEL_BYTES = 150 * 1024 * 1024
_VERSION_RE = re.compile(
    r"^v?(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)(?:-(?P<pre>[0-9A-Za-z.-]+))?"
    r"(?:\+[0-9A-Za-z.-]+)?$"
)
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class UpdateError(RuntimeError):
    """The update cannot be safely checked or installed."""


def _version_key(value: str) -> tuple[Any, ...]:
    if not isinstance(value, str):
        raise UpdateError("Release version is not a string.")
    match = _VERSION_RE.fullmatch(value)
    if not match:
        raise UpdateError(f"Unsupported AXIOM release version: {value!r}.")
    core = tuple(int(match.group(key)) for key in ("major", "minor", "patch"))
    prerelease = match.group("pre")
    if prerelease is None:
        return core, 1, ()
    parts: list[tuple[int, int | str]] = []
    for token in prerelease.split("."):
        if not token:
            raise UpdateError(f"Invalid prerelease identifier in {value!r}.")
        parts.append((0, int(token)) if token.isdigit() else (1, token.lower()))
    return core, 0, tuple(parts)


def compare_versions(left: str, right: str) -> int:
    """Compare AXIOM semantic versions, including numeric prerelease identifiers."""
    a, b = _version_key(left), _version_key(right)
    return (a > b) - (a < b)


def _read_url(url: str, *, max_bytes: int, timeout: float = 20.0) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/json, application/octet-stream",
            "User-Agent": "AXIOM-CLI-Updater",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read(max_bytes + 1)
    except HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("The requested AXIOM release/package is not published yet.") from None
        raise UpdateError(f"Update server returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"Could not reach the update server: {exc}.") from None
    if len(payload) > max_bytes:
        raise UpdateError("Update download exceeded the configured size limit.")
    return payload


def fetch_latest_release(channel: str = "beta") -> dict[str, Any] | None:
    """Fetch the highest published, non-draft stable or beta GitHub release."""
    if channel not in {"stable", "beta"}:
        raise ValueError("Update channel must be 'stable' or 'beta'.")
    payload = _read_url(GITHUB_RELEASES_URL, max_bytes=4 * 1024 * 1024)
    try:
        releases = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise UpdateError("GitHub returned invalid release metadata.") from None
    if not isinstance(releases, list):
        raise UpdateError("GitHub returned an unexpected releases response.")
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for release in releases:
        if not isinstance(release, dict) or release.get("draft") is True:
            continue
        if channel == "stable" and release.get("prerelease") is True:
            continue
        tag = release.get("tag_name")
        try:
            key = _version_key(tag)
        except UpdateError:
            continue
        candidates.append((key, release))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def check_for_update(current_version: str, channel: str = "beta") -> dict[str, Any]:
    latest = fetch_latest_release(channel)
    if latest is None:
        return {
            "channel": channel,
            "current_version": current_version,
            "latest_version": None,
            "update_available": False,
            "release": None,
        }
    latest_version = latest["tag_name"].removeprefix("v")
    available = compare_versions(latest_version, current_version) > 0
    return {
        "channel": channel,
        "current_version": current_version,
        "latest_version": latest_version,
        "update_available": available,
        "release": {
            "tag_name": latest["tag_name"],
            "name": latest.get("name") or latest["tag_name"],
            "html_url": latest.get("html_url"),
            "published_at": latest.get("published_at"),
            "prerelease": latest.get("prerelease") is True,
        },
    }


def _validate_wheel(payload: bytes, expected_version: str) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise UpdateError("Downloaded wheel failed ZIP integrity validation.")
            names = archive.namelist()
            metadata_name = next(
                (name for name in names if name.endswith(".dist-info/METADATA")),
                None,
            )
            if metadata_name is None:
                raise UpdateError("Downloaded wheel is missing package metadata.")
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
    except (zipfile.BadZipFile, OSError, KeyError):
        raise UpdateError("Downloaded AXIOM wheel is not a valid Python wheel.") from None
    package_name = (metadata.get("Name") or "").lower().replace("_", "-")
    if package_name != PACKAGE_NAME or metadata.get("Version") != expected_version:
        raise UpdateError("Wheel metadata does not match the requested AXIOM package/version.")
    if "axiom/webui/index.html" not in names:
        raise UpdateError("Wheel is missing the bundled AXIOM Control Center.")


def _wheel_for_version(version: str) -> tuple[bytes, str]:
    url = PYPI_JSON_URL.format(version=quote(version, safe=""))
    payload = _read_url(url, max_bytes=4 * 1024 * 1024)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise UpdateError("PyPI returned invalid package metadata.") from None
    info = document.get("info") if isinstance(document, dict) else None
    normalized_name = str(info.get("name", "")).lower().replace("_", "-") if isinstance(info, dict) else ""
    if normalized_name != PACKAGE_NAME or info.get("version") != version:
        raise UpdateError("PyPI package identity/version does not match the GitHub release.")
    distributions = document.get("urls")
    if not isinstance(distributions, list):
        raise UpdateError("PyPI did not publish a wheel for this release.")
    wheels = [
        item for item in distributions
        if isinstance(item, dict)
        and item.get("packagetype") == "bdist_wheel"
        and isinstance(item.get("url"), str)
        and isinstance(item.get("digests"), dict)
        and isinstance(item["digests"].get("sha256"), str)
        and _SHA256_RE.fullmatch(item["digests"]["sha256"])
        and item.get("yanked") is not True
    ]
    if not wheels:
        raise UpdateError("No non-yanked, SHA-256-verifiable Python wheel is published for this release.")
    wheel = next((item for item in wheels if item.get("filename", "").endswith("py3-none-any.whl")), wheels[0])
    download_url = wheel["url"]
    parsed_url = urlsplit(download_url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "files.pythonhosted.org" or parsed_url.username or parsed_url.password:
        raise UpdateError("PyPI supplied a wheel URL outside the trusted HTTPS package-file host.")
    wheel_bytes = _read_url(download_url, max_bytes=MAX_WHEEL_BYTES, timeout=60.0)
    actual = hashlib.sha256(wheel_bytes).hexdigest()
    if not hmac.compare_digest(actual, wheel["digests"]["sha256"]):
        raise UpdateError("Downloaded AXIOM wheel failed its PyPI SHA-256 verification.")
    _validate_wheel(wheel_bytes, version)
    return wheel_bytes, str(wheel.get("filename") or f"axiom_all-{version}-py3-none-any.whl")


def download_and_install(version: str) -> dict[str, Any]:
    """Verify and install one exact PyPI wheel using the active Python interpreter."""
    _version_key(version)
    wheel_bytes, filename = _wheel_for_version(version)
    with tempfile.TemporaryDirectory(prefix="axiom-update-") as temporary:
        wheel_path = Path(temporary) / Path(filename).name
        wheel_path.write_bytes(wheel_bytes)
        from axiom.core.artifact_security import inspect_release_artifact

        inspection = inspect_release_artifact(wheel_path)
        if inspection.get("artifact", {}).get("format") != "wheel" or inspection.get("status") != "pass":
            summary = inspection.get("summary", {})
            raise UpdateError(
                "Downloaded AXIOM wheel failed full RECORD/archive inspection "
                f"(failed checks: {summary.get('failed', 'unknown')}; warnings: {summary.get('warnings', 'unknown')})."
            )
        command = [
            sys.executable, "-m", "pip", "install", "--upgrade", "--no-input",
            "--disable-pip-version-check", str(wheel_path),
        ]
        try:
            completed = subprocess.run(command, check=False, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise UpdateError(f"Could not run pip to install the verified AXIOM wheel: {exc}.") from None
        if completed.returncode != 0:
            raise UpdateError(
                f"pip could not install AXIOM v{version} (exit code {completed.returncode}). "
                "No AXIOM workspace, model, account, or browser data was changed."
            )
    return {
        "installed_version": version,
        "distribution": PACKAGE_NAME,
        "wheel": filename,
        "verified_sha256": hashlib.sha256(wheel_bytes).hexdigest(),
        "user_data_preserved": True,
        "restart_required": True,
    }


__all__ = [
    "UpdateError", "compare_versions", "fetch_latest_release",
    "check_for_update", "download_and_install",
]
