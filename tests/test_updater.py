"""Tests for safe release discovery, wheel verification and package installation."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from axiom.core import updater


def _wheel(version: str = "9.9.9", *, tamper_record: bool = False, unsafe_member: bool = False) -> bytes:
    stream = io.BytesIO()
    dist_info = f"axiom_all-{version}.dist-info"
    files = {
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: axiom-all\nVersion: {version}\n\n"
        ).encode("utf-8"),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        "axiom/__init__.py": b'"""AXIOM test wheel."""\n',
        "axiom/webui/index.html": b"<main>AXIOM</main>",
    }
    if unsafe_member:
        files["../outside.txt"] = b"must not be installed"
    rows = []
    for name, payload in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
        rows.append([name, f"sha256={digest}", str(len(payload))])
    rows.append([f"{dist_info}/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[f"{dist_info}/RECORD"] = record.getvalue().encode("utf-8")
    if tamper_record:
        files["axiom/__init__.py"] = b"tampered after RECORD creation\n"
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return stream.getvalue()

def test_semantic_release_comparison_handles_numeric_beta_tags_and_stable_releases():
    assert updater.compare_versions("0.3.0-beta.10", "0.3.0-beta.9") > 0
    assert updater.compare_versions("0.3.0", "0.3.0-beta.99") > 0
    assert updater.compare_versions("0.4.0-beta.1", "0.3.9") > 0
    assert updater.compare_versions("v0.3.0-beta.7", "0.3.0-beta.7") == 0


def test_latest_release_skips_drafts_and_selects_channel(monkeypatch):
    release_list = [
        {"tag_name": "v0.4.0-beta.2", "draft": False, "prerelease": True},
        {"tag_name": "v0.3.0", "draft": False, "prerelease": False},
        {"tag_name": "v9.0.0", "draft": True, "prerelease": False},
    ]
    monkeypatch.setattr(
        updater,
        "_read_url",
        lambda url, **kwargs: json.dumps(release_list).encode("utf-8"),
    )
    assert updater.fetch_latest_release("beta")["tag_name"] == "v0.4.0-beta.2"
    assert updater.fetch_latest_release("stable")["tag_name"] == "v0.3.0"


def test_check_for_update_reports_current_build_ahead_of_release(monkeypatch):
    monkeypatch.setattr(
        updater,
        "fetch_latest_release",
        lambda channel: {"tag_name": "v0.3.0-beta.7", "name": "Beta 7", "html_url": "https://example.invalid"},
    )
    result = updater.check_for_update("0.3.0-beta.8", "beta")
    assert result["update_available"] is False
    assert result["latest_version"] == "0.3.0-beta.7"


def test_download_and_install_verifies_wheel_and_uses_current_python(monkeypatch):
    wheel = _wheel()
    digest = hashlib.sha256(wheel).hexdigest()
    pypi = {
        "info": {"name": "axiom-all", "version": "9.9.9"},
        "urls": [{
            "packagetype": "bdist_wheel",
            "filename": "axiom_all-9.9.9-py3-none-any.whl",
            "url": "https://files.pythonhosted.org/axiom.whl",
            "digests": {"sha256": digest},
            "yanked": False,
        }],
    }

    def fake_urlopen(request, timeout=20):
        url = request.full_url
        if "pypi.org" in url:
            return io.BytesIO(json.dumps(pypi).encode("utf-8"))
        return io.BytesIO(wheel)

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)
    commands = []

    def fake_run(command, *, check, timeout):
        commands.append(command)
        downloaded_wheel = command[-1]
        with zipfile.ZipFile(downloaded_wheel) as archive:
            assert "axiom/webui/index.html" in archive.namelist()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    result = updater.download_and_install("9.9.9")
    assert result["verified_sha256"] == digest
    assert result["user_data_preserved"] is True
    assert result["restart_required"] is True
    assert commands[0][0:5] == [
        updater.sys.executable, "-m", "pip", "install", "--upgrade",
    ]
    assert "--no-input" in commands[0]


def test_download_rejects_bad_hash_before_invoking_pip(monkeypatch):
    wheel = _wheel()
    pypi = {
        "info": {"name": "axiom-all", "version": "9.9.9"},
        "urls": [{
            "packagetype": "bdist_wheel",
            "filename": "axiom_all-9.9.9-py3-none-any.whl",
            "url": "https://files.pythonhosted.org/axiom.whl",
            "digests": {"sha256": "0" * 64},
            "yanked": False,
        }],
    }

    def fake_urlopen(request, timeout=20):
        if "pypi.org" in request.full_url:
            return io.BytesIO(json.dumps(pypi).encode("utf-8"))
        return io.BytesIO(wheel)

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)
    monkeypatch.setattr(updater.subprocess, "run", lambda *args, **kwargs: pytest.fail("pip must not run"))
    with pytest.raises(updater.UpdateError, match="SHA-256"):
        updater.download_and_install("9.9.9")


def test_wheel_rejects_wrong_package_or_missing_webui():
    invalid = io.BytesIO()
    with zipfile.ZipFile(invalid, "w") as archive:
        archive.writestr("other.dist-info/METADATA", "Name: other\nVersion: 9.9.9\n\n")
    with pytest.raises(updater.UpdateError, match="does not match"):
        updater._validate_wheel(invalid.getvalue(), "9.9.9")


def test_download_rejects_wheel_record_tampering_before_invoking_pip(monkeypatch):
    wheel = _wheel(tamper_record=True)
    digest = hashlib.sha256(wheel).hexdigest()
    pypi = {
        "info": {"name": "axiom-all", "version": "9.9.9"},
        "urls": [{
            "packagetype": "bdist_wheel",
            "filename": "axiom_all-9.9.9-py3-none-any.whl",
            "url": "https://files.pythonhosted.org/axiom.whl",
            "digests": {"sha256": digest},
            "yanked": False,
        }],
    }

    def fake_urlopen(request, timeout=20):
        if "pypi.org" in request.full_url:
            return io.BytesIO(json.dumps(pypi).encode("utf-8"))
        return io.BytesIO(wheel)

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)
    monkeypatch.setattr(updater.subprocess, "run", lambda *args, **kwargs: pytest.fail("pip must not run"))
    with pytest.raises(updater.UpdateError, match="RECORD/archive inspection"):
        updater.download_and_install("9.9.9")


def test_download_rejects_wheel_archive_traversal_before_invoking_pip(monkeypatch):
    wheel = _wheel(unsafe_member=True)
    digest = hashlib.sha256(wheel).hexdigest()
    pypi = {
        "info": {"name": "axiom-all", "version": "9.9.9"},
        "urls": [{
            "packagetype": "bdist_wheel",
            "filename": "axiom_all-9.9.9-py3-none-any.whl",
            "url": "https://files.pythonhosted.org/axiom.whl",
            "digests": {"sha256": digest},
            "yanked": False,
        }],
    }

    def fake_urlopen(request, timeout=20):
        if "pypi.org" in request.full_url:
            return io.BytesIO(json.dumps(pypi).encode("utf-8"))
        return io.BytesIO(wheel)

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)
    monkeypatch.setattr(updater.subprocess, "run", lambda *args, **kwargs: pytest.fail("pip must not run"))
    with pytest.raises(updater.UpdateError, match="RECORD/archive inspection"):
        updater.download_and_install("9.9.9")


def test_download_rejects_non_pypi_wheel_host_before_fetching(monkeypatch):
    pypi = {
        "info": {"name": "axiom-all", "version": "9.9.9"},
        "urls": [{
            "packagetype": "bdist_wheel",
            "filename": "axiom_all-9.9.9-py3-none-any.whl",
            "url": "https://attacker.example/axiom.whl",
            "digests": {"sha256": "a" * 64},
            "yanked": False,
        }],
    }
    calls = []

    def fake_urlopen(request, timeout=20):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps(pypi).encode("utf-8"))

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)
    monkeypatch.setattr(updater.subprocess, "run", lambda *args, **kwargs: pytest.fail("pip must not run"))
    with pytest.raises(updater.UpdateError, match="trusted HTTPS package-file host"):
        updater.download_and_install("9.9.9")
    assert len(calls) == 1 and "pypi.org" in calls[0]


def test_release_tags_normalize_to_canonical_pypi_prerelease_versions():
    assert updater.pypi_version_for_release("v0.3.0-beta.7") == "0.3.0b7"
    assert updater.pypi_version_for_release("0.3.0-beta.10") == "0.3.0b10"
    assert updater.pypi_version_for_release("0.3.0-alpha.2") == "0.3.0a2"
    assert updater.pypi_version_for_release("0.3.0-rc.3") == "0.3.0rc3"
    assert updater.pypi_version_for_release("0.3.0") == "0.3.0"


def test_download_uses_pep440_version_for_pypi_and_wheel_metadata(monkeypatch):
    version = "0.3.0-beta.7"
    normalized = "0.3.0b7"
    wheel = _wheel(normalized)
    digest = hashlib.sha256(wheel).hexdigest()
    pypi = {
        "info": {"name": "axiom-all", "version": normalized},
        "urls": [{
            "packagetype": "bdist_wheel",
            "filename": f"axiom_all-{normalized}-py3-none-any.whl",
            "url": "https://files.pythonhosted.org/axiom.whl",
            "digests": {"sha256": digest},
            "yanked": False,
        }],
    }
    requests = []

    def fake_urlopen(request, timeout=20):
        requests.append(request.full_url)
        if "pypi.org" in request.full_url:
            assert request.full_url.endswith(f"/{normalized}/json")
            return io.BytesIO(json.dumps(pypi).encode("utf-8"))
        return io.BytesIO(wheel)

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)
    installed = []
    monkeypatch.setattr(
        updater.subprocess,
        "run",
        lambda command, *, check, timeout: installed.append(command) or SimpleNamespace(returncode=0),
    )
    result = updater.download_and_install(version)
    assert result["installed_version"] == version
    assert result["verified_sha256"] == digest
    assert requests[0].endswith(f"/{normalized}/json")
    assert installed and installed[0][-1].endswith(f"axiom_all-{normalized}-py3-none-any.whl")


def test_release_version_normalization_rejects_unsupported_prerelease_forms():
    with pytest.raises(updater.UpdateError, match="Unsupported AXIOM prerelease label"):
        updater.pypi_version_for_release("0.3.0-preview.1")
