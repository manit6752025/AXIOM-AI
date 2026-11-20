"""Tests for safe release discovery, wheel verification and package installation."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from axiom.core import updater


def _wheel(version: str = "9.9.9") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "axiom_all-9.9.9.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: axiom-all\nVersion: {version}\n\n",
        )
        archive.writestr("axiom/webui/index.html", "<main>AXIOM</main>")
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
