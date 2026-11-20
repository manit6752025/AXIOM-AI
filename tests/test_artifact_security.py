from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from axiom.core.artifact_security import inspect_release_artifact


def _wheel(path: Path, *, tamper_record: bool = False) -> Path:
    members = {
        "demo/__init__.py": b"VALUE = 1\n",
        "demo-1.2.3.dist-info/METADATA": b"Metadata-Version: 2.1\nName: demo\nVersion: 1.2.3\n",
        "demo-1.2.3.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    rows: list[list[str]] = []
    for name, payload in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        rows.append([name, f"sha256={digest}", str(len(payload))])
    rows.append(["demo-1.2.3.dist-info/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    members["demo-1.2.3.dist-info/RECORD"] = record.getvalue().encode()
    if tamper_record:
        members["demo/__init__.py"] = b"VALUE = 99\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_inspect_valid_wheel_verifies_record_hashes_without_extracting(tmp_path: Path):
    artifact = _wheel(tmp_path / "demo-1.2.3-py3-none-any.whl")
    before = artifact.read_bytes()

    report = inspect_release_artifact(artifact)

    assert report["status"] == "pass"
    assert report["artifact"]["format"] == "wheel"
    assert report["summary"]["files_inspected"] == 4
    assert any(item["id"] == "artifact.wheel.record_hashes" and item["status"] == "pass" for item in report["checks"])
    assert artifact.read_bytes() == before
    assert list(tmp_path.iterdir()) == [artifact]


def test_inspect_wheel_detects_tampered_file_hash(tmp_path: Path):
    artifact = _wheel(tmp_path / "demo-1.2.3-py3-none-any.whl", tamper_record=True)

    report = inspect_release_artifact(artifact)

    assert report["status"] == "fail"
    assert any(item["id"] == "artifact.wheel.record_hashes" and item["status"] == "fail" for item in report["checks"])


def test_inspect_wheel_rejects_traversal_member(tmp_path: Path):
    artifact = _wheel(tmp_path / "demo-1.2.3-py3-none-any.whl")
    with zipfile.ZipFile(artifact, "a") as archive:
        archive.writestr("../outside.txt", b"not extracted")

    report = inspect_release_artifact(artifact)

    assert report["status"] == "fail"
    assert any(item["id"] == "artifact.archive.path" for item in report["checks"])


def test_inspect_wheel_rejects_symlink_member(tmp_path: Path):
    artifact = _wheel(tmp_path / "demo-1.2.3-py3-none-any.whl")
    link = zipfile.ZipInfo("demo/linked.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(artifact, "a") as archive:
        archive.writestr(link, "../outside.py")

    report = inspect_release_artifact(artifact)

    assert report["status"] == "fail"
    assert any(item["id"] == "artifact.archive.links" for item in report["checks"])


def test_inspect_zip_sdist_rejects_traversal_and_never_executes_setup(tmp_path: Path):
    artifact = tmp_path / "demo-1.2.3.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("demo-1.2.3/pyproject.toml", b"[build-system]\n")
        archive.writestr("demo-1.2.3/setup.py", b"raise RuntimeError('must not run')\n")
        archive.writestr("../../outside.txt", b"payload")

    report = inspect_release_artifact(artifact)

    assert report["status"] == "fail"
    assert report["artifact"]["format"] == "sdist-zip"
    assert any(item["id"] == "artifact.archive.path" for item in report["checks"])
    assert not (tmp_path.parent / "outside.txt").exists()


def test_inspect_tar_sdist_rejects_symlink_members(tmp_path: Path):
    artifact = tmp_path / "demo-1.2.3.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        manifest = tarfile.TarInfo("demo-1.2.3/pyproject.toml")
        manifest.size = len(b"[build-system]\n")
        archive.addfile(manifest, io.BytesIO(b"[build-system]\n"))
        link = tarfile.TarInfo("demo-1.2.3/shortcut")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)

    report = inspect_release_artifact(artifact)

    assert report["status"] == "fail"
    assert report["artifact"]["format"] == "sdist-tar"
    assert any(item["id"] == "artifact.archive.links" for item in report["checks"])


def test_inspect_rejects_unsupported_artifact_and_symlink_input(tmp_path: Path):
    unsupported = tmp_path / "package.bin"
    unsupported.write_bytes(b"not a release artifact")
    with pytest.raises(ValueError, match="Supported artifact formats"):
        inspect_release_artifact(unsupported)

    target = _wheel(tmp_path / "demo-1.2.3-py3-none-any.whl")
    link = tmp_path / "alias.whl"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks are unavailable on this platform: {exc}")
    with pytest.raises(ValueError, match="symbolic link"):
        inspect_release_artifact(link)


def test_inspect_rejects_empty_archive_and_malformed_zip(tmp_path: Path):
    malformed = tmp_path / "broken.zip"
    malformed.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="malformed"):
        inspect_release_artifact(malformed)
