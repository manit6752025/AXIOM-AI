from __future__ import annotations

import io
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
from packaging.version import Version

from scripts.verify_package_artifacts import PackageArtifactError, verify_package_artifacts


ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
VERSION = str(Version(PROJECT["version"]))
DIST_INFO = f"axiom_all-{VERSION}.dist-info"
WHEEL_NAME = f"axiom_all-{VERSION}-py3-none-any.whl"


def _wheel(path: Path, *, extra: list[tuple[str, bytes, int]] | None = None, entry_point: str = "axiom = axiom.cli.main:main", include_css: bool = True) -> None:
    metadata = (
        f"Metadata-Version: 2.1\nName: {PROJECT['name']}\nVersion: {VERSION}\n"
        f"Requires-Python: {PROJECT['requires-python']}\n"
    )
    entries = [
        ("axiom/__init__.py", b'"""AXIOM package."""\n', 0),
        ("axiom/webui/index.html", b"<!doctype html><html></html>", 0),
        ("axiom/webui/assets/app.js", b"console.log('AXIOM');", 0),
        *([("axiom/webui/assets/app.css", b"body{margin:0}", 0)] if include_css else []),
        (f"{DIST_INFO}/METADATA", metadata.encode(), 0),
        (f"{DIST_INFO}/WHEEL", b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n", 0),
        (f"{DIST_INFO}/entry_points.txt", f"[console_scripts]\n{entry_point}\n".encode(), 0),
    ]
    entries.extend(extra or [])
    with zipfile.ZipFile(path, "w") as archive:
        for name, content, file_type in entries:
            info = zipfile.ZipInfo(name)
            if file_type:
                info.external_attr = file_type << 16
            archive.writestr(info, content)


def _sdist(path: Path, *, extra: tuple[str, bytes, int] | None = None, include_css: bool = True) -> None:
    root = f"axiom_all-{VERSION}"
    files = {
        "pyproject.toml": (ROOT / "pyproject.toml").read_bytes(),
        "README.md": (ROOT / "README.md").read_bytes(),
        "axiom/__init__.py": b'"""AXIOM package."""\n',
        "axiom/webui/index.html": b"<!doctype html><html></html>",
        "axiom/webui/assets/app.js": b"console.log('AXIOM');",
        **({"axiom/webui/assets/app.css": b"body{margin:0}"} if include_css else {}),
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if extra:
            name, content, kind = extra
            info = tarfile.TarInfo(f"{root}/{name}")
            info.type = kind
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content) if kind == tarfile.REGTYPE else None)


def _dist(tmp_path: Path, *, wheel_extra=None, wheel_entry=None, sdist_extra=None, include_css: bool = True, include_sdist_css: bool = True) -> Path:
    (tmp_path / WHEEL_NAME).write_bytes(b"")
    _wheel(tmp_path / WHEEL_NAME, extra=wheel_extra, entry_point=wheel_entry or "axiom = axiom.cli.main:main", include_css=include_css)
    _sdist(tmp_path / f"axiom_all-{VERSION}.tar.gz", extra=sdist_extra, include_css=include_sdist_css)
    return tmp_path


def test_verifies_wheel_and_sdist_bundle_cli_and_control_center(tmp_path: Path) -> None:
    report = verify_package_artifacts(_dist(tmp_path))
    assert report["project"] == PROJECT["name"]
    assert report["version"] == PROJECT["version"]
    assert report["wheel"]["entry_point"] == "axiom = axiom.cli.main:main"
    assert report["wheel"]["bundled_webui"] is True
    assert report["sdist"]["bundled_webui"] is True
    assert report["wheel_installed_and_imported"] is False


def test_rejects_wheel_path_traversal_and_special_files(tmp_path: Path) -> None:
    with pytest.raises(PackageArtifactError, match="traversal|invalid member path"):
        verify_package_artifacts(_dist(tmp_path, wheel_extra=[("../escape.py", b"bad", 0)]))


def test_rejects_wheel_symlink_members(tmp_path: Path) -> None:
    with pytest.raises(PackageArtifactError, match="symbolic link|special file"):
        verify_package_artifacts(_dist(tmp_path, wheel_extra=[("axiom/evil", b"target", 0o120777)]))


def test_rejects_case_colliding_wheel_paths(tmp_path: Path) -> None:
    with pytest.raises(PackageArtifactError, match="duplicate or case-colliding"):
        verify_package_artifacts(_dist(tmp_path, wheel_extra=[("AXIOM/__init__.py", b"shadow", 0)]))


def test_rejects_wrong_cli_entry_point(tmp_path: Path) -> None:
    with pytest.raises(PackageArtifactError, match="CLI entry point"):
        verify_package_artifacts(_dist(tmp_path, wheel_entry="axiom = wrong.module:app"))


def test_rejects_sdist_symbolic_links(tmp_path: Path) -> None:
    with pytest.raises(PackageArtifactError, match="link or special file"):
        verify_package_artifacts(_dist(tmp_path, sdist_extra=("axiom/webui/linked", b"target", tarfile.SYMTYPE)))


def test_rejects_wheel_missing_control_center_css_asset(tmp_path: Path) -> None:
    with pytest.raises(PackageArtifactError, match="JavaScript or CSS assets"):
        verify_package_artifacts(_dist(tmp_path, include_css=False))


def test_rejects_sdist_missing_control_center_css_asset(tmp_path: Path) -> None:
    with pytest.raises(PackageArtifactError, match="JavaScript or CSS assets"):
        verify_package_artifacts(_dist(tmp_path, include_sdist_css=False))
