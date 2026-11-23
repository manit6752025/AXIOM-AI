#!/usr/bin/env python3
"""Build-output verifier for AXIOM's cross-platform wheel and source archive."""
from __future__ import annotations

import argparse
import json
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from configparser import ConfigParser
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

MAX_ARCHIVE_MEMBERS = 25_000
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 1536 * 1024 * 1024
EXPECTED_ENTRY_POINT = "axiom = axiom.cli.main:main"


class PackageArtifactError(ValueError):
    """A built distribution is incomplete, unsafe, or incompatible."""


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise PackageArtifactError("Archive contains an invalid member path.")
    normalized = name.rstrip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or re.match(r"^[A-Za-z]:", normalized):
        raise PackageArtifactError(f"Archive contains an absolute member path: {name!r}")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise PackageArtifactError(f"Archive contains a traversal or ambiguous member path: {name!r}")
    return normalized


def _check_case_collisions(names: list[str]) -> None:
    observed: set[str] = set()
    for name in names:
        folded = name.casefold()
        if folded in observed:
            raise PackageArtifactError(f"Archive contains duplicate or case-colliding member: {name}")
        observed.add(folded)


def _check_required_payload(paths: set[str], label: str) -> None:
    required = {"axiom/__init__.py", "axiom/webui/index.html"}
    missing = sorted(required - paths)
    if missing:
        raise PackageArtifactError(f"{label} is missing required application files: {', '.join(missing)}")
    assets = [path for path in paths if path.startswith("axiom/webui/assets/")]
    if not any(path.endswith(".js") for path in assets) or not any(path.endswith(".css") for path in assets):
        raise PackageArtifactError(f"{label} is missing the built Control Center JavaScript or CSS assets.")


def _verify_wheel(path: Path, project: dict[str, Any]) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) == 0 or len(infos) > MAX_ARCHIVE_MEMBERS:
                raise PackageArtifactError("Wheel member count is outside the supported bounds.")
            names: list[str] = []
            total = 0
            for info in infos:
                name = _safe_member_name(info.filename)
                names.append(name)
                if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                    raise PackageArtifactError(f"Wheel member exceeds the per-file size bound: {name}")
                total += info.file_size
                if total > MAX_TOTAL_BYTES:
                    raise PackageArtifactError("Wheel expands beyond the 1.5 GiB size limit.")
                file_type = stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise PackageArtifactError(f"Wheel contains a symbolic link or special file: {name}")
            _check_case_collisions(names)
            bad_member = archive.testzip()
            if bad_member is not None:
                raise PackageArtifactError(f"Wheel CRC validation failed for {bad_member}")
            files = {name for name, info in zip(names, infos) if not info.is_dir()}
            dist_info = [name.rsplit("/", 1)[0] for name in files if name.endswith(".dist-info/METADATA")]
            if len(dist_info) != 1:
                raise PackageArtifactError("Wheel must contain exactly one dist-info/METADATA file.")
            prefix = dist_info[0]
            metadata = Parser().parsestr(archive.read(prefix + "/METADATA").decode("utf-8"))
            if metadata.get("Name", "").casefold().replace("_", "-") != str(project["name"]).casefold().replace("_", "-"):
                raise PackageArtifactError("Wheel project name does not match pyproject.toml.")
            from packaging.version import Version
            if Version(metadata.get("Version", "")) != Version(str(project["version"])):
                raise PackageArtifactError("Wheel version does not match pyproject.toml.")
            if metadata.get("Requires-Python") != str(project.get("requires-python", "")):
                raise PackageArtifactError("Wheel Python compatibility metadata does not match pyproject.toml.")
            wheel_metadata = archive.read(prefix + "/WHEEL").decode("utf-8")
            if "Root-Is-Purelib: true" not in wheel_metadata:
                raise PackageArtifactError("The current AXIOM wheel must be a platform-independent pure-Python wheel.")
            entries_name = prefix + "/entry_points.txt"
            config = ConfigParser()
            config.read_string(archive.read(entries_name).decode("utf-8"))
            if config.get("console_scripts", "axiom", fallback="").strip() != "axiom.cli.main:main":
                raise PackageArtifactError("Wheel is missing the supported axiom CLI entry point.")
            _check_required_payload(files, "Wheel")
            return {
                "filename": path.name,
                "format": "wheel",
                "member_count": len(infos),
                "expanded_bytes": total,
                "entry_point": EXPECTED_ENTRY_POINT,
                "bundled_webui": True,
            }
    except (OSError, zipfile.BadZipFile, UnicodeError, KeyError, ValueError) as exc:
        if isinstance(exc, PackageArtifactError):
            raise
        raise PackageArtifactError(f"Could not safely inspect wheel {path.name}: {exc}") from None


def _verify_sdist(path: Path, project: dict[str, Any]) -> dict[str, Any]:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) == 0 or len(members) > MAX_ARCHIVE_MEMBERS:
                raise PackageArtifactError("Source archive member count is outside the supported bounds.")
            names: list[str] = []
            total = 0
            for member in members:
                name = _safe_member_name(member.name)
                names.append(name)
                if not (member.isfile() or member.isdir()):
                    raise PackageArtifactError(f"Source archive contains a link or special file: {name}")
                if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                    raise PackageArtifactError(f"Source archive member exceeds the per-file size bound: {name}")
                total += member.size
                if total > MAX_TOTAL_BYTES:
                    raise PackageArtifactError("Source archive expands beyond the 1.5 GiB size limit.")
            _check_case_collisions(names)
            roots = {name.split("/", 1)[0] for name in names}
            if len(roots) != 1:
                raise PackageArtifactError("Source archive entries must share exactly one top-level directory.")
            prefix = next(iter(roots))
            paths = {name[len(prefix) + 1:] for name in names if name.startswith(prefix + "/")}
            required = {"pyproject.toml", "README.md"}
            if not required.issubset(paths):
                raise PackageArtifactError("Source archive is missing pyproject.toml or README.md.")
            _check_required_payload(paths, "Source archive")
            project_file = archive.extractfile(prefix + "/pyproject.toml")
            if project_file is None:
                raise PackageArtifactError("Source archive pyproject.toml is unreadable.")
            # Confirm source metadata agrees with the checkout that generated it.
            from tomllib import loads
            source_project = loads(project_file.read().decode("utf-8"))["project"]
            if source_project.get("name") != project["name"] or source_project.get("version") != project["version"]:
                raise PackageArtifactError("Source archive project identity differs from the checkout.")
            return {
                "filename": path.name,
                "format": "sdist",
                "member_count": len(members),
                "expanded_bytes": total,
                "bundled_webui": True,
            }
    except (OSError, tarfile.TarError, UnicodeError, KeyError, ValueError) as exc:
        if isinstance(exc, PackageArtifactError):
            raise
        raise PackageArtifactError(f"Could not safely inspect source archive {path.name}: {exc}") from None


def verify_package_artifacts(dist: Path, *, install_wheel: bool = False) -> dict[str, Any]:
    """Check one wheel and one source archive without extracting archive members."""
    root = Path(__file__).resolve().parents[1]
    try:
        import tomllib
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    except (OSError, KeyError, ValueError) as exc:
        raise PackageArtifactError(f"Could not read the project package metadata: {exc}") from None
    dist = Path(dist)
    if dist.is_symlink() or not dist.is_dir():
        raise PackageArtifactError("Distribution output must be a real directory, not a symbolic link.")
    dist = dist.resolve()
    if not dist.is_dir():
        raise PackageArtifactError("Distribution output must be a real directory.")
    wheels = sorted(path for path in dist.iterdir() if path.is_file() and path.suffix == ".whl" and not path.is_symlink())
    sdists = sorted(path for path in dist.iterdir() if path.is_file() and path.name.endswith(".tar.gz") and not path.is_symlink())
    if len(wheels) != 1 or len(sdists) != 1:
        raise PackageArtifactError("Expected exactly one .whl and one .tar.gz source distribution in the output directory.")
    wheel_report = _verify_wheel(wheels[0], project)
    sdist_report = _verify_sdist(sdists[0], project)
    if install_wheel:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", str(wheels[0])],
            check=True,
        )
        # Leave the checkout so an editable/source-tree import cannot satisfy
        # the check in place of the installed wheel.
        with tempfile.TemporaryDirectory(prefix="axiom-wheel-import-") as working_directory:
            subprocess.run(
                [sys.executable, "-I", "-c", "import axiom; assert axiom.__file__"],
                check=True,
                cwd=working_directory,
            )
    return {
        "schema_version": 1,
        "project": project["name"],
        "version": project["version"],
        "python_requires": project.get("requires-python"),
        "wheel": wheel_report,
        "sdist": sdist_report,
        "wheel_installed_and_imported": bool(install_wheel),
        "scope_note": "Archives were inspected without extraction; this verifies package structure and importability, not every optional runtime or hardware backend.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--install-wheel", action="store_true", help="Install the verified wheel without dependencies and import axiom.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable verification report.")
    args = parser.parse_args()
    try:
        report = verify_package_artifacts(args.dist, install_wheel=args.install_wheel)
    except PackageArtifactError as exc:
        print(f"Package verification failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Verified AXIOM {report['version']} wheel + sdist; bundled Control Center assets and CLI entry point are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
