"""Safe, non-executing inspection of Python release artifacts.

The inspector never extracts or imports artifact contents. Wheel RECORD hashes
are checked in-place, while source archives are checked for traversal, links,
special files, duplicate paths, and expansion limits before any file is read.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import re
import stat
import tarfile
import unicodedata
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

_SCHEMA_VERSION = 1
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_MEMBERS = 10_000
_MAX_TOTAL_UNCOMPRESSED = 1024 * 1024 * 1024
_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200
_WHEEL_SIGNATURES = (".dist-info/RECORD.jws", ".dist-info/RECORD.p7s")


def _check(check_id: str, title: str, status: str, detail: str, remediation: str | None = None) -> dict[str, str | None]:
    return {
        "id": check_id,
        "title": title,
        "status": status,
        "detail": detail,
        "remediation": remediation,
    }


def _safe_member_name(name: str) -> bool:
    """Reject archive names that could escape a target directory on any OS."""
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        return False
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return False
    raw_parts = name.split("/")
    if raw_parts[-1] == "":
        raw_parts = raw_parts[:-1]
    if not raw_parts:
        return False
    reserved = {"CON", "PRN", "AUX", "NUL"}
    for part in raw_parts:
        if part in {"", ".", ".."} or ":" in part or part.endswith((".", " ")):
            return False
        device_stem = part.split(".", 1)[0].upper()
        if device_stem in reserved or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", device_stem):
            return False
    return True


def _safe_artifact_file(path: str | Path) -> tuple[Path, BinaryIO, int]:
    """Open a local artifact without following symlinked path components on POSIX."""
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if not candidate.name or len(candidate.parts) < 2:
        raise ValueError("Release artifact must be a regular file path.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    try:
        if os.name == "posix" and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(candidate.anchor, os.O_RDONLY | os.O_DIRECTORY)
            try:
                for component in candidate.parts[1:-1]:
                    next_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    os.close(directory_fd)
                    directory_fd = next_fd
                descriptor = os.open(
                    candidate.name,
                    flags | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            finally:
                os.close(directory_fd)
        else:
            if candidate.is_symlink():
                raise ValueError("Release artifact path must not be a symbolic link.")
            descriptor = os.open(candidate, flags)
        stream = os.fdopen(descriptor, "rb")
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            stream.close()
            raise ValueError("Release artifact must be a regular file.")
        if info.st_size <= 0 or info.st_size > _MAX_ARTIFACT_BYTES:
            stream.close()
            raise ValueError("Release artifact is empty or exceeds the 256 MiB inspection limit.")
        return candidate, stream, info.st_size
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("Release artifact could not be opened safely.") from exc


def _finish(checks: list[dict[str, str | None]], *, file_count: int) -> dict[str, Any]:
    failed = sum(item["status"] == "fail" for item in checks)
    warnings = sum(item["status"] == "warn" for item in checks)
    passed = sum(item["status"] == "pass" for item in checks)
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "fail" if failed else "warn" if warnings else "pass",
        "summary": {"passed": passed, "warnings": warnings, "failed": failed, "files_inspected": file_count},
        "checks": checks,
    }


def _zip_inventory(archive: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], list[dict[str, str | None]], int]:
    checks: list[dict[str, str | None]] = []
    infos = archive.infolist()
    if len(infos) > _MAX_MEMBERS:
        raise ValueError("Archive exceeds the 10,000-entry inspection limit.")

    seen: set[str] = set()
    total_size = 0
    regular_files: list[zipfile.ZipInfo] = []
    unsafe = False
    for info in infos:
        name = info.filename
        is_directory = info.is_dir()
        if not is_directory and name.endswith("/"):
            unsafe = True
            checks.append(_check(
                "artifact.archive.path", "Archive member paths", "fail",
                "A non-directory archive member has a directory-style trailing slash.",
                "Rebuild the artifact with file and directory entries represented consistently.",
            ))
            continue
        normalized = name[:-1] if is_directory and name.endswith("/") else name
        folded = unicodedata.normalize("NFC", normalized).casefold()
        if not _safe_member_name(name):
            unsafe = True
            checks.append(_check(
                "artifact.archive.path", "Archive member paths", "fail",
                "The archive contains an absolute, traversal, empty-component, or cross-platform ambiguous member path.",
                "Rebuild the artifact with normalized relative POSIX paths.",
            ))
            continue
        if folded in seen:
            unsafe = True
            checks.append(_check(
                "artifact.archive.duplicates", "Duplicate archive paths", "fail",
                "The archive contains duplicate or case-colliding member paths.",
                "Rebuild the artifact without duplicate paths.",
            ))
            continue
        seen.add(folded)

        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type == stat.S_IFLNK:
            unsafe = True
            checks.append(_check(
                "artifact.archive.links", "Archive links", "fail",
                "The archive contains a symbolic link; release inspection does not follow or extract links.",
                "Rebuild the artifact using regular files and directories only.",
            ))
            continue
        if not is_directory and file_type not in {0, stat.S_IFREG}:
            unsafe = True
            checks.append(_check(
                "artifact.archive.special", "Special archive entries", "fail",
                "The archive contains a non-regular file entry.",
                "Rebuild the artifact using regular files and directories only.",
            ))
            continue
        if info.flag_bits & 0x1:
            unsafe = True
            checks.append(_check(
                "artifact.archive.encrypted", "Encrypted archive entries", "fail",
                "The archive contains encrypted entries that cannot be inspected reliably.",
                "Publish an unencrypted release artifact.",
            ))
            continue
        if is_directory:
            continue
        if info.file_size < 0 or info.file_size > _MAX_MEMBER_BYTES:
            unsafe = True
            checks.append(_check(
                "artifact.archive.member_size", "Archive member sizes", "fail",
                "An archive member is negative-sized or exceeds the 256 MiB per-file inspection limit.",
                "Rebuild the artifact with bounded individual files.",
            ))
            continue
        total_size += info.file_size
        if total_size > _MAX_TOTAL_UNCOMPRESSED:
            raise ValueError("Archive exceeds the 1 GiB total uncompressed inspection limit.")
        if info.file_size > 1024 * 1024 and info.file_size / max(info.compress_size, 1) > _MAX_COMPRESSION_RATIO:
            unsafe = True
            checks.append(_check(
                "artifact.archive.compression", "Archive compression ratio", "fail",
                "An archive member exceeds the maximum allowed compression ratio.",
                "Rebuild the artifact without extreme compression ratios.",
            ))
            continue
        regular_files.append(info)

    if not unsafe:
        checks.append(_check(
            "artifact.archive.structure", "Archive structure", "pass",
            f"Inspected {len(regular_files)} regular file entries with safe relative paths and no links or duplicate names.",
        ))
    return regular_files, checks, total_size


def _zip_read_bounded(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    if info.file_size > limit:
        raise ValueError("Archive metadata file exceeds its inspection size limit.")
    with archive.open(info, "r") as member:
        data = member.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Archive metadata file exceeds its inspection size limit.")
    return data


def _verify_wheel(archive: zipfile.ZipFile, filename: str) -> dict[str, Any]:
    infos, checks, total_size = _zip_inventory(archive)
    by_name = {info.filename: info for info in infos}
    dist_info_roots = {
        "/".join(info.filename.split("/")[:i + 1])
        for info in infos
        for i, part in enumerate(info.filename.split("/"))
        if part.endswith(".dist-info")
    }
    if len(dist_info_roots) != 1:
        checks.append(_check(
            "artifact.wheel.dist_info", "Wheel metadata directory", "fail",
            "A wheel must contain exactly one .dist-info metadata directory.",
            "Rebuild the wheel with one valid distribution metadata directory.",
        ))
        return _finish(checks, file_count=len(infos))

    dist_info = next(iter(dist_info_roots))
    metadata_name = f"{dist_info}/METADATA"
    wheel_name = f"{dist_info}/WHEEL"
    record_name = f"{dist_info}/RECORD"
    missing = [name.rsplit("/", 1)[-1] for name in (metadata_name, wheel_name, record_name) if name not in by_name]
    if missing:
        checks.append(_check(
            "artifact.wheel.required_metadata", "Wheel required metadata", "fail",
            f"The wheel is missing required metadata files: {', '.join(missing)}.",
            "Rebuild the wheel using a standards-compliant Python build backend.",
        ))
        return _finish(checks, file_count=len(infos))

    metadata = BytesParser().parsebytes(_zip_read_bounded(archive, by_name[metadata_name], 1024 * 1024))
    wheel_metadata = BytesParser().parsebytes(_zip_read_bounded(archive, by_name[wheel_name], 64 * 1024))
    package_name = metadata.get("Name", "").strip()
    package_version = metadata.get("Version", "").strip()
    wheel_version = wheel_metadata.get("Wheel-Version", "").strip()
    tags = wheel_metadata.get_all("Tag", [])
    if not package_name or not package_version or not wheel_version.startswith(("1.", "2.")) or not tags:
        checks.append(_check(
            "artifact.wheel.metadata", "Wheel metadata validity", "fail",
            "The wheel's Name, Version, Wheel-Version, or Tag metadata is missing or malformed.",
            "Rebuild the wheel with valid Core Metadata and WHEEL metadata.",
        ))
    else:
        checks.append(_check(
            "artifact.wheel.metadata", "Wheel metadata validity", "pass",
            f"Wheel metadata identifies {package_name} {package_version} with {len(tags)} compatibility tag(s).",
        ))
    expected_dist_info = f"{re.sub(r'[-_.]+', '_', package_name).lower()}-{package_version}.dist-info"
    if dist_info != expected_dist_info:
        checks.append(_check(
            "artifact.wheel.dist_info_name", "Wheel metadata directory name", "fail",
            "The .dist-info directory name does not match the embedded distribution Name and Version.",
            "Rebuild the wheel with a correctly named .dist-info directory.",
        ))
    else:
        checks.append(_check(
            "artifact.wheel.dist_info_name", "Wheel metadata directory name", "pass",
            "The .dist-info directory name matches the embedded distribution Name and Version.",
        ))

    parts = Path(filename).name[:-4].split("-")
    if len(parts) not in {5, 6}:
        checks.append(_check(
            "artifact.wheel.filename", "Wheel filename", "fail",
            "The artifact filename does not follow the five- or six-part wheel filename convention.",
            "Use the normalized distribution-version-build-python-abi-platform wheel filename.",
        ))
    else:
        normalized_file_name = re.sub(r"[-_.]+", "_", parts[0]).lower()
        normalized_metadata_name = re.sub(r"[-_.]+", "_", package_name).lower()
        if normalized_file_name != normalized_metadata_name or parts[1] != package_version:
            checks.append(_check(
                "artifact.wheel.filename", "Wheel filename", "fail",
                "The wheel filename distribution or version does not match its embedded METADATA.",
                "Rebuild or rename the wheel so its filename matches the embedded Name and Version.",
            ))
        else:
            checks.append(_check(
                "artifact.wheel.filename", "Wheel filename", "pass",
                "The wheel filename agrees with its embedded distribution name and version.",
            ))

    record_bytes = _zip_read_bounded(archive, by_name[record_name], _MAX_RECORD_BYTES)
    try:
        rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8"))))
    except (UnicodeError, csv.Error):
        rows = []
    if not rows:
        checks.append(_check(
            "artifact.wheel.record", "Wheel RECORD manifest", "fail",
            "The wheel RECORD file is empty or cannot be parsed as UTF-8 CSV.",
            "Rebuild the wheel so its RECORD manifest is valid.",
        ))
        return _finish(checks, file_count=len(infos))

    record_entries: dict[str, tuple[str, str]] = {}
    record_invalid = False
    for row in rows:
        if len(row) != 3:
            record_invalid = True
            break
        path_value, hash_value, size_value = row
        if not _safe_member_name(path_value) or path_value in record_entries:
            record_invalid = True
            break
        record_entries[path_value] = (hash_value, size_value)

    actual_files = set(by_name)
    signatures = {name for name in actual_files if name.endswith(_WHEEL_SIGNATURES)}
    actual_payload = actual_files - signatures
    if not record_invalid and set(record_entries) != actual_payload:
        record_invalid = True
    if record_invalid:
        checks.append(_check(
            "artifact.wheel.record", "Wheel RECORD manifest", "fail",
            "The RECORD manifest contains invalid, duplicate, missing, or unexpected file entries.",
            "Rebuild the wheel with a complete and internally consistent RECORD file.",
        ))
        return _finish(checks, file_count=len(infos))

    verified = 0
    mismatch = False
    for name, info in by_name.items():
        if name in signatures:
            continue
        hash_value, size_value = record_entries[name]
        if name == record_name:
            if hash_value or size_value:
                mismatch = True
            continue
        if not hash_value or not size_value:
            mismatch = True
            continue
        algorithm, separator, expected = hash_value.partition("=")
        if (
            not separator
            or algorithm.lower() not in {"sha256", "sha384", "sha512"}
            or not re.fullmatch(r"[A-Za-z0-9_-]+", expected)
        ):
            mismatch = True
            continue
        try:
            expected_size = int(size_value)
            hasher = hashlib.new(algorithm.lower())
            with archive.open(info, "r") as member:
                actual_size = 0
                while chunk := member.read(1024 * 1024):
                    actual_size += len(chunk)
                    hasher.update(chunk)
            actual_digest = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode("ascii")
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
            mismatch = True
            continue
        if actual_size != expected_size or not hmac_compare(actual_digest, expected):
            mismatch = True
            continue
        verified += 1

    if mismatch:
        checks.append(_check(
            "artifact.wheel.record_hashes", "Wheel RECORD hashes", "fail",
            "At least one wheel file does not match its SHA-256-or-stronger RECORD hash or recorded size.",
            "Discard the artifact and rebuild it from a trusted source.",
        ))
    else:
        checks.append(_check(
            "artifact.wheel.record_hashes", "Wheel RECORD hashes", "pass",
            f"Verified {verified} wheel file hash(es) and recorded sizes against RECORD.",
        ))
    if signatures:
        checks.append(_check(
            "artifact.wheel.signatures", "Wheel signatures", "warn",
            "The wheel contains RECORD signature sidecars; this inspector checks package RECORD hashes but does not verify signature trust.",
            "Verify signature sidecars with the publisher's documented trust chain when signatures are required.",
        ))
    return _finish(checks, file_count=len(infos))


def hmac_compare(left: str, right: str) -> bool:
    """Constant-time ASCII equality for manifest digests."""
    import hmac

    try:
        left_bytes = left.encode("ascii")
        right_bytes = right.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


def _verify_zip_sdist(archive: zipfile.ZipFile) -> dict[str, Any]:
    infos, checks, _total = _zip_inventory(archive)
    names = {info.filename for info in infos}
    if not any(name.endswith("/pyproject.toml") or name.endswith("/setup.py") for name in names):
        checks.append(_check(
            "artifact.sdist.build_metadata", "Source distribution metadata", "warn",
            "No pyproject.toml or setup.py was found in the archive; build metadata could not be confirmed.",
            "Include a standard Python source-distribution build manifest.",
        ))
    else:
        checks.append(_check(
            "artifact.sdist.build_metadata", "Source distribution metadata", "pass",
            "A Python source-distribution build manifest is present; it was not executed.",
        ))
    checks.append(_check(
        "artifact.sdist.execution", "Artifact execution", "pass",
        "Archive contents were inspected as data only; no setup.py, build backend, or project code was executed.",
    ))
    return _finish(checks, file_count=len(infos))


def _verify_tar_sdist(stream: BinaryIO) -> dict[str, Any]:
    checks: list[dict[str, str | None]] = []
    seen: set[str] = set()
    count = 0
    total_size = 0
    build_manifest = False
    try:
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                count += 1
                if count > _MAX_MEMBERS:
                    raise ValueError("Archive exceeds the 10,000-entry inspection limit.")
                if not _safe_member_name(member.name):
                    checks.append(_check(
                        "artifact.archive.path", "Archive member paths", "fail",
                        "The archive contains an absolute, traversal, empty-component, or cross-platform ambiguous member path.",
                        "Rebuild the source archive with normalized relative POSIX paths.",
                    ))
                    continue
                normalized = unicodedata.normalize("NFC", member.name.rstrip("/")).casefold()
                if normalized in seen:
                    checks.append(_check(
                        "artifact.archive.duplicates", "Duplicate archive paths", "fail",
                        "The archive contains duplicate member paths.",
                        "Rebuild the source archive without duplicate paths.",
                    ))
                    continue
                seen.add(normalized)
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    checks.append(_check(
                        "artifact.archive.links", "Links and special files", "fail",
                        "The source archive contains a link or special filesystem entry.",
                        "Rebuild the source archive using regular files and directories only.",
                    ))
                    continue
                if member.isfile():
                    if member.size < 0 or member.size > _MAX_MEMBER_BYTES:
                        raise ValueError("An archive member exceeds the 256 MiB per-file inspection limit.")
                    total_size += member.size
                    if total_size > _MAX_TOTAL_UNCOMPRESSED:
                        raise ValueError("Archive exceeds the 1 GiB total uncompressed inspection limit.")
                    if PurePosixPath(member.name).name in {"pyproject.toml", "setup.py"}:
                        build_manifest = True
            if count == 0:
                raise ValueError("Source archive contains no entries.")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError("Source archive is malformed or could not be inspected safely.") from exc

    if not any(item["id"] == "artifact.archive.structure" for item in checks):
        if any(item["status"] == "fail" for item in checks):
            checks.append(_check(
                "artifact.archive.structure", "Archive structure", "fail",
                "The source archive contains unsafe paths or filesystem entries.",
                "Rebuild the archive with unique relative paths and regular files/directories only.",
            ))
        else:
            checks.append(_check(
                "artifact.archive.structure", "Archive structure", "pass",
                f"Inspected {count} source archive entries with safe paths and regular file types.",
            ))
    checks.append(_check(
        "artifact.sdist.build_metadata", "Source distribution metadata",
        "pass" if build_manifest else "warn",
        "A pyproject.toml or setup.py build manifest is present; it was not executed."
        if build_manifest else
        "No pyproject.toml or setup.py was found in the archive; build metadata could not be confirmed.",
        None if build_manifest else "Include a standard Python source-distribution build manifest.",
    ))
    checks.append(_check(
        "artifact.sdist.execution", "Artifact execution", "pass",
        "Archive contents were inspected as data only; no setup.py, build backend, or project code was executed.",
    ))
    return _finish(checks, file_count=count)


def inspect_release_artifact(path: str | Path) -> dict[str, Any]:
    """Inspect a wheel or source archive without extracting or executing it."""
    artifact_path, stream, size_bytes = _safe_artifact_file(path)
    try:
        filename = artifact_path.name
        if filename.lower().endswith(".whl"):
            try:
                with zipfile.ZipFile(stream, "r") as archive:
                    report = _verify_wheel(archive, filename)
            except (zipfile.BadZipFile, OSError, RuntimeError, EOFError) as exc:
                raise ValueError("Wheel archive is malformed or could not be inspected safely.") from exc
            artifact_format = "wheel"
        elif filename.lower().endswith((".tar.gz", ".tgz")):
            report = _verify_tar_sdist(stream)
            artifact_format = "sdist-tar"
        elif filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(stream, "r") as archive:
                    report = _verify_zip_sdist(archive)
            except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                raise ValueError("ZIP source archive is malformed or could not be inspected safely.") from exc
            artifact_format = "sdist-zip"
        else:
            raise ValueError("Supported artifact formats are .whl, .tar.gz, .tgz, and .zip.")
        report["artifact"] = {"name": filename, "format": artifact_format, "size_bytes": size_bytes}
        return report
    finally:
        stream.close()
