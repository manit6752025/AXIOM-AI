from __future__ import annotations

import io
import json
import logging
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

_LOG = logging.getLogger(__name__)


class DatasetCleaningResult:
    def __init__(
        self,
        source: Path,
        output: Path,
        total_lines: int,
        kept: int,
        removed_invalid: int,
        removed_empty: int,
        removed_duplicates: int,
        removed_non_objects: int,
    ):
        self.source = source
        self.output = output
        self.total_lines = total_lines
        self.kept = kept
        self.removed_invalid = removed_invalid
        self.removed_empty = removed_empty
        self.removed_duplicates = removed_duplicates
        self.removed_non_objects = removed_non_objects
        self.removed_schema = removed_non_objects


class DatasetPathError(OSError):
    """Raised when a cleaning destination would violate source safety."""


def _bounded_utf8_lines(
    stream: BinaryIO,
    *,
    max_bytes: int,
    max_rows: int,
    max_line_bytes: int,
):
    read_bytes = rows = 0
    while True:
        remaining = max_bytes - read_bytes
        if remaining <= 0:
            if stream.read(1):
                raise ValueError("Dataset exceeds the cleaning byte limit.")
            return

        raw = stream.readline(min(max_line_bytes + 1, remaining + 1))
        if not raw:
            return
        if len(raw) > remaining:
            raise ValueError("Dataset exceeds the cleaning byte limit.")
        read_bytes += len(raw)
        if len(raw) > max_line_bytes:
            raise ValueError("Dataset contains a row above the cleaning line limit.")
        rows += 1
        if rows > max_rows:
            raise ValueError("Dataset exceeds the cleaning row limit.")

        body = raw.removesuffix(b"\r\n").removesuffix(b"\n").removesuffix(b"\r")
        if b"\r" in body:
            raise ValueError("CR-only line endings are not supported by bounded cleaning.")
        try:
            yield body.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("Dataset contains invalid UTF-8.") from exc


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _same_path(source: Path, destination: Path) -> bool:
    try:
        if destination.exists() and os.path.samefile(source, destination):
            return True
    except OSError:
        _LOG.debug(
            "samefile check failed for %s and %s; comparing resolved paths instead",
            source,
            destination,
            exc_info=True,
        )

    return source.resolve(strict=False) == destination.resolve(strict=False)


def clean_jsonl(
    source: Path,
    output: Path,
    *,
    overwrite: bool = True,
    source_stream: BinaryIO | None = None,
    output_parent_fd: int | None = None,
    output_name: str | None = None,
    max_bytes: int | None = None,
    max_rows: int | None = None,
    max_line_bytes: int | None = None,
) -> DatasetCleaningResult:
    source = Path(source)
    output = Path(output)

    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a boolean")
    if output_parent_fd is not None and source_stream is None:
        raise ValueError("A pinned output directory requires a pinned source stream.")
    if output_parent_fd is None and output_name is not None:
        raise ValueError("output_name requires a pinned output directory.")
    if output_parent_fd is not None and overwrite:
        raise ValueError("Pinned workspace output cannot replace an existing file.")
    limits = (max_bytes, max_rows, max_line_bytes)
    if any(value is not None for value in limits) and any(
        type(value) is not int or value < 1 for value in limits
    ):
        raise ValueError("Cleaning limits must be positive integers.")
    if max_bytes is not None and source_stream is None:
        raise ValueError("Bounded cleaning requires an open binary source stream.")

    if source_stream is None and not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    if output_parent_fd is not None:
        if (
            not isinstance(output_name, str)
            or not output_name
            or output_name in {".", ".."}
            or "/" in output_name
            or "\\" in output_name
            or any(ord(character) < 32 or ord(character) == 127 for character in output_name)
        ):
            raise ValueError("A single output filename is required for a pinned directory.")
        if not overwrite:
            try:
                os.stat(output_name, dir_fd=output_parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("Output path already exists.")

    if output_parent_fd is None and output.exists() and output.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {output}")

    if output_parent_fd is None and not overwrite and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"Output path already exists: {output}")

    if output_parent_fd is None and _same_path(source, output):
        raise DatasetPathError(
            "Output path must differ from the source dataset; "
            "cleaning in place is not supported."
        )

    seen: set[str] = set()

    total_lines = 0
    kept = 0
    removed_invalid = 0
    removed_empty = 0
    removed_duplicates = 0
    removed_non_objects = 0

    if output_parent_fd is None:
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    file_descriptor: int | None = None

    try:
        if source_stream is None:
            input_stream = source.open("r", encoding="utf-8")
            input_lines = input_stream
        elif max_bytes is not None:
            assert max_rows is not None and max_line_bytes is not None
            input_stream = source_stream
            input_lines = _bounded_utf8_lines(
                source_stream,
                max_bytes=max_bytes,
                max_rows=max_rows,
                max_line_bytes=max_line_bytes,
            )
        else:
            input_stream = io.TextIOWrapper(source_stream, encoding="utf-8", newline=None)
            input_lines = input_stream
        with input_stream as infile:
            if output_parent_fd is None:
                file_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{output.name}.",
                    suffix=".tmp",
                    dir=output.parent,
                )
            else:
                assert output_name is not None
                for _ in range(5):
                    candidate_name = f".axiom-clean-{secrets.token_hex(12)}.tmp"
                    try:
                        file_descriptor = os.open(
                            candidate_name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=output_parent_fd,
                        )
                        temporary_name = candidate_name
                        break
                    except FileExistsError:
                        continue
                if file_descriptor is None or temporary_name is None:
                    raise FileExistsError("Could not reserve a temporary output file.")
            outfile = os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            )
            file_descriptor = None

            with outfile:
                for raw_line in input_lines:
                    total_lines += 1
                    line = raw_line.strip()

                    if not line:
                        removed_empty += 1
                        continue

                    try:
                        item: Any = json.loads(
                            line,
                            parse_constant=_reject_non_json_constant,
                        )
                    except (json.JSONDecodeError, ValueError):
                        removed_invalid += 1
                        continue

                    if not isinstance(item, dict):
                        removed_non_objects += 1
                        continue

                    fingerprint = json.dumps(
                        item,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

                    if fingerprint in seen:
                        removed_duplicates += 1
                        continue

                    seen.add(fingerprint)

                    outfile.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    kept += 1

                outfile.flush()
                os.fsync(outfile.fileno())

        if output_parent_fd is not None:
            assert output_name is not None
            if overwrite:
                os.replace(
                    temporary_name,
                    output_name,
                    src_dir_fd=output_parent_fd,
                    dst_dir_fd=output_parent_fd,
                )
                temporary_name = None
            else:
                os.link(
                    temporary_name,
                    output_name,
                    src_dir_fd=output_parent_fd,
                    dst_dir_fd=output_parent_fd,
                    follow_symlinks=False,
                )
                try:
                    os.unlink(temporary_name, dir_fd=output_parent_fd)
                except OSError:
                    _LOG.warning(
                        "Could not remove temporary output %s",
                        temporary_name,
                        exc_info=True,
                    )
                else:
                    temporary_name = None
        elif overwrite:
            os.replace(temporary_name, output)
            temporary_name = None
        else:
            # A same-directory hard link publishes the completed file atomically
            # and fails if another process created the destination first.
            os.link(temporary_name, output)
            try:
                os.unlink(temporary_name)
            except OSError:
                _LOG.warning(
                    "Could not remove temporary output %s",
                    temporary_name,
                    exc_info=True,
                )
            else:
                temporary_name = None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temporary_name is not None:
            try:
                if output_parent_fd is None:
                    os.unlink(temporary_name)
                else:
                    os.unlink(temporary_name, dir_fd=output_parent_fd)
            except FileNotFoundError:
                _LOG.debug(
                    "Temporary file %s was already removed",
                    temporary_name,
                )
            except OSError:
                _LOG.warning(
                    "Could not remove temporary output %s",
                    temporary_name,
                    exc_info=True,
                )

    return DatasetCleaningResult(
        source=source,
        output=output,
        total_lines=total_lines,
        kept=kept,
        removed_invalid=removed_invalid,
        removed_empty=removed_empty,
        removed_duplicates=removed_duplicates,
        removed_non_objects=removed_non_objects,
    )
