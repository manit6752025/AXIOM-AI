"""Versioned, allowlisted import/export for local workspace metadata.

Archives intentionally exclude project configuration, credentials, MCP grants,
integration endpoints/commands, prediction-job state, datasets, and model weights.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import stat
import tempfile
import zipfile
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from axiom.core.workspace_context import active_workspace_root
from axiom.path_safety import open_regular_file

from axiom.core.agent_profiles import AgentProfileStore
from axiom.core.experiments import ExperimentStore
from axiom.core.storage import file_lock
from axiom.core.workflows import (
    WORKFLOW_RECORD_LIMIT,
    WORKFLOW_SCHEMA_VERSION,
    WorkflowStore,
    validate_workflow_graph,
)
from axiom.evaluation.store import EvaluationStore
from axiom.models.registry import Model, ModelRegistry

ARCHIVE_FORMAT = "application/vnd.axiom.workspace-metadata+zip"
ARCHIVE_SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
MAX_COLLECTION_BYTES = 4 * 1024 * 1024
MAX_COLLECTIONS = 5
_COLLECTION_FILES = {
    "models": "models.json",
    "experiments": "experiments.json",
    "workflows": "workflows.json",
    "evaluations": "evaluations.json",
    "agent_profiles": "agent-profiles.json",
}
_METADATA_LOCK_FILES = (
    ".agent-profiles.lock",
    ".evaluations.lock",
    ".experiments.lock",
    ".metadata-transfer.lock",
    ".models.lock",
    ".workflows.lock",
)
_EXCLUDED = [
    "axiom.yaml and all project configuration",
    "credentials, environment variables, API keys, and tokens",
    "MCP tool-permission profiles and active grants",
    "external integration commands and endpoint URLs",
    "prediction-job history and runtime traces",
    "datasets, model weights, checkpoints, and generated artifacts",
    "lock files, caches, and temporary files",
]


def _workspace_root(workspace: str | Path | None) -> Path:
    candidate = active_workspace_root() if workspace is None else Path(workspace).expanduser()
    if candidate.is_symlink():
        raise ValueError("Workspace directory must not be a symbolic link.")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Workspace directory could not be resolved safely.") from exc
    if not root.is_dir():
        raise ValueError("Workspace path must be a directory.")
    return root


def _metadata_directory(root: Path, *, create: bool = False) -> Path:
    directory = root / ".axiom"
    if directory.is_symlink():
        raise ValueError("The workspace .axiom directory must not be a symbolic link.")
    if directory.exists() and not directory.is_dir():
        raise ValueError("The workspace .axiom path is not a directory.")
    if create:
        directory.mkdir(mode=0o700, parents=False, exist_ok=True)
    return directory


@contextmanager
def _locked_metadata_directory(root: Path, *, create: bool) -> Iterator[int | None]:
    """Lock every supported metadata collection in a stable order.

    POSIX operations use a no-follow directory descriptor so an attacker cannot
    redirect a lock or registry operation by swapping the .axiom path. Other
    platforms use the existing cross-process file-lock primitive after the
    same symlink checks.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    supports_dir_fd = (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in getattr(os, "supports_dir_fd", set())
        and os.mkdir in getattr(os, "supports_dir_fd", set())
    )
    directory = root / ".axiom"
    if not supports_dir_fd:
        _metadata_directory(root, create=create)
        if not directory.exists():
            yield None
            return
        with ExitStack() as stack:
            for name in _METADATA_LOCK_FILES:
                lock_path = directory / name
                if lock_path.is_symlink():
                    raise ValueError(f"Workspace metadata lock must not be a symbolic link: {name}")
                stack.enter_context(file_lock(lock_path))
            yield None
        return

    try:
        root_fd = os.open(str(root), flags | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise ValueError("Workspace root could not be opened safely for metadata transfer.") from exc
    directory_fd: int | None = None
    lock_descriptors: list[int] = []
    try:
        try:
            directory_fd = os.open(".axiom", flags | getattr(os, "O_CLOEXEC", 0), dir_fd=root_fd)
        except FileNotFoundError:
            if not create:
                yield None
                return
            try:
                os.mkdir(".axiom", mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            directory_fd = os.open(".axiom", flags | getattr(os, "O_CLOEXEC", 0), dir_fd=root_fd)
        import fcntl
        for name in _METADATA_LOCK_FILES:
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_fd,
            )
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f"Workspace metadata lock must be a regular file: {name}")
                if info.st_mode & 0o077:
                    os.fchmod(descriptor, 0o600)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except BaseException:
                os.close(descriptor)
                raise
            lock_descriptors.append(descriptor)
        yield directory_fd
    except OSError as exc:
        raise ValueError("Workspace metadata locks could not be acquired safely.") from exc
    finally:
        try:
            if directory_fd is not None:
                try:
                    import fcntl
                    for descriptor in reversed(lock_descriptors):
                        try:
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                        finally:
                            os.close(descriptor)
                finally:
                    os.close(directory_fd)
        finally:
            os.close(root_fd)


def _read_existing_collection_file(
    directory: Path,
    filename: str,
    *,
    directory_fd: int | None,
) -> bytes | None:
    descriptor: int | None = None
    try:
        if directory_fd is not None:
            descriptor = os.open(
                filename,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_COLLECTION_BYTES:
                    raise ValueError(f"Workspace metadata file is not a bounded regular file: {filename}")
                payload = stream.read(MAX_COLLECTION_BYTES + 1)
        else:
            path = directory / filename
            if path.is_symlink():
                raise ValueError(f"Workspace metadata file must not be a symbolic link: {filename}")
            with open_regular_file(path) as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_COLLECTION_BYTES:
                    raise ValueError(f"Workspace metadata file is not a bounded regular file: {filename}")
                payload = stream.read(MAX_COLLECTION_BYTES + 1)
        if len(payload) > MAX_COLLECTION_BYTES:
            raise ValueError(f"Workspace metadata file exceeds the 4 MiB limit: {filename}")
        return payload
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Workspace metadata file could not be read safely: {filename}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_collection_file(
    directory: Path,
    filename: str,
    payload: bytes,
    *,
    directory_fd: int | None,
    suffix: str = "import",
) -> None:
    temporary: str | None = None
    descriptor: int | None = None
    try:
        if directory_fd is not None:
            temporary = f".{filename}.{secrets.token_hex(16)}.{suffix}"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_fd,
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary,
                filename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary = None
            return
        fd, temporary = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=f".{suffix}", dir=directory
        )
        descriptor = fd
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, directory / filename)
        temporary = None
    except OSError as exc:
        raise ValueError(f"Workspace metadata file could not be committed safely: {filename}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                if directory_fd is not None:
                    os.unlink(temporary, dir_fd=directory_fd)
                else:
                    os.unlink(temporary)
            except OSError:
                pass


def _remove_collection_file(directory: Path, filename: str, *, directory_fd: int | None) -> None:
    try:
        if directory_fd is not None:
            os.unlink(filename, dir_fd=directory_fd)
        else:
            (directory / filename).unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ValueError(f"Workspace metadata file could not be removed safely: {filename}") from exc


def _read_collection(root: Path, name: str) -> Any:
    directory = _metadata_directory(root)
    filename = _COLLECTION_FILES[name]
    path = directory / filename
    if path.is_symlink():
        raise ValueError(f"Workspace metadata file must not be a symbolic link: {filename}")
    if name == "models":
        return [item.to_dict() for item in ModelRegistry(directory, create=False).list()]
    if name == "experiments":
        return ExperimentStore(directory)._read()
    if name == "workflows":
        return WorkflowStore(directory)._read()
    if name == "evaluations":
        return EvaluationStore(directory)._read()
    if name == "agent_profiles":
        return AgentProfileStore(root)._load_unlocked()
    raise ValueError(f"Unsupported workspace metadata collection: {name}")


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Workspace metadata contains non-JSON values.") from exc


def export_workspace_metadata(
    workspace: str | Path | None = None,
    *,
    collections: list[str] | None = None,
) -> bytes:
    """Create a bounded ZIP containing validated, allowlisted metadata only."""
    root = _workspace_root(workspace)
    selected = list(_COLLECTION_FILES) if collections is None else list(collections)
    if not selected or len(selected) > MAX_COLLECTIONS or len(set(selected)) != len(selected):
        raise ValueError("Select between one and five unique metadata collections.")
    if any(name not in _COLLECTION_FILES for name in selected):
        raise ValueError("One or more workspace metadata collections are unsupported.")

    with _locked_metadata_directory(root, create=False):
        members: dict[str, bytes] = {}
        collection_info: dict[str, dict[str, Any]] = {}
        total = 0
        for name in selected:
            payload = _json_bytes(_read_collection(root, name))
            if len(payload) > MAX_COLLECTION_BYTES:
                raise ValueError(f"Workspace metadata collection '{name}' exceeds the 4 MiB limit.")
            total += len(payload)
            if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("Workspace metadata export exceeds the 20 MiB uncompressed limit.")
            member = "metadata/" + _COLLECTION_FILES[name]
            members[member] = payload
            decoded = json.loads(payload)
            count = len(decoded) if isinstance(decoded, list) else len(decoded.get("profiles", []))
            collection_info[name] = {
                "file": member,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "records": count,
            }

        manifest = {
            "format": ARCHIVE_FORMAT,
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "collections": collection_info,
            "excluded": list(_EXCLUDED),
        }
        manifest_bytes = _json_bytes(manifest)
        if total + len(manifest_bytes) > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("Workspace metadata archive including its manifest exceeds the 20 MiB uncompressed limit.")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for name, payload in [("manifest.json", manifest_bytes), *members.items()]:
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                bundle.writestr(info, payload)
        result = archive.getvalue()
        if len(result) > MAX_ARCHIVE_BYTES:
            raise ValueError("Workspace metadata archive exceeds the 24 MiB compressed limit.")
        return result


def _validate_models(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > ModelRegistry.MAX_MODELS:
        raise ValueError("Model metadata must be a JSON array of at most 1000 entries.")
    models: list[Model] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Model metadata contains an invalid entry.")
        try:
            model = Model(**item)
            ModelRegistry._check_duplicate(models, model)
        except (TypeError, ValueError) as exc:
            raise ValueError("Model metadata contains invalid or duplicate records.") from exc
        models.append(model)
    return [model.to_dict() for model in models]


def _validate_experiments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 1000:
        raise ValueError("Experiment metadata must contain at most 1000 records.")
    ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("Experiment metadata contains an invalid record.")
        record_id = record.get("id")
        version = record.get("schema_version", ExperimentStore.SCHEMA_VERSION)
        tags = record.get("tags", [])
        if not isinstance(record_id, str) or not record_id or record_id in ids:
            raise ValueError("Experiment metadata contains missing or duplicate IDs.")
        if type(version) is not int or version != ExperimentStore.SCHEMA_VERSION:
            raise ValueError("Experiment metadata uses an unsupported schema version.")
        if not isinstance(tags, list) or len(tags) > 20 or any(
            not isinstance(tag, str) or len(tag) > 64 for tag in tags
        ):
            raise ValueError("Experiment metadata contains invalid tags.")
        ids.add(record_id)
        normalized.append({**record, "schema_version": ExperimentStore.SCHEMA_VERSION, "tags": tags})
    return normalized


def _validate_workflows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > WORKFLOW_RECORD_LIMIT:
        raise ValueError(f"Workflow metadata must contain at most {WORKFLOW_RECORD_LIMIT} designs.")
    ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("Workflow metadata contains an invalid design.")
        workflow_id = record.get("id")
        name = record.get("name")
        nodes = record.get("nodes")
        edges = record.get("edges")
        if not isinstance(workflow_id, str) or not workflow_id or workflow_id in ids:
            raise ValueError("Workflow metadata contains missing or duplicate IDs.")
        if not isinstance(name, str) or not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("Workflow metadata contains an invalid graph.")
        if not all(isinstance(node, dict) for node in nodes) or not all(
            isinstance(edge, dict) for edge in edges
        ):
            raise ValueError("Workflow metadata contains invalid nodes or connections.")
        try:
            validate_workflow_graph(name, nodes, edges)
        except (AttributeError, TypeError) as exc:
            raise ValueError("Workflow metadata contains an invalid graph.") from exc
        version = record.get("schema_version", WORKFLOW_SCHEMA_VERSION)
        revision = record.get("revision", 1)
        versions = record.get("versions", [])
        if (
            type(version) is not int or version != WORKFLOW_SCHEMA_VERSION
            or type(revision) is not int or revision < 1
            or not isinstance(versions, list) or len(versions) > 10
            or any(not isinstance(item, dict) for item in versions)
        ):
            raise ValueError("Workflow metadata contains an unsupported revision history.")
        ids.add(workflow_id)
        normalized.append({
            **record,
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "revision": revision,
            "versions": versions,
        })
    return normalized


def _validate_evaluations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > EvaluationStore.MAX_RECORDS:
        raise ValueError("Evaluation metadata exceeds the 1000-report limit.")
    ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("Evaluation metadata contains an invalid report.")
        report_id = record.get("id")
        version = record.get("schema_version", EvaluationStore.SCHEMA_VERSION)
        if not isinstance(report_id, str) or not report_id or report_id in ids:
            raise ValueError("Evaluation metadata contains missing or duplicate IDs.")
        if type(version) is not int or version != EvaluationStore.SCHEMA_VERSION:
            raise ValueError("Evaluation metadata uses an unsupported schema version.")
        ids.add(report_id)
        normalized.append({**record, "schema_version": EvaluationStore.SCHEMA_VERSION})
    return normalized


def _validate_agent_profiles(value: Any) -> dict[str, Any]:
    return AgentProfileStore.__new__(AgentProfileStore)._validate_document(value)


def _validate_collection(name: str, value: Any) -> Any:
    validators = {
        "models": _validate_models,
        "experiments": _validate_experiments,
        "workflows": _validate_workflows,
        "evaluations": _validate_evaluations,
        "agent_profiles": _validate_agent_profiles,
    }
    if name not in validators:
        raise ValueError(f"Unsupported workspace metadata collection: {name}")
    return validators[name](value)


def _merge_collection(name: str, current: Any, incoming: Any) -> Any:
    if name == "agent_profiles":
        existing = {item["id"]: item for item in current["profiles"]}
        additions = []
        for item in incoming["profiles"]:
            if item["id"] == "default":
                continue
            if item["id"] in existing or any(
                old["name"].casefold() == item["name"].casefold() for old in existing.values()
            ):
                raise ValueError("Agent profile import conflicts with an existing profile ID or name.")
            additions.append(item)
        combined = {**current, "profiles": [existing["default"], *[
            item for item in existing.values() if item["id"] != "default"
        ], *additions]}
        return _validate_agent_profiles(combined)

    if not isinstance(current, list) or not isinstance(incoming, list):
        raise ValueError(f"Invalid {name} metadata for merge.")
    seen = {item["id"] for item in current}
    if any(item["id"] in seen for item in incoming):
        raise ValueError(f"{name.replace('_', ' ').title()} import conflicts with an existing record ID.")
    combined = [*current, *incoming]
    if name == "models":
        # Also reject name, source, and content-identity duplicates, not just ID collisions.
        checked: list[Model] = []
        for item in combined:
            model = Model(**item)
            ModelRegistry._check_duplicate(checked, model)
            checked.append(model)
    return _validate_collection(name, combined)


def _read_archive(data: bytes) -> dict[str, Any]:
    if not isinstance(data, bytes) or not data or len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Workspace metadata archive must be between 1 byte and 24 MiB.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("Workspace metadata archive is not a valid ZIP file.") from exc
    with archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("Workspace metadata archive has duplicate entries or no manifest.")
        allowed = {"manifest.json", *(
            "metadata/" + filename for filename in _COLLECTION_FILES.values()
        )}
        total = 0
        for info in infos:
            mode = info.external_attr >> 16
            if (
                info.filename not in allowed
                or info.is_dir()
                or (mode and not stat.S_ISREG(mode))
                or info.flag_bits & 0x1
                or info.file_size > MAX_COLLECTION_BYTES
                or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            ):
                raise ValueError("Workspace metadata archive contains an unsupported or unsafe entry.")
            total += info.file_size
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("Workspace metadata archive exceeds the uncompressed size limit.")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (KeyError, UnicodeError, json.JSONDecodeError, RuntimeError, zipfile.BadZipFile) as exc:
            raise ValueError("Workspace metadata archive manifest is invalid.") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != ARCHIVE_FORMAT
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != ARCHIVE_SCHEMA_VERSION
            or not isinstance(manifest.get("collections"), dict)
            or not 1 <= len(manifest["collections"]) <= MAX_COLLECTIONS
        ):
            raise ValueError("Unsupported workspace metadata archive format or version.")
        expected_names = {"manifest.json"}
        result: dict[str, Any] = {}
        for name, descriptor in manifest["collections"].items():
            if name not in _COLLECTION_FILES or not isinstance(descriptor, dict):
                raise ValueError("Workspace metadata archive lists an unsupported collection.")
            member = "metadata/" + _COLLECTION_FILES[name]
            expected_names.add(member)
            if descriptor.get("file") != member or member not in names:
                raise ValueError("Workspace metadata archive is missing a declared collection.")
            try:
                payload = archive.read(member)
            except (RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
                raise ValueError(f"Workspace metadata collection '{name}' could not be read safely.") from exc
            if (
                type(descriptor.get("size_bytes")) is not int
                or descriptor["size_bytes"] != len(payload)
                or not isinstance(descriptor.get("sha256"), str)
                or descriptor["sha256"] != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError(f"Workspace metadata collection '{name}' failed its checksum.")
            try:
                decoded = json.loads(payload)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Workspace metadata collection '{name}' is invalid JSON.") from exc
            validated = _validate_collection(name, decoded)
            expected_count = len(validated) if isinstance(validated, list) else len(validated["profiles"])
            if type(descriptor.get("records")) is not int or descriptor["records"] != expected_count:
                raise ValueError(f"Workspace metadata collection '{name}' has an invalid record count.")
            result[name] = validated
        if set(names) != expected_names:
            raise ValueError("Workspace metadata archive includes undeclared files.")
        return result


def import_workspace_metadata(
    data: bytes,
    workspace: str | Path | None = None,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Validate a complete archive, then merge or replace collections atomically per file."""
    if type(replace) is not bool:
        raise ValueError("replace must be a boolean.")
    root = _workspace_root(workspace)
    incoming = _read_archive(data)
    # Fail with the explicit path-safety diagnostic before attempting to open
    # lock files under a pre-existing .axiom symlink. The locked helper repeats
    # no-follow checks to protect against replacements after this preflight.
    _metadata_directory(root)
    with _locked_metadata_directory(root, create=True) as directory_fd:
        directory = _metadata_directory(root)
        before: dict[str, bytes | None] = {}
        after: dict[str, bytes] = {}
        for name, values in incoming.items():
            filename = _COLLECTION_FILES[name]
            destination = directory / filename
            if destination.is_symlink():
                raise ValueError(f"Workspace metadata file must not be a symbolic link: {filename}")
            current = _read_collection(root, name)
            combined = values if replace else _merge_collection(name, current, values)
            payload = _json_bytes(combined)
            if len(payload) > MAX_COLLECTION_BYTES:
                raise ValueError(f"Imported '{name}' metadata exceeds the 4 MiB storage limit.")
            before[filename] = _read_existing_collection_file(
                directory, filename, directory_fd=directory_fd
            )
            after[filename] = payload

        replaced: list[str] = []
        try:
            for filename, payload in after.items():
                _write_collection_file(
                    directory, filename, payload, directory_fd=directory_fd, suffix="import"
                )
                replaced.append(filename)
            if directory_fd is not None:
                os.fsync(directory_fd)
        except BaseException as exc:
            rollback_errors: list[str] = []
            for filename in reversed(replaced):
                previous = before[filename]
                try:
                    if previous is None:
                        _remove_collection_file(directory, filename, directory_fd=directory_fd)
                    else:
                        _write_collection_file(
                            directory, filename, previous, directory_fd=directory_fd, suffix="rollback"
                        )
                except (OSError, ValueError):
                    rollback_errors.append(filename)
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    rollback_errors.extend(filename for filename in replaced if filename not in rollback_errors)
            if rollback_errors:
                raise OSError(
                    "Workspace metadata import failed and rollback could not restore: "
                    + ", ".join(dict.fromkeys(rollback_errors))
                ) from exc
            if isinstance(exc, OSError):
                raise ValueError("Workspace metadata import could not be committed; prior files were restored.") from exc
            raise

    return {
        "imported": {
            name: (
                len(values) if isinstance(values, list)
                else len(values["profiles"]) if replace
                else sum(profile["id"] != "default" for profile in values["profiles"])
            )
            for name, values in incoming.items()
        },
        "mode": "replace" if replace else "merge",
        "excluded": list(_EXCLUDED),
    }
