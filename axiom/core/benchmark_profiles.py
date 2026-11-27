"""Reproducible, workspace-local profiles for native inference benchmark runs.

Profiles retain workload settings, allowlisted hardware/software provenance and
summary metrics. Prompt text and model paths are never persisted.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.hardware import detect_hardware
from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import workspace_metadata_dir
from axiom.version import __version__

SCHEMA_VERSION = 1
MAX_STORE_BYTES = 4 * 1024 * 1024
MAX_PROFILES = 100
MAX_PROFILE_NAME = 64
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")
_LOCK = ".benchmark-profiles.lock"
_MODEL_CONFIG_FILES = (
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "generation_config.json",
)
_METRICS = (
    "total_wall_time_ms",
    "measured_wall_time_ms",
    "median_elapsed_ms",
    "p95_elapsed_ms",
    "median_prefill_ms",
    "median_decode_ms",
    "median_time_to_first_token_ms",
    "median_generation_tokens_per_second",
    "completion_tokens",
    "prompt_tokens",
    "aggregate_tokens_per_second",
)


class BenchmarkProfileError(ValueError):
    """Invalid or unsafe benchmark profile operation."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _trusted_model_directory(model_path: str | Path) -> Path | None:
    """Resolve model identifiers through AXIOM's filesystem-derived trusted inventory.

    The supplied value is passed only to AXIOM's validating inventory matcher;
    it is not used to construct or probe a filesystem path before validation.
    """
    try:
        from axiom.core.engine import (
            _allowed_model_roots,
            _is_link_or_reparse_point,
            _path_is_within_root,
            resolve_local_model_directory,
        )

        # Inspect the caller-supplied lexical path before the resolver returns
        # a canonical target. This catches a model directory reached through a
        # symlinked ancestor as well as a symlink at the final component.
        if _is_link_or_reparse_point(os.fspath(model_path)):
            raise BenchmarkProfileError(
                "The selected model directory must not use symbolic links or junctions."
            )
        resolved = Path(resolve_local_model_directory(os.fspath(model_path), "Base model")).resolve(strict=True)
        trusted_roots = tuple(Path(root).resolve(strict=True) for root in _allowed_model_roots())
        if not resolved.is_absolute() or not any(
            _path_is_within_root(resolved, root) for root in trusted_roots
        ):
            return None
        return resolved
    except BenchmarkProfileError:
        raise
    except ValueError as exc:
        # Keep the model inventory's explicit symlink rejection visible to
        # callers. Treating this particular trust-boundary violation as an
        # absent fingerprint would let a benchmark profile accept a path that
        # the engine deliberately rejects.
        if "symlinked path component" in str(exc):
            raise BenchmarkProfileError(
                "The selected model directory must not use symbolic links or junctions."
            ) from None
        return None
    except (OSError, RuntimeError, TypeError):
        return None


def _model_config_fingerprint(model_path: str | Path) -> str | None:
    """Hash fixed-name model configuration metadata without following file links."""
    root = _trusted_model_directory(model_path)
    if root is None:
        return None
    # The directory itself is returned by AXIOM's bounded trusted-root walk.
    # Verify that the directory identity does not change while its fixed-name
    # metadata entries are being opened.
    try:
        root_before = root.lstat()
    except OSError:
        raise BenchmarkProfileError("The selected model directory could not be inspected safely.") from None
    if not stat.S_ISDIR(root_before.st_mode):
        raise BenchmarkProfileError("The selected model directory must be a regular directory.")
    root_reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if root_reparse and getattr(root_before, "st_file_attributes", 0) & root_reparse:
        raise BenchmarkProfileError("The selected model directory must not be a reparse point.")

    parts: list[tuple[str, str]] = []
    total = 0
    for name in _MODEL_CONFIG_FILES:
        if name != os.path.basename(name) or name not in _MODEL_CONFIG_FILES:
            raise BenchmarkProfileError("Model configuration filename is invalid.")
        path = root / name
        try:
            entry = path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise BenchmarkProfileError("Model configuration metadata could not be inspected safely.") from None
        if stat.S_ISLNK(entry.st_mode) or (
            root_reparse and getattr(entry, "st_file_attributes", 0) & root_reparse
        ):
            raise BenchmarkProfileError("Model configuration metadata must not use symbolic links or junctions.")
        if not stat.S_ISREG(entry.st_mode) or entry.st_size < 0 or entry.st_size > 2 * 1024 * 1024:
            raise BenchmarkProfileError("A model configuration file exceeds the benchmark provenance limit.")
        total += entry.st_size
        if total > 8 * 1024 * 1024:
            raise BenchmarkProfileError("Model configuration metadata exceeds the 8 MiB provenance limit.")

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if os.name == "posix":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                before = os.fstat(stream.fileno())
                identity = (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns)
                opened_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                if not stat.S_ISREG(before.st_mode) or identity != opened_identity:
                    raise BenchmarkProfileError("Model configuration metadata changed while opening.")
                raw = stream.read(2 * 1024 * 1024 + 1)
                after = os.fstat(stream.fileno())
                after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                if after_identity != opened_identity or len(raw) != entry.st_size:
                    raise BenchmarkProfileError("Model configuration metadata changed during provenance inspection.")
        except BenchmarkProfileError:
            raise
        except OSError:
            raise BenchmarkProfileError("Model configuration metadata could not be read safely.") from None
        finally:
            if "descriptor" in locals() and descriptor >= 0:
                os.close(descriptor)
        parts.append((name, hashlib.sha256(raw).hexdigest()))

    try:
        root_after = root.lstat()
    except OSError:
        raise BenchmarkProfileError("The selected model directory changed during provenance inspection.") from None
    if (root_before.st_dev, root_before.st_ino) != (root_after.st_dev, root_after.st_ino):
        raise BenchmarkProfileError("The selected model directory changed during provenance inspection.")
    return hashlib.sha256(_canonical(parts)).hexdigest() if parts else None


def _hardware_record() -> dict[str, Any]:
    hardware = detect_hardware()
    ram = getattr(hardware, "ram_gb", None)
    vram = getattr(hardware, "vram_gb", None)
    return {
        "os_name": str(getattr(hardware, "os_name", platform.system()))[:64],
        "os_release": platform.release()[:64],
        "architecture": str(getattr(hardware, "architecture", platform.machine()))[:64],
        "cpu_cores": int(getattr(hardware, "cpu_cores", os.cpu_count() or 1)),
        "ram_bytes": int(round(ram * (1024 ** 3))) if isinstance(ram, (int, float)) and not isinstance(ram, bool) and ram > 0 else None,
        "gpu_name": str(hardware.gpu_name)[:128] if getattr(hardware, "gpu_name", None) else None,
        "vram_bytes": int(round(vram * (1024 ** 3))) if isinstance(vram, (int, float)) and not isinstance(vram, bool) and vram > 0 else None,
        "cuda_available": bool(getattr(hardware, "cuda_available", False)),
    }


def _software_record() -> dict[str, Any]:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    return {
        "axiom_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
    }


def _validated_report(report: Any) -> dict[str, Any]:
    required = {"schema_version", "engine", "model", "architecture", "model_type", "device", "precision",
                "memory_mode", "checkpoint_format", "warmup_runs_requested", "warmup_runs_completed",
                "iterations_requested", "iterations_completed", "max_new_tokens", "sampling", "metrics",
                "runs", "memory_stats", "generated_at"}
    if not isinstance(report, dict) or not required.issubset(report):
        raise BenchmarkProfileError("The benchmark result is missing required provenance or metrics.")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("engine") != "axiom-native-engine":
        raise BenchmarkProfileError("Only AXIOM Native Engine schema-v1 benchmark results can be profiled.")
    if not isinstance(report.get("model"), str) or not report["model"] or len(report["model"]) > 128:
        raise BenchmarkProfileError("The benchmark model label is invalid.")
    iterations = report.get("iterations_completed")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 5:
        raise BenchmarkProfileError("The benchmark result has an invalid measured iteration count.")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise BenchmarkProfileError("The benchmark metrics are malformed.")
    safe_metrics: dict[str, int | float] = {}
    for key in _METRICS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) < 1e15:
            raise BenchmarkProfileError(f"The benchmark metric {key} is invalid.")
        safe_metrics[key] = round(float(value), 4)
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != iterations:
        raise BenchmarkProfileError("The benchmark run list does not match its measured iteration count.")
    safe_runs = []
    for index, item in enumerate(runs, 1):
        if not isinstance(item, dict) or item.get("iteration") != index:
            raise BenchmarkProfileError("The benchmark run ordering is invalid.")
        row: dict[str, Any] = {"iteration": index}
        for key in ("elapsed_ms", "prefill_ms", "decode_ms", "time_to_first_token_ms", "generation_tokens_per_second"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) < 1e15:
                raise BenchmarkProfileError(f"A benchmark run metric {key} is invalid.")
            row[key] = round(float(value), 4)
        for key in ("prompt_tokens", "completion_tokens"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000_000:
                raise BenchmarkProfileError(f"A benchmark run count {key} is invalid.")
            row[key] = value
        stop_reason = item.get("stop_reason")
        if not isinstance(stop_reason, str) or stop_reason not in {"eos", "stop_sequence", "max_new_tokens", "cancelled", "unknown"}:
            stop_reason = "other"
        row["stop_reason"] = stop_reason
        safe_runs.append(row)
    sampling = report.get("sampling")
    if not isinstance(sampling, dict):
        raise BenchmarkProfileError("Benchmark sampling configuration is malformed.")
    expected_sampling = {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed"}
    if set(sampling) != expected_sampling:
        raise BenchmarkProfileError("Benchmark sampling configuration has unsupported fields.")
    sampling_bounds = {
        "temperature": (0.0, 2.0),
        "top_p": (0.000001, 1.0),
        "top_k": (0, 100_000),
        "min_p": (0.0, 1.0),
        "repetition_penalty": (0.1, 5.0),
    }
    for key, (minimum, maximum) in sampling_bounds.items():
        value = sampling[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
            raise BenchmarkProfileError(f"Benchmark sampling configuration contains an invalid {key} value.")
    if sampling["seed"] is not None and (isinstance(sampling["seed"], bool) or not isinstance(sampling["seed"], int) or sampling["seed"] < 0):
        raise BenchmarkProfileError("Benchmark sampling seed is invalid.")
    memory_stats = report.get("memory_stats")
    if not isinstance(memory_stats, dict) or len(memory_stats) > 32:
        raise BenchmarkProfileError("Benchmark memory counters are malformed.")
    safe_memory = {}
    for key, value in memory_stats.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z_]{1,64}", key):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchmarkProfileError("Benchmark memory counters must be non-negative integers.")
        safe_memory[key] = value
    for key in ("architecture", "model_type", "device", "precision", "memory_mode", "checkpoint_format"):
        value = report.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 128):
            raise BenchmarkProfileError(f"Benchmark field {key} is invalid.")
    for key in ("warmup_runs_requested", "warmup_runs_completed", "iterations_requested", "max_new_tokens"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2048:
            raise BenchmarkProfileError(f"Benchmark field {key} is invalid.")
    return {
        "model": report["model"],
        "architecture": report.get("architecture"),
        "model_type": report.get("model_type"),
        "device": report.get("device"),
        "precision": report.get("precision"),
        "memory_mode": report.get("memory_mode"),
        "checkpoint_format": report.get("checkpoint_format"),
        "warmup_runs_requested": report["warmup_runs_requested"],
        "warmup_runs_completed": report["warmup_runs_completed"],
        "iterations_requested": report["iterations_requested"],
        "iterations_completed": iterations,
        "max_new_tokens": report["max_new_tokens"],
        "sampling": sampling,
        "metrics": safe_metrics,
        "runs": safe_runs,
        "memory_stats": safe_memory,
    }


def _validate_document(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "profiles"} or type(raw.get("schema_version")) is not int or raw["schema_version"] != SCHEMA_VERSION:
        raise BenchmarkProfileError("The benchmark profile store has an unsupported schema.")
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or len(profiles) > MAX_PROFILES:
        raise BenchmarkProfileError("The benchmark profile list exceeds its limit.")
    output = []
    ids, names = set(), set()
    for item in profiles:
        if not isinstance(item, dict) or set(item) != {"id", "name", "created_at", "fingerprint", "prompt_sha256", "prompt_chars", "model_config_sha256", "workload", "hardware", "software"}:
            raise BenchmarkProfileError("A benchmark profile has unsupported fields.")
        identity = item["id"]
        name = item["name"]
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity) or identity in ids:
            raise BenchmarkProfileError("A benchmark profile ID is invalid or duplicated.")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name.casefold() in names:
            raise BenchmarkProfileError("A benchmark profile name is invalid or duplicated.")
        created_at = item["created_at"]
        if not isinstance(created_at, str) or len(created_at) > 40:
            raise BenchmarkProfileError("A benchmark profile timestamp is invalid.")
        try:
            parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError
        except ValueError:
            raise BenchmarkProfileError("A benchmark profile timestamp must include a timezone.") from None
        fingerprint = item["fingerprint"]
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise BenchmarkProfileError("A benchmark profile fingerprint is invalid.")
        payload = {key: value for key, value in item.items() if key != "fingerprint"}
        if hashlib.sha256(_canonical(payload)).hexdigest() != fingerprint:
            raise BenchmarkProfileError("A benchmark profile fingerprint does not match its content.")
        for key, size in (("prompt_sha256", 64), ("model_config_sha256", 64)):
            value = item[key]
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", value)):
                raise BenchmarkProfileError("A benchmark profile digest is invalid.")
        if isinstance(item["prompt_chars"], bool) or not isinstance(item["prompt_chars"], int) or not 1 <= item["prompt_chars"] <= 12000:
            raise BenchmarkProfileError("A benchmark profile prompt length is invalid.")
        if not isinstance(item["workload"], dict) or set(item["workload"]) != {"prompt_sha256", "prompt_chars", "config", "metrics"}:
            raise BenchmarkProfileError("A benchmark profile workload is malformed.")
        if item["workload"]["prompt_sha256"] != item["prompt_sha256"] or item["workload"]["prompt_chars"] != item["prompt_chars"]:
            raise BenchmarkProfileError("A benchmark profile workload fingerprint is inconsistent.")
        config = item["workload"]["config"]
        required_config = {"model", "architecture", "model_type", "checkpoint_format", "warmup_runs", "iterations", "max_new_tokens", "sampling", "device", "memory_mode", "precision", "adapter_used"}
        if not isinstance(config, dict) or set(config) != required_config or type(config["adapter_used"]) is not bool:
            raise BenchmarkProfileError("A benchmark profile workload configuration is malformed.")
        if not isinstance(config["model"], str) or not config["model"] or len(config["model"]) > 128:
            raise BenchmarkProfileError("A benchmark profile model label is invalid.")
        for key in ("architecture", "model_type", "checkpoint_format", "device", "memory_mode", "precision"):
            value = config[key]
            if value is not None and (not isinstance(value, str) or len(value) > 128):
                raise BenchmarkProfileError("A benchmark profile contains an invalid workload label.")
        for key, minimum, maximum in (("warmup_runs", 0, 2), ("iterations", 1, 5), ("max_new_tokens", 1, 128)):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise BenchmarkProfileError("A benchmark profile contains an invalid workload bound.")
        sampling = config.get("sampling")
        sampling_keys = {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed"}
        if not isinstance(sampling, dict) or set(sampling) != sampling_keys:
            raise BenchmarkProfileError("A benchmark profile sampling configuration is malformed.")
        for key, minimum, maximum in (("temperature", 0, 2), ("top_p", 0.000001, 1), ("top_k", 0, 100000), ("min_p", 0, 1), ("repetition_penalty", 0.1, 5)):
            value = sampling[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
                raise BenchmarkProfileError("A benchmark profile contains invalid sampling settings.")
        seed = sampling["seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 9_223_372_036_854_775_807):
            raise BenchmarkProfileError("A benchmark profile sampling seed is invalid.")
        metrics = item["workload"]["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != set(_METRICS):
            raise BenchmarkProfileError("A benchmark profile metrics record is malformed.")
        for metric in _METRICS:
            value = metrics[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) < 1e15:
                raise BenchmarkProfileError("A benchmark profile metric is invalid.")
        if not isinstance(item["hardware"], dict) or set(item["hardware"]) != {"os_name", "os_release", "architecture", "cpu_cores", "ram_bytes", "gpu_name", "vram_bytes", "cuda_available"}:
            raise BenchmarkProfileError("A benchmark profile hardware record is malformed.")
        hardware = item["hardware"]
        if type(hardware["cuda_available"]) is not bool or isinstance(hardware["cpu_cores"], bool) or not isinstance(hardware["cpu_cores"], int) or not 1 <= hardware["cpu_cores"] <= 4096:
            raise BenchmarkProfileError("A benchmark profile hardware record has invalid values.")
        for key in ("os_name", "os_release", "architecture"):
            if not isinstance(hardware[key], str) or not hardware[key] or len(hardware[key]) > 128:
                raise BenchmarkProfileError("A benchmark profile hardware label is invalid.")
        if hardware["gpu_name"] is not None and (not isinstance(hardware["gpu_name"], str) or len(hardware["gpu_name"]) > 128):
            raise BenchmarkProfileError("A benchmark profile GPU label is invalid.")
        for key in ("ram_bytes", "vram_bytes"):
            value = hardware[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1024 ** 5):
                raise BenchmarkProfileError("A benchmark profile memory capacity is invalid.")
        software = item["software"]
        if not isinstance(software, dict) or set(software) != {"axiom_version", "python_version", "torch_version"}:
            raise BenchmarkProfileError("A benchmark profile software record is malformed.")
        for key in ("axiom_version", "python_version"):
            if not isinstance(software[key], str) or not software[key] or len(software[key]) > 64:
                raise BenchmarkProfileError("A benchmark profile software version is invalid.")
        if software["torch_version"] is not None and (not isinstance(software["torch_version"], str) or len(software["torch_version"]) > 64):
            raise BenchmarkProfileError("A benchmark profile PyTorch version is invalid.")
        ids.add(identity); names.add(name.casefold()); output.append(item)
    return output


class BenchmarkProfileStore:
    """Persist explicit, bounded benchmark profiles inside one AXIOM workspace."""
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else workspace_metadata_dir()
        self.path = self.root / "benchmark-profiles.json"
        self.lock_path = self.root / _LOCK

    def _check_paths(self, *, create: bool = False) -> None:
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise BenchmarkProfileError("The benchmark profile store must not use symbolic links.")
        if self.root.exists() and not self.root.is_dir():
            raise BenchmarkProfileError("The benchmark profile metadata path is not a directory.")
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise BenchmarkProfileError("The benchmark profile store path changed unsafely.")

    def _read(self) -> list[dict[str, Any]]:
        self._check_paths()
        if not self.path.exists():
            return []
        try:
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                raise BenchmarkProfileError("The benchmark profile store exceeds its size or file-type limit.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except BenchmarkProfileError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise BenchmarkProfileError("The benchmark profile store could not be read safely.") from None
        return _validate_document(raw)

    def _write(self, profiles: list[dict[str, Any]]) -> None:
        content = json.dumps({"schema_version": SCHEMA_VERSION, "profiles": profiles}, ensure_ascii=False, indent=2, allow_nan=False)
        if len(content.encode("utf-8")) > MAX_STORE_BYTES:
            raise BenchmarkProfileError("The benchmark profile store exceeds its 4 MiB size limit.")
        self._check_paths(create=True)
        try:
            atomic_write_text(self.path, content)
            if os.name != "nt":
                os.chmod(self.path, 0o600)
        except OSError:
            raise BenchmarkProfileError("The benchmark profile store could not be saved safely.") from None

    def list(self, limit: int = MAX_PROFILES) -> dict[str, Any]:
        self._check_paths()
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PROFILES:
            raise BenchmarkProfileError("Benchmark profile limit must be from 1 through 100.")
        with file_lock(self.lock_path):
            profiles = self._read()
        profiles.sort(key=lambda item: item["created_at"], reverse=True)
        return {"schema_version": SCHEMA_VERSION, "total": len(profiles), "profiles": profiles[:limit],
                "privacy": {"prompt_content_included": False, "model_paths_included": False, "credentials_included": False}}

    def create(self, name: str, report: dict[str, Any], prompt: str, model_path: str | Path, *, adapter_used: bool = False) -> dict[str, Any]:
        if not isinstance(name, str) or not _NAME.fullmatch(name.strip()):
            raise BenchmarkProfileError("Benchmark profile names must be 1–64 safe ASCII characters.")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
            raise BenchmarkProfileError("Benchmark profile prompts must contain 1–12000 characters.")
        workload = _validated_report(report)
        self._check_paths()
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        workload_record = {"prompt_sha256": prompt_digest, "prompt_chars": len(prompt), "config": {
            "model": workload["model"], "architecture": workload["architecture"], "model_type": workload["model_type"],
            "checkpoint_format": workload["checkpoint_format"],
            "warmup_runs": workload["warmup_runs_requested"], "iterations": workload["iterations_requested"],
            "max_new_tokens": workload["max_new_tokens"], "sampling": workload["sampling"],
            "device": workload["device"], "memory_mode": workload["memory_mode"], "precision": workload["precision"],
            "adapter_used": bool(adapter_used),
        }, "metrics": workload["metrics"]}
        hardware = _hardware_record()
        software = _software_record()
        payload = {
            "id": uuid.uuid4().hex,
            "name": name.strip(),
            "created_at": _now(),
            "prompt_sha256": prompt_digest,
            "prompt_chars": len(prompt),
            "model_config_sha256": _model_config_fingerprint(model_path),
            "workload": workload_record,
            "hardware": hardware,
            "software": software,
        }
        fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
        record = {**payload, "fingerprint": fingerprint}
        with file_lock(self.lock_path):
            profiles = self._read()
            if any(item["name"].casefold() == record["name"].casefold() for item in profiles):
                raise BenchmarkProfileError("A benchmark profile with this name already exists in the workspace.")
            profiles.insert(0, record)
            self._write(profiles[:MAX_PROFILES])
        return record

    def compare(self, left_id: str, right_id: str) -> dict[str, Any]:
        self._check_paths()
        if not re.fullmatch(r"[0-9a-f]{32}", left_id or "") or not re.fullmatch(r"[0-9a-f]{32}", right_id or "") or left_id == right_id:
            raise BenchmarkProfileError("Two distinct valid benchmark profile IDs are required.")
        with file_lock(self.lock_path):
            profiles = self._read()
        by_id = {item["id"]: item for item in profiles}
        if left_id not in by_id or right_id not in by_id:
            raise BenchmarkProfileError("One or both benchmark profiles were not found in this workspace.")
        left, right = by_id[left_id], by_id[right_id]
        same = {
            "hardware": left["hardware"] == right["hardware"],
            "software": left["software"] == right["software"],
            "model": left["workload"]["config"]["model"] == right["workload"]["config"]["model"],
            "model_metadata": left["model_config_sha256"] is not None and left["model_config_sha256"] == right["model_config_sha256"],
            "workload": left["workload"]["config"] == right["workload"]["config"],
            "same_prompt": left["prompt_sha256"] == right["prompt_sha256"],
        }
        comparable = all(same.values())
        metrics = {}
        for key in ("median_elapsed_ms", "p95_elapsed_ms", "median_generation_tokens_per_second", "aggregate_tokens_per_second"):
            before = left["workload"]["metrics"].get(key)
            after = right["workload"]["metrics"].get(key)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                metrics[key] = {"left": before, "right": after, "delta": round(after - before, 4)}
        return {"schema_version": SCHEMA_VERSION, "comparable": comparable, "match": same,
                "left": {"id": left_id, "name": left["name"]}, "right": {"id": right_id, "name": right["name"]},
                "metrics": metrics, "note": "A hardware/software/model/workload mismatch means this comparison is descriptive, not a controlled benchmark."}
