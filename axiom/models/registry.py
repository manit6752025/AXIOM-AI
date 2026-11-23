from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from axiom.core.resource_ids import legacy_resource_id, new_resource_id

from axiom.core.storage import atomic_write_text, file_lock


@dataclass
class Model:
    name: str
    source: str
    format: str = "unknown"
    parameters: str | None = None
    quantization: str | None = None
    id: str = field(default_factory=new_resource_id)
    architecture: str | None = None
    license: str | None = None
    tags: list[str] = field(default_factory=list)
    favorite: bool = False
    content_sha256: str | None = None
    size_bytes: int | None = None
    workspace_labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("Model name cannot be empty.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise TypeError("Model source cannot be empty.")
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-f0-9]{32}", self.id):
            raise TypeError("Model ID must be a 32-character lowercase hexadecimal value.")
        for label, value, limit in (
            ("format", self.format, 64),
            ("parameters", self.parameters, 80),
            ("quantization", self.quantization, 80),
            ("architecture", self.architecture, 160),
            ("license", self.license, 120),
        ):
            if value is not None and (not isinstance(value, str) or len(value) > limit):
                raise TypeError(f"Model {label} metadata is invalid.")
        if (
            not isinstance(self.tags, list)
            or len(self.tags) > 32
            or any(not isinstance(tag, str) or not tag.strip() or len(tag) > 40 for tag in self.tags)
        ):
            raise TypeError("Model tags must be a list of up to 32 short labels.")
        if type(self.favorite) is not bool:
            raise TypeError("Model favorite must be a boolean.")
        if self.content_sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", self.content_sha256):
            raise TypeError("Model content_sha256 must be a 64-character hexadecimal SHA-256.")
        if self.size_bytes is not None and (
            type(self.size_bytes) is not int
            or not 0 <= self.size_bytes <= 9_007_199_254_740_991
        ):
            raise TypeError("Model size_bytes must be a non-negative safe integer.")
        if (
            not isinstance(self.workspace_labels, list)
            or len(self.workspace_labels) > 32
            or any(
                not isinstance(label, str) or not label.strip() or len(label) > 40
                for label in self.workspace_labels
            )
        ):
            raise TypeError("Model workspace_labels must be a list of up to 32 short labels.")

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """Local AXIOM model registry."""

    MAX_MODELS = 1000
    MAX_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        root: Path | None = None,
        *,
        create: bool = True,
    ):
        self.root = Path(root) if root is not None else Path(".axiom")
        self.registry_file = self.root / "models.json"
        self.lock_file = self.root / ".models.lock"

        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            with file_lock(self.lock_file):
                if not self.registry_file.exists():
                    atomic_write_text(self.registry_file, "[]\n")

    def _read(self) -> list[Model]:
        if not self.registry_file.exists():
            return []
        if not self.registry_file.is_file():
            raise ValueError(
                f"Invalid model registry: expected a file at {self.registry_file}"
            )

        try:
            data = json.loads(self.registry_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid model registry: {self.registry_file}") from exc

        if not isinstance(data, list):
            raise ValueError(  # noqa: TRY004
                f"Invalid model registry: expected a JSON array in {self.registry_file}"
            )
        if len(data) > self.MAX_MODELS:
            raise ValueError("Invalid model registry: more than 1000 records.")

        models: list[Model] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid model registry entry at index {index}.")  # noqa: TRY004

            try:
                migrated = dict(item)
                if "id" not in migrated:
                    migrated["id"] = legacy_resource_id(
                        "model",
                        f"{migrated.get('name', '')}:{migrated.get('source', '')}",
                    )
                models.append(Model(**migrated))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid model registry entry at index {index}."
                ) from exc

        ids = [model.id for model in models]
        if len(ids) != len(set(ids)):
            raise ValueError("Invalid model registry: duplicate model IDs.")
        return models

    def list(self) -> list[Model]:
        return self._read()

    def get(self, model_id: str) -> Model | None:
        return next((model for model in self._read() if model.id == model_id), None)

    def add(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be a Model instance.")

        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            models = self._read()

            self._check_duplicate(models, model)

            models.append(model)

            self._write(models)

    @staticmethod
    def _source_identity(source: str) -> str:
        value = source.strip()
        path = Path(value).expanduser()
        if value.startswith((".", "~", "/")) or path.is_absolute() or path.exists():
            return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))
        return value.rstrip("/").casefold()

    def _write(self, models: list[Model]) -> None:
        if len(models) > self.MAX_MODELS:
            raise ValueError("Model registry cannot contain more than 1000 records.")
        content = json.dumps([item.to_dict() for item in models], indent=2, ensure_ascii=False) + "\n"
        if len(content.encode("utf-8")) > self.MAX_BYTES:
            raise ValueError("Model registry exceeds its 4 MB storage limit.")
        atomic_write_text(self.registry_file, content)

    @classmethod
    def _check_duplicate(cls, models: list[Model], model: Model, *, exclude_id: str | None = None) -> None:
        for existing in models:
            if existing.id == exclude_id:
                continue
            if existing.id == model.id:
                raise ValueError(f"Model ID already exists: {model.id}")
            if existing.name.casefold() == model.name.casefold():
                raise ValueError(f"Model already exists: {model.name}")
            if cls._source_identity(existing.source) == cls._source_identity(model.source):
                raise ValueError(f"Model source already registered: {model.source}")
            if (
                model.content_sha256
                and existing.content_sha256
                and model.content_sha256.casefold() == existing.content_sha256.casefold()
            ):
                raise ValueError(f"Model content already registered: {model.name}")

    def update(self, model_id: str, changes: dict) -> Model | None:
        with file_lock(self.lock_file):
            models = self._read()
            index = next((index for index, model in enumerate(models) if model.id == model_id), None)
            if index is None:
                return None
            candidate = Model(**{**models[index].to_dict(), **changes, "id": model_id})
            self._check_duplicate(models, candidate, exclude_id=model_id)
            models[index] = candidate
            self._write(models)
            return candidate

    def remove(self, model_id: str) -> Model | None:
        with file_lock(self.lock_file):
            models = self._read()
            removed = next((model for model in models if model.id == model_id), None)
            if removed is None:
                return None
            remaining = [model for model in models if model.id != model_id]
            self._write(remaining)
            return removed

    def import_models(self, values: list[dict]) -> list[Model]:
        if not isinstance(values, list) or len(values) > 1000:
            raise ValueError("Model import must contain at most 1000 entries.")
        try:
            incoming = [Model(**value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("Model import contains invalid metadata.") from exc

        with file_lock(self.lock_file):
            models = self._read()
            for index, model in enumerate(incoming):
                self._check_duplicate(models + incoming[:index], model)
            combined = [*models, *incoming]
            self._write(combined)
        return incoming
