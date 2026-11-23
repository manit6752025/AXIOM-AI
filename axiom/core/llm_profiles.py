"""Workspace-scoped, secret-free profiles for the unified LLM client."""

from __future__ import annotations

import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.resource_ids import new_resource_id
from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import workspace_metadata_dir

SCHEMA_VERSION = 1
MAX_PROFILES = 25
MAX_STORE_BYTES = 64 * 1024
PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,47}$")
PROFILE_ID = re.compile(r"^[0-9a-f]{32}$")
SUPPORTED_PROVIDERS = frozenset({
    "axiom-local",
    "ollama-local",
    "openai",
    "anthropic",
    "gemini",
    "deepseek",
    "moonshot",
    "openrouter",
    "openai-compatible",
})
_LOCK_NAME = ".llm-provider-profiles.lock"


class LLMProfileError(ValueError):
    """Invalid or unsafe workspace-local provider profile operation."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class LLMProfileStore:
    """Persist non-secret provider metadata under the active workspace's .axiom directory.

    API keys, prompts, system instructions and conversation history are deliberately
    excluded from the schema. Cloud credentials are accepted only by connect/chat calls.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else workspace_metadata_dir()
        self.path = self.root / "llm-provider-profiles.json"
        self.lock_path = self.root / _LOCK_NAME

    def _check_paths(self, *, create: bool = False) -> None:
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise LLMProfileError("The provider-profile store must not use symbolic links.")
        if self.root.exists() and not self.root.is_dir():
            raise LLMProfileError("The workspace metadata path is not a directory.")
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_path.is_symlink():
            raise LLMProfileError("The provider-profile store path changed unsafely.")
        if self.root.exists() and os.name != "nt":
            try:
                os.chmod(self.root, 0o700)
            except OSError as exc:
                raise LLMProfileError("Could not restrict workspace provider-profile directory permissions.") from exc

    @staticmethod
    def _normalize_profile(
        *,
        name: Any,
        provider: Any,
        model: Any,
        base_url: Any,
    ) -> tuple[str, str, str, str]:
        if not isinstance(name, str):
            raise LLMProfileError("Provider profile name must be text.")
        normalized_name = name.strip()
        if not PROFILE_NAME.fullmatch(normalized_name):
            raise LLMProfileError("Provider profile names must be 1–48 safe characters and start with a letter or number.")
        if not isinstance(provider, str) or provider not in SUPPORTED_PROVIDERS:
            raise LLMProfileError("Choose a supported AXIOM LLM provider.")
        if (
            not isinstance(model, str)
            or not model.strip()
            or len(model) > 256
            or any(ord(character) < 32 or ord(character) == 127 for character in model)
        ):
            raise LLMProfileError("Provider model IDs must contain 1–256 printable characters.")
        if base_url is not None and not isinstance(base_url, str):
            raise LLMProfileError("Provider base URL must be text.")
        raw_base_url = base_url.strip() if isinstance(base_url, str) else ""
        if provider == "openai-compatible":
            try:
                from axiom.core.llm_client import _base_url

                normalized_base_url = _base_url(provider, raw_base_url)
            except Exception as exc:
                raise LLMProfileError("Custom provider profiles require a valid public HTTPS /v1 endpoint.") from exc
        else:
            if raw_base_url:
                raise LLMProfileError("A custom base URL is supported only for the OpenAI-compatible provider.")
            normalized_base_url = ""
        return normalized_name, provider, model.strip(), normalized_base_url

    def _read(self) -> list[dict[str, Any]]:
        self._check_paths()
        if not self.path.exists():
            return []
        try:
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES:
                raise LLMProfileError("The workspace provider-profile store is not a bounded regular file.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except LLMProfileError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise LLMProfileError("The workspace provider-profile store could not be read safely.") from exc
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "profiles"}:
            raise LLMProfileError("The workspace provider-profile document is malformed.")
        version = raw.get("schema_version")
        profiles = raw.get("profiles")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise LLMProfileError("Unsupported workspace provider-profile schema version.")
        if not isinstance(profiles, list) or len(profiles) > MAX_PROFILES:
            raise LLMProfileError("The workspace provider-profile list is invalid or exceeds its limit.")

        normalized: list[dict[str, Any]] = []
        ids: set[str] = set()
        names: set[str] = set()
        expected_keys = {"id", "name", "provider", "model", "base_url", "created_at", "updated_at"}
        for item in profiles:
            if not isinstance(item, dict) or set(item) != expected_keys:
                # Unknown fields fail closed, in particular fields such as api_key or secret.
                raise LLMProfileError("A provider profile contains unsupported or secret-bearing fields.")
            profile_id = item.get("id")
            if not isinstance(profile_id, str) or not PROFILE_ID.fullmatch(profile_id):
                raise LLMProfileError("The provider-profile store contains an invalid ID.")
            name, provider, model, base_url = self._normalize_profile(
                name=item.get("name"),
                provider=item.get("provider"),
                model=item.get("model"),
                base_url=item.get("base_url"),
            )
            created_at = item.get("created_at")
            updated_at = item.get("updated_at")
            if (
                not isinstance(created_at, str) or not created_at or len(created_at) > 40
                or not isinstance(updated_at, str) or not updated_at or len(updated_at) > 40
            ):
                raise LLMProfileError("The provider-profile timestamps are invalid.")
            if profile_id in ids or name.casefold() in names:
                raise LLMProfileError("Provider profile IDs and names must be unique.")
            ids.add(profile_id)
            names.add(name.casefold())
            normalized.append({
                "id": profile_id,
                "name": name,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "created_at": created_at,
                "updated_at": updated_at,
            })
        return normalized

    def list(self) -> dict[str, Any]:
        profiles = self._read()
        return {
            "schema_version": SCHEMA_VERSION,
            "profiles": profiles,
            "total": len(profiles),
            "max_profiles": MAX_PROFILES,
            "credentials_persisted": False,
        }

    def get(self, profile_id: str) -> dict[str, Any] | None:
        return next((item for item in self._read() if item["id"] == profile_id), None)

    def _write(self, profiles: list[dict[str, Any]]) -> None:
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "profiles": profiles},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
        if len(payload.encode("utf-8")) > MAX_STORE_BYTES:
            raise LLMProfileError("The workspace provider-profile store exceeds its 64 KiB limit.")
        atomic_write_text(self.path, payload)
        if os.name != "nt":
            try:
                os.chmod(self.path, 0o600)
            except OSError as exc:
                raise LLMProfileError("Could not secure workspace provider-profile permissions.") from exc

    def save(self, *, name: str, provider: str, model: str, base_url: str | None = None) -> dict[str, Any]:
        normalized = self._normalize_profile(
            name=name, provider=provider, model=model, base_url=base_url,
        )
        normalized_name, normalized_provider, normalized_model, normalized_url = normalized
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            profiles = self._read()
            existing = next(
                (item for item in profiles if item["name"].casefold() == normalized_name.casefold()),
                None,
            )
            now = _now()
            if existing is None:
                if len(profiles) >= MAX_PROFILES:
                    raise LLMProfileError(f"The workspace provider-profile limit of {MAX_PROFILES} has been reached.")
                profile = {
                    "id": new_resource_id(),
                    "name": normalized_name,
                    "provider": normalized_provider,
                    "model": normalized_model,
                    "base_url": normalized_url,
                    "created_at": now,
                    "updated_at": now,
                }
                profiles.insert(0, profile)
            else:
                profile = {
                    **existing,
                    "name": normalized_name,
                    "provider": normalized_provider,
                    "model": normalized_model,
                    "base_url": normalized_url,
                    "updated_at": now,
                }
                profiles = [profile if item["id"] == existing["id"] else item for item in profiles]
            self._write(profiles)
            return dict(profile)

    def delete(self, profile_id: str) -> dict[str, Any] | None:
        if not isinstance(profile_id, str) or not PROFILE_ID.fullmatch(profile_id):
            raise LLMProfileError("Invalid provider-profile ID.")
        self._check_paths(create=True)
        with file_lock(self.lock_path):
            self._check_paths(create=True)
            profiles = self._read()
            removed = next((item for item in profiles if item["id"] == profile_id), None)
            if removed is None:
                return None
            self._write([item for item in profiles if item["id"] != profile_id])
            return dict(removed)
