from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from axiom.core.storage import atomic_write_text, file_lock
from axiom.core.workspace_context import workspace_metadata_dir


class IntegrationType(str, Enum):
    MCP = "mcp"
    CLI = "cli"
    HTTP = "http"
    A2A = "a2a"


MCP_TRUST_STATES = frozenset({"unverified", "trusted", "blocked"})
MCP_OWNER_TYPES = frozenset({"unknown", "workspace", "external"})
MCP_CONNECTION_STATES = frozenset({"not_inspected", "connected", "failed", "timeout"})
MCP_CONNECTION_ERRORS = frozenset({
    "handshake_failed",
    "handshake_timeout",
    "schema_inspection_failed",
    "schema_inspection_timeout",
})


@dataclass
class AgentIntegration:
    name: str
    type: IntegrationType
    command: str | None = None
    url: str | None = None
    description: str = ""
    trust_status: str = "unverified"
    owner_type: str = "unknown"
    owner_label: str | None = None
    connection_status: str = "not_inspected"
    last_inspected_at: str | None = None
    last_connection_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type.value,
            "command": self.command,
            "url": self.url,
            "description": self.description,
            "trust_status": self.trust_status,
            "owner_type": self.owner_type,
            "owner_label": self.owner_label,
            "connection_status": self.connection_status,
            "last_inspected_at": self.last_inspected_at,
            "last_connection_error": self.last_connection_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentIntegration:
        if not isinstance(data, dict):
            raise TypeError("Integration entry must be a JSON object.")

        try:
            name = data["name"]
            integration_type = IntegrationType(data["type"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Integration entry is missing valid name/type.") from exc

        if not isinstance(name, str) or not name.strip():
            raise TypeError("Integration name cannot be empty.")

        command = data.get("command")
        url = data.get("url")
        description = data.get("description", "")

        if command is not None and not isinstance(command, str):
            raise ValueError("Integration command must be a string or null.")
        if url is not None and not isinstance(url, str):
            raise ValueError("Integration URL must be a string or null.")
        if not isinstance(description, str):
            raise TypeError("Integration description must be a string.")

        trust_status = data.get("trust_status", "unverified")
        owner_type = data.get("owner_type", "unknown")
        owner_label = data.get("owner_label")
        connection_status = data.get("connection_status", "not_inspected")
        last_inspected_at = data.get("last_inspected_at")
        last_connection_error = data.get("last_connection_error")

        if trust_status not in MCP_TRUST_STATES:
            raise ValueError("Integration trust status is invalid.")
        if owner_type not in MCP_OWNER_TYPES:
            raise ValueError("Integration owner type is invalid.")
        if owner_label is not None:
            if not isinstance(owner_label, str):
                raise TypeError("Integration owner label must be a string or null.")
            owner_label = owner_label.strip()
            if not owner_label:
                owner_label = None
            elif len(owner_label) > 96 or any(not character.isprintable() for character in owner_label):
                raise ValueError("Integration owner label must be at most 96 printable characters.")
        if connection_status not in MCP_CONNECTION_STATES:
            raise ValueError("Integration connection status is invalid.")
        if last_inspected_at is not None:
            if not isinstance(last_inspected_at, str) or len(last_inspected_at) > 64:
                raise ValueError("Integration inspection timestamp is invalid.")
            try:
                datetime.fromisoformat(last_inspected_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("Integration inspection timestamp is invalid.") from exc
        if last_connection_error is not None and last_connection_error not in MCP_CONNECTION_ERRORS:
            raise ValueError("Integration connection error is invalid.")
        if connection_status == "not_inspected" and (
            last_inspected_at is not None or last_connection_error is not None
        ):
            raise ValueError("Uninspected integration cannot have connection result metadata.")

        return cls(
            name=name,
            type=integration_type,
            command=command,
            url=url,
            description=description,
            trust_status=trust_status,
            owner_type=owner_type,
            owner_label=owner_label,
            connection_status=connection_status,
            last_inspected_at=last_inspected_at,
            last_connection_error=last_connection_error,
        )


class IntegrationRegistry:
    """Local registry for external AI-agent integrations."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        create: bool = True,
    ):
        self.root = Path(root) if root is not None else workspace_metadata_dir()
        self.path = self.root / "integrations.json"
        self.lock_file = self.root / ".integrations.lock"

        if not create:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            if not self.path.exists():
                atomic_write_text(self.path, "[]\n")

    def _ensure_safe_registry_path(self) -> None:
        """Reject symlinked or non-regular registry paths before metadata writes."""
        if self.root.is_symlink() or self.path.is_symlink() or self.lock_file.is_symlink():
            raise ValueError("Integration registry path must not be a symbolic link.")
        if not self.root.exists() or not self.root.is_dir():
            raise LookupError("Integration registry does not exist.")
        if not self.path.exists() or not self.path.is_file():
            raise LookupError("Integration registry file does not exist.")

    def _read(self) -> list[AgentIntegration]:
        if not self.path.exists():
            return []
        if not self.path.is_file():
            raise ValueError(
                f"Invalid integrations registry: expected a file at {self.path}"
            )

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid integrations registry: {self.path}") from exc

        if not isinstance(data, list):
            raise ValueError(  # noqa: TRY004
                f"Invalid integrations registry: expected a JSON array in {self.path}"
            )

        integrations: list[AgentIntegration] = []
        for index, item in enumerate(data):
            try:
                integrations.append(AgentIntegration.from_dict(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid integration registry entry at index {index}."
                ) from exc

        return integrations

    def list(self) -> list[AgentIntegration]:
        return self._read()

    def add(self, integration: AgentIntegration) -> None:
        if not isinstance(integration, AgentIntegration):
            raise TypeError("integration must be an AgentIntegration instance.")

        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            integrations = self._read()

            if any(
                item.name.lower() == integration.name.lower() for item in integrations
            ):
                raise ValueError(f"Integration already exists: {integration.name}")

            integrations.append(integration)

            atomic_write_text(
                self.path,
                json.dumps(
                    [item.to_dict() for item in integrations],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
            )


    def update_mcp_trust(
        self,
        name: str,
        *,
        trust_status: str,
        owner_type: str = "unknown",
        owner_label: str | None = None,
    ) -> AgentIntegration:
        """Persist explicit trust and user-declared ownership for one MCP server."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP server name cannot be empty.")
        if trust_status not in MCP_TRUST_STATES:
            raise ValueError("Trust status must be unverified, trusted, or blocked.")
        if owner_type not in MCP_OWNER_TYPES:
            raise ValueError("Owner type must be unknown, workspace, or external.")
        if owner_label is not None and not isinstance(owner_label, str):
            raise ValueError("Owner label must be a string or null.")
        changes = {
            "trust_status": trust_status,
            "owner_type": owner_type,
            "owner_label": owner_label,
        }
        self._ensure_safe_registry_path()
        with file_lock(self.lock_file):
            integrations = self._read()
            updated: AgentIntegration | None = None
            remaining: list[AgentIntegration] = []
            for item in integrations:
                if item.type is IntegrationType.MCP and item.name.casefold() == name.strip().casefold():
                    updated = AgentIntegration.from_dict({**item.to_dict(), **changes})
                    remaining.append(updated)
                else:
                    remaining.append(item)
            if updated is None:
                raise LookupError(f"MCP server not found: {name}")
            atomic_write_text(
                self.path,
                json.dumps([item.to_dict() for item in remaining], indent=2, ensure_ascii=False) + "\n",
            )
            return updated

    def record_mcp_connection(
        self,
        name: str,
        *,
        connection_status: str,
        last_inspected_at: str | None = None,
        last_connection_error: str | None = None,
    ) -> AgentIntegration:
        """Persist the latest metadata-only connection inspection result."""
        if connection_status not in {"connected", "failed", "timeout"}:
            raise ValueError("Connection status must be connected, failed, or timeout.")
        if last_inspected_at is None:
            last_inspected_at = datetime.now(UTC).isoformat()
        changes = {
            "connection_status": connection_status,
            "last_inspected_at": last_inspected_at,
            "last_connection_error": last_connection_error,
        }
        self._ensure_safe_registry_path()
        with file_lock(self.lock_file):
            integrations = self._read()
            updated: AgentIntegration | None = None
            remaining: list[AgentIntegration] = []
            for item in integrations:
                if item.type is IntegrationType.MCP and item.name.casefold() == name.casefold():
                    updated = AgentIntegration.from_dict({**item.to_dict(), **changes})
                    remaining.append(updated)
                else:
                    remaining.append(item)
            if updated is None:
                raise LookupError(f"MCP server not found: {name}")
            atomic_write_text(
                self.path,
                json.dumps([item.to_dict() for item in remaining], indent=2, ensure_ascii=False) + "\n",
            )
            return updated

    def remove(self, name: str) -> bool:
        if not isinstance(name, str) or not name.strip():
            raise TypeError("Integration name cannot be empty.")

        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            integrations = self._read()

            remaining = [
                item for item in integrations if item.name.lower() != name.lower()
            ]

            if len(remaining) == len(integrations):
                return False

            atomic_write_text(
                self.path,
                json.dumps(
                    [item.to_dict() for item in remaining],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
            )

            return True
