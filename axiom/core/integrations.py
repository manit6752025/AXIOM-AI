from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from axiom.core.storage import atomic_write_text, file_lock


class IntegrationType(str, Enum):
    MCP = "mcp"
    CLI = "cli"
    HTTP = "http"
    A2A = "a2a"


@dataclass
class AgentIntegration:
    name: str
    type: IntegrationType
    command: str | None = None
    url: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type.value,
            "command": self.command,
            "url": self.url,
            "description": self.description,
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

        return cls(
            name=name,
            type=integration_type,
            command=command,
            url=url,
            description=description,
        )


class IntegrationRegistry:
    """Local registry for external AI-agent integrations."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        create: bool = True,
    ):
        self.root = Path(root) if root is not None else Path(".axiom")
        self.path = self.root / "integrations.json"
        self.lock_file = self.root / ".integrations.lock"

        if not create:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            if not self.path.exists():
                atomic_write_text(self.path, "[]\n")

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
