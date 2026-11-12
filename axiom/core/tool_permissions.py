"""Default-deny MCP tool permissions for local AXIOM workspaces."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from axiom.core.storage import atomic_write_text, file_lock

TOOL_DESCRIPTIONS = {
    "axiom_info": "Read AXIOM release information.",
    "axiom_workspace_context": (
        "Read bounded local project, hardware, and developer-tool context; "
        "suggestions are not executed."
    ),
    "axiom_model_list": "List locally registered models.",
    "axiom_model_info": "Inspect a local model path.",
    "axiom_dataset_inspect": "Inspect a local dataset.",
    "axiom_dataset_clean": "Write a cleaned dataset copy.",
    "axiom_system_info": "Read local hardware information.",
    "axiom_training_plan": "Create a hardware-aware training estimate.",
    "axiom_supercompress_status": "Read optional SuperCompress configuration status.",
    "axiom_supercompress": "Send context to the configured compression service.",
}
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")


def _default_config() -> dict[str, Any]:
    return {
        "active_profile": "default",
        "profiles": {"default": {"name": "default", "allowed_tools": []}},
    }


class ToolPermissionStore:
    """Persist named tool allowlists; an absent entry never grants access."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".axiom")
        self.policy_file = self.root / "tool_permissions.json"
        self.lock_file = self.root / ".tool_permissions.lock"

    def _read(self) -> dict[str, Any]:
        if self.root.is_symlink() or self.policy_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid tool permission path: {self.policy_file}")
        if not self.policy_file.exists():
            return _default_config()
        if not self.policy_file.is_file() or self.policy_file.stat().st_size > 256_000:
            raise ValueError(f"Invalid tool permission file: {self.policy_file}")
        try:
            config = json.loads(self.policy_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid tool permission file: {self.policy_file}") from exc
        if (
            not isinstance(config, dict)
            or set(config) != {"active_profile", "profiles"}
            or not isinstance(config.get("profiles"), dict)
            or not 1 <= len(config["profiles"]) <= 25
        ):
            raise ValueError(f"Invalid tool permission file: {self.policy_file}")
        active = config.get("active_profile")
        if not isinstance(active, str) or active not in config["profiles"]:
            raise ValueError(f"Invalid active MCP profile in {self.policy_file}")
        for name, profile in config["profiles"].items():
            if (
                not isinstance(name, str)
                or not _PROFILE_NAME.fullmatch(name)
                or not isinstance(profile, dict)
                or set(profile) != {"name", "allowed_tools"}
                or profile.get("name") != name
                or not isinstance(profile.get("allowed_tools"), list)
                or len(profile["allowed_tools"]) > len(TOOL_DESCRIPTIONS)
                or any(
                    not isinstance(tool, str) or tool not in TOOL_DESCRIPTIONS
                    for tool in profile["allowed_tools"]
                )
                or len(set(profile["allowed_tools"])) != len(profile["allowed_tools"])
            ):
                raise ValueError(f"Invalid MCP profile in {self.policy_file}")
        return config

    def list(self) -> dict[str, Any]:
        config = self._read()
        return {
            "active_profile": config["active_profile"],
            "profiles": [config["profiles"][name] for name in sorted(config["profiles"])],
            "tools": [
                {"name": name, "description": description}
                for name, description in TOOL_DESCRIPTIONS.items()
            ],
        }

    def allows(self, tool_name: str) -> bool:
        if tool_name not in TOOL_DESCRIPTIONS:
            return False
        config = self._read()
        active = config["profiles"][config["active_profile"]]
        return tool_name in active["allowed_tools"]

    def require(self, tool_name: str) -> None:
        if not self.allows(tool_name):
            raise PermissionError(
                f"MCP tool '{tool_name}' is denied by the active AXIOM profile. "
                "Grant it in the Control Center or with the AXIOM CLI."
            )

    def save_profile(self, name: str, allowed_tools: list[str]) -> dict[str, Any]:
        self._validate_profile(name, allowed_tools)

        if self.root.is_symlink() or self.policy_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid tool permission path: {self.policy_file}")
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            config = self._read()
            if name not in config["profiles"] and len(config["profiles"]) >= 25:
                raise ValueError("A workspace can contain at most 25 MCP profiles.")
            profile = {"name": name, "allowed_tools": sorted(allowed_tools)}
            config["profiles"][name] = profile
            self._write(config)
        return profile

    def create_profile(self, name: str) -> dict[str, Any]:
        self._validate_profile(name, [])
        if self.root.is_symlink() or self.policy_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid tool permission path: {self.policy_file}")
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            config = self._read()
            if name in config["profiles"]:
                raise ValueError(f"MCP profile already exists: {name}")
            if len(config["profiles"]) >= 25:
                raise ValueError("A workspace can contain at most 25 MCP profiles.")
            profile = {"name": name, "allowed_tools": []}
            config["profiles"][name] = profile
            self._write(config)
        return profile

    def set_active(self, name: str) -> None:
        if self.root.is_symlink() or self.policy_file.is_symlink() or self.lock_file.is_symlink():
            raise ValueError(f"Invalid tool permission path: {self.policy_file}")
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_file):
            config = self._read()
            if name not in config["profiles"]:
                raise ValueError(f"Unknown MCP profile: {name}")
            config["active_profile"] = name
            self._write(config)

    def _write(self, config: dict[str, Any]) -> None:
        serialized = json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False)
        atomic_write_text(self.policy_file, serialized + "\n")

    @staticmethod
    def _validate_profile(name: str, allowed_tools: list[str]) -> None:
        if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
            raise ValueError("Profile names must be 1 to 48 letters, digits, underscores, or hyphens.")
        if not isinstance(allowed_tools, list) or len(allowed_tools) > len(TOOL_DESCRIPTIONS):
            raise ValueError("Profile tool list is invalid.")
        if any(not isinstance(tool, str) or tool not in TOOL_DESCRIPTIONS for tool in allowed_tools):
            raise ValueError("Profile contains an unknown MCP tool.")
        if len(set(allowed_tools)) != len(allowed_tools):
            raise ValueError("Profile tool list contains duplicates.")
