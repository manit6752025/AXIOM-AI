"""Request-scoped workspace selection for concurrent local API requests."""
from __future__ import annotations

from contextvars import ContextVar, Token
from pathlib import Path

_Workspace = tuple[Path, str]
_active_workspace: ContextVar[_Workspace | None] = ContextVar("axiom_active_workspace", default=None)


def workspace_context_active() -> bool:
    """Return whether this execution context selected an explicit workspace."""
    return _active_workspace.get() is not None


def active_workspace_root() -> Path:
    """Return the selected workspace root, or the process working directory for CLI use."""
    current = _active_workspace.get()
    return current[0] if current is not None else Path.cwd()


def active_workspace_id() -> str | None:
    """Return the selected workspace ID inside a scoped request/task context."""
    current = _active_workspace.get()
    return current[1] if current is not None else None


def workspace_metadata_dir() -> Path:
    """Return the workspace metadata directory while preserving existing CLI semantics."""
    current = _active_workspace.get()
    return current[0] / ".axiom" if current is not None else Path(".axiom")


def set_workspace_context(root: Path, workspace_id: str) -> Token[_Workspace | None]:
    """Select a workspace for the current async request/task context."""
    return _active_workspace.set((Path(root), workspace_id))


def reset_workspace_context(token: Token[_Workspace | None]) -> None:
    """Restore the caller's previous workspace after a request completes."""
    _active_workspace.reset(token)
