"""Public structural types for the AXIOM v1 SDK."""
from __future__ import annotations

from types import NoneType
from typing import Any, Literal, TypedDict

JSONValue = NoneType | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class APIErrorDetails(TypedDict, total=False):
    code: str
    message: str
    details: JSONValue


class APIErrorEnvelope(TypedDict, total=False):
    request_id: str
    error: APIErrorDetails
    detail: str


class APIContractMetadata(TypedDict):
    api_contract_version: Literal[1]
    application_version: str
    openapi_url: str
    compatibility_policy: str


class WorkspaceSummary(TypedDict, total=False):
    id: str
    name: str
    created_at: str
    updated_at: str
    is_default: bool
    status: str
    managed: bool


class WorkspaceCatalog(TypedDict, total=False):
    schema_version: int
    active_workspace_id: str
    default_workspace_id: str
    workspaces: list[WorkspaceSummary]
    scope_note: str


class TelemetryHistorySettings(TypedDict):
    enabled: bool
    max_samples: int
    max_age_days: int


class TelemetryHistory(TypedDict, total=False):
    schema_version: int
    settings: TelemetryHistorySettings
    total: int
    samples: list[dict[str, Any]]
    privacy: dict[str, bool]


class WorkspaceBackupRestoreResult(TypedDict, total=False):
    schema_version: int
    workspace: WorkspaceSummary
    backup: dict[str, Any]
    scope_note: str


class AxiomClientOptions(TypedDict, total=False):
    base_url: str
    token: str | None
    workspace_id: str | None
    timeout: float
    headers: dict[str, str]
    user_agent: str
