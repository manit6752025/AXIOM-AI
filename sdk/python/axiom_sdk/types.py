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



class PredictionJobSummary(TypedDict, total=False):
    id: str
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    dataset_path: str
    output_path: str
    artifact_ready: bool
    rows_total: int
    rows_completed: int
    resumable: bool
    resumption_count: int
    error_code: str | None


class _PredictionJobStartRequired(TypedDict):
    model_path: str
    dataset_path: str
    output_path: str


class PredictionJobStartRequest(_PredictionJobStartRequired, total=False):
    adapter_path: str | None
    prompt_field: str
    reference_field: str
    prediction_field: str
    max_new_tokens: int
    temperature: float
    seed: int | None
    device: Literal["auto", "cpu", "cuda", "rocm", "mps", "xpu"]
    precision: Literal["auto", "fp32", "fp16", "bf16"]


class PredictionJobPage(TypedDict):
    schema_version: Literal[1]
    jobs: list[PredictionJobSummary]
    limit: int


class PredictionJobStatusResult(TypedDict):
    schema_version: Literal[1]
    job: PredictionJobSummary


class PredictionJobStartResult(TypedDict, total=False):
    schema_version: Literal[1]
    job: PredictionJobSummary
    limits: dict[str, int]
    scope: dict[str, Any]


class PredictionJobCancelResult(TypedDict, total=False):
    schema_version: Literal[1]
    job: PredictionJobSummary
    cancellation_accepted: bool
    note: str


class AxiomClientOptions(TypedDict, total=False):
    base_url: str
    token: str | None
    workspace_id: str | None
    timeout: float
    headers: dict[str, str]
    user_agent: str
