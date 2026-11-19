"""Workspace-local agent task profile API.

Task profile labels are descriptive metadata and are not authorization grants.
MCP tool policy remains enforced independently by the existing security router.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from axiom.core.agent_profiles import AgentProfileError, AgentProfileStore, TASK_CAPABILITIES

router = APIRouter(prefix="/api/agent-profiles", tags=["agent-profiles"])


class ProfileCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=64)
    task_capabilities: list[str] = Field(default_factory=list, max_length=len(TASK_CAPABILITIES))

    @field_validator("task_capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Task capabilities must be unique.")
        if any(item not in TASK_CAPABILITIES for item in value):
            raise ValueError("One or more task capabilities are not supported.")
        return value


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_capabilities: list[str] = Field(max_length=len(TASK_CAPABILITIES))

    @field_validator("task_capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Task capabilities must be unique.")
        if any(item not in TASK_CAPABILITIES for item in value):
            raise ValueError("One or more task capabilities are not supported.")
        return value


def _safe_call(callable_):
    try:
        return callable_()
    except AgentProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="The agent-profile store could not be accessed safely.") from exc


@router.get("")
def list_agent_profiles(
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100),
) -> dict:
    """Return a bounded page of versioned workspace-local agent profile metadata."""
    document = _safe_call(AgentProfileStore().load)
    profiles = document["profiles"]
    page = profiles[offset:offset + limit]
    return {
        **document,
        "profiles": page,
        "total": len(profiles),
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(page) < len(profiles),
        "capability_catalog": list(TASK_CAPABILITIES),
        "authorization_note": "Task capabilities are descriptive metadata and do not grant MCP/tool permissions.",
    }


@router.post("", status_code=201)
def create_agent_profile(request: ProfileCreate) -> dict:
    profile = _safe_call(lambda: AgentProfileStore().create(request.name, request.task_capabilities))
    return {"profile": profile}


@router.put("/{profile_id}")
def update_agent_profile(profile_id: str, request: ProfileUpdate) -> dict:
    profile = _safe_call(lambda: AgentProfileStore().update_capabilities(profile_id, request.task_capabilities))
    return {"profile": profile}


@router.delete("/{profile_id}")
def delete_agent_profile(profile_id: str) -> dict:
    return _safe_call(lambda: AgentProfileStore().delete(profile_id))
