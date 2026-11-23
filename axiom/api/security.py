from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from axiom.core.tool_permissions import ToolPermissionStore
from axiom.core.plugin_manifests import PluginManifestError, PluginManifestStore
from axiom.core.security_audit import audit_security_posture
from axiom.core.workspace_context import active_workspace_root
from axiom.core.security_events import SecurityEventError, SecurityEventStore, record_security_event
from axiom.core.supply_chain import SupplyChainError, audit_dependency_vulnerabilities, generate_spdx_sbom

router = APIRouter(prefix="/api/security", tags=["security"])


class DependencyAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_network: bool = Field(strict=True)
    timeout_seconds: int = Field(default=90, ge=1, le=300, strict=True)


class ToolProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    allowed_tools: list[str] = Field(max_length=32)


class PluginManifestRegistrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    manifest: dict[str, Any]


class PluginToolGrantInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=64)
    scopes: list[str] = Field(min_length=1, max_length=5)


class PluginPermissionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tools: list[PluginToolGrantInput] = Field(min_length=1, max_length=32)


class ActiveProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=48)


class ReviewSecurityEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reviewer_label: str = Field(
        min_length=2,
        max_length=48,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._-]{1,47}$",
    )


def _record_action_event(event_type: str) -> None:
    try:
        record_security_event(event_type)
    except (SecurityEventError, OSError, TimeoutError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The policy change may already have taken effect, but the security journal could not record it. Inspect the registry and journal before retrying.",
        ) from exc


@router.post("/tool-profiles/{profile_name}", status_code=201)
def create_tool_profile(profile_name: str) -> dict[str, Any]:
    try:
        profile = ToolPermissionStore().create_profile(profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail="Tool permission profile could not be created.") from exc
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("mcp.profile.created", "Created a default-deny MCP profile")
    _record_action_event("mcp.profile.created")
    return {"profile": profile}


@router.get("/tool-permissions")
def get_tool_permissions() -> dict[str, Any]:
    try:
        return ToolPermissionStore().list()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Tool permission policy is invalid.") from exc


@router.put("/tool-profiles/{profile_name}")
def save_tool_profile(profile_name: str, request: ToolProfileInput) -> dict[str, Any]:
    try:
        profile = ToolPermissionStore().save_profile(profile_name, request.allowed_tools)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail="Tool permission profile could not be saved.") from exc
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("mcp.permissions.saved", "Updated an MCP tool profile")
    _record_action_event("mcp.permissions.saved")
    return {"profile": profile}


@router.put("/active-profile")
def set_active_profile(request: ActiveProfileInput) -> dict[str, str]:
    try:
        ToolPermissionStore().set_active(request.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail="Active MCP profile could not be changed.") from exc
    from axiom.api.workspace import record_workspace_event

    record_workspace_event("mcp.profile.activated", "Changed the active MCP profile")
    _record_action_event("mcp.profile.activated")
    return {"active_profile": request.name}


@router.get("/plugins")
def list_plugin_manifests() -> dict[str, Any]:
    """Return declarative plugin manifests and their hash-bound permission state."""
    try:
        return PluginManifestStore().list()
    except (OSError, PluginManifestError):
        raise HTTPException(status_code=409, detail="Plugin manifest registry could not be read safely.") from None


@router.post("/plugins/manifests", status_code=201)
def register_plugin_manifest(request: PluginManifestRegistrationInput) -> dict[str, Any]:
    """Validate and register a declarative manifest; registration never grants access."""
    try:
        result = PluginManifestStore().register(request.manifest)
    except PluginManifestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError):
        raise HTTPException(status_code=500, detail="Plugin manifest could not be registered safely.") from None
    _record_action_event("plugin.manifest.registered")
    return result


@router.put("/plugins/{plugin_id}/permissions")
def grant_plugin_permissions(plugin_id: str, request: PluginPermissionsInput) -> dict[str, Any]:
    """Grant a subset of a registered manifest's declared tool scopes."""
    try:
        result = PluginManifestStore().grant(
            plugin_id,
            expected_sha256=request.manifest_sha256,
            tools=[item.model_dump() for item in request.tools],
        )
    except PluginManifestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError):
        raise HTTPException(status_code=500, detail="Plugin permissions could not be saved safely.") from None
    _record_action_event("plugin.permissions.granted")
    return result


@router.delete("/plugins/{plugin_id}/permissions")
def revoke_plugin_permissions(plugin_id: str) -> dict[str, Any]:
    """Revoke all grants for one manifest while leaving its metadata intact."""
    try:
        revoked = PluginManifestStore().revoke(plugin_id)
    except PluginManifestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, TimeoutError):
        raise HTTPException(status_code=500, detail="Plugin permissions could not be revoked safely.") from None
    if revoked:
        _record_action_event("plugin.permissions.revoked")
    return {"plugin_id": plugin_id, "revoked": revoked}


@router.get("/audit")
def security_audit() -> dict[str, Any]:
    """Return read-only configuration, permissions, and exposure findings."""
    try:
        return audit_security_posture(workspace=active_workspace_root())
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Security posture could not be audited safely.",
        ) from exc


@router.get("/events")
def list_security_events(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0, lt=2000),
) -> dict[str, Any]:
    """List bounded security events with chain integrity and review status."""
    try:
        return SecurityEventStore(active_workspace_root()).list_page(limit, offset)
    except SecurityEventError as exc:
        raise HTTPException(status_code=409, detail="Security event journal could not be verified safely.") from exc


@router.get("/events/verify")
def verify_security_events() -> dict[str, Any]:
    """Verify the complete retained security-event hash chain."""
    try:
        return SecurityEventStore(active_workspace_root()).verify()
    except SecurityEventError as exc:
        raise HTTPException(status_code=409, detail="Security event journal integrity verification failed.") from exc


@router.get("/events/export")
def export_security_events() -> dict[str, Any]:
    """Export the retained redacted journal as JSON; no credentials are included."""
    try:
        return SecurityEventStore(active_workspace_root()).export()
    except SecurityEventError as exc:
        raise HTTPException(status_code=409, detail="Security event journal could not be exported safely.") from exc


@router.post("/events/{event_id}/review")
def review_security_event(
    event_id: str,
    request: ReviewSecurityEventInput,
) -> dict[str, Any]:
    """Append a review acknowledgement; reviewer labels are self-attested."""
    try:
        result = SecurityEventStore(active_workspace_root()).review(
            event_id,
            request.reviewer_label,
        )
        return {**result, "reviewer_identity_verified": False}
    except SecurityEventError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 422
        raise HTTPException(
            status_code=status_code,
            detail="Security event review could not be recorded.",
        ) from exc



@router.get("/sbom")
def security_software_bill_of_materials() -> dict[str, Any]:
    """Return a bounded SPDX 2.3 inventory of installed Python distributions."""
    try:
        return generate_spdx_sbom()
    except SupplyChainError:
        raise HTTPException(status_code=409, detail="Software bill of materials could not be generated safely.") from None


@router.post("/dependency-audit")
def security_dependency_audit(request: DependencyAuditRequest) -> dict[str, Any]:
    """Run an advisory-backed dependency scan only after explicit network authorization."""
    try:
        return audit_dependency_vulnerabilities(
            allow_network=request.allow_network,
            timeout_seconds=request.timeout_seconds,
        )
    except SupplyChainError:
        raise HTTPException(status_code=422, detail="Dependency audit requires explicit authorization and valid bounded settings.") from None
