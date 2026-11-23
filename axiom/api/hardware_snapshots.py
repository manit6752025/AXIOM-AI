"""Explicit hardware snapshot history API."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from axiom.core.hardware_snapshots import (
    HardwareSnapshotError,
    HardwareSnapshotOptInRequired,
    HardwareSnapshotStore,
)

router = APIRouter(prefix="/api/workspace", tags=["hardware-snapshots"])


class HardwareSnapshotSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    max_snapshots: int = Field(default=20, ge=1, le=100, strict=True)
    max_age_days: int = Field(default=90, ge=1, le=3650, strict=True)


@router.get("/hardware-snapshots")
def get_hardware_snapshot_history() -> dict[str, Any]:
    """Read workspace-local, privacy-bounded snapshots without triggering a capture."""
    try:
        return HardwareSnapshotStore().list()
    except HardwareSnapshotError:
        raise HTTPException(
            status_code=500,
            detail="The local hardware snapshot history could not be read safely.",
        ) from None


@router.put("/hardware-snapshots/settings")
def update_hardware_snapshot_settings(request: HardwareSnapshotSettingsRequest) -> dict[str, Any]:
    """Explicitly enable/disable capture and configure bounded age/count retention."""
    try:
        return HardwareSnapshotStore().configure(
            enabled=request.enabled,
            max_snapshots=request.max_snapshots,
            max_age_days=request.max_age_days,
        )
    except HardwareSnapshotError:
        raise HTTPException(
            status_code=500,
            detail="The local hardware snapshot settings could not be saved safely.",
        ) from None


@router.post("/hardware-snapshots", status_code=201)
def capture_hardware_snapshot() -> dict[str, Any]:
    """Capture hardware once; this route never schedules or starts background sampling."""
    try:
        return HardwareSnapshotStore().capture()
    except HardwareSnapshotOptInRequired as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except HardwareSnapshotError:
        raise HTTPException(
            status_code=500,
            detail="The hardware snapshot could not be captured or saved safely.",
        ) from None



@router.delete("/hardware-snapshots")
def clear_hardware_snapshot_history() -> dict[str, Any]:
    """Clear every captured snapshot without disabling the user's retention choice."""
    try:
        return HardwareSnapshotStore().clear()
    except HardwareSnapshotError:
        raise HTTPException(
            status_code=500,
            detail="The local hardware snapshot history could not be cleared safely.",
        ) from None
