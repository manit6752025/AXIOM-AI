"""Passive runtime-discovery endpoint; never launches or contacts a runtime."""
from __future__ import annotations

from fastapi import APIRouter

from axiom.core.runtime_discovery import discover_runtime_connectors

router = APIRouter(prefix="/api/diagnostics", tags=["runtime-discovery"])


@router.get("/runtime-connectors")
def runtime_connectors() -> dict:
    """Return local presence signals without claiming a successful connection."""
    return discover_runtime_connectors()
