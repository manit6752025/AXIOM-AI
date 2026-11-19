"""Workspace diagnostics and privacy-preserving support bundle API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from axiom.core.diagnostics import create_support_bundle, diagnose_workspace
from axiom.core.roadmap import audit_roadmap

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


@router.get("/workspace")
def workspace_diagnostics() -> dict:
    """Run read-only workspace checks without returning config values."""
    try:
        return diagnose_workspace()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Workspace diagnostics could not be completed safely.") from exc


@router.get("/support-bundle")
def download_support_bundle() -> Response:
    """Download a support archive containing only privacy-safe metadata."""
    try:
        content = create_support_bundle()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The support bundle could not be generated safely.") from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="axiom-support-bundle.zip"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/roadmap")
def roadmap_audit() -> dict:
    """Return the real scoped-roadmap statuses for the Control Center."""
    try:
        return audit_roadmap()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="ROADMAP.md was not found in the current workspace or source checkout.") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="ROADMAP.md could not be audited safely.") from exc
