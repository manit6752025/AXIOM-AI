"""Read-only generated examples for the published OpenAPI contract."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from axiom.core.api_examples import ApiExamplesError, build_api_examples

router = APIRouter(prefix="/api/contract", tags=["api-contract"])


@router.get("/examples")
def get_api_contract_examples(request: Request) -> dict[str, Any]:
    """Return synthetic typed examples generated from the application OpenAPI schema."""
    try:
        return build_api_examples(request.app.openapi())
    except ApiExamplesError as exc:
        raise HTTPException(
            status_code=500,
            detail="The API contract examples could not be generated safely.",
        ) from exc
