"""Workspace-local Prompt Studio API.

Templates are validated metadata. The render endpoint performs bounded literal
placeholder substitution only; it never invokes models or tools.
"""
from __future__ import annotations

import json
import math
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from axiom.core.prompt_templates import (
    PromptTemplateConflict,
    PromptTemplateError,
    PromptTemplateStore,
)

router = APIRouter(prefix="/api/prompt-templates", tags=["prompt-studio"])


class PromptVariableInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    type: Literal["string", "integer", "number", "boolean", "choice"]
    required: bool = True
    default: Any = None
    description: str = Field(default="", max_length=256)
    choices: list[str] = Field(default_factory=list, max_length=32)
    min_length: int | None = Field(default=None, ge=0, le=16_384)
    max_length: int | None = Field(default=None, ge=0, le=16_384)
    minimum: int | float | None = None
    maximum: int | float | None = None


class PromptTemplateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)
    body: str = Field(min_length=1, max_length=65_536)
    variables: list[PromptVariableInput] = Field(default_factory=list, max_length=32)


class PromptTemplateUpdate(PromptTemplateCreate):
    expected_version: int = Field(ge=1, le=1_000_000)


class PromptRenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, Any] = Field(default_factory=dict)
    revision: int | None = Field(default=None, ge=1, le=1_000_000)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        if len(values) > 32:
            raise ValueError("No more than 32 prompt values may be supplied.")
        try:
            encoded = json.dumps(values, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("Prompt values must be JSON-compatible and finite.") from None
        if len(encoded.encode("utf-8")) > 512 * 1024:
            raise ValueError("Prompt render values exceed the 512 KiB request limit.")
        for name, value in values.items():
            if not isinstance(name, str) or len(name) > 64:
                raise ValueError("Prompt variable names are invalid.")
            if isinstance(value, str) and len(value) > 16_384:
                raise ValueError(f"Prompt variable {name} exceeds the value-size limit.")
            if isinstance(value, (dict, list)):
                raise ValueError(f"Prompt variable {name} must be a typed scalar value.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Prompt variable {name} must be finite.")
        return values


def _safe(call):
    try:
        return call()
    except PromptTemplateConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except PromptTemplateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except OSError:
        raise HTTPException(status_code=500, detail="Prompt template storage could not be accessed safely.") from None


@router.get("")
def list_prompt_templates(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=1_000_000),
    search: str = Query(default="", max_length=120),
) -> dict[str, Any]:
    """List prompt template metadata in the active workspace."""
    return _safe(lambda: PromptTemplateStore().list_page(limit=limit, offset=offset, search=search))


@router.post("", status_code=201)
def create_prompt_template(request: PromptTemplateCreate) -> dict[str, Any]:
    template = _safe(lambda: PromptTemplateStore().create(
        name=request.name,
        description=request.description,
        body=request.body,
        variables=[item.model_dump() for item in request.variables],
    ))
    return {"template": template}


@router.get("/{template_id}")
def get_prompt_template(template_id: str) -> dict[str, Any]:
    template = _safe(lambda: PromptTemplateStore().get(template_id))
    if template is None:
        raise HTTPException(status_code=404, detail="Prompt template was not found.")
    return {"template": template}


@router.put("/{template_id}")
def update_prompt_template(template_id: str, request: PromptTemplateUpdate) -> dict[str, Any]:
    template = _safe(lambda: PromptTemplateStore().update(
        template_id,
        expected_version=request.expected_version,
        name=request.name,
        description=request.description,
        body=request.body,
        variables=[item.model_dump() for item in request.variables],
    ))
    if template is None:
        raise HTTPException(status_code=404, detail="Prompt template was not found.")
    return {"template": template}


@router.get("/{template_id}/versions")
def list_prompt_template_versions(template_id: str) -> dict[str, Any]:
    versions = _safe(lambda: PromptTemplateStore().versions(template_id))
    if versions is None:
        raise HTTPException(status_code=404, detail="Prompt template was not found.")
    return {"template_id": template_id, "versions": versions, "total": len(versions)}


@router.get("/{template_id}/versions/{version}")
def get_prompt_template_version(template_id: str, version: int) -> dict[str, Any]:
    revision = _safe(lambda: PromptTemplateStore().get_revision(template_id, version))
    if revision is None:
        raise HTTPException(status_code=404, detail="Prompt template was not found.")
    return {"revision": revision}


@router.post("/{template_id}/render")
def render_prompt_template(template_id: str, request: PromptRenderRequest) -> dict[str, Any]:
    result = _safe(lambda: PromptTemplateStore().render(
        template_id, request.values, version=request.revision,
    ))
    if result is None:
        raise HTTPException(status_code=404, detail="Prompt template was not found.")
    return result


@router.delete("/{template_id}")
def delete_prompt_template(
    template_id: str,
    expected_version: int = Query(ge=1, le=1_000_000),
) -> dict[str, Any]:
    deleted = _safe(lambda: PromptTemplateStore().delete(
        template_id, expected_version=expected_version,
    ))
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt template was not found.")
    return {"deleted_id": template_id, "deleted": True}
