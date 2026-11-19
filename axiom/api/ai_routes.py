"""FastAPI routes for the beta.6 AI features."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from axiom.api.optimization import plan_optimization
from axiom.api.policy_audit import audit_model

router = APIRouter()


class OptimizeRequest(BaseModel):
    model: str
    agent_type: str = "general-agent"
    use_case: str = "assistant"
    privacy: str = "local"
    latency: str = "low"
    target_tokens_per_second: float = 10.0
    sequence_length: int = 2048
    quantization: str = "auto"


class AuditRequest(BaseModel):
    path: str


@router.post("/api/models/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    return plan_optimization(
        model=req.model,
        agent_type=req.agent_type,
        use_case=req.use_case,
        privacy=req.privacy,
        latency=req.latency,
        target_tokens_per_second=req.target_tokens_per_second,
        sequence_length=req.sequence_length,
        quantization=req.quantization,
    ).to_dict()


@router.post("/api/models/policy-audit")
def policy_audit(req: AuditRequest) -> dict[str, Any]:
    try:
        return audit_model(req.path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model path does not exist")
