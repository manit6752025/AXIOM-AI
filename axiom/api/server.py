from __future__ import annotations

import os
import logging
import hashlib
import re
import uuid
import hmac
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from axiom.api.evaluation import router as evaluation_router
from axiom.api.model_studio import router as model_studio_router
from axiom.api.fine_tuning import router as fine_tuning_router
from axiom.api.diagnostics import router as diagnostics_router
from axiom.api.runtime_discovery import router as runtime_discovery_router
from axiom.api.agent_profiles import router as agent_profiles_router
from axiom.api.inference import router as inference_router
from axiom.api.engine import router as engine_router
from axiom.api.openai_compat import router as openai_compat_router
from axiom.api.llm_client import router as llm_client_router
from axiom.api.server_optimizer_routes import register_optimizer_routes
from axiom.api.security import router as security_router
from axiom.api.workspace import router as workspace_router
from axiom.api.workspaces import router as workspaces_router
from axiom.api.workflows import router as workflows_router
from axiom.version import __version__

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_logger = logging.getLogger(__name__)
API_CONTRACT_VERSION = 1

_API_SCOPE_TOKEN = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
_API_SCOPE_PUBLIC_PATHS = frozenset({
    "/api/health",
    "/api/meta",
    "/api/ui/build",
    "/api/openapi.json",
    "/api/docs",
    "/api/redoc",
    "/api/docs/oauth2-redirect",
})
_READ_ONLY_POST_PATHS = frozenset({
    # These handlers validate or calculate plans only. Unlisted POST routes
    # intentionally require write scope (fail closed).
    "/api/workflows/validate",
    "/api/models/optimize",
    "/api/models/policy-audit",
    "/api/optimizer/plan",
    "/api/workspace/training-plan",
    "/api/workspace/dataset-cleaning-preview",
    "/api/workspace/dataset-quality",
    "/api/workspace/models/inspect",
    "/api/engine/adapters/compatibility",
    "/api/engine/quantization/estimate",
})


def _configured_api_scope_tokens() -> tuple[str | None, str | None]:
    """Read optional read/write API tokens, rejecting partial or weak configuration."""
    read_token = os.environ.get("AXIOM_API_READ_TOKEN")
    write_token = os.environ.get("AXIOM_API_WRITE_TOKEN")
    if read_token is None and write_token is None:
        return None, None
    if not read_token or not write_token:
        raise RuntimeError(
            "AXIOM_API_READ_TOKEN and AXIOM_API_WRITE_TOKEN must be configured together."
        )
    if not _API_SCOPE_TOKEN.fullmatch(read_token) or not _API_SCOPE_TOKEN.fullmatch(write_token):
        raise RuntimeError(
            "AXIOM API scope tokens must be 32–256 ASCII characters using letters, digits, '.', '_', '~', or '-'."
        )
    if hmac.compare_digest(read_token, write_token):
        raise RuntimeError("AXIOM API read and write tokens must be different.")
    return read_token, write_token


def _required_api_scope(method: str, path: str) -> str:
    """Return the minimum scope for a request; unknown mutations require write."""
    normalized_method = method.upper()
    if normalized_method in {"GET", "HEAD"}:
        return "read"
    if normalized_method == "POST" and (
        path in _READ_ONLY_POST_PATHS
        or re.fullmatch(r"/api/workflows/[A-Za-z0-9_-]{1,64}/dry-run", path)
    ):
        return "read"
    return "write"


def _is_same_origin_control_center_request(request: Any) -> bool:
    """Keep the bundled same-origin Control Center usable with API tokens enabled."""
    origin = request.headers.get("origin")
    host = request.headers.get("host")
    fetch_site = request.headers.get("sec-fetch-site", "")
    return bool(
        origin
        and host
        and origin == f"{request.url.scheme}://{host}"
        and fetch_site in {"", "same-origin"}
    )


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
    model_path: str


def create_app() -> Any:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse

    from axiom.api.optimization import plan_optimization
    from axiom.api.policy_audit import audit_model

    api_read_token, api_write_token = _configured_api_scope_tokens()

    app = FastAPI(
        title="AXIOM API",
        version=__version__,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in os.environ.get(
                "AXIOM_CORS_ORIGINS",
                "http://localhost:4173,http://127.0.0.1:4173,http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id

        path = request.url.path
        if (
            api_read_token is not None
            and path.startswith(("/api/", "/v1/"))
            and path not in _API_SCOPE_PUBLIC_PATHS
            and request.method.upper() != "OPTIONS"
            and not _is_same_origin_control_center_request(request)
        ):
            required_scope = _required_api_scope(request.method, path)
            authorization = request.headers.get("authorization", "")
            scheme, separator, supplied_token = authorization.partition(" ")
            supplied_token = supplied_token.strip() if separator and scheme.lower() == "bearer" else ""
            has_write_scope = bool(
                supplied_token and hmac.compare_digest(supplied_token, api_write_token or "")
            )
            has_read_scope = bool(
                supplied_token and hmac.compare_digest(supplied_token, api_read_token or "")
            )
            if not (has_write_scope or has_read_scope):
                return error_response(
                    request,
                    "authentication_required",
                    "A valid AXIOM API token is required.",
                    {"required_scope": required_scope},
                    401,
                    {"WWW-Authenticate": "Bearer"},
                )
            if required_scope == "write" and not has_write_scope:
                return error_response(
                    request,
                    "insufficient_scope",
                    "This operation requires the write scope.",
                    {"required_scope": "write"},
                    403,
                )

        from axiom.core.workspace_context import reset_workspace_context, set_workspace_context
        from axiom.core.workspaces import WorkspaceCatalogError, WorkspaceNotFoundError, WorkspaceRegistry

        workspace_token = None
        try:
            selected_workspace = WorkspaceRegistry().resolve(
                request.headers.get("X-AXIOM-Workspace-ID")
            )
            workspace_token = set_workspace_context(selected_workspace.root, selected_workspace.id)
        except WorkspaceNotFoundError:
            return error_response(
                request, "workspace_not_found", "The selected AXIOM workspace does not exist.",
                {"header": "X-AXIOM-Workspace-ID"}, 404,
            )
        except (OSError, WorkspaceCatalogError):
            return error_response(
                request, "workspace_unavailable", "The selected AXIOM workspace is unavailable.",
                {"header": "X-AXIOM-Workspace-ID"}, 400,
            )

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-AXIOM-Workspace-ID"] = selected_workspace.id
            if path in {"/ui", "/ui/"}:
                # Revalidate the app shell while hashed JS/CSS assets remain cacheable.
                response.headers["Cache-Control"] = "no-cache, max-age=0"
            return response
        finally:
            if workspace_token is not None:
                reset_workspace_context(workspace_token)

    def error_response(
        request: Request,
        code: str,
        message: str,
        detail: Any,
        status: int,
        headers: dict[str, str] | None = None,
    ) -> Any:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        return JSONResponse(
            status_code=status,
            content={
                "error": {"code": code, "message": message, "request_id": request_id},
                "detail": detail,
                "request_id": request_id,
            },
            headers={**(headers or {}), "X-Request-ID": request_id},
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> Any:
        message = exc.detail if isinstance(exc.detail, str) else "Request rejected"
        return error_response(request, "http_error", message, exc.detail, exc.status_code, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Any:
        details = [
            {
                "loc": item.get("loc", []),
                "msg": item.get("msg", "Invalid value"),
                "type": item.get("type", "value_error"),
            }
            for item in exc.errors()
        ]
        return error_response(
            request,
            "validation_error",
            "Request validation failed",
            details,
            422,
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> Any:
        request_id = getattr(request.state, "request_id", "unknown")
        _logger.error(
            "Unhandled AXIOM API error (request_id=%s)",
            request_id,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return error_response(
            request,
            "internal_error",
            "Internal server error",
            "Internal server error",
            500,
        )

    register_optimizer_routes(app)
    app.include_router(workspace_router)
    app.include_router(workspaces_router)
    app.include_router(workflows_router)
    app.include_router(security_router)
    app.include_router(evaluation_router)
    app.include_router(model_studio_router)
    app.include_router(fine_tuning_router)
    app.include_router(diagnostics_router)
    app.include_router(runtime_discovery_router)
    app.include_router(agent_profiles_router)
    app.include_router(inference_router)
    app.include_router(engine_router)
    app.include_router(openai_compat_router)
    app.include_router(llm_client_router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/meta")
    def api_metadata() -> dict[str, str | int]:
        return {
            "api_contract_version": API_CONTRACT_VERSION,
            "application_version": __version__,
            "openapi_url": "/api/openapi.json",
            "compatibility_policy": "Additive changes stay in v1; breaking changes require a parallel /api/v2 contract.",
        }

    @app.get("/api/ui/build")
    def ui_build_metadata() -> Any:
        """Expose a cache-free fingerprint so open Control Center tabs can self-refresh."""
        index_path = Path(__file__).resolve().parent.parent / "webui" / "index.html"
        try:
            fingerprint = hashlib.sha256(index_path.read_bytes()).hexdigest()
        except OSError:
            raise HTTPException(status_code=503, detail="The bundled Control Center is not available.") from None
        return JSONResponse(
            {"fingerprint": fingerprint},
            headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
        )

    @app.post("/api/models/optimize")
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

    @app.post("/api/models/policy-audit")
    def policy_audit(req: AuditRequest) -> dict[str, Any]:
        return audit_model(req.model_path)

    webui_dir = Path(__file__).resolve().parent.parent / "webui"
    if (webui_dir / "index.html").is_file():
        from fastapi.staticfiles import StaticFiles

        app.mount("/ui", StaticFiles(directory=webui_dir, html=True), name="ui")

    return app
