"""Synchronous, dependency-free HTTP client for AXIOM API v1."""
from __future__ import annotations

import json
import math
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, BinaryIO, Mapping
from urllib.parse import quote

from .errors import AxiomAPIError, AxiomConfigurationError, AxiomNetworkError
from .types import (
    APIContractMetadata,
    PredictionJobCancelResult,
    PredictionJobPage,
    PredictionJobStartRequest,
    PredictionJobStartResult,
    PredictionJobResumeResult,
    PredictionJobStatusResult,
    TelemetryHistory,
    WorkspaceBackupRestoreResult,
    WorkspaceCatalog,
)

DEFAULT_BASE_URL = "http://127.0.0.1:40507"
API_CONTRACT_VERSION = 1


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect before urllib can contact another origin."""

    def _reject_redirect(self, request, response, code, message, headers):
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "AXIOM SDK redirects are disabled.",
            headers,
            response,
        )

    http_error_300 = _reject_redirect
    http_error_301 = _reject_redirect
    http_error_302 = _reject_redirect
    http_error_303 = _reject_redirect
    http_error_305 = _reject_redirect
    http_error_307 = _reject_redirect
    http_error_308 = _reject_redirect

    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class AxiomClient:
    """Synchronous client for a local or securely hosted AXIOM API v1 service."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        token: str | None = None,
        workspace_id: str | None = None,
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
        user_agent: str = "axiom-ai-sdk-python/1.0.0",
    ) -> None:
        self._base_url = self._validate_base_url(base_url)
        if token is not None and (
            not isinstance(token, str)
            or not token.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in token)
        ):
            raise AxiomConfigurationError("Bearer token must be a non-empty string without control characters.")
        if workspace_id is not None and (
            not isinstance(workspace_id, str)
            or not workspace_id.strip()
            or len(workspace_id) > 128
            or any(ord(character) < 32 or ord(character) == 127 for character in workspace_id)
        ):
            raise AxiomConfigurationError("Workspace ID must be a non-empty printable string of at most 128 characters.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 900:
            raise AxiomConfigurationError("Request timeout must be greater than 0 and at most 900 seconds.")
        if (
            not isinstance(user_agent, str)
            or not user_agent.strip()
            or len(user_agent) > 256
            or any(ord(character) < 32 or ord(character) == 127 for character in user_agent)
        ):
            raise AxiomConfigurationError("User-Agent must contain 1 to 256 printable characters.")
        self.token = token.strip() if token is not None else None
        self.workspace_id = workspace_id.strip() if workspace_id is not None else None
        self.timeout = float(timeout)
        self.user_agent = user_agent.strip()
        self.headers = self._validate_extra_headers(headers if headers is not None else {})
        self._closed = False
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    @property
    def base_url(self) -> str:
        """Return the validated API origin; callers cannot replace it in-place."""
        return self._base_url

    @staticmethod
    def _validate_base_url(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 2048:
            raise AxiomConfigurationError("base_url must be a non-empty HTTP(S) origin.")
        parsed = urllib.parse.urlsplit(value.strip())
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise AxiomConfigurationError("base_url must be an HTTP(S) origin without credentials, path, query, or fragment.")
        try:
            port = parsed.port
        except ValueError:
            raise AxiomConfigurationError("base_url contains an invalid port.") from None
        if port is not None and not 1 <= port <= 65535:
            raise AxiomConfigurationError("base_url port must be between 1 and 65535.")
        local = parsed.hostname.casefold().rstrip(".") in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not local:
            raise AxiomConfigurationError("Non-loopback AXIOM services must use HTTPS.")
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _validate_extra_headers(headers: Mapping[str, str]) -> dict[str, str]:
        if not isinstance(headers, Mapping):
            raise AxiomConfigurationError("headers must be a mapping of string names and values.")
        normalized: dict[str, str] = {}
        seen_names: set[str] = set()
        protected = {"authorization", "x-request-id", "x-axiom-workspace-id", "host", "content-length"}
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str) or not key.strip():
                raise AxiomConfigurationError("Additional headers must have non-empty string names and string values.")
            folded_name = key.casefold()
            if folded_name in protected:
                raise AxiomConfigurationError(f"The {key} header is managed by AxiomClient.")
            if folded_name in seen_names:
                raise AxiomConfigurationError(f"The {key} header was supplied more than once.")
            seen_names.add(folded_name)
            if any(ord(character) < 32 or ord(character) == 127 for character in key + value):
                raise AxiomConfigurationError("Additional headers must not contain control characters.")
            normalized[key] = value
        return normalized

    def _url(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        if (
            not isinstance(path, str)
            or not path.startswith("/api/")
            or path.startswith("//")
            or "\\" in path
            or "#" in path
        ):
            raise AxiomConfigurationError("SDK paths must be relative AXIOM /api/ routes.")
        if any(part in {".", ".."} for part in path.split("/")):
            raise AxiomConfigurationError("SDK paths must not contain dot segments.")
        url = self.base_url + path
        if params:
            flattened: list[tuple[str, str]] = []
            for key, value in params.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    flattened.extend((str(key), str(item)) for item in value)
                else:
                    flattened.append((str(key), str(value)))
            query = urllib.parse.urlencode(flattened, doseq=True)
            if query:
                url += ("&" if "?" in url else "?") + query
        return url

    @staticmethod
    def _decode_payload(raw: bytes) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _api_error(status: int, headers: Mapping[str, str], payload: Any) -> AxiomAPIError:
        request_id = headers.get("X-Request-ID") if hasattr(headers, "get") else None
        body = payload if isinstance(payload, dict) else {}
        error = body.get("error") if isinstance(body.get("error"), dict) else {}
        code = error.get("code") if isinstance(error.get("code"), str) else None
        message = (
            error.get("message") if isinstance(error.get("message"), str)
            else body.get("detail") if isinstance(body.get("detail"), str)
            else body.get("message") if isinstance(body.get("message"), str)
            else f"AXIOM API request failed with HTTP {status}."
        )
        if not request_id and isinstance(body.get("request_id"), str):
            request_id = body["request_id"]
        return AxiomAPIError(
            message,
            status_code=status,
            code=code,
            request_id=request_id,
            details=body.get("details", error.get("details")),
        )

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        body: bytes | None = None,
        content_type: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, Mapping[str, str], bytes]:
        if self._closed:
            raise AxiomConfigurationError("AxiomClient is closed.")
        if not isinstance(method, str) or method.upper() not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}:
            raise AxiomConfigurationError("HTTP method is not supported by the AXIOM SDK.")
        method = method.upper()
        if json_body is not None and body is not None:
            raise AxiomConfigurationError("Supply json_body or body, not both.")
        request_headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
            **self.headers,
            **dict(headers or {}),
        }
        request_id = secrets.token_hex(16)
        request_headers["X-Request-ID"] = request_id
        if self.workspace_id:
            request_headers["X-AXIOM-Workspace-ID"] = self.workspace_id
        if self.token:
            request_headers["Authorization"] = f"Bearer {self.token}"
        data = body
        if json_body is not None:
            try:
                data = json.dumps(json_body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError):
                raise AxiomConfigurationError("Request body must contain finite JSON-compatible values.") from None
            request_headers["Content-Type"] = "application/json"
        elif body is not None and content_type:
            request_headers["Content-Type"] = content_type
        selected_timeout = self.timeout if timeout is None else timeout
        if isinstance(selected_timeout, bool) or not isinstance(selected_timeout, (int, float)) or not 0 < selected_timeout <= 900:
            raise AxiomConfigurationError("Request timeout must be greater than 0 and at most 900 seconds.")
        request = urllib.request.Request(
            self._url(path, params),
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            response = self._opener.open(request, timeout=float(selected_timeout))
            with response:
                response_status = int(response.status)
                response_headers = response.headers
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            raise self._api_error(exc.code, exc.headers or {}, self._decode_payload(raw)) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            message = "AXIOM request timed out." if isinstance(reason, TimeoutError) else "AXIOM service could not be reached."
            raise AxiomNetworkError(message, request_id=request_id) from None
        if not 200 <= response_status < 300:
            raise self._api_error(response_status, response_headers, self._decode_payload(raw))
        return response_status, response_headers, raw

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        timeout: float | None = None,
    ) -> Any:
        _status, _headers, raw = self._send(
            method, path, params=params, json_body=json_body, timeout=timeout
        )
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise AxiomNetworkError("AXIOM returned a successful response that was not valid JSON.") from None

    def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: float | None = None,
    ) -> bytes:
        _status, _headers, raw = self._send(
            method, path, params=params, body=body, content_type=content_type, timeout=timeout
        )
        return raw

    def health(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/health")

    def meta(self) -> APIContractMetadata:
        return self.request_json("GET", "/api/meta")

    def openapi(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/openapi.json")

    def runtime(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/workspace/runtime")

    def models(self, *, limit: int = 100, offset: int = 0, search: str | None = None) -> dict[str, Any]:
        return self.request_json("GET", "/api/workspace/models", params={"limit": limit, "offset": offset, "search": search})

    def datasets(self, *, limit: int = 100, offset: int = 0, search: str | None = None) -> dict[str, Any]:
        return self.request_json("GET", "/api/workspace/datasets", params={"limit": limit, "offset": offset, "search": search})

    def workspaces(self) -> WorkspaceCatalog:
        return self.request_json("GET", "/api/workspaces")

    def create_workspace(self, name: str) -> dict[str, Any]:
        return self.request_json("POST", "/api/workspaces", json_body={"name": name})

    def rename_workspace(self, workspace_id: str, name: str) -> dict[str, Any]:
        return self.request_json("PATCH", f"/api/workspaces/{quote(workspace_id, safe='')}", json_body={"name": name})

    def download_workspace_backup(self, workspace_id: str, *, timeout: float | None = None) -> bytes:
        selected_timeout = 900 if timeout is None else timeout
        return self.request_bytes(
            "GET", f"/api/workspaces/{quote(workspace_id, safe='')}/backup", timeout=selected_timeout
        )

    def download_workspace_backup_to(
        self,
        workspace_id: str,
        output: BinaryIO,
        *,
        timeout: float = 900,
        chunk_size: int = 1024 * 1024,
    ) -> int:
        if (
            not isinstance(workspace_id, str)
            or not workspace_id.strip()
            or len(workspace_id) > 128
            or any(ord(character) < 32 or ord(character) == 127 for character in workspace_id)
        ):
            raise AxiomConfigurationError("Workspace ID must contain 1 to 128 printable characters.")
        """Stream a workspace archive into a caller-owned binary file-like object.

        Unlike download_workspace_backup(), this method does not buffer the
        full archive in memory. The caller controls the output's lifecycle and
        should use a new/private file or a temporary file where appropriate.
        """
        if not callable(getattr(output, "write", None)):
            raise AxiomConfigurationError("output must be a writable binary stream.")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 4096 <= chunk_size <= 8 * 1024 * 1024:
            raise AxiomConfigurationError("chunk_size must be between 4 KiB and 8 MiB.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 900:
            raise AxiomConfigurationError("Request timeout must be greater than 0 and at most 900 seconds.")
        if self._closed:
            raise AxiomConfigurationError("AxiomClient is closed.")
        request_id = secrets.token_hex(16)
        headers = {
            "Accept": "application/zip",
            "User-Agent": self.user_agent,
            **self.headers,
            "X-Request-ID": request_id,
        }
        if self.workspace_id:
            headers["X-AXIOM-Workspace-ID"] = self.workspace_id
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self._url(f"/api/workspaces/{quote(workspace_id, safe='')}/backup"),
            headers=headers,
            method="GET",
        )
        written = 0
        try:
            with self._opener.open(request, timeout=float(timeout)) as response:
                if not 200 <= int(response.status) < 300:
                    raise self._api_error(int(response.status), response.headers, self._decode_payload(response.read()))
                while True:
                    block = response.read(chunk_size)
                    if not block:
                        break
                    result = output.write(block)
                    if result is not None and result != len(block):
                        raise AxiomNetworkError("The backup output stream accepted only part of a response chunk.", request_id=request_id)
                    written += len(block)
        except urllib.error.HTTPError as exc:
            raise self._api_error(exc.code, exc.headers or {}, self._decode_payload(exc.read())) from None
        except AxiomAPIError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            message = "AXIOM backup download timed out." if isinstance(reason, TimeoutError) else "AXIOM backup could not be downloaded."
            raise AxiomNetworkError(message, request_id=request_id) from None
        return written

    def restore_workspace_backup(
        self, archive: bytes, name: str, *, timeout: float = 900
    ) -> WorkspaceBackupRestoreResult:
        if not isinstance(archive, bytes) or not archive:
            raise AxiomConfigurationError("Backup archive must be a non-empty bytes object.")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 64 or any(
            ord(character) < 32 or ord(character) == 127 for character in name
        ):
            raise AxiomConfigurationError("Restored workspace name must contain 1 to 64 printable characters.")
        _status, _headers, raw = self._send(
            "POST",
            "/api/workspaces/backup/restore",
            params={"name": name},
            body=archive,
            content_type="application/zip",
            timeout=timeout,
        )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise AxiomNetworkError("AXIOM returned an invalid workspace restore response.") from None

    def engine_status(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/engine/status")

    def engine_telemetry(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/engine/telemetry")

    def telemetry_history(self, *, limit: int = 500) -> TelemetryHistory:
        return self.request_json("GET", "/api/engine/telemetry-history", params={"limit": limit})

    def configure_telemetry_history(
        self, *, enabled: bool, max_samples: int = 120, max_age_days: int = 30
    ) -> TelemetryHistory:
        return self.request_json(
            "PUT", "/api/engine/telemetry-history/settings",
            json_body={"enabled": enabled, "max_samples": max_samples, "max_age_days": max_age_days},
        )

    def record_telemetry_sample(self) -> dict[str, Any]:
        return self.request_json("POST", "/api/engine/telemetry-history")

    def clear_telemetry_history(self) -> TelemetryHistory:
        return self.request_json("DELETE", "/api/engine/telemetry-history")

    def knowledge_bases(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/workspace/knowledge-bases")

    def create_knowledge_base(self, name: str, description: str = "") -> dict[str, Any]:
        return self.request_json("POST", "/api/workspace/knowledge-bases", json_body={"name": name, "description": description})

    def search_knowledge_base(self, base_id: str, query: str, **options: Any) -> dict[str, Any]:
        return self.request_json(
            "POST", f"/api/workspace/knowledge-bases/{quote(base_id, safe='')}/search",
            json_body={"query": query, **options},
        )


    @staticmethod
    def _validate_prediction_job_id(job_id: str) -> None:
        if (
            not isinstance(job_id, str)
            or len(job_id) != 32
            or any(character not in "0123456789abcdef" for character in job_id)
        ):
            raise AxiomConfigurationError("Prediction job ID must be 32 lowercase hexadecimal characters.")

    def prediction_jobs(self, *, limit: int = 20) -> PredictionJobPage:
        """List workspace-local prediction-job summaries without returning generated text."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise AxiomConfigurationError("Prediction job list limit must be between 1 and 50.")
        return self.request_json("GET", "/api/prediction-jobs", params={"limit": limit})

    def start_prediction_job(self, request: PredictionJobStartRequest) -> PredictionJobStartResult:
        """Start a bounded local JSONL prediction job; this consumes local compute."""
        if not isinstance(request, Mapping):
            raise AxiomConfigurationError("Prediction job start request must be a mapping.")
        payload = dict(request)
        allowed = {
            "model_path", "adapter_path", "dataset_path", "output_path",
            "prompt_field", "reference_field", "prediction_field",
            "max_new_tokens", "temperature", "seed", "device", "precision",
        }
        if set(payload) - allowed:
            raise AxiomConfigurationError("Prediction job start request contains unsupported fields.")
        for field, maximum in (("model_path", 4096), ("dataset_path", 512), ("output_path", 512)):
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                raise AxiomConfigurationError(f"Prediction job start requires a non-empty {field} of at most {maximum} characters.")
        defaults = {"prompt_field": "prompt", "reference_field": "reference", "prediction_field": "prediction"}
        field_values = []
        field_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,159}$")
        for field, default in defaults.items():
            value = payload.get(field, default)
            if not isinstance(value, str) or not field_pattern.fullmatch(value):
                raise AxiomConfigurationError(f"Prediction job {field} must be a valid JSON field name of at most 160 characters.")
            field_values.append(value)
        if len(set(field_values)) != len(field_values):
            raise AxiomConfigurationError("Prompt, reference, and prediction fields must be different.")
        if not payload["dataset_path"].startswith("data/"):
            raise AxiomConfigurationError("Prediction input must be a workspace-relative path under data/.")
        output_path = payload["output_path"]
        if (
            not output_path.startswith("data/")
            or output_path.count("/") != 1
            or not output_path.lower().endswith(".jsonl")
        ):
            raise AxiomConfigurationError("Prediction output must be a direct-child data/NAME.jsonl path.")
        if output_path == payload["dataset_path"]:
            raise AxiomConfigurationError("Prediction output must differ from the source dataset.")
        if "max_new_tokens" in payload and (
            isinstance(payload["max_new_tokens"], bool)
            or not isinstance(payload["max_new_tokens"], int)
            or not 1 <= payload["max_new_tokens"] <= 64
        ):
            raise AxiomConfigurationError("max_new_tokens must be an integer between 1 and 64.")
        if "temperature" in payload and (
            isinstance(payload["temperature"], bool)
            or not isinstance(payload["temperature"], (int, float))
            or not math.isfinite(payload["temperature"])
            or not 0 <= payload["temperature"] <= 2
        ):
            raise AxiomConfigurationError("temperature must be a finite number between 0 and 2.")
        if "seed" in payload and payload["seed"] is not None and (
            isinstance(payload["seed"], bool)
            or not isinstance(payload["seed"], int)
            or not 0 <= payload["seed"] <= 2**63 - 1
        ):
            raise AxiomConfigurationError("seed must be an integer between 0 and 2^63-1.")
        adapter = payload.get("adapter_path")
        if adapter is not None and (
            not isinstance(adapter, str) or not adapter.strip() or len(adapter) > 4096
        ):
            raise AxiomConfigurationError("adapter_path must be omitted or contain 1 to 4096 characters.")
        device = payload.get("device", "auto")
        precision = payload.get("precision", "auto")
        if not isinstance(device, str) or device not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}:
            raise AxiomConfigurationError("device is not a supported local prediction device.")
        if not isinstance(precision, str) or precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise AxiomConfigurationError("precision is not a supported local prediction precision.")
        if device == "mps" and precision == "bf16":
            raise AxiomConfigurationError("BF16 precision is not supported on Metal/MPS.")
        return self.request_json("POST", "/api/prediction-jobs", json_body=payload)

    def prediction_job(self, job_id: str) -> PredictionJobStatusResult:
        """Read a single workspace-local prediction job summary."""
        self._validate_prediction_job_id(job_id)
        return self.request_json("GET", f"/api/prediction-jobs/{quote(job_id, safe='')}")

    def resume_prediction_job(self, job_id: str) -> PredictionJobResumeResult:
        """Resume an interrupted job after AXIOM verifies its saved inputs and checkpoint."""
        self._validate_prediction_job_id(job_id)
        return self.request_json("POST", f"/api/prediction-jobs/{quote(job_id, safe='')}/resume")

    def cancel_prediction_job(self, job_id: str) -> PredictionJobCancelResult:
        """Request cooperative cancellation of a workspace-local prediction job."""
        self._validate_prediction_job_id(job_id)
        return self.request_json("POST", f"/api/prediction-jobs/{quote(job_id, safe='')}/cancel")

    def plugin_manifests(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/security/plugins")

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "AxiomClient":
        if self._closed:
            raise AxiomConfigurationError("AxiomClient is closed.")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
