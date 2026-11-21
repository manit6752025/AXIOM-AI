"""Synchronous, dependency-free HTTP client for AXIOM API v1."""
from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, BinaryIO, Mapping
from urllib.parse import quote

from .errors import AxiomAPIError, AxiomConfigurationError, AxiomNetworkError
from .types import APIContractMetadata, TelemetryHistory, WorkspaceBackupRestoreResult, WorkspaceCatalog

DEFAULT_BASE_URL = "http://127.0.0.1:40507"
API_CONTRACT_VERSION = 1


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent forwarding bearer credentials to a redirect target."""

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
            raise AxiomConfigurationError("Workspace ID must be a non-empty string of at most 128 characters.")
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
        self.headers = self._validate_extra_headers(headers or {})
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
            parsed.port
        except ValueError:
            raise AxiomConfigurationError("base_url contains an invalid port.") from None
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
