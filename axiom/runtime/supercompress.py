from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from urllib import error, request
from urllib.parse import urlsplit, urlunsplit

from axiom.version import __version__

DEFAULT_BASE_URL = "https://api.supercompress.dev"
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30


class _NoRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> request.Request:
        raise error.HTTPError(
            newurl,
            code,
            "HTTP redirects are disabled for provider requests.",
            headers,
            fp,
        )


_OPENER = request.build_opener(_NoRedirectHandler())


def _open_request(req: request.Request, *, timeout: int):
    return _OPENER.open(req, timeout=timeout)


@dataclass
class CompressionResult:
    compressed_text: str
    original_tokens: int | None
    kept_tokens: int | None
    tokens_saved: int | None
    savings_pct: float | None
    compression_risk: str | None
    policy_name: str | None
    mode: str | None


def _configured_base_url() -> str:
    raw_url = os.getenv("SUPERCOMPRESS_API_BASE", DEFAULT_BASE_URL).strip()
    try:
        parsed = urlsplit(raw_url)
    except ValueError as exc:
        raise RuntimeError(
            "SUPERCOMPRESS_API_BASE must be an absolute HTTP(S) URL "
            "without credentials."
        ) from exc

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "SUPERCOMPRESS_API_BASE must be an absolute HTTP(S) URL "
            "without credentials."
        )

    try:
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError
    except ValueError as exc:
        raise RuntimeError("SUPERCOMPRESS_API_BASE must contain a valid port.") from exc

    return raw_url.rstrip("/")


def redacted_base_url() -> str:
    """Return a safe endpoint description for status or diagnostics output."""

    raw_url = os.getenv("SUPERCOMPRESS_API_BASE", DEFAULT_BASE_URL).strip()
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return "<invalid endpoint>"

    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<invalid endpoint>"

    if parsed.scheme not in {"http", "https"} or not hostname:
        return "<invalid endpoint>"

    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"

    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, host, path, "", ""))


def _read_limited(response: object) -> bytes:
    reader = getattr(response, "read", None)
    if not callable(reader):
        raise RuntimeError("SuperCompress response was not readable.")  # noqa: TRY004

    try:
        body = reader(MAX_RESPONSE_BYTES + 1)
    except OSError as exc:
        raise RuntimeError("SuperCompress response could not be read.") from exc
    if not isinstance(body, bytes):
        raise TypeError("SuperCompress response was not valid bytes.")
    if len(body) > MAX_RESPONSE_BYTES:
        raise RuntimeError("SuperCompress response exceeded the 10 MiB safety limit.")
    return body


def _optional_non_negative_int(
    data: dict,
    key: str,
) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"SuperCompress response field {key!r} must be a non-negative integer."
        )
    return value


def _optional_string(data: dict, key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"SuperCompress response field {key!r} must be a string.")
    return value


def compress_context(
    context: str,
    query: str,
    *,
    budget_ratio: float = 0.35,
    ccr: bool = False,
) -> CompressionResult:
    if not isinstance(context, str) or not isinstance(query, str):
        raise TypeError("Context and query must be strings.")

    if (
        isinstance(budget_ratio, bool)
        or not isinstance(budget_ratio, (int, float))
        or not math.isfinite(budget_ratio)
        or not 0 < budget_ratio <= 1
    ):
        raise ValueError("budget_ratio must be a finite number in (0, 1].")

    if not isinstance(ccr, bool):
        raise TypeError("ccr must be a boolean.")

    api_key = os.getenv("SUPERCOMPRESS_API_KEY")

    if not api_key:
        raise RuntimeError("SUPERCOMPRESS_API_KEY is not configured.")

    base_url = _configured_base_url()

    payload = {
        "context": context,
        "query": query,
        "budget_ratio": budget_ratio,
        "ccr": ccr,
    }

    req = request.Request(
        f"{base_url}/api/v1/compress",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "User-Agent": f"AXIOM-AI/{__version__}",
        },
        method="POST",
    )

    try:
        with _open_request(
            req,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            body = _read_limited(response)
            data = json.loads(body.decode("utf-8"))
    except error.HTTPError as exc:
        try:
            body = _read_limited(exc).decode("utf-8", errors="replace")
        except RuntimeError:
            body = "response body unavailable"
        finally:
            exc.close()

        if exc.code == 503 and "neural_unavailable" in body:
            raise RuntimeError(
                "SuperCompress neural compression is currently unavailable. "
                "The remote service accepted the request but its neural "
                "compression backend is not configured."
            ) from exc

        raise RuntimeError(f"SuperCompress HTTP {exc.code}: {body[:512]}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"SuperCompress connection failed: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise RuntimeError(f"SuperCompress connection failed: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("SuperCompress returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise TypeError("SuperCompress response must be a JSON object.")

    compressed = data.get("compressed_text")
    if compressed is None:
        compressed = data.get("compressed")

    if not isinstance(compressed, str):
        raise TypeError("SuperCompress response did not contain compressed text.")

    original = _optional_non_negative_int(data, "original_tokens")
    kept = _optional_non_negative_int(data, "kept_tokens")
    saved = _optional_non_negative_int(data, "tokens_saved")

    if original is not None and kept is not None and kept > original:
        raise RuntimeError("SuperCompress response kept more tokens than the original.")

    savings = data.get("kv_savings_pct")

    if savings is None and original and kept is not None:
        savings = ((original - kept) / original) * 100

    if savings is not None and (
        isinstance(savings, bool)
        or not isinstance(savings, (int, float))
        or not math.isfinite(savings)
    ):
        raise RuntimeError(
            "SuperCompress response field 'kv_savings_pct' must be finite."
        )

    return CompressionResult(
        compressed_text=compressed,
        original_tokens=original,
        kept_tokens=kept,
        tokens_saved=saved,
        savings_pct=savings,
        compression_risk=_optional_string(data, "compression_risk"),
        policy_name=_optional_string(data, "policy_name"),
        mode=_optional_string(data, "mode"),
    )
