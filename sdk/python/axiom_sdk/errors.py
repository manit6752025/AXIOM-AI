"""Structured errors returned by the AXIOM HTTP API."""
from __future__ import annotations

from typing import Any


class AxiomError(Exception):
    """Base class for client-side AXIOM SDK failures."""


class AxiomConfigurationError(AxiomError, ValueError):
    """The client configuration is unsafe or invalid."""


class AxiomNetworkError(AxiomError, OSError):
    """The AXIOM service could not be reached or a request timed out."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class AxiomAPIError(AxiomError):
    """An HTTP error returned by the AXIOM API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str | None = None,
        request_id: str | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.details = details

    def __repr__(self) -> str:
        return (
            f"AxiomAPIError(status_code={self.status_code!r}, code={self.code!r}, "
            f"request_id={self.request_id!r}, message={str(self)!r})"
        )
