"""Safe references to administrator-provisioned environment secrets.

Only variable names may be persisted. Secret values are resolved on demand and
must remain inside the process handling the outbound provider request.
"""
from __future__ import annotations

import os
import re

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MAX_SECRET_VALUE_CHARS = 512


class SecretReferenceError(ValueError):
    """An environment-backed secret reference is invalid or unavailable."""


def normalize_secret_reference(value: object) -> str | None:
    """Validate a reference name; never accept or infer a secret value."""
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not _ENVIRONMENT_NAME.fullmatch(value):
        raise SecretReferenceError("Environment secret references must be valid variable names.")
    return value


def resolve_secret_reference(reference: object) -> str:
    """Resolve a previously validated environment reference for one request."""
    try:
        name = normalize_secret_reference(reference)
    except SecretReferenceError:
        raise SecretReferenceError("The configured credential reference is invalid.") from None
    if name is None:
        raise SecretReferenceError("No credential environment reference is configured.")
    value = os.environ.get(name)
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_SECRET_VALUE_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SecretReferenceError("The configured credential environment variable is missing or invalid.")
    return value
