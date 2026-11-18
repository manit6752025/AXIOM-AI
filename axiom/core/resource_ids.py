"""Canonical identity helpers for persisted AXIOM resources.

New persisted resource IDs are UUID4 values rendered as 32 lowercase hexadecimal
characters. Legacy IDs are migrated deterministically and are never re-issued.
"""
from __future__ import annotations

import re
import uuid

RESOURCE_ID_FORMAT = "uuid4-hex-v1"
RESOURCE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_LEGACY_ID_NAMESPACE = uuid.NAMESPACE_URL


def new_resource_id() -> str:
    """Create a collision-resistant ID for a new persisted resource."""
    return uuid.uuid4().hex


def is_resource_id(value: object) -> bool:
    """Return whether value is a canonical lowercase 128-bit resource ID."""
    return isinstance(value, str) and RESOURCE_ID_PATTERN.fullmatch(value) is not None


def legacy_resource_id(kind: str, identity: str) -> str:
    """Return a deterministic UUID5 ID for a legacy resource lacking an ID.

    The namespace and axiom:<kind>:<identity> name are part of the v1
    migration contract. Do not change them after a resource has been migrated.
    """
    if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", kind):
        raise ValueError("Legacy resource kind must be a short lowercase identifier.")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Legacy resource identity must be a non-empty string.")
    return uuid.uuid5(_LEGACY_ID_NAMESPACE, f"axiom:{kind}:{identity}").hex
