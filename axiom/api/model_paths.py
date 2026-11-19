"""Trusted filesystem boundaries for AXIOM model and optimizer paths.

User-supplied paths are canonicalized with ``realpath`` and accepted only when
a separator-aware prefix check proves that the resolved path stays inside an
AXIOM-owned root. No filesystem operation is performed on the unvalidated input.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

_MODEL_ROOT_NAMES = (
    "models",
    ".axiom/models",
)
_OUTPUT_ROOT_NAMES = (
    ".axiom/optimized",
    "optimized",
)


def _trusted_roots(names: Iterable[str]) -> tuple[Path, ...]:
    base = os.path.realpath(os.getcwd())
    roots: list[Path] = []
    for name in names:
        roots.append(Path(os.path.realpath(os.path.join(base, name))))
    return tuple(roots)


def allowed_model_roots() -> tuple[Path, ...]:
    """Return the trusted model roots for the current AXIOM workspace."""
    return _trusted_roots(_MODEL_ROOT_NAMES)


def _coerce_path(value: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TypeError("path must be a string or path-like value") from exc
    if not isinstance(raw, str):
        raise TypeError("path must resolve to a string")
    if not raw.strip():
        raise ValueError("path must not be empty")
    return raw


def _inside(candidate: str, roots: Iterable[Path]) -> Path:
    candidate_real = os.path.realpath(candidate)
    for root in roots:
        root_real = os.path.realpath(os.fspath(root))
        if candidate_real == root_real:
            return Path(root_real)
        if candidate_real.startswith(os.path.join(root_real, "")):
            return Path(candidate_real)
    raise ValueError("path is outside an AXIOM trusted filesystem boundary")


def validate_model_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a model path only when it is contained in a trusted model root."""
    return _inside(_coerce_path(value), allowed_model_roots())


def validate_output_path(value: str | os.PathLike[str]) -> Path:
    """Resolve an optimizer output path only inside trusted output roots."""
    return _inside(_coerce_path(value), _trusted_roots(_OUTPUT_ROOT_NAMES))


def validate_child_path(
    parent: str | os.PathLike[str] | Path, child: str | os.PathLike[str]
) -> Path:
    """Resolve a child path and require it to remain below the validated parent."""
    parent_real = os.path.realpath(_coerce_path(parent))
    candidate = os.path.realpath(os.path.join(parent_real, _coerce_path(child)))
    if candidate == parent_real:
        return Path(parent_real)
    if candidate.startswith(os.path.join(parent_real, "")):
        return Path(candidate)
    raise ValueError("child path escapes its trusted parent")


MODEL_ROOT = Path("models")
