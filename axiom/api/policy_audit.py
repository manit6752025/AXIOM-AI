from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axiom.api.model_paths import allowed_model_roots, validate_model_path

HINTS = (
    "safety",
    "guardrail",
    "policy",
    "moderation",
    "refusal",
    "content_filter",
    "safety_checker",
    "chat_template",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            # codeql[py/path-injection]
            path.read_text(encoding="utf-8")
        )  # codeql[py/path-injection]
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_within_trusted_model_roots(path: Path) -> bool:
    candidate = path.resolve()
    for trusted_root in allowed_model_roots():
        try:
            candidate.relative_to(trusted_root.resolve())
            return True
        except ValueError:
            continue
    return False


def _trusted_files(root: Path) -> tuple[Path, ...]:
    candidates = (root,) if root.is_file() else root.rglob("*")
    resolved_root = root.resolve()
    files: list[Path] = []
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            continue
        resolved_path = path.resolve(strict=True)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"model audit path escapes the trusted model root: {path}"
            ) from exc
        files.append(path)
        if len(files) == 5000:
            break
    return tuple(files)


def audit_model(model_path: str) -> dict[str, Any]:
    root = validate_model_path(model_path).resolve()
    base = root.parent if root.is_file() else root
    files = _trusted_files(root)

    indicators: set[str] = set()
    configs: list[str] = []

    for path in files:
        relative = str(path.relative_to(base))

        if any(hint in path.name.lower() for hint in HINTS):
            indicators.add(relative)

        if path.suffix.lower() == ".json" and path.name.lower() in {
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
        }:
            configs.append(relative)
            blob = json.dumps(
                _load_json(path),
                ensure_ascii=False,
            ).lower()

            for hint in HINTS:
                if hint in blob:
                    indicators.add(f"{relative}: contains '{hint}'")

    return {
        "model": str(root),
        "status": "review_required" if indicators else "no_obvious_policy_indicators",
        "policy_indicators": sorted(indicators),
        "config_files": sorted(configs),
        "scope": (
            "Audit only. AXIOM does not remove, disable, bypass, "
            "or weaken model safety mechanisms."
        ),
    }
