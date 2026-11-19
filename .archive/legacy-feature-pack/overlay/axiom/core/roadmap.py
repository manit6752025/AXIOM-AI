"""Parse and validate AXIOM's scoped 120-item roadmap without guessing status."""
from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_START = "<!-- AXIOM-ROADMAP-UPDATE:START -->"
_END = "<!-- AXIOM-ROADMAP-UPDATE:END -->"
_ITEM = re.compile(r"^\s*-\s+\[(R\d{3})\s*·\s*([^\]]+)\]\s*(.*?)\s*$")
_HEADING = re.compile(r"^###\s+(.+?)\s*$")
_ALLOWED = {
    "Implemented; tests passing": "implemented",
    "Planned": "planned",
    "In progress; validation pending": "in_progress",
    "In progress": "in_progress",
}
_EXPECTED_IDS = tuple(f"R{number:03d}" for number in range(1, 121))


def default_roadmap_path() -> Path:
    """Find the repository roadmap in the current directory or source checkout."""
    package_root = Path(__file__).resolve().parents[2]
    candidates = (Path.cwd() / "ROADMAP.md", package_root / "ROADMAP.md")
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve()
    raise FileNotFoundError("ROADMAP.md was not found in the current workspace or AXIOM source checkout.")


def audit_roadmap(path: str | Path | None = None) -> dict[str, Any]:
    """Return the 120 roadmap rows, a status roll-up and integrity diagnostics."""
    source = default_roadmap_path() if path is None else Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("Roadmap path must not be a symbolic link.")
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError("ROADMAP.md could not be resolved.") from exc
    if not source.is_file() or source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("ROADMAP.md must be a regular file no larger than 2 MiB.")
    try:
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("ROADMAP.md could not be read as UTF-8.") from exc

    start = content.find(_START)
    end = content.find(_END, start + len(_START)) if start >= 0 else -1
    if start < 0 or end < 0 or content.find(_START, start + len(_START)) >= 0:
        raise ValueError("ROADMAP.md does not contain one valid scoped-roadmap marker pair.")
    section_text = content[start + len(_START):end]

    group = "Uncategorised"
    items: list[dict[str, str]] = []
    for line in section_text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            group = re.sub(r"^\d+\s*·\s*", "", heading.group(1)).strip()
            continue
        match = _ITEM.match(line)
        if not match:
            continue
        item_id, status_label, title = match.groups()
        normalized_status = _ALLOWED.get(status_label.strip())
        if normalized_status is None:
            normalized_status = "unknown"
        items.append(
            {
                "id": item_id,
                "status": normalized_status,
                "status_label": status_label.strip(),
                "title": title.strip(),
                "group": group,
            }
        )

    found_ids = [item["id"] for item in items]
    duplicates = sorted(item_id for item_id, count in Counter(found_ids).items() if count > 1)
    found_set = set(found_ids)
    missing = [item_id for item_id in _EXPECTED_IDS if item_id not in found_set]
    unexpected = sorted(found_set - set(_EXPECTED_IDS))
    unknown_statuses = [item["id"] for item in items if item["status"] == "unknown"]
    counts = Counter(item["status"] for item in items)
    total = len(items)
    complete = counts["implemented"]
    in_progress = counts["in_progress"]
    planned = counts["planned"]
    integrity_ok = total == 120 and not duplicates and not missing and not unexpected and not unknown_statuses

    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source.name,
        "integrity": {
            "ok": integrity_ok,
            "expected_items": 120,
            "parsed_items": total,
            "missing_ids": missing,
            "duplicate_ids": duplicates,
            "unexpected_ids": unexpected,
            "unknown_status_ids": unknown_statuses,
        },
        "total": total,
        "counts": {
            "implemented": complete,
            "in_progress": in_progress,
            "planned": planned,
            "unknown": counts["unknown"],
        },
        "items": items,
    }
