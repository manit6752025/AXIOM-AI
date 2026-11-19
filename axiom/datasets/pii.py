"""Local-only, values-never-returned heuristics for common PII patterns."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

MAX_PII_TEXT_BYTES = 64 * 1024
MAX_PII_VALUES_PER_ROW = 10_000
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]{1,64}@[\w.-]{1,255}\.[A-Za-z]{2,63}(?![\w.-])")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{5,}\d(?!\w)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _passes_luhn(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def scan_row_for_pii(row: dict[str, Any]) -> dict[str, Any]:
    """Return category/field counts only; never return matched text."""
    counts: Counter[str] = Counter()
    fields: dict[str, set[str]] = defaultdict(set)
    values_scanned = skipped_long = nodes_visited = 0
    row_scan_truncated = False

    def scan_string(value: str, field_name: str) -> None:
        nonlocal values_scanned, skipped_long
        values_scanned += 1
        if len(value.encode("utf-8", errors="replace")) > MAX_PII_TEXT_BYTES:
            skipped_long += 1
            return

        matches = _EMAIL.findall(value)
        if matches:
            counts["email"] += len(matches)
            fields["email"].add(field_name)

        phone_matches = 0
        for candidate in _PHONE.findall(value):
            digits = re.sub(r"\D", "", candidate)
            if 7 <= len(digits) <= 15 and not (13 <= len(digits) <= 19 and _passes_luhn(digits)):
                phone_matches += 1
        if phone_matches:
            counts["phone_like"] += phone_matches
            fields["phone_like"].add(field_name)

        card_matches = 0
        for candidate in _CARD.findall(value):
            digits = re.sub(r"\D", "", candidate)
            if 13 <= len(digits) <= 19 and _passes_luhn(digits):
                card_matches += 1
        if card_matches:
            counts["payment_card"] += card_matches
            fields["payment_card"].add(field_name)

    stack: list[tuple[Any, str | None, bool]] = [(iter(row.items()), None, True)]
    while stack:
        iterator, parent_field, is_root = stack[-1]
        try:
            entry = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        nodes_visited += 1
        if is_root:
            field_name, value = str(entry[0]), entry[1]
        else:
            field_name, value = parent_field or "", entry
        if nodes_visited > MAX_PII_VALUES_PER_ROW:
            row_scan_truncated = True
            break
        if isinstance(value, str):
            scan_string(value, field_name)
        elif isinstance(value, dict):
            stack.append((iter(value.values()), field_name, False))
        elif isinstance(value, list):
            stack.append((iter(value), field_name, False))

    return {
        "values_scanned": values_scanned,
        "matches_by_type": {kind: counts[kind] for kind in ("email", "phone_like", "payment_card")},
        "fields_by_type": {kind: sorted(fields.get(kind, set())) for kind in ("email", "phone_like", "payment_card")},
        "long_values_skipped": skipped_long,
        "row_scan_truncated": row_scan_truncated,
    }
