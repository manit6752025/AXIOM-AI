from __future__ import annotations

import json
import re
from pathlib import Path


def test_e2e_coverage_inventory_matches_every_control_center_navigation_route():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "src/App.tsx").read_text(encoding="utf-8")
    start = app_source.index("const navItems")
    end = app_source.index("const pageMeta", start)
    navigation = app_source[start:end]
    route_ids = re.findall(r'\bid:\s*"([a-z][a-z0-9-]*)"', navigation)

    inventory_path = root / "docs/testing/e2e-workflow-coverage.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    pages = inventory["pages"]
    page_ids = [item["id"] for item in pages]

    assert inventory["schema_version"] == 1
    assert len(route_ids) == len(set(route_ids)), "Control Center navigation contains duplicate page IDs."
    assert len(page_ids) == len(set(page_ids)), "E2E coverage inventory contains duplicate page IDs."
    assert set(page_ids) == set(route_ids), {
        "navigation_without_e2e_inventory": sorted(set(route_ids) - set(page_ids)),
        "inventory_without_navigation": sorted(set(page_ids) - set(route_ids)),
    }

    for page in pages:
        assert page["label"].strip()
        assert page["journeys"], f"{page['id']} has no documented acceptance journeys."
        assert page["specs"], f"{page['id']} has no Playwright specification."
        for relative_path in page["specs"]:
            candidate = Path(relative_path)
            assert candidate.parts[0] == "e2e" and ".." not in candidate.parts
            spec_path = root / candidate
            assert spec_path.is_file(), f"{page['id']} references missing spec {relative_path}."
            source = spec_path.read_text(encoding="utf-8")
            assert re.search(r"\btest(?:\.(?:describe|skip|only|fixme))?\s*\(", source), (
                f"{relative_path} contains no Playwright test declaration."
            )
