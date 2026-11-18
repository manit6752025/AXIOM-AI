from __future__ import annotations

import json

from typer.testing import CliRunner

from axiom.cli.model_studio import studio_app


def test_model_studio_jobs_cli_supports_offset_search_and_json(monkeypatch):
    from axiom.cli import model_studio

    expected = {
        "jobs": [{"id": "a" * 32, "operation": "pull_native", "status": "completed", "updated_at": "2026-11-18"}],
        "total": 9, "limit": 3, "offset": 6, "has_more": False,
        "truncated": True, "scan_limit": 5000, "filters": {"search": "qwen"},
    }
    observed = {}
    def fake_page(*, limit, offset, search):
        observed.update(limit=limit, offset=offset, search=search)
        return expected
    monkeypatch.setattr(model_studio.studio, "list_jobs_page", fake_page)
    result = CliRunner().invoke(
        studio_app,
        ["jobs", "--limit", "3", "--offset", "6", "--search", "qwen", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert observed == {"limit": 3, "offset": 6, "search": "qwen"}
    assert json.loads(result.stdout) == expected
