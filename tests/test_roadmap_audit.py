from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom.core.roadmap import audit_roadmap


def _write_roadmap(path: Path, rows: list[str]) -> Path:
    path.write_text(
        "# Roadmap\n\n<!-- AXIOM-ROADMAP-UPDATE:START -->\n"
        "### 01 · Test group\n" + "\n".join(rows) +
        "\n<!-- AXIOM-ROADMAP-UPDATE:END -->\n",
        encoding="utf-8",
    )
    return path


def test_audit_parses_all_140_known_statuses_and_group_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    rows = []
    for number in range(1, 141):
        status = "Implemented; tests passing" if number <= 30 else "In progress; validation pending" if number <= 50 else "Planned"
        rows.append(f"- [R{number:03d} · {status}] Example deliverable {number}.")
    report = audit_roadmap(_write_roadmap(tmp_path / "ROADMAP.md", rows))
    assert report["integrity"]["ok"] is True
    assert report["total"] == 140
    assert report["counts"] == {"implemented": 30, "in_progress": 20, "planned": 90, "unknown": 0}
    assert report["items"][0]["group"] == "Test group"
    assert report["items"][-1]["id"] == "R140"
    json.dumps(report)


def test_audit_reports_missing_duplicate_and_unknown_statuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    rows = ["- [R001 · Implemented; tests passing] One", "- [R001 · Mystery] Duplicate"]
    report = audit_roadmap(_write_roadmap(tmp_path / "ROADMAP.md", rows))
    assert report["integrity"]["ok"] is False
    assert report["integrity"]["duplicate_ids"] == ["R001"]
    assert "R140" in report["integrity"]["missing_ids"]
    assert report["integrity"]["unknown_status_ids"] == ["R001"]


def test_audit_rejects_bad_markers_and_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "ROADMAP.md"
    bad.write_text("# no markers", encoding="utf-8")
    with pytest.raises(ValueError, match="marker pair"):
        audit_roadmap(bad)
    link = tmp_path / "linked.md"
    try:
        link.symlink_to(bad)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symbolic link"):
        audit_roadmap(link)


def test_audit_rejects_paths_outside_workspace_and_symlinked_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    outside_roadmap = _write_roadmap(external / "ROADMAP.md", [])
    monkeypatch.chdir(workspace)

    with pytest.raises(ValueError, match="inside the current workspace or AXIOM source checkout"):
        audit_roadmap(outside_roadmap)

    linked_directory = workspace / "external-link"
    try:
        linked_directory.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    with pytest.raises(ValueError, match="inside the current workspace or AXIOM source checkout"):
        audit_roadmap(linked_directory / "ROADMAP.md")


def test_audit_requires_the_canonical_roadmap_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    other_file = _write_roadmap(tmp_path / "NOT-ROADMAP.md", [])
    with pytest.raises(ValueError, match="named ROADMAP.md"):
        audit_roadmap(other_file)


def test_audit_preserves_the_four_repository_status_classes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    rows = []
    for number in range(1, 141):
        if number <= 29:
            status = "Implemented; tests passing"
        elif number <= 51:
            status = "In progress; validation pending"
        elif number <= 139:
            status = "Planned"
        else:
            status = "In progress"
        rows.append(f"- [R{number:03d} · {status}] Item {number}")
    report = audit_roadmap(_write_roadmap(tmp_path / "ROADMAP.md", rows))
    assert report["integrity"]["ok"] is True
    assert report["counts"] == {"implemented": 29, "in_progress": 23, "planned": 88, "unknown": 0}


def test_roadmap_cli_emits_machine_readable_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    import typer
    from typer.testing import CliRunner

    from axiom.cli.roadmap import roadmap_app

    rows = [f"- [R{number:03d} · Planned] Item {number}" for number in range(1, 141)]
    source = _write_roadmap(tmp_path / "ROADMAP.md", rows)
    app = typer.Typer()
    app.add_typer(roadmap_app, name="roadmap")
    result = CliRunner().invoke(app, ["roadmap", "audit", "--path", str(source), "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["integrity"]["ok"] is True
    assert payload["total"] == 140
    assert payload["counts"]["planned"] == 140
