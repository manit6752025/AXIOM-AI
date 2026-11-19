from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom.api.diagnostics import router


def test_workspace_diagnostics_api_is_read_only_and_reports_project(tmp_path: Path, monkeypatch):
    (tmp_path / "axiom.yaml").write_text("project:\n  name: api-test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/diagnostics/workspace")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["workspace"]["is_project"] is True
    assert not (tmp_path / ".axiom").exists()


def test_support_bundle_endpoint_returns_zip_with_no_config_values(tmp_path: Path, monkeypatch):
    (tmp_path / "axiom.yaml").write_text("project:\n  name: api-test\napi_key: DONT_EXPORT_ME\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/diagnostics/support-bundle")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert response.headers["cache-control"] == "no-store"
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"README.txt", "diagnostics.json", "environment.json"}
        content = b"".join(archive.read(name) for name in archive.namelist())
    assert b"DONT_EXPORT_ME" not in content


def test_roadmap_audit_endpoint_returns_live_source_statuses(tmp_path: Path, monkeypatch):
    roadmap = tmp_path / "ROADMAP.md"
    rows = [
        "# Roadmap",
        "<!-- AXIOM-ROADMAP-UPDATE:START -->",
        "### 01 · Platform foundation",
    ]
    for number in range(1, 141):
        status = "Implemented; tests passing" if number <= 29 else "In progress; validation pending" if number <= 51 else "Planned"
        rows.append(f"- [R{number:03d} · {status}] Example deliverable {number}.")
    rows.append("<!-- AXIOM-ROADMAP-UPDATE:END -->")
    roadmap.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/diagnostics/roadmap")
    assert response.status_code == 200
    payload = response.json()
    assert payload["integrity"]["ok"] is True
    assert payload["total"] == 140
    assert payload["counts"] == {"implemented": 29, "in_progress": 22, "planned": 89, "unknown": 0}
    assert payload["items"][0]["id"] == "R001"
    assert payload["items"][-1]["id"] == "R140"


def test_roadmap_audit_endpoint_does_not_accept_arbitrary_file_paths(tmp_path: Path, monkeypatch):
    current = tmp_path / "ROADMAP.md"
    current.write_text("""# Roadmap
<!-- AXIOM-ROADMAP-UPDATE:START -->
### Current workspace
- [R001 · Planned] Local workspace marker
<!-- AXIOM-ROADMAP-UPDATE:END -->
""", encoding="utf-8")
    other = tmp_path / "private.md"
    other.write_text("""# Private
<!-- AXIOM-ROADMAP-UPDATE:START -->
### Private file
- [R001 · Planned] Must never be returned from arbitrary path input
<!-- AXIOM-ROADMAP-UPDATE:END -->
""", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/diagnostics/roadmap", params={"path": str(other)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["title"] == "Local workspace marker"
    assert payload["items"][0]["group"] == "Current workspace"
