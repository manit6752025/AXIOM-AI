from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from axiom.api.server import create_app


ENDPOINTS = (
    "/api/health",
    "/api/meta",
    "/api/workspace/runtime",
    "/api/workspace/models?offset=0&limit=5",
    "/api/workspace/datasets",
    "/api/workspace/events",
    "/api/diagnostics/workspace",
    "/api/diagnostics/roadmap",
    "/api/diagnostics/runtime-connectors",
)


def test_all_api_playground_endpoints_respond_successfully(
    tmp_path: Path, monkeypatch
) -> None:
    roadmap = [
        "# Roadmap",
        "<!-- AXIOM-ROADMAP-UPDATE:START -->",
        "### 01 · Platform foundation",
    ]
    for number in range(1, 121):
        status = (
            "Implemented; tests passing"
            if number <= 29
            else "In progress; validation pending"
            if number <= 51
            else "Planned"
        )
        roadmap.append(
            f"- [R{number:03d} · {status}] Example deliverable {number}."
        )
    roadmap.append("<!-- AXIOM-ROADMAP-UPDATE:END -->")
    (tmp_path / "ROADMAP.md").write_text(
        "\n".join(roadmap) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        for endpoint in ENDPOINTS:
            response = client.get(endpoint)
            assert response.status_code == 200, (
                f"GET {endpoint} returned {response.status_code}: "
                f"{response.text[:300]}"
            )
