from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from axiom.api.server import create_app


def test_dataset_snapshot_api_creates_lists_and_verifies_workspace_snapshots(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    content = b'{"prompt":"PRIVATE_SNAPSHOT_API_VALUE"}\n'
    (data / "train.jsonl").write_bytes(content)
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        initial = client.get("/api/workspace/dataset-snapshots")
        assert initial.status_code == 200
        assert initial.json()["snapshots"] == []

        created = client.post("/api/workspace/dataset-snapshots", json={"path": "train.jsonl"})
        assert created.status_code == 201, created.text
        record = created.json()["snapshot"]
        manifest = record["manifest"]
        assert manifest["sha256"] == hashlib.sha256(content).hexdigest()
        assert manifest["file_size_bytes"] == len(content)
        assert "PRIVATE_SNAPSHOT_API_VALUE" not in created.text

        listed = client.get("/api/workspace/dataset-snapshots")
        assert listed.status_code == 200
        assert listed.json()["total"] == 1
        assert listed.json()["snapshots"][0]["manifest"]["snapshot_id"] == manifest["snapshot_id"]

        verified = client.get(f"/api/workspace/dataset-snapshots/{manifest['snapshot_id']}/verify")
        assert verified.status_code == 200
        assert verified.json()["valid"] is True
        assert verified.json()["lineage_valid"] is True


def test_dataset_snapshot_api_rejects_unsafe_sources_parent_and_byte_overflow(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "other.csv").write_text("value\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        traversal = client.post("/api/workspace/dataset-snapshots", json={"path": "../secret.jsonl"})
        wrong_format = client.post("/api/workspace/dataset-snapshots", json={"path": "other.csv"})
        unknown_parent = client.post("/api/workspace/dataset-snapshots", json={
            "path": "train.jsonl", "parent_snapshot_id": "a" * 32,
        })
        oversize = client.post("/api/workspace/dataset-snapshots", json={
            "path": "train.jsonl", "max_bytes": 1,
        })
        extra = client.post("/api/workspace/dataset-snapshots", json={
            "path": "train.jsonl", "max_bytes": 1024, "absolute_path": "/etc/passwd",
        })

        assert traversal.status_code == 400
        assert wrong_format.status_code == 422
        assert unknown_parent.status_code == 409
        assert oversize.status_code == 413
        assert extra.status_code == 422


def test_dataset_snapshot_api_links_parent_manifest_to_prior_snapshot(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        first = client.post("/api/workspace/dataset-snapshots", json={"path": "train.jsonl"}).json()["snapshot"]
        second = client.post("/api/workspace/dataset-snapshots", json={
            "path": "train.jsonl",
            "parent_snapshot_id": first["manifest"]["snapshot_id"],
        }).json()["snapshot"]

        assert second["manifest"]["parent_snapshot_id"] == first["manifest"]["snapshot_id"]
        assert second["manifest"]["parent_manifest_sha256"] == first["manifest_sha256"]


def test_dataset_snapshot_api_returns_not_found_for_unknown_snapshot(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        response = client.get(f"/api/workspace/dataset-snapshots/{'f' * 32}/verify")
    assert response.status_code == 404
