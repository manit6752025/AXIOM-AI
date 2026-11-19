import hashlib
import io
import json

from axiom.datasets.manifest import create_dataset_manifest


def test_manifest_reports_full_sha256_without_dataset_values():
    contents = b'{"text":"PRIVATE_MANIFEST_VALUE"}\n'
    parent_digest = hashlib.sha256(b"parent-manifest").hexdigest()
    report = create_dataset_manifest(
        io.BytesIO(contents),
        file_name="train.jsonl",
        relative_path="train.jsonl",
        file_size_bytes=len(contents),
        parent_manifest_sha256=parent_digest,
    )

    assert report["status"] == "complete"
    assert report["full_content_hashed"] is True
    assert report["sha256"] == hashlib.sha256(contents).hexdigest()
    assert report["lineage"] == {"relation": "derived_from", "parent_manifest_sha256": parent_digest}
    assert "PRIVATE_MANIFEST_VALUE" not in json.dumps(report)


def test_manifest_labels_bounded_prefix_hash_as_partial():
    contents = b"0123456789"
    report = create_dataset_manifest(
        io.BytesIO(contents),
        file_name="large.jsonl",
        relative_path="large.jsonl",
        file_size_bytes=len(contents),
        max_bytes=4,
    )

    assert report["status"] == "partial"
    assert report["full_content_hashed"] is False
    assert report["sha256"] is None
    assert report["bytes_hashed"] == 4
    assert report["partial_sha256"] == hashlib.sha256(contents[:4]).hexdigest()


def test_manifest_api_is_bounded_and_workspace_scoped(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    contents = b'{"text":"PRIVATE_API_VALUE"}\n'
    (data_dir / "train.jsonl").write_bytes(contents)
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        parent_digest = hashlib.sha256(b"parent-manifest").hexdigest()
        response = client.get(
            "/api/workspace/dataset-manifest",
            params={"path": "train.jsonl", "parent_manifest_sha256": parent_digest},
        )
        assert response.status_code == 200
        manifest = response.json()["manifest"]
        assert manifest["sha256"] == hashlib.sha256(contents).hexdigest()
        assert manifest["lineage"]["parent_manifest_sha256"] == parent_digest
        assert "PRIVATE_API_VALUE" not in response.text

        partial = client.get(
            "/api/workspace/dataset-manifest",
            params={"path": "train.jsonl", "max_bytes": 4},
        )
        assert partial.status_code == 200
        assert partial.json()["manifest"]["status"] == "partial"

        outside = client.get("/api/workspace/dataset-manifest", params={"path": "../train.jsonl"})
        assert outside.status_code == 400
