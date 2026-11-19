import json

from axiom.datasets.quality import compare_dataset_schemas, validate_dataset_profile


def test_dataset_quality_comparison_detects_missing_fields_and_type_conflicts():
    first = {
        "path": "train.jsonl",
        "profile": {
            "valid_rows": 4,
            "truncated": False,
            "fields": [
                {"name": "prompt", "types": {"string": 4}, "possibly_sensitive": False},
                {"name": "score", "types": {"integer": 4}, "possibly_sensitive": False},
            ],
        },
    }
    second = {
        "path": "validation.jsonl",
        "profile": {
            "valid_rows": 2,
            "truncated": True,
            "fields": [
                {"name": "prompt", "types": {"string": 2}, "possibly_sensitive": False},
                {"name": "score", "types": {"string": 2}, "possibly_sensitive": False},
                {"name": "label", "types": {"boolean": 2}, "possibly_sensitive": False},
            ],
        },
    }

    report = compare_dataset_schemas([first, second])
    fields = {field["name"]: field for field in report["fields"]}

    assert report["has_differences"] is True
    assert fields["score"]["type_mismatch"] is True
    assert fields["label"]["missing_from"] == ["train.jsonl"]
    assert report["datasets"][1]["truncated"] is True


def test_dataset_quality_required_fields_and_number_types():
    profile = {
        "fields": [
            {"name": "count", "types": {"integer": 2, "null": 1}},
        ],
    }

    assert validate_dataset_profile(
        profile,
        required_fields=["count"],
        expected_types={"count": "number"},
    ) == {"passed": True, "missing_fields": [], "type_mismatches": []}


def test_dataset_quality_api_compares_schemas_without_returning_values(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"prompt":"RAW_TRAIN_MARKER","count":2}\n',
        encoding="utf-8",
    )
    (data_dir / "validation.jsonl").write_text(
        '{"prompt":"RAW_VALIDATION_MARKER","count":"many","label":"RAW_LABEL_MARKER"}\n',
        encoding="utf-8",
    )
    (data_dir / "test.jsonl").write_text('{"prompt":"test","count":1}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/workspace/dataset-quality",
            json={
                "paths": ["train.jsonl", "validation.jsonl"],
                "required_fields": ["prompt", "extra"],
                "expected_types": {"count": "integer"},
                "max_bytes": 1024,
                "max_rows": 20,
                "max_line_bytes": 512,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["comparison"]["has_differences"] is True
        assert body["checks"]["train.jsonl"]["missing_fields"] == ["extra"]
        assert body["checks"]["validation.jsonl"]["type_mismatches"] == [
            {"field": "count", "expected": "integer", "observed": ["string"]},
        ]
        assert body["limits"]["max_rows"] == 20
        assert body["datasets"][0]["profile"]["pii_scan"]["opted_in"] is False
        serialized = json.dumps(body)
        assert "RAW_TRAIN_MARKER" not in serialized
        assert "RAW_VALIDATION_MARKER" not in serialized
        assert "RAW_LABEL_MARKER" not in serialized

        duplicate_paths = client.post(
            "/api/workspace/dataset-quality",
            json={"paths": ["train.jsonl", "train.jsonl"]},
        )
        assert duplicate_paths.status_code == 422

        excessive_budget = client.post(
            "/api/workspace/dataset-quality",
            json={
                "paths": ["train.jsonl", "validation.jsonl", "test.jsonl"],
                "max_bytes": 256 * 1024 * 1024,
            },
        )
        assert excessive_budget.status_code == 422


def test_dataset_quality_api_exposes_only_aggregate_diagnostics(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "first.jsonl").write_bytes(
        b'{"id":"same","prompt":"do not leak this text","email_template":"not private","contact":"secret.person@example.com"}\r\n'
        b'{"id":"same","prompt":"and not this","email_template":"not private","contact":"secret.person@example.com"}\n'
    )
    (data_dir / "second.jsonl").write_text('{"id":"different","prompt":"safe"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/workspace/dataset-quality",
            json={
                "paths": ["first.jsonl", "second.jsonl"],
                "duplicate_key_fields": ["id"],
                "duplicate_rate_threshold": 0.4,
                "text_fields": ["prompt"],
                "sensitive_field_rules": ["prompt"],
                "ignored_sensitive_fields": ["email_template"],
                "scan_pii": True,
                "max_bytes": 1024,
                "max_rows": 20,
                "max_line_bytes": 512,
            },
        )
        assert response.status_code == 200
        body = response.json()
        first = body["datasets"][0]["profile"]
        assert first["duplicate_key_summary"]["duplicate_rows"] == 1
        assert first["duplicate_key_summary"]["threshold_exceeded"] is True
        assert first["diagnostics"]["newline_style"] == "mixed"
        assert first["text_token_lengths"]["prompt"]["observations"] == 2
        assert first["possibly_sensitive_fields"] == ["prompt"]
        assert first["ignored_sensitive_fields"] == ["email_template"]
        assert first["pii_scan"]["opted_in"] is True
        assert first["pii_scan"]["matches_by_type"]["email"] == 2
        assert "do not leak this text" not in json.dumps(body)
        assert "secret.person@example.com" not in json.dumps(body)

        assert body["datasets"][1]["profile"]["pii_scan"]["status"] == "completed"

        profile = client.get(
            "/api/workspace/dataset-profile",
            params=[("path", "first.jsonl"), ("duplicate_key_fields", "id"), ("text_fields", "prompt")],
        )
        assert profile.status_code == 200
        assert profile.json()["profile"]["duplicate_key_summary"]["fields"] == ["id"]
