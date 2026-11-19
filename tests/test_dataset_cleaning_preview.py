import io
import json

from axiom.datasets.cleaning_preview import preview_jsonl_cleaning


def test_cleaning_preview_matches_cleaner_decisions_without_writing():
    source = (
        b'{"id":1,"text":"first"}\n'
        b'{"id":1,"text":"first"}\n'
        b'not json\n'
        b'\n'
        b'[1,2]\n'
        b'{"id":2}\n'
    )
    stream = io.BytesIO(source)

    report = preview_jsonl_cleaning(
        stream,
        file_name="train.jsonl",
        file_size_bytes=len(source),
    )

    assert report["preview_only"] is True
    assert report["output_written"] is False
    assert report["kept_rows"] == 2
    assert report["removed_duplicate_rows"] == 1
    assert report["removed_invalid_rows"] == 1
    assert report["removed_empty_rows"] == 1
    assert report["removed_non_object_rows"] == 1
    assert report["removed_schema_rows"] == 0
    assert report["rewritten_rows"] == 2
    assert report["removed_rows"] == 4
    assert stream.tell() == len(source)


def test_cleaning_preview_enforces_limits_and_marks_partial_reports():
    source = b'{"text":"' + b"x" * 200 + b'"}\n{"text":"later"}\n'
    report = preview_jsonl_cleaning(
        io.BytesIO(source),
        file_name="large.jsonl",
        file_size_bytes=len(source),
        max_bytes=64,
        max_rows=10,
        max_line_bytes=32,
    )

    assert report["truncated"] is True
    assert {"max_bytes", "max_line_bytes"}.issubset(report["truncation_reasons"])
    assert report["bytes_scanned"] <= 64
    assert report["overlong_rows"] == 1
    assert report["output_written"] is False


def test_cleaning_preview_explains_crlf_normalization(tmp_path):
    from axiom.datasets.cleaner import clean_jsonl

    contents = b'{"id": 1}\r\n'
    source = tmp_path / "crlf.jsonl"
    output = tmp_path / "cleaned.jsonl"
    source.write_bytes(contents)
    with source.open("rb") as stream:
        preview = preview_jsonl_cleaning(
            stream,
            file_name=source.name,
            file_size_bytes=len(contents),
        )

    clean_jsonl(source, output)

    assert preview["rewritten_rows"] == 1
    assert output.read_bytes() == b'{"id": 1}\n'


def test_cleaning_preview_api_is_bounded_and_non_destructive(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    contents = b'{"id":1,"value":"PRIVATE_PREVIEW_VALUE"}\n{"id":1,"value":"PRIVATE_PREVIEW_VALUE"}\n'
    source = data_dir / "train.jsonl"
    source.write_bytes(contents)
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/workspace/dataset-cleaning-preview",
            json={"path": "train.jsonl", "max_bytes": 1024, "max_rows": 10, "max_line_bytes": 512},
        )
        assert response.status_code == 200
        body = response.json()["preview"]
        assert body["removed_duplicate_rows"] == 1
        assert body["preview_only"] is True
        assert body["output_written"] is False
        assert "PRIVATE_PREVIEW_VALUE" not in json.dumps(body)
        assert source.read_bytes() == contents
        assert list(data_dir.iterdir()) == [source]
