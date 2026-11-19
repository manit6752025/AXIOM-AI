from http import HTTPStatus

from axiom.daemon_api import route


def test_dataset_profile_api_returns_report_without_raw_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(
        '{"text":"hello"}\n'
        '{"text":"hello"}\n'
        '{"email":"private-value"}\n'
        'invalid json\n',
        encoding="utf-8",
    )

    status, payload = route("/api/dataset/profile", {"path": ["data/train.jsonl"]})
    assert status == HTTPStatus.OK
    report = payload["profile"]
    assert report["valid_rows"] == 3
    assert report["invalid_rows"] == 1
    assert report["duplicate_rows"] == 1
    assert report["privacy_review_recommended"] is True
    assert "private-value" not in repr(payload)


def test_dataset_profile_api_rejects_wrong_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text('{"text":"x"}\n', encoding="utf-8")
    status, payload = route("/api/dataset/profile", {"path": ["data.txt"]})
    assert status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert payload["error"] == "unsupported_format"


def test_dataset_profile_api_rejects_outside_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    status, payload = route("/api/dataset/profile", {"path": ["../outside.jsonl"]})
    assert status == HTTPStatus.BAD_REQUEST
    assert payload["error"] == "invalid_path"
