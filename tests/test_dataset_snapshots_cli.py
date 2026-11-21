from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from axiom.cli.main import app


def test_dataset_snapshot_cli_creates_lists_and_verifies_immutable_history(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    content = b'{"text":"PRIVATE_CLI_SNAPSHOT"}\n'
    (data / "train.jsonl").write_bytes(content)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    created = runner.invoke(app, ["dataset", "snapshot", "data/train.jsonl", "--json"])
    assert created.exit_code == 0, created.output
    record = json.loads(created.output)["snapshot"]
    snapshot_id = record["manifest"]["snapshot_id"]
    assert record["manifest"]["sha256"] == __import__("hashlib").sha256(content).hexdigest()
    assert "PRIVATE_CLI_SNAPSHOT" not in created.output

    listed = runner.invoke(app, ["dataset", "snapshots", "--json"])
    assert listed.exit_code == 0, listed.output
    report = json.loads(listed.output)
    assert report["total"] == 1
    assert report["snapshots"][0]["manifest"]["snapshot_id"] == snapshot_id

    verified = runner.invoke(app, ["dataset", "snapshot-verify", snapshot_id, "--json"])
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.output)["valid"] is True


def test_dataset_snapshot_cli_rejects_files_outside_workspace_data(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["dataset", "snapshot", str(outside), "--json"])
    assert result.exit_code != 0
