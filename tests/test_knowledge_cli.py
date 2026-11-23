from __future__ import annotations

import json

from typer.testing import CliRunner

from axiom.cli.main import app


def test_knowledge_cli_create_ingest_list_and_remove_copy(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    source = documents / "private.md"
    source.write_text("# Private\nSecret document body.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    created = runner.invoke(app, ["--no-banner", "knowledge", "base", "create", "Engineering", "--json"])
    assert created.exit_code == 0, created.output
    base = json.loads(created.stdout)["knowledge_base"]
    ingested = runner.invoke(app, [
        "--no-banner", "knowledge", "ingest", base["id"], "documents/private.md", "--json",
    ])
    assert ingested.exit_code == 0, ingested.output
    document = json.loads(ingested.stdout)["document"]
    assert "Secret document body" not in ingested.stdout
    assert document["source_path"] == "documents/private.md"

    listed = runner.invoke(app, ["--no-banner", "knowledge", "documents", base["id"], "--json"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout)["documents"][0]["id"] == document["id"]
    assert "Secret document body" not in listed.stdout

    removed = runner.invoke(app, [
        "--no-banner", "knowledge", "remove-document", base["id"], document["id"], "--yes", "--json",
    ])
    assert removed.exit_code == 0, removed.output
    assert source.read_text(encoding="utf-8") == "# Private\nSecret document body."
    assert not (tmp_path / ".axiom" / "knowledge" / "documents" / f"{document['id']}.txt").exists()


def test_knowledge_cli_rejects_duplicate_names_and_unsafe_ingest(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("local", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    first = runner.invoke(app, ["--no-banner", "knowledge", "base", "create", "Engineering", "--json"])
    assert first.exit_code == 0, first.output
    duplicate = runner.invoke(app, ["--no-banner", "knowledge", "base", "create", "engineering"])
    assert duplicate.exit_code != 0
    base = json.loads(first.stdout)["knowledge_base"]
    unsafe = runner.invoke(app, ["--no-banner", "knowledge", "ingest", base["id"], "../notes.txt"])
    assert unsafe.exit_code != 0
