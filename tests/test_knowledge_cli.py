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



def test_knowledge_cli_search_returns_local_citations_and_honours_filters(tmp_path, monkeypatch):
    import json

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "runtime.md").write_text(
        "AXIOM local inference runtime stays on this workstation.", encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("Cooking notes are unrelated to local model serving.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    created = runner.invoke(app, ["--no-banner", "knowledge", "base", "create", "Search", "--json"])
    assert created.exit_code == 0, created.output
    base = json.loads(created.stdout)["knowledge_base"]
    for path in ("docs/runtime.md", "notes.txt"):
        ingested = runner.invoke(app, ["--no-banner", "knowledge", "ingest", base["id"], path, "--json"])
        assert ingested.exit_code == 0, ingested.output

    searched = runner.invoke(app, [
        "--no-banner", "knowledge", "search", base["id"], "--query", "AXIOM local inference",
        "--format", "md", "--source-prefix", "docs", "--json",
    ])
    assert searched.exit_code == 0, searched.output
    payload = json.loads(searched.stdout)
    assert payload["engine"] == "axiom-lexical-knowledge-search"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["citation"]["source_path"] == "docs/runtime.md"
    assert payload["results"][0]["citation"]["content_sha256"]
    assert payload["privacy"]["full_source_documents_returned"] is False



def test_knowledge_cli_search_prints_citations_and_supports_json(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "runtime.md").write_text(
        "AXIOM local inference runtime runs models on this workstation.",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    created = runner.invoke(app, ["--no-banner", "knowledge", "base", "create", "Search", "--json"])
    assert created.exit_code == 0, created.output
    base = json.loads(created.stdout)["knowledge_base"]
    ingested = runner.invoke(app, [
        "--no-banner", "knowledge", "ingest", base["id"], "docs/runtime.md", "--json",
    ])
    assert ingested.exit_code == 0, ingested.output

    result = runner.invoke(app, [
        "--no-banner", "knowledge", "search", base["id"], "--query", "AXIOM local inference", "--json",
    ])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["engine"] == "axiom-lexical-knowledge-search"
    assert report["results"][0]["citation"]["source_path"] == "docs/runtime.md"
    assert "AXIOM" in report["results"][0]["snippet"]

    filtered = runner.invoke(app, [
        "--no-banner", "knowledge", "search", base["id"], "--query", "AXIOM",
        "--source-prefix", "../outside", "--json",
    ])
    assert filtered.exit_code != 0
