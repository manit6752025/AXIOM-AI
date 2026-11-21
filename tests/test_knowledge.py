from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from axiom.core.knowledge import KnowledgeBaseStore, KnowledgeStoreError


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "workspace"
    (root / "documents").mkdir(parents=True)
    source = root / "documents" / "guide.md"
    source.write_text("# Private note\nThis source text must remain in the workspace.", encoding="utf-8")
    return root, source


def test_knowledge_base_ingests_supported_text_privately_and_verifies_integrity(tmp_path):
    workspace, source = _workspace(tmp_path)
    source_before = source.read_bytes()
    store = KnowledgeBaseStore(workspace)
    base = store.create_base("Engineering", "Local design documents")
    document = store.ingest_document(base["id"], "documents/guide.md")

    assert document["source_path"] == "documents/guide.md"
    assert document["format"] == "md"
    assert document["content_sha256"] == hashlib.sha256(source_before).hexdigest()
    assert document["size_bytes"] == len(source_before)
    assert store.read_document(base["id"], document["id"]) == source_before.decode("utf-8")
    assert source.read_bytes() == source_before

    listing = store.list_bases()
    assert listing["total"] == 1
    assert listing["knowledge_bases"][0]["document_count"] == 1
    docs = store.list_documents(base["id"])
    assert docs["documents"] == [document]
    index = (workspace / ".axiom" / "knowledge" / "index.json").read_text(encoding="utf-8")
    assert "Private note" not in index
    assert "Private design documents" not in index
    assert str(workspace) not in index


def test_knowledge_store_isolated_between_workspaces(tmp_path):
    workspace_a, _ = _workspace(tmp_path / "a")
    workspace_b, _ = _workspace(tmp_path / "b")
    store_a = KnowledgeBaseStore(workspace_a)
    store_b = KnowledgeBaseStore(workspace_b)

    base_a = store_a.create_base("Project A")
    base_b = store_b.create_base("Project B")
    assert store_a.get_base(base_a["id"]) is not None
    assert store_a.get_base(base_b["id"]) is None
    assert store_b.get_base(base_b["id"]) is not None
    assert store_b.get_base(base_a["id"]) is None


@pytest.mark.parametrize("path", [
    "../outside.txt",
    "/etc/passwd",
    "documents/../guide.md",
    "documents\\guide.md",
    "documents//guide.md",
])
def test_knowledge_ingestion_rejects_unsafe_paths(tmp_path, path):
    workspace, _ = _workspace(tmp_path)
    store = KnowledgeBaseStore(workspace)
    base = store.create_base("Safe")
    with pytest.raises(KnowledgeStoreError):
        store.ingest_document(base["id"], path)


def test_knowledge_ingestion_rejects_symlink_and_unsupported_formats(tmp_path):
    workspace, source = _workspace(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("external content", encoding="utf-8")
    (workspace / "documents" / "link.md").symlink_to(outside)
    (workspace / "documents" / "table.pdf").write_bytes(b"%PDF")
    store = KnowledgeBaseStore(workspace)
    base = store.create_base("Safe")

    with pytest.raises(KnowledgeStoreError):
        store.ingest_document(base["id"], "documents/link.md")
    with pytest.raises(KnowledgeStoreError, match="Supported"):
        store.ingest_document(base["id"], "documents/table.pdf")


def test_knowledge_ingestion_rejects_binary_empty_and_invalid_utf8(tmp_path):
    workspace, _ = _workspace(tmp_path)
    (workspace / "documents" / "empty.txt").write_text(" \n\t", encoding="utf-8")
    (workspace / "documents" / "binary.txt").write_bytes(b"prefix\x00suffix")
    (workspace / "documents" / "invalid.txt").write_bytes(b"\xff\xfe")
    store = KnowledgeBaseStore(workspace)
    base = store.create_base("Safe")

    for path in ("documents/empty.txt", "documents/binary.txt", "documents/invalid.txt"):
        with pytest.raises(KnowledgeStoreError):
            store.ingest_document(base["id"], path)


def test_knowledge_ingestion_enforces_file_size_limit(tmp_path):
    workspace, _ = _workspace(tmp_path)
    large = workspace / "documents" / "large.txt"
    large.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    store = KnowledgeBaseStore(workspace)
    base = store.create_base("Bounded")
    with pytest.raises(KnowledgeStoreError, match="8 MiB"):
        store.ingest_document(base["id"], "documents/large.txt")


def test_knowledge_store_rejects_duplicate_base_names_and_validates_ids(tmp_path):
    workspace, _ = _workspace(tmp_path)
    store = KnowledgeBaseStore(workspace)
    store.create_base("Engineering")
    with pytest.raises(KnowledgeStoreError, match="unique"):
        store.create_base("engineering")
    with pytest.raises(KnowledgeStoreError, match="ID"):
        store.get_base("../outside")


def test_knowledge_document_and_base_removal_preserve_source_files(tmp_path):
    workspace, source = _workspace(tmp_path)
    store = KnowledgeBaseStore(workspace)
    base = store.create_base("Remove")
    document = store.ingest_document(base["id"], "documents/guide.md")
    removed_doc = store.remove_document(base["id"], document["id"])
    assert removed_doc["id"] == document["id"]
    assert not (workspace / ".axiom" / "knowledge" / "documents" / f"{document['id']}.txt").exists()
    assert store.list_documents(base["id"])["documents"] == []
    base_two = store.create_base("Remove base")
    document_two = store.ingest_document(base_two["id"], "documents/guide.md")
    removed_base = store.remove_base(base_two["id"])
    assert removed_base["document_count"] == 1
    assert store.get_base(base_two["id"]) is None
    assert source.exists()
    assert not (workspace / ".axiom" / "knowledge" / "documents" / f"{document_two['id']}.txt").exists()
