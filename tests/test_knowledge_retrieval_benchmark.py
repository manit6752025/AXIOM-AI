"""Small offline retrieval-quality benchmark over AXIOM's operator documentation.

The corpus is copied into a temporary workspace and ingested through the same
production KnowledgeBaseStore used by the API/CLI/Runtime UI. The benchmark
measures document-level Recall@5, MRR@5 and nDCG@5 for fixed operator queries;
it does not load models, call hosted embedding services or modify the repository.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

from axiom.core.knowledge import KnowledgeBaseStore
from axiom.core.knowledge_search import search_knowledge_base


_CORPUS = (
    "docs/CLI-GUIDE.md",
    "docs/Diagnostics-and-Support.md",
    "docs/Local-Engine.md",
    "docs/Local-Knowledge.md",
    "docs/Plugin-Manifests.md",
    "docs/Workspace-Backup-Restore.md",
)
_QUERIES = (
    (
        "workspace backup archive manifest checksums SHA-256 restore safe paths",
        "docs/Workspace-Backup-Restore.md",
    ),
    (
        "plugin manifest registration permissions grants tool scopes revoke fingerprints",
        "docs/Plugin-Manifests.md",
    ),
    (
        "support bundle redact secrets credentials environment variables diagnostics",
        "docs/Diagnostics-and-Support.md",
    ),
    (
        "local engine prediction jobs cancellation progress persisted JSONL",
        "docs/Local-Engine.md",
    ),
    (
        "CLI guide reference doctor command JSON options",
        "docs/CLI-GUIDE.md",
    ),
    (
        "knowledge base hybrid retrieval embedding vectors citations chunk offsets",
        "docs/Local-Knowledge.md",
    ),
)


def test_documentation_corpus_meets_retrieval_quality_regression_thresholds(tmp_path: Path):
    """Pin practical lexical retrieval quality against six real product documents."""
    repository = Path(__file__).resolve().parents[1]
    documents_root = tmp_path / "operator-docs"
    documents_root.mkdir()
    for relative_path in _CORPUS:
        destination = documents_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative_path, destination)

    store = KnowledgeBaseStore(tmp_path)
    base = store.create_base(
        "AXIOM operator documentation",
        "Offline retrieval-quality regression corpus copied from the repository docs.",
    )
    for relative_path in _CORPUS:
        store.ingest_document(base["id"], relative_path)

    reciprocal_ranks = []
    recalls = []
    discounted_gains = []
    for query, relevant_path in _QUERIES:
        report = search_knowledge_base(
            store,
            base["id"],
            query,
            limit=20,
            max_documents=len(_CORPUS),
            chunk_size_chars=1200,
            chunk_overlap_chars=160,
        )
        # Count a document once at its highest-ranked chunk. A document with
        # many chunks must not occupy several positions in the document metric.
        ranked_documents = []
        for result in report["results"]:
            source_path = result["citation"]["source_path"]
            if source_path not in ranked_documents:
                ranked_documents.append(source_path)

        try:
            rank = ranked_documents.index(relevant_path) + 1
        except ValueError:
            rank = None

        reciprocal_ranks.append(1.0 / rank if rank is not None and rank <= 5 else 0.0)
        recalls.append(1.0 if rank is not None and rank <= 5 else 0.0)
        discounted_gains.append(1.0 / math.log2(rank + 1) if rank is not None and rank <= 5 else 0.0)

    mean_recall_at_5 = sum(recalls) / len(recalls)
    mean_reciprocal_rank_at_5 = sum(reciprocal_ranks) / len(reciprocal_ranks)
    mean_ndcg_at_5 = sum(discounted_gains) / len(discounted_gains)

    assert mean_recall_at_5 >= 0.95, {
        "recall_at_5": mean_recall_at_5,
        "queries": len(_QUERIES),
        "corpus": _CORPUS,
    }
    assert mean_reciprocal_rank_at_5 >= 0.80, {
        "mrr_at_5": mean_reciprocal_rank_at_5,
        "queries": len(_QUERIES),
        "corpus": _CORPUS,
    }
    assert mean_ndcg_at_5 >= 0.80, {
        "ndcg_at_5": mean_ndcg_at_5,
        "queries": len(_QUERIES),
        "corpus": _CORPUS,
    }
