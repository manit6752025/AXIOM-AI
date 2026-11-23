# Workspace knowledge bases and safe ingestion

AXIOM provides workspace-scoped knowledge bases that hold private copies of
local documents for later retrieval workflows. A knowledge base is stored
under the active workspace's .axiom/knowledge directory. Its index contains
metadata only; ingested text is stored separately in generated, random-ID files.
Different workspaces have separate indexes and document stores.

## Create and ingest

~~~bash
axiom knowledge base create "Engineering notes" --description "Local design documents"
axiom knowledge base list
axiom knowledge ingest BASE_ID documents/guide.md
axiom knowledge documents BASE_ID
axiom knowledge remove-document BASE_ID DOCUMENT_ID --yes
axiom knowledge base remove BASE_ID --yes
~~~

The Control Center's Runtime page has the same create/select/ingest/list/remove
workflow. The API uses /api/workspace/knowledge-bases and its document routes.
API responses and CLI list output contain metadata, not stored source text. Search results intentionally return bounded snippets together with SHA-256 hashes and character offsets for citations.

## Supported formats and bounds

The initial ingestion adapter accepts UTF-8 TXT, Markdown, RST, CSV, JSON, and
JSONL files. Each input is limited to 8 MiB, each knowledge base to 1,000
documents, each workspace to 64 knowledge bases, and the workspace knowledge
store to 1 GiB of ingested content. The index is limited to 4 MiB.

A path must be relative to the active workspace. Absolute paths, parent
traversal, empty path components, symlinks, hidden AXIOM-managed paths, and
unsupported file types are rejected. Empty text, NUL bytes, invalid UTF-8,
and unsupported control characters are rejected. Files are opened as regular
files without following a final symlink. Content is normalized to LF line
endings and recorded with a SHA-256 digest and character count.

## Privacy and deletion

Ingestion copies content into the active workspace's private AXIOM storage.
The source path is retained as workspace-relative metadata; the source file is
never modified. Content does not go to a network service, logs, runtime traces,
or API list responses. When a document or knowledge base is removed, AXIOM
deletes only its own generated copy. The original workspace files remain.

## Bounded lexical and hybrid retrieval

~~~bash
axiom knowledge search BASE_ID --query "local inference" --json
axiom knowledge search BASE_ID --query "local inference" --source-prefix documents/engineering
axiom knowledge search BASE_ID --query "local inference" --model-path ./models/my-encoder --vector-weight 0.35 --device cpu --json
~~~

Search uses deterministic BM25 lexical scoring, paragraph-aware character
chunking with overlap, and optional local dense-vector fusion. Vector inference
is performed only when a trusted local embedding model path is explicitly
provided; no hosted embedding provider or network lookup is used. You can
filter by format, workspace-relative source prefixes, and document IDs.

Requests are bounded to 64 documents, 128 chunks, 16 MiB of source bytes,
1,048,576 scanned characters, 8,192 query characters, and 20 results. Chunk
sizes are 256–4,096 characters with 0–512 character overlap. Search does not
change source documents or persist its results. Returned snippets are limited
to 320 characters and carry source path, format, content SHA-256, document ID,
chunk number, and start/end character offsets so callers can verify provenance.
BM25 and cosine-derived fusion scores are ranking signals, not calibrated
relevance probabilities. This is retrieval, not answer synthesis; AXIOM does
not claim that the returned snippets constitute a complete answer.


## Retrieval quality regression

The offline test `tests/test_knowledge_retrieval_benchmark.py` copies six actual
operator documents from the repository into a temporary workspace, ingests
them through `KnowledgeBaseStore`, and runs six fixed information-seeking
queries through the production BM25 search implementation. It reports
document-level Recall@5, mean reciprocal rank@5, and nDCG@5; duplicate chunks
from one document count only at that document's first rank. The regression
thresholds are Recall@5 ≥ 0.95, MRR@5 ≥ 0.80, and nDCG@5 ≥ 0.80.

This is a repeatable quality gate over AXIOM's current operator-document corpus,
not a universal relevance guarantee or external user-corpus benchmark. Before
changing the chunker or ranker, preserve the fixed query/relevance pairs and
review metric changes; extend the corpus with representative documents and
independently judged relevance labels as additional domains are introduced.
Scores remain ranking signals, not calibrated probabilities.
