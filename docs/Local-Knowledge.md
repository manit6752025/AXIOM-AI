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
API responses and CLI list output contain metadata, not stored source text.

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

This is a storage and ingestion layer. Chunking, embeddings, lexical/vector
retrieval, ranking, and citations are separate roadmap deliverables; a stored
document is not automatically searchable until those components are enabled.
