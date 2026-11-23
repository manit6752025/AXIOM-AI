# Local model provenance and security evidence

AXIOM can inspect a model directory without loading its weights, importing model
code, or contacting a remote service:

~~~bash
axiom model provenance ./models/qwen3-8b --json
axiom model provenance ./models/qwen3-8b --model-id 0123456789abcdef0123456789abcdef --json
~~~

The Control Center exposes the same operation from **Models → Discover local →
Inspect provenance**. The workspace API accepts a workspace-relative model
folder at `POST /api/workspace/models/provenance` and optionally a registered
model ID to attach declared metadata.

## Evidence collected

- Detected weight formats and bounded file inventory for safetensors, GGUF,
  and PyTorch binary formats.
- Streaming SHA-256 for each eligible weight file and a deterministic
  checkpoint digest derived from sorted relative paths, file sizes, and file
  digests.
- Names, sizes, and bounded SHA-256 digests of local `LICENSE`, `COPYING`,
  and `NOTICE` files.
- Symlinked files/directories skipped, plus explicit flags for pickle-capable
  PyTorch weight extensions.
- Optional registry source, URL, revision, attribution, license, and review
  fields, always labelled as user-supplied metadata rather than independently
  authenticated evidence.

## Safety and limits

The scan is read-only. It never loads model weights, invokes a framework,
executes `trust_remote_code`, downloads metadata, or calls a hosted service.
Symlinked entries are skipped. Traversal is bounded to 2,000 directories and
20,000 files. Per-file inventory and SHA-256 hashing are capped at 512 weight
files and 16 GiB of combined weight data; a larger checkpoint returns
`skipped_size_or_file_count_limit` rather than silently returning a partial
checkpoint digest. License evidence is capped at 64 candidate files, 16 MiB
aggregate candidate bytes, and 1 MiB per file. When the aggregate count or byte
limit is exceeded, AXIOM reports `skipped_count_or_total_size_limit` and returns
no partial license-file digest set; the report includes candidate counts/bytes
and the active limits. Symlink components in the selected model path are
rejected, not resolved through.

A SHA-256 digest identifies the exact bytes that were scanned. It does not
authenticate the publisher, prove that files came from a declared repository
revision, or detect malware. File extensions are format indicators, not proof
that a checkpoint is safe to load. In particular, `.bin`, `.pt`, and
`.pth` may contain pickle payloads and should not be loaded from untrusted
sources.

A local license file or registry license label is not a legal determination.
The user must review the actual license terms and any model-specific usage
restrictions before redistribution or commercial deployment. Registry review
states are human-entered records, not an AXIOM legal approval.

The report contains metadata and digests only. It does not include model
weights, README/license bodies, tokenizer content, or source text.
