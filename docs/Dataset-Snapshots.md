# Dataset snapshots

AXIOM dataset snapshots are workspace-local, immutable-by-application content
captures for JSONL files below the workspace `data/` directory. The original
dataset is never modified. Snapshot objects are stored below
`.axiom/dataset-snapshots/objects/` using their full SHA-256 digest as the
object name. Identical content is stored once even when multiple snapshot
records refer to it.

Each snapshot has a versioned manifest containing its 32-character snapshot
ID, creation time, workspace-relative source name, JSONL format, byte length,
SHA-256 digest and content-addressed object name. No JSONL row values are
copied into the manifest. A snapshot can reference only an earlier snapshot
in the same workspace; the child records the parent's `parent_snapshot_id`
and parent manifest hash.

The index is append-only through AXIOM's application interface and is
hash-chained. Each record hashes its manifest and previous record, while the
index stores a head hash. Listing validates the complete retained metadata
chain. The explicit Verify action additionally re-hashes the stored object
and compares both byte length and SHA-256. POSIX object files are made
read-only (mode `0400`) and metadata is owner-only; storage directories are
created with restrictive permissions. AXIOM exposes no snapshot delete or
overwrite operation.

Storage is bounded to 250 snapshot records, 1 GiB per dataset and 10 GiB of
unique indexed object content per workspace. API and CLI requests have
validated workspace-relative paths, reject symlinks, and stream source bytes
through a bounded temporary file before publishing the content object. A
source file that changes while it is being copied is rejected rather than
committed as a potentially inconsistent snapshot.

These checks provide local content-integrity and lineage evidence, not a
signed attestation or protection against an attacker who can rewrite both the
workspace index and every stored object. Back up the complete
`.axiom/dataset-snapshots/` directory together; copying only the index or
only the objects directory is not a complete snapshot archive.

CLI examples:

```bash
axiom dataset snapshot ./data/train.jsonl --json
axiom dataset snapshots --json
axiom dataset snapshot-verify <snapshot-id> --json
```
