# Full workspace backup and restore

AXIOM's full workspace backup is different from the privacy-preserving support
bundle and the metadata-only transfer archive. It captures regular project
files from the selected workspace, including configuration, the `.axiom`
metadata directory, datasets, local models, checkpoints, and generated
artifacts.

**Backups can contain secrets and private user data.** Project configuration,
credential files, tokens, private datasets, and model weights are not
redacted. The archive is not encrypted. Store it in a protected location,
restrict its access, and inspect its contents before sharing or uploading it
to another service.

## CLI

Create a new archive outside the workspace:

```bash
axiom workspace backup create \
  --workspace ./my-project \
  --output ./my-project-backup.zip \
  --json
```

Restore into a new workspace:

```bash
axiom workspace backup restore ./my-project-backup.zip \
  --name "Restored project" \
  --json
```

The output is create-only; existing archive files are never overwritten.
Restore validates the archive and publishes a new managed workspace. It does
not merge into or overwrite an existing workspace.

## Control Center

Open **Diagnostics → Full workspace backup and restore**. Download the full
backup for the current workspace, or choose a ZIP archive and a name for a new
restored workspace. The operation requires a confirmation because the archive
may contain sensitive data. The workspace switcher refreshes after a
successful restore; the current workspace remains active.

## API

- `GET /api/workspaces/{workspace_id}/backup` streams a ZIP64 archive for a
  registered workspace. The response is `application/zip`, has
  `Cache-Control: no-store`, and includes the backup manifest SHA-256 in
  `X-Axiom-Backup-Manifest-SHA256`.
- `POST /api/workspaces/backup/restore?name=...` accepts an
  `application/zip` body and restores into a new managed workspace.

The upload is streamed to a temporary file with a 105 GiB compressed-size
cap. The API does not accept an arbitrary destination path.

## Archive format and integrity

Format version 1 is a ZIP64 archive containing `manifest.json` and one entry
at `files/{relative-workspace-path}` for each regular file. The manifest lists
a canonical relative path, byte size, SHA-256, and sanitized file mode for
every entry. A checksum is calculated in a first bounded scan and recalculated
while the archive is written; a file that changes between either pass causes
the backup to fail.

Restore validates the manifest schema, declared limits, duplicate JSON keys,
archive inventory, canonical relative paths, case-insensitive path collisions,
ZIP entry types and compression methods, sizes, CRCs, and every SHA-256 before
registering the workspace. Absolute paths, parent-directory components,
Windows device names, symlinks, special files, duplicate entries, encrypted
entries, and unexpected ZIP contents are rejected. Extraction occurs in a new
private staging directory under the managed workspace root. The directory is
renamed into its final workspace ID only after the complete archive has been
extracted and verified.

SHA-256 detects accidental or deliberate changes relative to the manifest; it
does not authenticate the archive publisher. Anyone able to replace both the
archive and its manifest can recompute the hashes. Use a trusted transport or
an independently authenticated signature when publisher authenticity is
required.

## Resource limits and exclusions

| Limit | Value |
|---|---:|
| Regular files | 50,000 |
| Individual file | 50 GiB |
| Total uncompressed file data | 100 GiB |
| Compressed archive / upload | 105 GiB |
| Manifest | 32 MiB |
| Per-file I/O buffer | 1 MiB |

Virtual environments, dependency caches, symlinked files/directories, sockets,
devices, FIFOs, and other special filesystem entries are not backed up.
Symlinks are skipped, not dereferenced. Filenames that cannot be represented
safely across supported filesystems and case-insensitive path collisions are
rejected. File modes are restored with restrictive permissions; setuid,
setgid, sticky and group/other write permissions are never restored.

A backup is a workspace file snapshot, not a running-process snapshot. Active
jobs, external services, system packages, databases outside the workspace,
and external object stores are not transactionally captured. Stop applications
that are actively modifying important files before creating a backup.
