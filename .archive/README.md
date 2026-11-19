# Archived repository artifacts

This hidden archive preserves non-runtime development material so it does not clutter the active repository tree.

## `legacy-feature-pack/`

An earlier roadmap feature-pack manifest, scope/QA notes, status snapshot, and source overlay. It is retained only for historical comparison. It is not an installable patch, does not describe current test status, and must not be applied over the current checkout. The active implementation lives under `axiom/`; active product documentation lives under `docs/`; the current scoped roadmap is `ROADMAP.md`.

The old manifest describes the original feature-pack bundle and may reference source paths or hashes from that bundle rather than this archive layout.

## `patch-backups/`

Snapshots created while editing files. They are not loaded by the application or release workflow. The authoritative current files are outside this directory; use Git history to inspect previous revisions. These copies are retained only for historical comparison.

## Cleanup

The accidental zero-byte root files `axiom-ui@0.1.1` and `tsc` were removed. `.gitignore` now ignores `.axiom-patch-backups/` so future local patch snapshots are not accidentally added to the repository.
