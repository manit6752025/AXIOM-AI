# AXIOM Security Advisory — Runtime Bundle Path Escapes

**Advisory date:** 2026-10-08  
**Patched date:** 2026-10-08  
**Affected area:** `axiom.optimizer.model.create_runtime_bundle`
**Severity:** Medium  
**Status:** Patched

The review covered CodeQL alerts #10, #11, #12, #14, #19, #20, #22,
#23, #24, #25, and #26-#33. The confirmed exploitable issues were the
source and destination symlink cases described below; the other findings were
covered by the existing trusted-root validation or the same boundary review.

## Summary

CodeQL reported uncontrolled path use in the runtime-bundle builder. The
builder accepted a trusted model directory and copied selected files into a
trusted output directory, but nested source and destination paths were not
protected against symlink redirection.

## Impact

An attacker who could place a model directory or modify the output directory
could potentially:

1. Place a source symlink such as `config.json -> /etc/passwd` and cause a file
   outside the model root to be copied into the runtime bundle.
2. Place a destination-directory symlink and redirect copied files outside the
   trusted output root.

This affects local workflows that process untrusted model directories. It does
not provide remote access by itself; an attacker must already be able to
write files in a trusted model or output location.

## Discovery

The issues were identified by CodeQL path-expression alerts and confirmed by a
focused security review on 2026-10-08. The vulnerable behavior was in the
runtime bundling implementation; ordinary `..` traversal was already blocked
by the trusted-root helpers.

## Patch

The patched implementation:

- rejects source symlinks before inspection or copying;
- resolves every source file and requires it to remain below the resolved model
  root;
- resolves every destination before creating parent directories or copying;
- rejects destination symlinks and rejects resolved destinations outside the
  output root;
- writes the runtime profile only below the trusted output root.

These checks are intentionally performed for every discovered file rather than
relying only on validation of the top-level paths.

## Validation

Regression tests cover both source-file symlink disclosure and
destination-directory symlink redirection:

```bash
python -m pytest tests/test_optimizer_security.py tests/test_backend.py -q
```

The fix preserves the existing trusted model/output root policy and does not
follow symlinks as a convenience. A model containing symlinked runtime files
must be materialized into the trusted model root before bundling.

## Recommended action

Upgrade to the patched revision and re-run any runtime-bundle operation that
was performed from a model directory supplied by another user or archive.
Review previously generated bundles if the source or output directories were
writable by untrusted users.
