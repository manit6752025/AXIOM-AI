# Production-scale workspace backup validation

The default CI suite verifies the actual 50,000-entry file-count limit, the
50 GiB per-file rejection boundary, aggregate-byte rejection, and a 1 GiB
sparse-file archive/restore round trip. The complete 100 GiB configured
uncompressed-data limit is exercised by a separate manually dispatched job.

## Large-storage runner

Provision an isolated Linux self-hosted runner and add the label
`axiom-large-storage`. The workflow uses this runner only when explicitly
dispatched and defaults to the label set
`["self-hosted","linux","x64","axiom-large-storage"]`. Ensure the runner has
sufficient free space, reliable filesystem sparse-file support, and a
four-hour maintenance window. Keep it isolated from untrusted pull requests
and deployment credentials.

## What the full-size test does

The test creates two sparse regular files of exactly 50 GiB each, runs the
production ZIP64 backup path (including the initial and write-time SHA-256
passes), writes the archive to disk, restores into a new empty staging
directory, and checks the 100 GiB aggregate, per-file sizes, integrity result,
and sparse allocation where the filesystem reports allocated blocks. It
consumes the full logical input stream during hashing/compression and restore;
the sparse source merely avoids allocating 100 GiB of zero-filled input on
disk.

Run **Actions → AXIOM production-scale backup stress → Run workflow**. This
workflow must pass on real large-storage infrastructure before R138 is marked
fully complete. It is not executed by normal pushes.
