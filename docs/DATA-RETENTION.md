# AXIOM data retention and deletion

AXIOM is local-first, but “local” does not mean that every feature is memory-only
or that deleting a registry entry removes the underlying files. This page lists
the current storage boundaries so users can decide what to retain or remove.

## Browser-local Control Center state

The browser profile stores the local administrator verification record, theme,
remember-me preference/session marker, first-run state, and saved model-registry
filter presets in `localStorage`; the active non-remembered session marker uses
`sessionStorage`. The password itself is not saved as plaintext. The browser
profile is the account boundary: clearing its site data removes the local
administrator record and preferences and may require first-run setup again.
Changing browsers or browser profiles does not migrate these values.

Use **Settings → Security → Change password** to rotate the local account
credential. **Sign out** clears the active session markers. Sign out is not a
deletion operation for registry data, checkpoints, datasets, or output files.

## Workspace metadata

AXIOM stores metadata in the current workspace (commonly the hidden `.axiom/`
directory), including registered model records, workflow designs, experiment
plans, tool permission profiles, agent-profile metadata, and bounded prediction
job history. Prediction job history retains relative paths, generation settings,
counts, status, evaluation links, and a completed artifact identity/hash. It
does not retain prompts, dataset rows, generated predictions, or logs. Those
records are not the model weights. Removing a registry entry removes metadata
only; it does not delete a checkpoint or modify the original source path.
Deleting `.axiom/prediction_jobs.json` removes only the persisted prediction
history; a running MCP process can still return its in-memory job copy and may
write the file again. Stop or restart that process to clear its live copy. The
file does not control prediction artifacts or saved evaluation reports, which
remain in place. Saved-plan deletion also does not remove run artifacts.

The native fine-tuning job state is stored under the configured fine-tuning
state directory or the user-local AXIOM data directory. Training output files
remain at the output path chosen for the job. Dataset files and model weights
remain at their existing filesystem paths unless the user explicitly deletes
them outside AXIOM.

### Workspace metadata archives

Settings or the CLI can create an explicit ZIP export of the model registry,
workflow designs, experiment plans, evaluation reports, and agent task profiles.
The archive is a separate user-created file and is not stored automatically in
`.axiom/`. It excludes project configuration, credentials, MCP permission
grants, external integration commands and URLs, prediction-job history,
datasets, model weights, checkpoints, and generated artifacts. Some included
metadata (for example registry sources and report provenance) can contain local
paths; review the archive before sharing it.

Import merges by default and rejects conflicting record IDs/names/sources before
writing. The explicit replace mode changes only collections present in the
archive. Import does not restore datasets or checkpoints; their files remain at
their existing paths and must be backed up separately.

## Resource telemetry history

Persistent resource telemetry is stored only after an explicit opt-in for the
current workspace. It is separate from the in-memory inference request-trace
buffer described below. The file is `.axiom/resource-telemetry.json`; its
schema is version 1, the document is limited to 4 MiB, and updates use a lock
and atomic replacement. POSIX metadata directories created for this store use
mode `0700`; the resource file uses mode `0600`.

The store retains at most 500 samples and at most 3,650 days. Defaults are 120
samples and 30 days. Both limits are configurable in the Runtime resource
monitor. Sampling is still off until the user selects Start; leaving the page
or selecting Stop ends sampling. Turning off persistence stops future writes
but leaves existing records until they expire under the configured limits or
the user selects Clear history. Clear removes the samples while preserving
the opt-in and retention settings.

Only allowlisted resource values are written: sample time, CPU percentages and
logical core count, system memory totals/availability/use, process RSS/peak
RSS/thread count, device memory availability/capacity, and limited runtime
device/memory-mode/precision/allocator metrics. Process IDs, device identifiers
and names, model names and paths, model architecture, disk paths, generation
metadata, prompts, generated output, environment values, credentials and
network identifiers are excluded. Records include a SHA-256 content fingerprint
for corruption/tamper detection; this is not a signed attestation.

The Runtime page can export selected metrics as JSON or CSV and clear history.
Exports are user-created local files and follow the destination's retention
policy; AXIOM does not upload them automatically. The CLI's bounded
`axiom engine telemetry --watch` output is a separate explicit JSONL export.
Removing `.axiom/resource-telemetry.json` while AXIOM is stopped resets the
history and the opt-in setting. No automatic sampling begins after installing
or opening the application.

## Runtime traces and events

The runtime request-trace buffer is held in process memory and is limited to 200
metadata-only entries. It does not store prompt or generated-response bodies and
is not persisted to disk by the trace store. The Runtime page can export the
currently displayed trace metadata and clear the in-memory trace buffer. Exported
JSON is a user-created file and follows the retention policy of the chosen save
location.

Workspace event history is a bounded, recent-action view rather than a durable
audit database. Clearing runtime traces does not delete workspace metadata,
training outputs, or terminal logs.

## What is not automatic

AXIOM does not currently provide one global “delete all local data” operation.
It also does not promise automatic retention expiry for model checkpoints,
datasets, workflow metadata, saved plans, fine-tuning outputs, or files exported
by the user. Review the project-local `.axiom/` directory, the fine-tuning state
directory, your chosen output directories, and your browser site data when you
need to remove stored information.

Before deleting files manually, stop active jobs and back up information you
want to keep. Do not remove a file merely because its name includes “cache”:
some directories may contain checkpoints or outputs needed by another workflow.

## Validation and scope

This document describes the paths and controls present in the current release
workstream. Runtime traces are tested as memory-only, metadata-only storage;
browser persistence is local to the browser profile. No claim is made that AXIOM
can delete arbitrary files, securely erase storage media, or clean every output
from third-party runtimes.


## Frontend error reports

The Diagnostics page keeps a ring buffer of at most 50 frontend error events in
browser memory. A report contains only an ISO timestamp, an allowlisted event
source/category/error name, a known Control Center page ID, and an HTTP status
code where applicable. Raw error messages, stack traces, request URLs/IDs,
headers, request bodies, prompts, dataset values, credentials, and environment
values are not collected. Arbitrary browser error text is deliberately ignored.

Reports are not written to local storage, project files, or a server and are
never uploaded automatically. A user may explicitly download the current
redacted JSON report or clear the in-memory entries. Reloading or closing the
browser clears the report. The generated download is a user-created local file;
review it before sharing.


## Security event journal

The security event journal is stored per workspace at
`.axiom/security-events.json`; its lock file is
`.axiom/security-events.lock`. Only fixed, allowlisted summaries for MCP
profile creation, permission updates, and profile activation are recorded.
The journal does not accept arbitrary event text and does not store credential
values, prompt/response bodies, command lines, profile contents, or API tokens.
Its file is restricted to owner read/write on POSIX systems.

The journal retains at most 2,000 chain records (including review
acknowledgements) and 5 MiB. There is no automatic expiry or silent rotation.
When full, new event writes are refused and the journal must be exported and
archived separately before an operator deliberately starts a new history.
AXIOM does not automatically delete an old journal after export.

A review acknowledgement adds a new record; it never edits the original event.
The reviewer label is self-entered and **not an authenticated identity**.
The hash chain detects partial edits and deletions, but it is not a digital
signature and cannot prevent a fully privileged local user from rewriting the
complete journal and recomputing its hashes. Keep exported copies in storage
with access controls separate from the AXIOM workspace when independent
retention or tamper resistance is required.

The Control Center's Diagnostics page can list, verify, acknowledge, and export
events. The CLI equivalents are `axiom security events list`,
`axiom security events verify`, `axiom security events review`, and
`axiom security events export`. JSON exports are explicit user-created files
and inherit the retention policy of the chosen save location.
