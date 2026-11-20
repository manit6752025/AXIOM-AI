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
