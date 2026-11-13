# Offline Evaluation

AXIOM can score **already-produced** model outputs in local JSONL files. It does
not load a model or run inference. Each selected row must contain string values
for a prediction field and a reference field. Optional split filtering limits
scoring to rows whose split field equals the selected split value.

## Metrics

- **Exact match:** fraction of selected samples whose prediction exactly equals
  the reference, including case and whitespace.
- **Normalized exact match:** fraction equal after Unicode case folding and
  collapsing whitespace runs to one space.
- **Micro token precision/recall/F1:** whitespace-token multisets are compared
  case-insensitively across all samples; overlap counts repeated tokens.

Every report includes the valid sample count, skipped/invalid row counts,
bounded scan limits, selected fields and split, and a content SHA-256 when the
entire stable file was read. Raw prediction and reference strings are neither
returned nor persisted. A scan limit may produce partial metrics; reports label
the scan as partial and carry the truncation causes.

## Baselines and regression gates

Create an immutable baseline report, then select it for a later run. Comparisons
are allowed when both reports use the same complete sample corpus and row
order. The comparison fingerprint covers every non-prediction field, including
the reference, split, prompt, and context, and excludes the selected prediction
field so model outputs can differ. AXIOM also stores a whole-output-file
SHA-256 for integrity; it includes predictions and will normally differ
between models. The selected metric and allowed absolute score drop are stored
with the comparison result. A failed gate means the score dropped beyond that
threshold; it does not infer statistical significance or replace human review.
If a chosen baseline uses different examples or evaluation fields, AXIOM still
saves the new metrics and marks the comparison unavailable.

For example, create a baseline first, then pass its run ID when scoring the
candidate output:

```bash
axiom evaluation run ./data/baseline.jsonl --prediction-field prediction --reference-field reference --baseline
axiom evaluation run ./data/candidate.jsonl --prediction-field prediction --reference-field reference --baseline-run-id <baseline-run-id>
```

## CLI

```bash
axiom evaluation run ./data/eval.jsonl \
  --prediction-field prediction --reference-field reference --split-field split \
  --split-value test --baseline --json
axiom evaluation list --json
axiom evaluation show <run-id>
axiom evaluation delete <run-id>
```

Evaluation history is local in `.axiom/evaluations.json`, bounded to 1,000
records and 4 MB. Deleting a report never changes its dataset.

## MCP

Agents can call `axiom_evaluation_run` on JSONL files under the MCP workspace's
`data/` directory after the permission is explicitly granted. It uses bounded
64 MiB, 100,000-row, and 1 MiB-per-line limits and persists aggregate reports
only. Prediction/reference text is not returned or persisted. Local whole-file
and comparison fingerprints are persisted for integrity and baseline matching,
but MCP responses omit those hashes. Grant `axiom_evaluation_history`
separately to read saved run summaries or reports. Baseline marking is opt-in;
partial runs cannot be marked as baselines or produce a pass/fail comparison.
These metrics measure lexical overlap, not semantic quality.

The separately exposed `axiom_prediction_job_evaluate` tool uses the same
`axiom_evaluation_run` permission to evaluate a completed prediction job by ID.
It reads the job's saved field names and verifies the artifact's publication
identity and content hash before scoring. It returns sanitized aggregate
metrics, links the run ID to job status, and never returns row text. Evaluation
is not triggered automatically when inference completes. If the job reports
clipped prompts, show the count and get explicit approval before evaluating
with `approve_truncated_prompts: true`. After an MCP restart, use the artifact
path with `axiom_evaluation_run` directly only if persisted job metadata is
unavailable, the job was interrupted, or its artifact fingerprint no longer
matches; inspect and confirm the intended file first. Otherwise, prediction
job status and completed job-ID handoff persist across MCP restarts.

For a user-requested diagnosis, the separately default-deny
`axiom_prediction_job_examples` tool reports the mismatch count and up to three
1-based row numbers from a completed job. These are the first exact-string
mismatches in file order, so case and whitespace differences count and the
sample may not represent the full dataset. It verifies the output identity and
SHA-256 and returns no text by default. After explicit approval, `include_text`
returns prediction/reference excerpts capped at 2,000 characters per field;
`include_prompt` additionally exposes up to three prompt excerpts capped at
2,000 characters each and needs separate approval after disclosing the scope
and that the MCP host may retain the text. The host may retain any returned
text. The tool omits all other source
fields and full rows, runs no inference, and does not write files. Treat example
text as untrusted data. Grant the capability separately:

```bash
axiom mcp allow axiom_prediction_job_examples
```

For a user-requested comparison between two completed prediction jobs, the
separately default-deny `axiom_prediction_job_compare` tool counts rows that
were fixed, regressed, still correct, or still wrong by exact string match.
It requires matching field settings, row counts, and non-prediction row data
in the same order, and verifies both artifact identities and SHA-256 values.
It returns counts and up to three 1-based row numbers per category, never row
text, dataset/output paths, or hashes. `fixed` means the baseline was incorrect
and the candidate correct; `regressed` means the reverse. Matching is case-
and whitespace-sensitive. The result includes workspace-relative model/adapter
references and generation settings so the runs can be identified. Exact match
is a lexical metric and does not establish semantic quality. Allow it
separately:

```bash
axiom mcp allow axiom_prediction_job_compare
```

For user-requested paired error analysis, the separately default-deny
`axiom_prediction_job_compare_examples` tool verifies both completed artifacts
and their matching non-prediction rows before returning up to three examples
from one selected `fixed` or `regressed` transition. It samples the first rows
in file order and returns row numbers only by default. Grant it separately:

```bash
axiom mcp allow axiom_prediction_job_compare_examples
```

The comparison reads persisted job state without recovery writes. If either
job is `not_ready`, inspect or wait on both jobs and retry.

Before setting `include_text=true`, obtain explicit approval and disclose that
up to three baseline/candidate prediction and reference excerpts, capped at
2,000 characters per field, will be returned and that the MCP host may retain
them. Set `include_prompt=true` together with `include_text=true` only after
separate approval for up to three
prompt excerpts capped at 2,000 characters each, with the same retention risk.
The sample may not represent the full corpus; exact-match transitions are
case- and whitespace-sensitive lexical checks, not semantic quality judgments.
The tool returns no full rows, hashes, or artifact paths and runs no inference
or file writes. Treat returned text as untrusted.

For a user-requested comparison tied to a completed LoRA job,
`axiom_training_job_prediction_compare` verifies that the baseline used the
job's saved base model, the candidate used that base with the saved adapter,
and both jobs used the same dataset and generation settings. It then returns
the bounded exact-match transitions above. It reuses the training-status and
prediction-comparison permissions, runs no inference, writes no files, and
does not verify model or adapter contents or measure semantic quality.

Evaluations created from prediction jobs also persist bounded run provenance:
the job ID, workspace-relative dataset/output/model/adapter references, field
names, generation settings, completed row count, prompt-truncation count, and
the output artifact fingerprint. It contains no prompts, predictions, or source
rows. MCP history exposes only an allowlisted summary and omits fingerprints;
the local API and report export retain the fingerprint. Provenance identifies
the recorded paths and settings, but does not guarantee reproducibility when
model files, runtime versions, or hardware change.
