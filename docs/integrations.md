# AXIOM Agent Integrations

AXIOM is designed to sit underneath external AI-agent runtimes as a local
inspection, planning, and bounded execution layer.

## Supported integration families

### MCP

Model Context Protocol is the preferred integration path.

It allows an agent runtime to discover AXIOM capabilities as tools.

Examples include OpenClaw and PicoClaw.

Agents can begin with the read-only `axiom_workspace_context` tool for one
structured snapshot of project layout, hardware, external developer-tool
availability, the tools allowed by the active profile, and a suggested next
command. It omits absolute paths, executable paths, and configuration
contents. Suggestions are never executed; agents should get user approval
before running a suggested shell command. Hardware detection may query
`nvidia-smi` locally when available; the tool does not modify workspace files
or access the network. Each MCP tool, including this one, remains denied until
it is granted in the active AXIOM tool profile.

The separately permissioned `axiom_workspace_assets` tool lists up to 200
workspace-relative model directories and JSONL dataset paths, with a 50,000
entry scan budget per category. The response distinguishes reaching the
result cap from a truncated or incomplete scan. It does not read file contents
or follow symlinked entries, and omits the absolute workspace path from its
response.
Both workspace tools use the MCP server process's working directory as the
workspace root. Start the server from the target project or set the client's
working-directory option where supported. Pass asset paths to
`axiom_model_info` or `axiom_dataset_inspect` for details. MCP permissions are
stored in `.axiom/tool_permissions.json` under that same root, so run
`axiom mcp allow` from the project where the server will run.

`axiom_dataset_inspect` profiles a workspace-relative JSONL path with a
64 MiB, 100,000-row, and 1 MiB-per-line limit. It returns field names, observed
types, null rates, duplicate summaries, and scan diagnostics without row
values or absolute paths. Optional required-field and expected-type checks
can be included in the same call; required fields mean present somewhere in
the profiled dataset, not present in every row. Validation is marked
inconclusive for truncated profiles, invalid rows, or when the field-tracking
cap is reached or field names are shortened to fit the report.
Pass `duplicate_key_fields` with an optional `duplicate_rate_threshold` to
summarize repeated IDs, or `text_fields` to request token-length histograms.
Sensitive-looking field names are always flagged heuristically. Scanning field
values for PII patterns is off by default and requires `scan_pii: true` for
that call.

The separately permissioned `axiom_training_preflight` tool checks a local
model directory, validates dataset text using the trainer's bounded rules,
checks training-package metadata, and includes a heuristic hardware estimate.
It reads dataset text locally but never returns its values, loads model
weights, creates a job, writes files, or accesses the network. Package checks
read presence and version metadata only; they do not import or initialize
training dependencies or verify a device backend. Its verdict is
`blockers_found`, `review_needed`, or `no_known_blockers`; the final status is
not a guarantee that training will load or fit. AXIOM's current local trainer
supports LoRA, not QLoRA, and the generic planner may suggest a QLoRA fallback.
Hardware status `warn` means only the generic estimate predicts a fit; actual
loading was not tested. The estimate also cannot verify Apple MPS or CPU fit.

The separately permissioned `axiom_training_job_start`,
`axiom_training_job_status`, and `axiom_training_job_cancel` tools expose the
existing local LoRA trainer to agents. Start accepts workspace-relative model
and JSONL paths plus an adapter destination below an existing `outputs/`
directory. It creates a local job record and launches training, which consumes
compute and later writes adapter files; it does not download models or execute
remote code. It retains the trainer's dataset/step limits and single-active-job
lock. Model directories with symbolic links or junctions are rejected. MCP job
responses include workspace-relative paths and selected progress/result fields;
local logs, absolute paths, and detailed failure text are omitted. Cancellation
is cooperative at training boundaries and remains available until adapter
finalization begins. The server instructions ask agents to run preflight, show
the exact configuration, and wait for an explicit user request before starting;
granting start permission itself gives the agent standing capability.

```bash
axiom mcp allow axiom_training_job_start
axiom mcp allow axiom_training_job_status
axiom mcp allow axiom_training_job_cancel
```

The separately permissioned `axiom_engine_preflight` checks a workspace-relative
trusted model and optional LoRA adapter before inference. It accepts a prompt
token target up to 8,192, an output limit up to 256 tokens, an optional device
and precision, and optional caller-supplied RAM/VRAM budgets. It reads model
metadata and, for a safetensors base with an adapter, checks adapter tensor
shapes and base tensor names. For GGUF plus LoRA, compatibility remains
`not_checked`; AXIOM's native loader checks adapter shapes when inference is
explicitly run. The tool reports runtime readiness and estimates memory without
loading model weights. No inference runs, files are written, network is
accessed, or absolute paths returned. Pass the same device and precision to
`axiom_engine_generate`. Use RAM budgets for CPU/MPS and VRAM budgets for
CUDA/ROCm/XPU. Without a compatible adapter check, a budget relevant to the
selected device, or an explicit precision, its assessment remains
`review_needed`. A `preflight_clear` assessment is conditional on the supplied
budget and never guarantees that the model will load or generate successfully.

```bash
axiom mcp allow axiom_engine_preflight
```

The separately permissioned `axiom_engine_generate` tool runs one bounded
request through AXIOM's native local model engine. It accepts workspace-relative
trusted model and optional adapter paths, limits prompts to 8,000 characters
and completions to 256 tokens, accepts optional device and precision choices,
and returns the generated text with basic
token/timing metadata. AXIOM does not persist the prompt or response, access
the network, or write output files; an MCP host may retain the tool call in its
own transcript. The result reports whether the native engine truncated the
prompt to fit context. The tool runs synchronously without a wall-clock timeout
or MCP cancellation, so model loading and generation can take time. Because it
uses local compute and returns prompt-derived content, the server instructions
require an explicit user request before invocation.

```bash
axiom mcp allow axiom_engine_generate
```

The separately permissioned `axiom_prediction_job_start`,
`axiom_prediction_job_status`, and `axiom_prediction_job_cancel` tools bridge
local inference to the offline evaluator. Start accepts a JSONL input below
`data/`, workspace-relative model and optional adapter paths, prompt/reference/
prediction field names, and a new `data/NAME.jsonl` destination. It validates
all input rows before launch, then caps work at 100 rows, 48 MiB total input,
768 KiB per input line, 8,000 prompt characters per row, and 64 generated
tokens per row. Generation accepts a temperature and optional seed. Output is
capped at 64 MiB total, 1 MiB per line, and 16,000 prediction characters per
row. Every source field is preserved and the selected prediction field is
added or replaced. Output is written to a hidden temporary file and atomically
published only after all rows complete; failure or cancellation discards
partial output, and an existing destination is never overwritten. The
completed artifact can be passed to `axiom_evaluation_run`.

Start consumes local compute and writes generated text into that local
artifact. When permitted, run `axiom_engine_preflight` first. Server
instructions require showing the exact configuration and output file, then
waiting for the user's explicit go-ahead before starting. Job status omits
prompts, predictions, logs, and absolute paths. Job state is in memory, so an MCP process restart
clears status; AXIOM removes orphaned partial artifacts on the next prediction
job start. Prediction jobs share AXIOM's single local compute slot with
fine-tuning. The `prompts_truncated` count signals that the native engine
clipped one or more prompts to fit model context; review a nonzero count before
using the artifact for evaluation. Safe `error_code` values identify failure
stages without exposing local exception text. Descriptor-relative platforms
pin and recheck the output parent before and after publication. On platforms
without those filesystem operations, path checks still reject static traversal
and links, but cannot prevent another local process from moving the parent
directory during publication. Each operation has its own default-deny
permission:

```bash
axiom mcp allow axiom_prediction_job_start
axiom mcp allow axiom_prediction_job_status
axiom mcp allow axiom_prediction_job_cancel
```

`axiom_prediction_job_evaluate` accepts a completed job ID, reads the saved
prediction/reference fields from that job, and verifies the artifact's
published file identity and SHA-256 before scoring it. It returns sanitized
aggregate metrics and links the evaluation run ID back to the job; prompts and
predictions are not returned. It never runs automatically when inference
finishes and uses the existing `axiom_evaluation_run` permission. Get approval
before saving aggregate metrics. If `prompts_truncated` is nonzero, show the
count and obtain explicit approval before setting
`approve_truncated_prompts: true`. If an MCP restart cleared the job ID, use
`axiom_evaluation_run` directly on the still-existing output path.
Saved evaluations from prediction jobs retain the job ID, workspace-relative
input/output/model references, generation settings, row and truncation counts,
and a local output fingerprint. The MCP history redacts that fingerprint and
returns only bounded provenance metadata; no prompt, prediction, or source row
is included. The metadata helps attribute scores across server restarts, but
does not guarantee exact reproduction if model files, runtime versions, or
hardware change.

The separately permissioned `axiom_dataset_clean_preview` reports bounded
keep/remove counts and a proposed output path for a workspace-relative JSONL
file. It returns no row values and writes no file. A partial scan includes its
truncation reason and its counts cover only the scanned portion; overlong rows are reported as unclassified because the
cleaner may still keep them. CR-only line endings and invalid UTF-8 are flagged
because the preview cannot promise matching cleaner results for those inputs.
The preview is advisory because the source can change between calls; re-preview
immediately before the separately permissioned `axiom_dataset_clean` write
action. That action only accepts workspace-relative paths, writes a new
`.jsonl` file, requires an existing parent directory, and refuses to overwrite
an existing output. It repeats the preview and enforces the same byte, row, and
line limits while writing. Review the counts and proposed destination with the
user. Both paths use descriptor-relative access where the OS supports it and
checked workspace-relative paths elsewhere, including Windows. On systems
without descriptor-relative filesystem operations, concurrent changes to a
parent directory can race the path checks; the fallback is scoped against
static traversal and refuses existing outputs, but does not pin directory
handles against another process changing the workspace at the same time.

The separately permissioned `axiom_evaluation_run` tool scores precomputed
prediction JSONL under the workspace's `data/` directory and saves a local
aggregate report. It does not load a model, run inference, or access the
network. It returns counts, exact-match, normalized exact-match, token F1,
confidence intervals, and optional baseline comparison without prediction or
reference text. Its limits are 64 MiB, 100,000 rows, and 1 MiB per line. Partial
runs are labeled partial, cannot become baselines, and receive no pass/fail
comparison. The whole-output hash includes predictions; a separate complete
corpus fingerprint covers all other row fields, including references and
prompt/context, and enables comparisons across different output files. Both
hashes are persisted locally but omitted from MCP responses. These lexical
metrics do not establish semantic quality, and a `model_label` is user-supplied
metadata. Grant `axiom_evaluation_run` to create runs and
`axiom_evaluation_history` separately to read stored aggregates.
When a selected baseline has different examples or evaluation fields, the new
run is still saved and its comparison is reported as unavailable.

### CLI

AXIOM can also integrate with local agent binaries.

### HTTP

The current HTTP surface is the local daemon. It binds to loopback by default
and exposes read-only health, project-summary, hardware, action, and tool
information. It is not a hosted API or a remote agent service.

### A2A

Future versions can expose agent-to-agent capabilities for systems that support A2A-style communication.

## Philosophy

AXIOM should make local AI work legible and give agents bounded access to
supported local operations.

External agents decide:

- what the user wants
- what tools to call
- how to communicate

AXIOM currently handles:

- model metadata and local model inspection;
- JSONL dataset inspection, validation, statistics, and cleaning;
- hardware detection and conservative training-plan estimates;
- opt-in local LoRA training jobs with status and cooperative cancellation;
- bounded offline evaluation and supported local inference;
- local project state and developer-tool discovery.

External runtimes still decide what the user wants, which tools to call, and
how to communicate. Work outside AXIOM's supported local operations—such as
hosted model training or deployment—remains with those runtimes. A training
plan does not mean a job ran, offline lexical metrics do not establish semantic
quality, and discovered developer tools are not evidence of authentication.

## Optional Headroom proxy

[Headroom](https://github.com/headroomlabs-ai/headroom) is an external local
context-compression proxy. When it is installed separately, its documented
entry points are:

```bash
headroom doctor
headroom proxy
headroom dashboard
```

AXIOM's role is limited to discovering or guiding a developer toward Headroom;
it does not bundle, install, start, or supervise it. Starting the proxy or
dashboard is always the user's explicit action. AXIOM makes no savings claim for Headroom;
any effect must be measured in the user's own workload and environment.

This separation keeps the architecture modular and prevents AXIOM from becoming tied to a single agent runtime.
