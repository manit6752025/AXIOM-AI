# Local LoRA fine-tuning

AXIOM includes an opt-in local training path using a purpose-built, bounded PyTorch loop and PEFT LoRA adapters. It creates an adapter; it does not update or overwrite the base model. AXIOM does not rely on the Transformers `Trainer` loop, which helps keep the training behaviour under AXIOM's control as the upstream Trainer API evolves.

## Requirements

Install the optional training dependencies in the same Python environment used to run AXIOM:

```bash
python3 -m pip install -e '.[training]'
```

Prepare a Transformers model directory on disk that already contains `config.json`, model weights, and tokenizer files. AXIOM sets `local_files_only=True` when loading the model/tokenizer and does not run remote custom model code. A GGUF file is not a Transformers model directory.

Create a UTF-8 JSONL file. Each non-empty line must be an object containing a string field such as `text`:

```jsonl
{"text":"User: What is 2 + 2?\nAssistant: 4."}
{"text":"User: Name a primary color.\nAssistant: Blue."}
```

Records using string `prompt` and `completion` fields, or a `messages` array containing `content` strings, are also accepted. The default text field is `text`.

## CLI

```bash
axiom --no-banner train --help
axiom --no-banner train run \
  --model-path "$HOME/models/my-transformers-model" \
  --dataset "$HOME/data/train.jsonl" \
  --output-dir "$HOME/runs/my-lora-adapter" \
  --max-steps 20 --sequence-length 512 --lora-rank 8 \
  --checkpoint-interval 5 --output-policy empty
axiom --no-banner train jobs --limit 20 --offset 0
axiom --no-banner train status JOB_ID
```

`train run` is an actual training operation and runs until completion or failure. Progress, loss when available, and the final adapter directory are recorded. The history is stored under `${AXIOM_FINETUNE_HOME:-~/.local/share/axiom/fine-tuning}/jobs`.

The `train jobs` command accepts `--limit` (1–100) and `--offset` (0–5,000), and its JSON output includes total, next-page, and bounded-scan metadata. The Control Center Training panel exposes the same history in pages of ten and lets you select a previous job to inspect its logs and output.

## Checkpoints, resume, and output policies

The `train run` command and Training panel expose two controls:

- `--checkpoint-interval N` (0–1,000): save a recoverable checkpoint every N completed optimizer steps. Zero disables checkpointing. When enabled, AXIOM also saves the final step even when it is not an exact interval boundary.
- `--output-policy new|empty|resume`: `new` requires a path that does not exist; `empty` (the default) accepts a new or existing empty directory; `resume` requires an existing AXIOM output directory containing only verified checkpoints and no final adapter files.

Checkpoints are stored below `.axiom-checkpoints/` in the selected output directory. AXIOM writes each checkpoint to a temporary directory, records a SHA-256 manifest for the adapter configuration, adapter weights, and optimizer state, then atomically publishes the completed checkpoint directory. The latest three complete checkpoints are retained. Interrupted temporary directories are never selected for resume. Before deserializing optimizer state, AXIOM verifies the manifest and uses PyTorch's restricted `weights_only` loader.

A checkpoint contains the trainable LoRA adapter, optimizer state, dataset sampling order/position, and Python/PyTorch random-number-generator states. Resuming restores those values and continues at the next optimizer step. The same local base-model path, dataset path and exact dataset bytes, text field, sequence length, learning rate, LoRA rank, output directory, training seed, and compute device must be used. The requested maximum step count may be increased, but it cannot be below the checkpoint's completed step. The checkpoint interval may be changed for later saves.

The base-model compatibility fingerprint includes relative file names, sizes, modification metadata, and a content hash of `config.json`; it does not re-read every potentially multi-gigabyte weight file. It detects ordinary local changes while keeping resume preflight bounded, but is not a cryptographic content identity for the entire model tree.

Example: resume an interrupted run into the same output directory, using the same model, dataset, and training settings:

```bash
axiom --no-banner train run \
  --model-path "$HOME/models/my-transformers-model" \
  --dataset "$HOME/data/train.jsonl" \
  --output-dir "$HOME/runs/my-lora-adapter" \
  --max-steps 40 --sequence-length 512 --lora-rank 8 \
  --learning-rate 0.0001 --checkpoint-interval 5 --output-policy resume
```

Only completed checkpoint steps are recoverable. If a process exits or is cancelled between checkpoint intervals, later uncheckpointed steps are repeated when the run resumes from the most recent verified checkpoint. A completed run writes final adapter files at the output-directory root; that completed output is not a resume target. Use a separate output directory for a new run.

## Job lifecycle and cancellation contract

Local LoRA job records expose the following explicit states:

`queued → running → finalizing → completed`

Cancellation follows a separate cooperative path:

`queued/running → cancelling → cancelled`

A worker can also reach `failed` from any active state. Terminal states
(`completed`, `cancelled`, and `failed`) cannot be reopened. The backend
validates each transition while holding the shared job-state lock; it rejects
transitions such as `finalizing → cancelling` and
`cancelled → running`.

Every job summary exposes a `cancellation` object with `supported`,
`requested`, `requestable`, `mode`, `boundary`, and
`unavailable_reason`. A cancellation request changes the state to
`cancelling`; it does not claim that the worker has stopped. The worker checks
for cancellation before beginning and at safe model-loading/device-placement
boundaries and between optimizer steps. If a request arrives while an optimizer
step is already in progress, that step is allowed to finish and its completed
step count/loss are persisted before the worker records `cancelled`. Once
`finalizing` begins, adapter persistence is not interrupted. Clients should
use `requestable` rather than infer cancellation capability from a hard-coded
list of status strings, and should wait for the terminal `cancelled` state
before treating the run as stopped.

The CLI supports `axiom train cancel JOB_ID [--json]`; the API endpoint is
`POST /api/fine-tuning/jobs/{job_id}/cancel`. Repeated cancellation requests
are idempotent while the job is already `cancelling`. Requests against
finalizing or terminal jobs return the unchanged state and an explanatory
`unavailable_reason`.

**Workflow execution boundary:** saved visual workflows and their dry-runs
remain design/planning-only and always report `execution_status:
not_executable`. The lifecycle above describes actual local fine-tuning jobs;
it does not imply that workflow graph nodes can be executed or cancelled. A
workflow runner still needs explicit per-node input/output contracts and
execution policies before graph execution can safely be introduced.

## Control Center and API

Open **Training** in the Control Center. The **Local LoRA fine-tuning** panel lets you supply local paths and bounded training settings, start a job, inspect progress, and request cancellation. Cancellation stops at a safe training boundary; the panel shows `cancelling` until the worker confirms `cancelled`. After training steps finish, the job enters `finalizing` while adapter files are saved, and cancellation is no longer accepted. The API endpoints are `GET /api/fine-tuning/runtime`, `GET /api/fine-tuning/jobs`, `GET /api/fine-tuning/jobs/{job_id}`, `POST /api/fine-tuning/jobs`, and `POST /api/fine-tuning/jobs/{job_id}/cancel`.

The MCP server also exposes `axiom_training_job_start`, `axiom_training_job_status`, `axiom_training_job_wait`, and `axiom_training_job_cancel`. Start is default-deny and needs its own explicit workspace permission; wait reuses the status permission and needs no separate grant. MCP start accepts workspace-relative inputs and an output under an existing `outputs/` folder. Its instructions tell agents to run preflight, present the exact configuration, and wait for the user's explicit request before starting; granting the start tool permission gives the agent standing capability. Wait checks persisted state for up to 30 seconds and returns `completed`, `cancelled`, `failed`, `timed_out`, `not_found`, or `state_unavailable`; call it again after `timed_out`, and inspect status after `state_unavailable`. If runner ownership cannot be verified, the worker may still be active; do not start a replacement job and check status again later. A saved record that cannot be read is reported as `state_unavailable`, not `not_found`.

MCP inputs are revalidated in the background worker, and the JSONL dataset is opened with descriptor-relative path checks. The adapter destination is checked again immediately before saving to refuse files added during model loading or training. Transformers still opens model weights by path, so a concurrent local process with write access to the workspace can race path checks while model files load or adapter files save. Avoid changing the model, dataset, or destination while a job is running.

Use `axiom_training_job_eval_overlap_preflight` to check a completed run's
recorded training dataset against an explicitly selected evaluation file. It
reuses the training-job status and overlap-preflight permissions and returns
safe base-model and adapter references for the next model preflight. Training
stores a fingerprint of the exact raw bytes consumed by its parser in local
job state; status responses omit the digest. POSIX job storage uses mode 0700
for its folder and 0600 for files. On Windows, storage inherits ACLs from
`AXIOM_FINETUNE_HOME`, which should be user-only when other local accounts are
present. The overlap scan checks that fingerprint against the current file
and reports `matches`, `changed_since_training`, or `unavailable` without returning the
digest. Changed files, older jobs without a fingerprint, and scans that cannot
verify a stable snapshot make the overlap assessment inconclusive; inspect the
reason code and retry after correcting the input. A matching snapshot and no
exact overlap still do not establish held-out data. The model and adapter
references are paths only, not content-verified artifacts; run
`axiom_engine_preflight` before inference.

For a full evaluation plan, `axiom_training_job_validation_preflight` combines
the job-linked overlap check, base/adapter compatibility checks, and prompt
truncation estimates. It accepts two new output paths directly under `data/`
and returns both prediction configurations without starting jobs, loading
weights, or writing files. It requires the existing status, overlap,
engine-preflight, and prediction-preflight permissions. Review its results and
show both configurations before running either prediction. By default, overlap
checks prompts but not evaluation references; set
`include_reference_in_overlap=true` to include references and confirm the
returned `overlap_configuration.reference_checked` value. `plan_only` means a
plan was assembled, not that a run is ready or approved. Use
`actionability.status`: resolve and rerun `blocked` findings; disclose
`review_required` findings and wait for the user's decision;
`clear_for_user_approval` means show both configurations and still get explicit
approval before starting. Clear means no known preflight blockers or review
findings, not guaranteed loading, fit, quality, or held-out data. Never start
a variant with a blocked engine check; resolve or recheck a blocked prompt
preflight before either run. Explain review-needed or inconclusive checks,
detected overlap, and expected or inconclusive truncation, then wait for an
explicit informed decision to proceed. Run the predictions sequentially because
AXIOM allows one local compute job at a time.

After both requested prediction jobs complete, use
`axiom_training_job_prediction_compare` to verify that the baseline used the
training job's saved base model and the candidate used its saved adapter with
the same dataset and generation settings before comparing exact-match
transitions. It uses the existing training-status and prediction-comparison
permissions. This is a lexical diagnostic; it does not verify model/adapter
contents or semantic quality, run inference, or write files.

## Limits and honest scope

- Base model path must be a local Transformers model directory; no model is downloaded by AXIOM.
- The training extra requires compatible versions of PyTorch (>=2.2), Transformers (>=5), PEFT (>=0.20), and Accelerate (>=1.1). The runtime endpoint checks installed versions and prevents starting a job with an older dependency set.
- Dataset is limited to 50 MiB, 10,000 rows, and 1 MiB per line; individual text values are limited to 300,000 characters.
- Steps: 1–1,000; sequence length: 32–2,048 tokens; LoRA rank: 1–64; learning rate: 0.000001–0.01.
- A single local fine-tuning job is allowed at a time to avoid accidental competing CPU/GPU workloads.
- The MCP trainer rejects symbolic links and junctions inside model directories and limits its directory metadata scan to 50,000 entries.
- Output policy defaults to a new or empty directory; resume is allowed only for an interrupted checkpoint-only directory with matching inputs. When enabled, the latest three complete checkpoints are retained below `.axiom-checkpoints/`.
- These limits do not guarantee that a model fits available memory. Start with a small model, short sequence length, and a few steps. GPU capability and architecture-specific LoRA compatibility are checked by the installed runtime, not inferred from the planning estimate.
- This is supervised causal-LM LoRA fine-tuning, not full-weight training, QLoRA, preference training, or a general workflow executor.
