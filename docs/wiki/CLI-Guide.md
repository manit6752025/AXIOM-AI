# AXIOM CLI Guide

Use the CLI for repeatable commands and automation independent of the Control Center UI.

## First commands

    axiom version
    axiom --help
    axiom guide
    axiom summary --json
    axiom status --json
    axiom doctor --json

Prefer supported JSON output for scripts, check exit codes, and handle failures. Do not parse human-oriented prose if structured output is available.

## Projects and configuration

    axiom init my-ai
    axiom project validate
    axiom config validate
    axiom info

## Models

    axiom model list
    axiom model inspect ./models/my-model
    axiom model search llama --path models
    axiom model info org/model
    axiom model analyze org/model

Remote metadata and downloads require network access. Model inspection does not execute inference.

## Datasets

    axiom dataset inspect ./data/train.jsonl
    axiom dataset validate ./data/train.jsonl
    axiom dataset stats ./data/train.jsonl
    axiom dataset profile ./data/train.jsonl
    axiom dataset manifest ./data/train.jsonl
    axiom dataset preview-clean ./data/train.jsonl

Dataset support focuses on JSONL. Structural validity does not prove semantic quality or compliance. PII checks, where supported, are opt-in heuristics. `preview-clean` does not write a file; `clean` is a separate explicit write operation.

## Reproducible plan snapshots and diffs

    axiom train plan 7 --method qlora --save --name "Qwen baseline" --json
    axiom experiment list --json
    axiom experiment print <experiment-id> > saved-plan.json
    axiom experiment diff <baseline-id> <candidate-id> --json

`experiment print` emits deterministic, sorted JSON from the immutable saved
snapshot. Its SHA-256 fingerprint covers the saved planning inputs, captured
hardware, planner provenance, and exact estimate; it deliberately excludes
display-only labels, notes, tags, and creation time. `experiment diff` compares
two snapshots field by field and reports input, hardware, planner, and estimate
changes. Neither command reruns the planner, starts a training job, or claims
that the saved estimate is a benchmark.

New snapshots record the planner algorithm version and AXIOM version. Older
records remain printable, but unknown provenance is reported as unknown rather
than inferred from the currently installed build. A matching fingerprint means
the stored snapshot content matches; it does not guarantee identical runtime
behavior, dependencies, hardware availability, or training results.

The same read-only snapshot and diff operations are available at
`GET /api/workspace/experiments/{id}/reproducible` and
`GET /api/workspace/experiments/{left_id}/diff/{right_id}`, and from the
Training page's **Saved plan tools** section.

## Workspace metadata transfer

    axiom workspace metadata export --output ./axiom-workspace-metadata.zip
    axiom workspace metadata export --workspace ./my-project --output ./backup.zip --json
    axiom workspace metadata import ./axiom-workspace-metadata.zip
    axiom workspace metadata import ./backup.zip --workspace ./my-project --replace --json

The versioned ZIP contains only the model registry, workflow designs, experiment
plans, evaluation reports, and agent task profiles. Import validates the entire
archive and its SHA-256 checksums before writing. It merges by default, preserves
existing records, and rejects ID/name/source conflicts without a partial import.
Use `--replace` only when intentionally replacing collections present in the
archive; collections omitted from the archive are left untouched. Export refuses
to overwrite an existing destination unless `--force` is supplied.

The same transfer is available in **Settings → Workspace metadata transfer**
and through `GET /api/workspace/metadata/export` /
`POST /api/workspace/metadata/import`.

Archives exclude `axiom.yaml`, credentials, environment variables, API keys,
MCP grants, external integration commands and URLs, prediction-job history,
runtime traces, datasets, model weights, checkpoints, generated artifacts,
locks, and caches. Registry and evaluation records can still contain local
paths; inspect an archive before sharing it. The ZIP has a 24 MiB compressed
limit, 20 MiB total uncompressed limit, and a 4 MiB per-collection limit.

## Dataset and AI inspection JSON

```bash
axiom dataset profile ./data/train.jsonl --json
axiom ai hf-search qwen --limit 10 --json
axiom ai policy-audit ./models/my-model --json
axiom ai plan --model ./models/my-model --json
```

These commands emit versioned JSON suitable for scripts. Dataset profiling
returns aggregate diagnostics without row values; Hugging Face search returns
model metadata without downloading weights; policy audit reports indicators
without changing model files; optimization planning reports estimates rather
than measured throughput or a runtime guarantee.
## Hardware and planning

    axiom system info
    axiom train plan 7 --method auto
    axiom train plan 7 --method lora
    axiom train plan 7 --method qlora
    axiom ai plan --model ./models/my-model

Fit, speed, and VRAM figures are estimates, not measured performance or a guarantee that a runtime will succeed.

## Workflow, evaluation, and diagnostics

    axiom workflow list
    axiom workflow show <workflow-id>
    axiom workflow validate <workflow-id>
    axiom evaluation run ./data/eval.jsonl --prediction-field prediction --reference-field reference
    axiom doctor --json
    axiom roadmap audit --json
    axiom support-bundle --output ./axiom-support-bundle.zip

Workflow commands inspect designs. Evaluation scores precomputed predictions; it does not run inference.

## Tools and MCP

    axiom tools list
    axiom tools doctor
    axiom mcp serve
    axiom mcp permissions

Tool discovery does not prove vendor authentication. MCP is a separate stdio integration and its tools are denied by default until explicitly allowed.

Check `axiom <command> --help` on the installed build for the exact supported flags. Related: [Getting Started](Getting-Started), [Diagnostics](Workspace-Diagnostics-and-Support-Bundles).
