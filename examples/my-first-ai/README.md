# My First AI Project with AXIOM

Build a small, local-first AI project scaffold, learn what each setting means, validate it, and make a training plan before spending time or compute. This starter is intentionally safe: it does **not** download model weights, upload data, or start training automatically.

## What you will do

1. Check that AXIOM is installed.
2. Inspect and customise `axiom.yaml`.
3. Validate the project and configuration.
4. Inspect a tiny illustrative JSONL dataset (optional).
5. Create a training plan and understand its limits.

## 1. Check your installation

```bash
axiom version
axiom doctor
axiom --help
```

If `axiom` is not found, follow the [installer guide](../../installers/README.md).

## 2. Understand the project file

Open `axiom.yaml`. It is a starter template, not a ready-to-train configuration:

- `project.name` identifies this workspace; change it to your project name.
- `model.base` and `model.source` are unset until you choose a supported model and source reference.
- `dataset.source` is unset until you point AXIOM at a dataset. The format is JSONL.
- `training.method` and `training.learning_rate` are unset deliberately; choose values only after reviewing your model and hardware.
- `training.epochs` is a placeholder, not a recommendation.
- `evaluation.enabled` enables evaluation configuration; it does not generate predictions by itself.
- `deployment.target: local` records your intended target; it does not deploy a model.

Keep secrets and private data out of this example. Do not commit API tokens, private datasets, model weights, or generated support bundles.

## 3. Validate the scaffold

```bash
axiom project validate
axiom config validate
```

The untouched template intentionally leaves the model and dataset unconfigured. If validation reports missing inputs, configure those paths before asking AXIOM for a concrete plan. Use `axiom --help` and the [CLI guide](../../docs/CLI-GUIDE.md) if command syntax differs in your installed version.

## 4. Inspect the sample JSONL (optional)

`data/sample.jsonl` contains two illustrative conversation records. Each line is a separate JSON object. This is a format example only—not a useful training corpus, a guarantee of compatibility with every model, or a recommendation to train.

```bash
axiom dataset inspect data/sample.jsonl
axiom dataset validate data/sample.jsonl
```

Before using real data, verify that you have the rights to use it, remove secrets and personal information, and confirm the exact schema expected by your chosen training workflow.

## 5. Plan before training

```bash
axiom train plan 7 --method qlora
```

A planning command estimates a possible run; it does not launch training and does not guarantee the model fits. Architecture, sequence length, precision, dataset schema, hardware, and optional dependencies all matter. Read the [local fine-tuning guide](../../docs/Local-Fine-Tuning.md) before starting any training job.

## 6. Explore the browser interface (optional)

```bash
axiom gui
```

AXIOM serves the Control Center locally. Use the URL printed in the terminal and press Ctrl-C to stop it. Do not expose the local API to an untrusted network.

## Suggested next steps

- Replace the sample data with a dataset you are permitted to use.
- Choose a model supported by the current runtime and check its licence.
- Run `axiom system info` and compare estimates with your actual memory budget.
- Review a dry-run plan before training.
- Read the [changelog](../../docs/CHANGELOG.md) and [API/data safety guidance](../../docs/API.md) before connecting external services.

**Starter status:** documentation and sample data only. These steps do not download a model, start training, perform remote inference, or deploy anything.
