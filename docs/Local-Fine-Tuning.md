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
  --max-steps 20 --sequence-length 512 --lora-rank 8
axiom --no-banner train jobs
axiom --no-banner train status JOB_ID
```

`train run` is an actual training operation and runs until completion or failure. Progress, loss when available, and the final adapter directory are recorded. The history is stored under `${AXIOM_FINETUNE_HOME:-~/.local/share/axiom/fine-tuning}/jobs`.

## Control Center and API

Open **Training** in the Control Center. The **Local LoRA fine-tuning** panel lets you supply local paths and bounded training settings, start a job, inspect progress, and request cancellation. Cancellation stops at a safe training boundary; the panel shows `cancelling` until the worker confirms `cancelled`. After training steps finish, the job enters `finalizing` while adapter files are saved, and cancellation is no longer accepted. The API endpoints are `GET /api/fine-tuning/runtime`, `GET /api/fine-tuning/jobs`, `GET /api/fine-tuning/jobs/{job_id}`, `POST /api/fine-tuning/jobs`, and `POST /api/fine-tuning/jobs/{job_id}/cancel`.

The MCP server also exposes `axiom_training_job_start`, `axiom_training_job_status`, and `axiom_training_job_cancel`. Each is default-deny and needs its own explicit workspace permission. MCP start accepts workspace-relative inputs and an output under an existing `outputs/` folder. Its instructions tell agents to run preflight, present the exact configuration, and wait for the user's explicit request before starting; granting the start tool permission gives the agent standing capability.

MCP inputs are revalidated in the background worker, and the JSONL dataset is opened with descriptor-relative path checks. The adapter destination is checked again immediately before saving to refuse files added during model loading or training. Transformers still opens model weights by path, so a concurrent local process with write access to the workspace can race path checks while model files load or adapter files save. Avoid changing the model, dataset, or destination while a job is running.

## Limits and honest scope

- Base model path must be a local Transformers model directory; no model is downloaded by AXIOM.
- The training extra requires compatible versions of PyTorch (>=2.2), Transformers (>=5), PEFT (>=0.20), and Accelerate (>=1.1). The runtime endpoint checks installed versions and prevents starting a job with an older dependency set.
- Dataset is limited to 50 MiB, 10,000 rows, and 1 MiB per line; individual text values are limited to 300,000 characters.
- Steps: 1–1,000; sequence length: 32–2,048 tokens; LoRA rank: 1–64; learning rate: 0.000001–0.01.
- A single local fine-tuning job is allowed at a time to avoid accidental competing CPU/GPU workloads.
- The MCP trainer rejects symbolic links and junctions inside model directories and limits its directory metadata scan to 50,000 entries.
- The output must be new or empty. Only the generated adapter and tokenizer files are written to the output directory; the base-model directory and dataset are inputs.
- These limits do not guarantee that a model fits available memory. Start with a small model, short sequence length, and a few steps. GPU capability and architecture-specific LoRA compatibility are checked by the installed runtime, not inferred from the planning estimate.
- This is supervised causal-LM LoRA fine-tuning, not full-weight training, QLoRA, preference training, or a general workflow executor.
