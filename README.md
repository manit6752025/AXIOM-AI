<div align="center">

![AXIOM. Know what fits before you train.](docs/assets/axiom-logo.svg)

[![Status](https://img.shields.io/badge/status-beta-f59e0b?style=flat-square&labelColor=0f172a)](https://github.com/manit6752025/AXIOM-AI)
[![Version](https://img.shields.io/badge/version-v0.3.0--beta.8-f59e0b?style=flat-square&labelColor=0f172a)](https://github.com/manit6752025/AXIOM-AI/releases)
[![License](https://img.shields.io/badge/license-MIT-7c3aed?style=flat-square&labelColor=0f172a)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&labelColor=0f172a&logo=python&logoColor=white)](https://www.python.org/)
[![Local first](https://img.shields.io/badge/local--first-yes-22c55e?style=flat-square&labelColor=0f172a)](#privacy-and-safety)
[![MCP server](https://img.shields.io/badge/MCP-server-7c3aed?style=flat-square&labelColor=0f172a)](#mcp-server-for-ai-agents)

# AXIOM AI

**Check your model, data, and computer before you start.**
Inspect models, clean datasets, read your hardware, plan training, run local inference, and let AI agents use all of it safely. From a terminal, a browser window, or an MCP client.

[Quick start](#quick-start) · [Features](#features) · [Control Center](#control-center) · [MCP](#mcp-server-for-ai-agents) · [Docs](#documentation) · [Report an issue](https://github.com/manit6752025/AXIOM-AI/issues)

![AXIOM workflow: inspect, validate, plan, run](docs/assets/axiom-flow.svg)

![AXIOM terminal demo](docs/assets/terminal.svg)

</div>

## Why AXIOM

Most AI projects fail early for boring reasons: the model does not fit, the dataset has broken rows, or the setup is wrong. AXIOM catches those problems in seconds, on your own machine, before you spend hours of GPU time.

| Check | Plan | Act |
| --- | --- | --- |
| Inspect model folders and Hugging Face repos | Fit estimates for your RAM and VRAM | LoRA fine-tuning |
| Validate and clean JSONL datasets | Training plans you can save and compare | Local text generation with the AXIOM Engine or Ollama |
| Diagnose your workspace | Agent-specific optimization profiles | Generate predictions, score them, and gate regressions |

AXIOM is in beta. Plans are estimates, and real results depend on your model, data, and hardware.

## Quick start

```bash
python -m pip install --upgrade --pre axiom-all
axiom init my-ai
cd my-ai
axiom doctor
axiom system info
axiom train plan 7 --method qlora
axiom gui
```

| Step | Command | What you get |
| --- | --- | --- |
| 1 | `axiom init my-ai` | A starter project with `data`, `models`, `experiments`, `evaluations`, and `outputs` |
| 2 | `axiom doctor` | Pass, warn, or fail checks for your setup, with next steps |
| 3 | `axiom system info` | Your OS, CPU, RAM, GPU, VRAM, and model fit estimates |
| 4 | `axiom train plan 7 --method qlora` | A hardware-aware plan for a 7B model. No job is started |
| 5 | `axiom gui` | The Control Center in your browser |

Lost? Run `axiom guide`. It reads your current folder and suggests the next commands.

## Install

<details open>
<summary><b>pip (all platforms)</b></summary>

```bash
python -m pip install --upgrade --pre axiom-all
axiom version
```

</details>

<details>
<summary><b>Debian, Ubuntu, or Linux Mint (64-bit Intel/AMD)</b></summary>

```bash
curl -fsSLo /tmp/axiom-archive-keyring.gpg https://manit6752025.github.io/AXIOM-AI/apt/axiom-archive-keyring.gpg
sudo install -Dm0644 /tmp/axiom-archive-keyring.gpg /usr/share/keyrings/axiom-archive-keyring.gpg
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/axiom-archive-keyring.gpg] https://manit6752025.github.io/AXIOM-AI/apt stable main' | sudo tee /etc/apt/sources.list.d/axiom.list
sudo apt update
sudo apt install axiom-ai
```

</details>

<details>
<summary><b>macOS or Linux installer</b></summary>

```bash
curl -fsSL https://raw.githubusercontent.com/manit6752025/AXIOM-AI/main/installers/install.sh | bash
axiom version
```

The installer puts `axiom` in `~/.local/bin`. If the command is not found, add that folder to your `PATH` and reopen your terminal.

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
irm https://raw.githubusercontent.com/manit6752025/AXIOM-AI/main/installers/install.ps1 | iex
axiom version
```

If PowerShell cannot find `axiom`, close and reopen it.

</details>

<details>
<summary><b>Docker</b></summary>

```bash
docker pull ghcr.io/manit6752025/axiom-ai:latest
docker run --rm -p 127.0.0.1:8765:8765 ghcr.io/manit6752025/axiom-ai:latest
```

Open `http://127.0.0.1:8765/ui/`. Other images: `axiom-ai-inference`, `axiom-ai-training`, `axiom-ai-full`. See the [container guide](docs/Container-Image.md).

</details>

Downloads and checksums are on [GitHub Releases](https://github.com/manit6752025/AXIOM-AI/releases). Packaging recipes (APT, AUR, Homebrew, npm, Scoop, WinGet) live in [`packaging/`](packaging). More options: [installers/README.md](installers/README.md).

## Features

| Area | What it does | Start with |
| --- | --- | --- |
| **Models** | Inspect local folders, analyze Hugging Face repos without downloading weights, keep a registry | `axiom model inspect ./models/my-model` |
| **Datasets** | Validate JSONL, find duplicates, estimate tokens, write a cleaned copy | `axiom dataset validate ./data/train.jsonl` |
| **Hardware** | Detect CPU, RAM, GPU, VRAM, and estimate which model sizes fit | `axiom system info` |
| **Training** | Hardware-aware plans, saved experiments, optional local LoRA fine-tuning | `axiom train plan 7 --method qlora` |
| **AXIOM Engine** | Direct local inference on PyTorch, no Ollama needed | `axiom engine status` |
| **Ollama** | Service control, model pulls, streaming, benchmarking | `axiom infer status` |
| **Model Studio** | Search and install GGUF models, convert to GGUF | `axiom model studio wizard` |
| **Prompt Studio** | Build typed, versioned prompt templates and safely preview literal substitutions | `axiom prompt list` |
| **Agent Optimizer** | Eight workload profiles that pick quantization and build a runtime bundle | `axiom optimize profiles` |
| **Evaluation** | Score predictions offline and gate regressions against a baseline | `axiom evaluation run` |
| **Unified LLM client** | One client for the native engine, Ollama, and cloud providers | `axiom llm providers` |
| **Mobile bridge** | Token-protected OpenAI-compatible endpoint for mobile clients | `axiom mobile-bridge` |
| **MCP server** | Permissioned, default-deny tools for AI agents | `axiom mcp serve` |
| **Developer tools** | Find and set up coding-agent CLIs and model APIs | `axiom tools list` |
| **Diagnostics** | Doctor, status, summary, privacy-safe support bundle | `axiom doctor` |
| **Control Center** | Browser interface for all of the above | `axiom gui` |

## Common tasks

<details>
<summary><b>Models and the registry</b></summary>

```bash
axiom model inspect ./models/my-model
axiom model info Qwen/Qwen2.5-0.5B-Instruct
axiom model pull Qwen/Qwen2.5-0.5B-Instruct --max-disk-usage 85
axiom model add my-model ./models/my-model --format safetensors --tag chat
axiom model list
```

`model info` and `model analyze` look up Hugging Face models without downloading weights. `model pull` refuses to download if the model will not fit on disk. For gated models, sign in with `axiom hf login`. Your token is never printed. More: [Model Registry](docs/Model-Registry.md).

</details>

<details>
<summary><b>Datasets</b></summary>

```bash
axiom dataset validate ./data/train.jsonl
axiom dataset stats ./data/train.jsonl
axiom dataset clean ./data/train.jsonl --output ./data/train.cleaned.jsonl
```

- `validate` exits non-zero on invalid rows, so it works as a CI gate.
- `clean` removes invalid, empty, duplicate, and non-object rows. Your original file is never changed.
- Token counts are rough estimates.

</details>

<details>
<summary><b>Training and fine-tuning</b></summary>

```bash
axiom train plan 13 --method lora --save --name "13B LoRA" --tag baseline
axiom experiment list
python -m pip install -e '.[training]'
axiom train run --model-path ./models/my-model --dataset ./data/train.jsonl --output-dir ./runs/my-adapter --max-steps 20
```

A plan gives you precision, LoRA rank, batch size, sequence length, learning rate, and an estimated VRAM fit. Planning never starts a job. See the [fine-tuning guide](docs/Local-Fine-Tuning.md).

</details>

<details>
<summary><b>AXIOM Engine (direct local inference)</b></summary>

```bash
axiom engine status
axiom engine estimate --model-path ~/models/my-llama-model --context-length 4096 --ram-budget-gib 16 --vram-budget-gib 8
axiom engine run --model-path ~/models/my-llama-model --prompt "Explain AXIOM in one sentence." --max-new-tokens 128
```

- Runs Llama, Mistral, Qwen2, and Qwen3 text models on PyTorch. It does not call Ollama or Transformers `model.generate`.
- Inputs: floating-point safetensors and complete dense GGUF files. GGUF is dequantized to floating point, so there are no low-bit memory savings. EXL2, GPTQ, AWQ, and pickle `.bin` files are rejected.
- Memory modes: full load, `layer_streaming`, or `disk_streaming` (slower, uses less memory).
- Pick `--device` (CPU, CUDA, ROCm, Metal, XPU) and `--precision auto|fp32|fp16|bf16`.
- A local OpenAI-compatible endpoint is at `/v1/chat/completions` (text chat only, no tool calling).

Details: [AXIOM Engine](docs/Local-Engine.md).

</details>

<details>
<summary><b>Ollama and Model Studio</b></summary>

```bash
axiom infer status
axiom infer run qwen2.5:0.5b --prompt "Say hello in one sentence."
axiom model studio wizard
axiom model studio install-gguf REPO_ID FILENAME --name my-local-model
axiom model studio convert-gguf ./models/my-model --llama-cpp-dir ./llama.cpp --output ./out/model.gguf --quantization Q4_K_M
```

Add `--native` to `install-gguf` to install into AXIOM's own engine directory instead of Ollama. `convert-gguf` needs a local `llama.cpp` checkout. Guides: [Local Inference](docs/Local-Inference.md), [Model Studio](docs/Model-Studio.md), [Benchmarking](docs/Inference-Benchmarking.md).

</details>

<details>
<summary><b>Agent Optimizer</b></summary>

```bash
axiom optimize profiles
axiom optimize run --model ./models/my-model --profile 1 --target-tps 10
```

| # | Profile | Best for |
| --- | --- | --- |
| 1 | Coding Agent | Code generation, debugging, repository work |
| 2 | Reasoning Agent | Multi-step planning and analysis |
| 3 | Research Agent | Summaries, synthesis, long context |
| 4 | General Assistant | Everyday local assistant tasks |
| 5 | Automation Agent | Tool calling and structured workflows |
| 6 | Math Agent | Symbolic and numerical problems |
| 7 | Writing Agent | Drafting, editing, style changes |
| 8 | Multilingual Agent | Translation and mixed-language tasks |

Your target tokens per second is a target, not a promise. `axiom ai policy-audit` only reports; AXIOM never removes or bypasses model safety controls.

</details>

<details>
<summary><b>Offline evaluation</b></summary>

```bash
axiom evaluation run ./data/predictions.jsonl --prediction-field prediction --reference-field reference --name "baseline" --baseline
axiom evaluation run ./data/new.jsonl --prediction-field prediction --reference-field reference --baseline-run-id RUN_ID --regression-tolerance 0.05
axiom evaluation list
```

Scores predictions you already have. No model is run. Metrics: `exact_match`, `normalized_exact_match`, `token_f1_micro`. Mark one run as the baseline, then later runs report pass or fail against it. Guide: [Evaluation](docs/Evaluation.md).

</details>

<details>
<summary><b>Unified LLM client and mobile bridge</b></summary>

```bash
axiom llm providers
axiom llm chat deepseek-chat "Explain KV caching briefly." --provider deepseek --send-to-provider
axiom mobile-bridge --listen-lan --port 40509
```

Providers: AXIOM Native Engine, Ollama, OpenAI, Anthropic, Gemini, DeepSeek, Moonshot/Kimi, OpenRouter, and public OpenAI-compatible HTTPS endpoints. Cloud chat needs explicit `--send-to-provider`, and keys are never saved.

The mobile bridge serves only `/v1/models` and `/v1/chat/completions` behind a random bearer token. It uses plain HTTP, so only run it on a trusted LAN and never port-forward it. See the [Unified LLM client guide](docs/Unified-LLM-Client.md).

</details>

<details>
<summary><b>Developer tools, integrations, workflows</b></summary>

```bash
axiom tools list
axiom tools plan claude-code
axiom tools install opencode --yes
axiom integration add my-agent --type mcp --command "my-agent-cli serve"
axiom workflow list
axiom agent-profile list
```

- `tools install` previews by default. AXIOM never runs a remote install script for you.
- Integration types: `mcp`, `cli`, `http`, `a2a`.
- Workflow designs are design-only and never executed.

See [Developer tools](docs/developer-tools-contract.md), [Integrations](docs/integrations.md), [Agent Profiles](docs/Agent-Profiles.md).

</details>

## MCP server for AI agents

AXIOM can run as an [MCP](https://modelcontextprotocol.io/) server so AI agents can inspect your models, datasets, and hardware through a controlled door.

**Every tool is denied by default.** You choose what each profile may call.

#### Set up

Point your MCP client at AXIOM. Start it from your project folder, because the workspace root is the server's working directory and permissions are stored in that project's `.axiom/` folder.

```json
{
  "mcpServers": {
    "axiom": {
      "command": "axiom",
      "args": ["mcp", "serve"]
    }
  }
}
```

Allow only what you need, from the same project root:

```bash
axiom mcp allow axiom_workspace_context
axiom mcp allow axiom_workspace_assets
axiom mcp allow axiom_system_info
axiom mcp allow axiom_model_info
axiom mcp allow axiom_dataset_inspect
```

Take access back at any time:

```bash
axiom mcp revoke axiom_dataset_inspect
```

Use profiles for stricter or separate access:

```bash
axiom mcp profile create readonly
axiom mcp allow axiom_system_info --profile readonly
axiom mcp profile use readonly
```

#### Rules every tool follows

- Paths are workspace-relative. Absolute paths, credentials, and config contents are never returned.
- Read-only tools write nothing, run no inference, and use no network.
- Tools that write files, start jobs, or use compute are separate permissions, and agents must show the exact configuration and get explicit user approval first.
- Row text from datasets is off by default. Where a tool can return excerpts, it needs `include_text=true` after approval, capped at 2,000 characters per field. The MCP host may keep what is returned.
- Returned text is untrusted data, never instructions.
- `clear` or `no_known_blockers` results mean no known problems. They do not guarantee that a model loads, fits, or performs well.
- Lexical metrics (exact match, normalized exact match, token F1) do not measure semantic quality.

#### Start here

`axiom_workspace_context` returns project structure, hardware, developer-tool availability, the tools your profile allows, and a suggested next command, all in one call. Suggested commands are never run without user approval. `axiom_workspace_assets` finds model folders and JSONL datasets (up to 200 results per category) and returns paths only.

#### Tools

##### Workspace and hardware

| Tool | What it does |
| --- | --- |
| `axiom_info` | AXIOM version and capabilities |
| `axiom_workspace_context` | Privacy-bounded project, hardware, and developer-tool snapshot |
| `axiom_workspace_assets` | Find model directories and JSONL datasets |
| `axiom_system_info` | Detected hardware |
| `axiom_model_list` | Registered models |
| `axiom_model_info` | Inspect a local model folder |
| `axiom_training_plan` | Hardware-aware training plan |

##### Datasets

| Tool | What it does |
| --- | --- |
| `axiom_dataset_inspect` | Bounded JSONL profile: row counts, field types, duplicates, warnings. Scans at most 64 MiB, 100,000 rows, 1 MiB per line. PII value scanning is off unless `scan_pii=true` |
| `axiom_dataset_clean_preview` | Keep/remove counts and a proposed output path. Writes nothing |
| `axiom_dataset_clean` | Write a new cleaned JSONL copy. Never overwrites an existing file |

##### Training

| Tool | What it does |
| --- | --- |
| `axiom_training_preflight` | Check LoRA inputs and estimates without loading weights |
| `axiom_training_eval_overlap_preflight` | Exact-overlap check between training and evaluation JSONL |
| `axiom_training_job_eval_overlap_preflight` | Same check, using a completed job's recorded dataset |
| `axiom_training_job_validation_preflight` | Plan a base-versus-adapter comparison with overlap, compatibility, and truncation checks |
| `axiom_training_job_start` | Start a bounded local LoRA job (up to 1,000 steps). Writes adapters under `outputs/` |
| `axiom_training_job_status` | Progress and a bounded loss trace |
| `axiom_training_job_wait` | Wait up to 30 seconds for an outcome. Uses the status permission |
| `axiom_training_job_cancel` | Cancel before the adapter is finalized |
| `axiom_training_job_prediction_compare` | Verify and compare a base/adapter prediction pair |

##### Inference and prediction jobs

| Tool | What it does |
| --- | --- |
| `axiom_engine_preflight` | Check a model and optional adapter before inference |
| `axiom_engine_generate` | One bounded local generation (prompt up to 8,000 characters, up to 256 tokens) |
| `axiom_prediction_job_preflight` | Estimate prompt clipping for a JSONL dataset |
| `axiom_prediction_job_validation_preflight` | Combine prompt and engine checks before a run |
| `axiom_prediction_job_start` | Generate an evaluation-ready JSONL artifact (up to 100 rows, 64 tokens per row) |
| `axiom_prediction_job_status` | Counts and workspace-relative paths |
| `axiom_prediction_job_wait` | Wait up to 30 seconds. Uses the status permission |
| `axiom_prediction_job_cancel` | Cancel before artifact finalization |
| `axiom_prediction_job_evaluate` | Score a completed job. Uses the `axiom_evaluation_run` permission |
| `axiom_prediction_job_examples` | Up to three verified mismatches |
| `axiom_prediction_job_compare` | Exact-match transitions and metric deltas between two jobs |
| `axiom_prediction_job_compare_examples` | Paired examples from one transition |

##### Evaluation

| Tool | What it does |
| --- | --- |
| `axiom_evaluation_preflight` | Preview metrics and data quality without saving |
| `axiom_evaluation_run` | Score predictions and save aggregate metrics (up to 64 MiB, 100,000 rows) |
| `axiom_evaluation_history` | Read saved results |
| `axiom_evaluation_compare` | Compare two complete saved evaluations. Uses the history permission |
| `axiom_evaluation_compare_examples` | Paired examples from a saved comparison |
| `axiom_evaluation_examples` | Up to three verified mismatches |

##### Workflows

Workflow designs are never executed. Every response marks them `execution_status: not_executable`.

| Tool | What it does |
| --- | --- |
| `axiom_workflow_read` | List summaries or fetch a validated design |
| `axiom_workflow_compare` | Structural diff against one retained revision (the store keeps 10) |
| `axiom_workflow_dry_run` | Planning estimate for `training.plan` nodes. Needs `axiom_workflow_read` and `axiom_training_plan` |
| `axiom_workflow_save` | Save a user-approved graph (up to 64 nodes, 128 connections, 200 designs per workspace) |

##### Compression

| Tool | What it does |
| --- | --- |
| `axiom_supercompress_status` | Whether SuperCompress is configured |
| `axiom_supercompress` | Compress context before inference |

#### Permissions that go together

| To use | Also allow |
| --- | --- |
| `axiom_workflow_dry_run` | `axiom_workflow_read`, `axiom_training_plan` |
| `axiom_training_job_wait` | `axiom_training_job_status` |
| `axiom_prediction_job_wait` | `axiom_prediction_job_status` |
| `axiom_prediction_job_evaluate` | `axiom_evaluation_run` |
| `axiom_training_job_eval_overlap_preflight` | `axiom_training_job_status`, `axiom_training_eval_overlap_preflight` |
| `axiom_training_job_validation_preflight` | training status, overlap, engine preflight, and prediction preflight permissions |
| `axiom_prediction_job_validation_preflight` | `axiom_engine_preflight`, `axiom_prediction_job_preflight` |
| `axiom_training_job_prediction_compare` | `axiom_training_job_status`, `axiom_prediction_job_compare` |

#### Suggested agent workflow

1. Call `axiom_workspace_context`, then `axiom_workspace_assets`.
2. Inspect with `axiom_model_info` and `axiom_dataset_inspect`.
3. Run the matching preflight tool.
4. Show the user the exact configuration and output path.
5. Start a job only after explicit approval.
6. Poll with the wait tool. After `timed_out`, wait again. After `state_unavailable` or `recovery_unverified`, check status later and do not start a replacement job.
7. Evaluate, then compare against a baseline.

## Control Center

```bash
axiom gui
```

Opens in your browser, served from your computer on port `40507`. Use `--port` or `AXIOM_GUI_PORT` to change it, `--port 0` for a free port, and `--no-open` to skip opening the browser. Press `Ctrl-C` to stop.

![AXIOM Control Center](docs/assets/webui.svg)

Panels: Models, Runtime, AXIOM Engine, Fine-tuning, Model Studio, Agent Optimizer and Quantizer, Agent Profiles, Prompt Studio, and an API Playground. The local API has interactive docs at `/api/docs`. See [API.md](docs/API.md).

## How it fits together

```mermaid
flowchart LR
  CLI[axiom CLI] --> CORE[AXIOM core]
  GUI[Control Center] --> API[Local API on 127.0.0.1]
  API --> CORE
  MCP[MCP server] --> PERM{Tool permissions}
  PERM --> CORE
  CORE --> FILES[(Your models, data, and workspace)]
  CORE -.->|optional, when you ask| NET[Hugging Face, Ollama, providers]
```

## Privacy and safety

- **Local by default.** Inspection, validation, cleaning, planning, and evaluation run on your computer. Model weights and datasets are not uploaded.
- **Network is opt-in.** Commands that use the network say so.
- **Loopback only.** The Control Center and local API bind to `127.0.0.1`.
- **Default-deny agents.** MCP tools are blocked until you allow them, per profile.
- **Files are protected.** Dataset cleaning never edits the original, and path checks reject symlink tricks. See the [security advisory](docs/SECURITY-ADVISORY-2026-10.md).
- **No secrets in output.** Tokens and API keys are never printed or stored.
- **Honest labels.** Fit values and token counts are heuristics.

Found a vulnerability? Read [SECURITY.md](SECURITY.md). Need help? Run `axiom support-bundle` (diagnostics only, no datasets, models, config values, or logs), review the ZIP, and attach it to an [issue](https://github.com/manit6752025/AXIOM-AI/issues).

## Automation
Local security checks include `axiom doctor --strict --json`,
`axiom security audit --json`, and
`axiom security artifact inspect ./dist/<artifact> --json`. The artifact
inspector checks installed package RECORD integrity, wheel hashes, and source-archive
safety without extracting or executing release contents; see
[Diagnostics and Support](docs/Diagnostics-and-Support.md).


Most commands accept `--json` and print a versioned payload:

```bash
axiom system info --json
axiom dataset validate ./data/train.jsonl --json
axiom doctor --json
```

`axiom daemon --json` prints readiness records for supervisors and scripts.

## Cheat sheet

| I want to... | Run |
| --- | --- |
| See all commands | `axiom --help` |
| Get guided next steps | `axiom guide` |
| Check my setup | `axiom doctor` |
| Quick snapshot | `axiom summary` |
| Check a model folder | `axiom model inspect ./models/my-model` |
| Check a dataset | `axiom dataset validate ./data/train.jsonl` |
| See my hardware | `axiom system info` |
| Plan training | `axiom train plan 7 --method qlora` |
| Generate text locally | `axiom infer run qwen2.5:0.5b --prompt "Hi"` |
| Open the browser UI | `axiom gui` |
| Tab completion | `axiom completion bash` |
| Make a support bundle | `axiom support-bundle` |
| See what is planned | `axiom roadmap` |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `axiom: command not found` | Add `~/.local/bin` to your `PATH` and reopen the terminal. On Windows, reopen PowerShell. |
| Control Center will not start | Run `axiom doctor`. From source, run `npm run build` first. |
| Port already in use | Run `axiom gui --port 0`, or set `AXIOM_GUI_PORT`. |
| Training or Engine commands fail | Run `python -m pip install -e '.[training]'`. |
| Still stuck | Run `axiom support-bundle` and open an [issue](https://github.com/manit6752025/AXIOM-AI/issues). |

## Documentation

- **Start here:** [My First AI](examples/my-first-ai/README.md) · [Installers](installers/README.md) · [CLI guide](docs/CLI-GUIDE.md) · [Wiki](https://github.com/manit6752025/AXIOM-AI/wiki)
- **Features:** [Model Registry](docs/Model-Registry.md) · [Model Studio](docs/Model-Studio.md) · [Fine-tuning](docs/Local-Fine-Tuning.md) · [Engine](docs/Local-Engine.md) · [Local Inference](docs/Local-Inference.md) · [Prompt Studio](docs/wiki/Prompt-Studio.md) · [Evaluation](docs/Evaluation.md) · [Unified LLM client](docs/Unified-LLM-Client.md) · [MCP](#mcp-server-for-ai-agents)
- **Reference:** [API](docs/API.md) · [Data retention](docs/DATA-RETENTION.md) · [Diagnostics](docs/Diagnostics-and-Support.md) · [Security](SECURITY.md) · [Governance](docs/GOVERNANCE.md) · [Accessibility](docs/ACCESSIBILITY.md)
- **Project:** [Changelog](docs/CHANGELOG.md) · [Roadmap](ROADMAP.md) · [Releases](docs/releases.md)

## Contributing

Good bug reports include your OS, Python version, AXIOM version, and the exact command you ran. Never attach API keys, private datasets, or model weights. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) first.

```bash
git clone https://github.com/manit6752025/AXIOM-AI.git
cd AXIOM-AI
python -m pip install -e '.[test]'
npm install
npm run build
```

Run these before opening a pull request:

```bash
python -m pytest -q
npm run lint
npm run build
npm run e2e
```

The web UI is React, TypeScript, and Vite. The CLI and API are Python (Typer, FastAPI, Pydantic). Pull requests run quality gates and CodeQL, and release workflows publish to PyPI, npm, and GHCR.

## License

AXIOM is available under the [MIT License](LICENSE).

<br>

<div align="center">

**Built with care by [@manit6752025](https://github.com/manit6752025) and [@taxcollector23](https://github.com/taxcollector23)**

<sub>If AXIOM saved you a bad training run, give it a star.</sub>

</div>
