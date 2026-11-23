# Real accelerator validation

AXIOM has a separate manual hardware workflow for validating the actual native
adapter load and forward path on CUDA, ROCm, Apple MPS, and Intel XPU. It is
deliberately separate from the hosted CPU quality matrix: a mocked device
availability response or a CPU-only runner must never be reported as
accelerator execution.

## Runner labels

Provision one self-hosted runner per backend and configure the required
hardware-enabled PyTorch build before running the workflow. The workflow does
not replace the runner's PyTorch build; it installs AXIOM and the test helpers
only.

| Backend | Required runner labels JSON | Required runtime |
|---|---|---|
| CUDA | `["self-hosted","linux","x64","axiom-cuda"]` | CUDA-enabled PyTorch, working NVIDIA driver and GPU |
| ROCm | `["self-hosted","linux","x64","axiom-rocm"]` | ROCm-enabled PyTorch, working AMD ROCm stack and GPU |
| MPS | `["self-hosted","macOS","ARM64","axiom-mps"]` | Apple Silicon, MPS-enabled PyTorch |
| Intel XPU | `["self-hosted","linux","x64","axiom-xpu"]` | XPU-enabled PyTorch, Intel GPU driver and runtime |

Do not put untrusted code on these self-hosted runners. Keep them isolated to
this repository, use ephemeral runners where possible, and avoid exposing
registry or deployment credentials to the hardware job.

## Run

Open **Actions → AXIOM real accelerator validation → Run workflow**. Select
the physical backend and provide the exact runner-label JSON for that backend.
The workflow fails before inference if the selected accelerator is unavailable
or PyTorch was built without the required backend.

The test creates a valid tiny Llama safetensors checkpoint and a real LoRA
adapter, loads both through AXIOM's native engine, runs the forward pass on the
selected accelerator, and checks finite logits plus a measurable adapter
effect. It does not count device metadata, mocked dispatch, or CPU fallback as
accelerator execution.

Record one successful run per backend before marking R095 complete. Backends
that the project does not ship or claim to support should be explicitly
removed from the supported-device contract rather than silently skipped.
