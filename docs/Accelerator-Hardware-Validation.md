# Physical accelerator validation

AXIOM has a single real checkpoint execution test for accelerator backends:
`tests/test_native_engine.py::test_native_lora_adapter_executes_on_configured_accelerator`.
It builds a tiny valid safetensors base model and LoRA adapter, loads both with
`NativeLlamaRuntime.from_directory`, runs the forward path on the selected
device, and checks finite logits and a measurable adapter effect.

The GitHub-hosted standard macOS and M2 xlarge runners expose an Apple
paravirtual device without usable Metal 2.0 support in the current runner
environment; they must not be counted as MPS execution. Real device validation
uses registered self-hosted runners with explicit labels.

| Device | Required runner labels |
| --- | --- |
| CUDA | `self-hosted, Linux, X64, axiom-cuda` |
| ROCm | `self-hosted, Linux, X64, axiom-rocm` |
| Apple Metal/MPS | `self-hosted, macOS, ARM64, axiom-mps` |
| Intel XPU | `self-hosted, Linux, X64, axiom-xpu` |
| GitHub-hosted CUDA | Organization larger runner named `axiom-cuda-t4` |

GitHub also documents a GPU larger-runner configuration with one NVIDIA
Tesla T4 (16 GiB VRAM), available to organizations on GitHub Team or GitHub
Enterprise Cloud. An organization owner must create a larger runner named
`axiom-cuda-t4`, grant this repository access to it, and configure its image
with the NVIDIA driver. The manual `cuda-github-hosted-t4` option then checks
that the physical device is actually a T4 before executing the adapter test.
The current repository is under a personal owner, so this hosted larger-runner
route requires moving it to an eligible organization; no access or hardware
is assumed by this workflow.

Each runner must have the vendor driver/runtime and a PyTorch build compatible
with that device preinstalled or installable by the configured environment.
The workflow verifies actual runtime availability before running the
safetensors/LoRA forward test; a missing device or incompatible build fails,
rather than converting the result to a successful skip.

Run from a maintainer account after registering the matching hardware runner:

```bash
gh workflow run accelerator-integration.yml --ref implement/roadmap -f device=mps
gh workflow run accelerator-integration.yml --ref implement/roadmap -f device=cuda
gh workflow run accelerator-integration.yml --ref implement/roadmap -f device=rocm
gh workflow run accelerator-integration.yml --ref implement/roadmap -f device=xpu
```

A successful CPU fallback or a mocked device availability result does not
satisfy these hardware acceptance checks. Keep a device pending until its
corresponding workflow completes successfully on physical hardware.
