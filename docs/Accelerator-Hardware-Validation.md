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
