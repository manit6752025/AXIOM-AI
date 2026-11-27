from __future__ import annotations

from pathlib import Path

import yaml


def test_accelerator_workflow_requires_manual_dispatch_and_real_device_probes():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (root / ".github/workflows/accelerator-integration.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    docs = (root / "docs/Accelerator-Hardware-Validation.md").read_text(encoding="utf-8")

    assert list(workflow["on"]) == ["workflow_dispatch"]
    choices = set(workflow["on"]["workflow_dispatch"]["inputs"]["device"]["options"])
    assert choices == {"cuda", "cuda-github-hosted-t4", "rocm", "mps", "xpu"}

    expected = {
        "cuda": ("cuda", ["self-hosted", "Linux", "X64", "axiom-cuda"], "torch.version.cuda", "torch.cuda.is_available()"),
        "cuda-github-hosted-t4": ("cuda", "axiom-cuda-t4", "torch.version.cuda", "torch.cuda.is_available()"),
        "rocm": ("rocm", ["self-hosted", "Linux", "X64", "axiom-rocm"], "torch.version.hip", "torch.cuda.is_available()"),
        "mps": ("mps", ["self-hosted", "macOS", "ARM64", "axiom-mps"], "torch.backends.mps.is_built()", "torch.backends.mps.is_available()"),
        "xpu": ("xpu", ["self-hosted", "Linux", "X64", "axiom-xpu"], "torch.xpu", "torch.xpu.is_available()"),
    }
    jobs = workflow["jobs"]
    for device, (backend, runner, runtime_probe, availability_probe) in expected.items():
        job = jobs[device]
        assert job["if"] == f"inputs.device == '{device}'"
        assert job["env"]["AXIOM_ACCELERATOR_DEVICE"] == backend
        assert job["runs-on"] == runner
        run_steps = "\n".join(step.get("run", "") for step in job["steps"])
        assert runtime_probe in run_steps
        assert availability_probe in run_steps
        assert "tests/test_native_engine.py::test_native_lora_adapter_executes_on_configured_accelerator" in run_steps

    hosted_probe = "\n".join(step.get("run", "") for step in jobs["cuda-github-hosted-t4"]["steps"])
    assert "nvidia-smi" in hosted_probe
    assert 'assert "T4" in name' in hosted_probe
    assert "compute_cap" not in hosted_probe

    assert "axiom-cuda-t4" in docs
    assert "Metal/MPS" in docs
    assert "paravirtual" in docs
    assert "self-hosted" in docs
