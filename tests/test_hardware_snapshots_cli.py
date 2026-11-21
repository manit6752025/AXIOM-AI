from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

import axiom.cli.main as cli_main
from axiom.core import hardware_snapshots as snapshots_core
from axiom.core.hardware_snapshots import HardwareSnapshotStore

runner = CliRunner()


def test_system_hardware_snapshot_commands_manage_opt_in_capture_and_history(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = HardwareSnapshotStore(tmp_path / ".axiom")
    monkeypatch.setattr(cli_main, "HardwareSnapshotStore", lambda: store)
    monkeypatch.setattr(snapshots_core, "detect_hardware", lambda: SimpleNamespace(
        os_name="Linux", architecture="x86_64", cpu_cores=12, ram_gb=48.0,
        gpu_name=None, vram_gb=None, cuda_available=False,
        private_host="DO-NOT-STORE",
    ))

    enabled = runner.invoke(cli_main.app, ["system", "snapshots", "enable", "--max-snapshots", "3", "--max-age-days", "20", "--json"])
    assert enabled.exit_code == 0, enabled.output
    assert json.loads(enabled.stdout)["settings"]["enabled"] is True

    captured = runner.invoke(cli_main.app, ["system", "snapshots", "capture", "--json"])
    assert captured.exit_code == 0, captured.output
    assert "DO-NOT-STORE" not in captured.stdout

    listed = runner.invoke(cli_main.app, ["system", "snapshots", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout)["total"] == 1

    disabled = runner.invoke(cli_main.app, ["system", "snapshots", "disable", "--json"])
    assert disabled.exit_code == 0, disabled.output
    assert json.loads(disabled.stdout)["total"] == 1
    assert json.loads(disabled.stdout)["settings"]["enabled"] is False

    cleared = runner.invoke(cli_main.app, ["system", "snapshots", "clear", "--yes", "--json"])
    assert cleared.exit_code == 0, cleared.output
    assert json.loads(cleared.stdout)["total"] == 0
