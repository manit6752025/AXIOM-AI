from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from axiom.cli.main import app

runner = CliRunner()


def _private_workspace(root: Path) -> tuple[Path, tuple[str, ...]]:
    root.mkdir(parents=True)
    secrets = (
        "CLI_SUPPORT_CONFIG_SECRET",
        "CLI_SUPPORT_DATA_SECRET",
        "CLI_SUPPORT_ENV_SECRET",
    )
    (root / "axiom.yaml").write_text(
        f"project:\n  name: bundle-test\napi_key: {secrets[0]}\n",
        encoding="utf-8",
    )
    data = root / "data"
    data.mkdir()
    (data / "private.jsonl").write_text(
        json.dumps({"text": secrets[1]}) + "\n",
        encoding="utf-8",
    )
    return root, secrets


def test_support_bundle_cli_writes_private_redacted_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace, secrets = _private_workspace(tmp_path / "workspace")
    monkeypatch.setenv("CLI_SUPPORT_SENTINEL", secrets[2])
    output = tmp_path / "support.zip"

    result = runner.invoke(
        app,
        ["support-bundle", "--workspace", str(workspace), "--output", str(output), "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "created"
    assert payload["bytes"] == output.stat().st_size
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"README.txt", "diagnostics.json", "environment.json"}
        joined = b"".join(archive.read(name) for name in archive.namelist())
    for secret in (*secrets, str(workspace)):
        assert secret.encode() not in joined
    if os.name == "posix":
        assert output.stat().st_mode & 0o077 == 0


def test_support_bundle_cli_does_not_overwrite_without_force(tmp_path: Path):
    workspace, _ = _private_workspace(tmp_path / "workspace")
    output = tmp_path / "existing.zip"
    output.write_bytes(b"preserve-existing-file")

    result = runner.invoke(
        app,
        ["support-bundle", "--workspace", str(workspace), "--output", str(output), "--json"],
    )

    assert result.exit_code != 0
    assert output.read_bytes() == b"preserve-existing-file"


def test_support_bundle_cli_force_replacement_restricts_permissions(tmp_path: Path):
    workspace, _ = _private_workspace(tmp_path / "workspace")
    output = tmp_path / "existing.zip"
    output.write_bytes(b"replace-this")
    if os.name == "posix":
        output.chmod(0o644)

    result = runner.invoke(
        app,
        ["support-bundle", "--workspace", str(workspace), "--output", str(output), "--force", "--json"],
    )

    assert result.exit_code == 0, result.stdout
    with zipfile.ZipFile(output) as archive:
        assert "diagnostics.json" in archive.namelist()
    if os.name == "posix":
        assert output.stat().st_mode & 0o077 == 0


def test_support_bundle_cli_rejects_symlink_output(tmp_path: Path):
    workspace, _ = _private_workspace(tmp_path / "workspace")
    target = tmp_path / "target.zip"
    target.write_bytes(b"preserve-target")
    output = tmp_path / "alias.zip"
    try:
        output.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks are not available: {exc}")

    result = runner.invoke(
        app,
        ["support-bundle", "--workspace", str(workspace), "--output", str(output), "--force", "--json"],
    )

    assert result.exit_code != 0
    assert target.read_bytes() == b"preserve-target"
    assert output.is_symlink()
