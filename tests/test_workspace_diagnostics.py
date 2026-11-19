from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from axiom.core.diagnostics import create_support_bundle, diagnose_workspace, discover_workspace_root


def _project(root: Path, config: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (root / "axiom.yaml").write_text(config, encoding="utf-8")
    return root


def test_discovery_uses_nearest_project_and_honors_explicit_override(tmp_path: Path):
    root = _project(tmp_path / "project", "project:\n  name: demo\n")
    nested = root / "data" / "raw"
    nested.mkdir(parents=True)
    assert discover_workspace_root(cwd=nested) == root.resolve()
    other = _project(tmp_path / "other")
    assert discover_workspace_root(other, cwd=nested) == other.resolve()


def test_discovery_rejects_symlink_and_non_directory(tmp_path: Path):
    actual = _project(tmp_path / "actual")
    link = tmp_path / "link"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(ValueError, match="symbolic link"):
        discover_workspace_root(link)
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        discover_workspace_root(file_path)


def test_invalid_config_reports_problem_without_echoing_values(tmp_path: Path):
    secret = "EXAMPLE-PRIVATE-CREDENTIAL-DO-NOT-LEAK"
    root = _project(tmp_path / "project", f"project: [{secret}\n")
    report = diagnose_workspace(root)
    config_check = next(check for check in report["checks"] if check["id"] == "workspace.config.invalid")
    assert config_check["status"] == "fail"
    assert secret not in json.dumps(report)


def test_valid_project_returns_stable_bounded_check_shape(tmp_path: Path):
    root = _project(tmp_path / "project", "project:\n  name: demo\ntraining:\n  epochs: 1\n")
    (root / "data").mkdir()
    report = diagnose_workspace(root)
    assert report["schema_version"] == 1
    assert report["status"] in {"pass", "warn", "fail"}
    assert report["workspace"]["is_project"] is True
    assert all({"id", "title", "status", "severity", "detail", "remediation"} <= set(check) for check in report["checks"])
    assert len({check["id"] for check in report["checks"]}) == len(report["checks"])


def test_support_bundle_contains_only_allowlisted_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _project(tmp_path / "private-project", "project:\n  name: private\napi_key: TOP_SECRET_CONFIG_SENTINEL\n")
    data = root / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"private_text":"DATA_SENTINEL"}\n', encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setenv("SUPPORT_BUNDLE_SECRET", "ENV_SECRET_SENTINEL")
    archive_bytes = create_support_bundle()
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == {"README.txt", "diagnostics.json", "environment.json"}
        joined = b"".join(archive.read(name) for name in archive.namelist())
        diagnostics = json.loads(archive.read("diagnostics.json"))
    for sentinel in (b"TOP_SECRET_CONFIG_SENTINEL", b"DATA_SENTINEL", b"ENV_SECRET_SENTINEL", str(root).encode()):
        assert sentinel not in joined
    assert diagnostics["privacy"]["config_values_included"] is False
    assert diagnostics["privacy"]["absolute_paths_included"] is False


def test_metadata_symlink_is_reported_without_following_it(tmp_path: Path):
    root = _project(tmp_path / "project", "project:\n  name: demo\n")
    target = _project(tmp_path / "outside")
    try:
        (root / "models").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    report = diagnose_workspace(root)
    check = next(c for c in report["checks"] if c["id"] == "workspace.path.models")
    assert check["status"] == "fail"
    assert "outside" not in check["detail"]
