from __future__ import annotations

import os
from pathlib import Path

import pytest

from axiom.core import mcp_paths


def _select_path_mode(monkeypatch: pytest.MonkeyPatch, *, fallback: bool) -> None:
    native_supported = mcp_paths._descriptor_access_supported()
    if not fallback and not native_supported:
        pytest.skip("Descriptor-relative no-follow file access is unavailable")
    monkeypatch.setattr(mcp_paths, "_descriptor_access_supported", lambda: not fallback)


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    (root / "data" / "sample.jsonl").write_bytes(b'{"text":"safe"}\n')
    return root


@pytest.mark.parametrize("fallback", [False, True], ids=["descriptor-relative", "portable-fallback"])
def test_open_workspace_file_reads_regular_file_in_both_path_modes(tmp_path, monkeypatch, fallback):
    root = _workspace(tmp_path, monkeypatch)
    _select_path_mode(monkeypatch, fallback=fallback)

    resolved, stream = mcp_paths.open_mcp_workspace_file("data/sample.jsonl", root)
    with stream:
        assert stream.read() == b'{"text":"safe"}\n'
    assert resolved == root / "data" / "sample.jsonl"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " ",
        "/etc/passwd",
        "../outside.jsonl",
        "data/../../outside.jsonl",
        "data//sample.jsonl",
        "data/./sample.jsonl",
        r"data\\sample.jsonl",
        r"C:\\private\\sample.jsonl",
        "data/" + ("x" * 4100),
    ],
)
def test_open_workspace_file_rejects_unsafe_relative_path_syntax(tmp_path, monkeypatch, raw):
    root = _workspace(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        mcp_paths.open_mcp_workspace_file(raw, root)


@pytest.mark.parametrize("fallback", [False, True], ids=["descriptor-relative", "portable-fallback"])
def test_open_workspace_file_rejects_symlinked_workspace_root(tmp_path, monkeypatch, fallback):
    root = _workspace(tmp_path, monkeypatch)
    _select_path_mode(monkeypatch, fallback=fallback)
    alias = tmp_path / "workspace-alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="workspace root"):
        mcp_paths.open_mcp_workspace_file("data/sample.jsonl", alias)


@pytest.mark.parametrize("fallback", [False, True], ids=["descriptor-relative", "portable-fallback"])
def test_open_workspace_file_rejects_symlinked_intermediate_directory(tmp_path, monkeypatch, fallback):
    root = _workspace(tmp_path, monkeypatch)
    _select_path_mode(monkeypatch, fallback=fallback)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jsonl").write_text("outside", encoding="utf-8")
    data = root / "linked-data"
    try:
        data.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError):
        mcp_paths.open_mcp_workspace_file("linked-data/secret.jsonl", root)


@pytest.mark.parametrize("fallback", [False, True], ids=["descriptor-relative", "portable-fallback"])
def test_open_workspace_file_rejects_final_component_symlink(tmp_path, monkeypatch, fallback):
    root = _workspace(tmp_path, monkeypatch)
    _select_path_mode(monkeypatch, fallback=fallback)
    secret = tmp_path / "secret.jsonl"
    secret.write_text("outside", encoding="utf-8")
    link = root / "data" / "secret.jsonl"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    with pytest.raises(ValueError):
        mcp_paths.open_mcp_workspace_file("data/secret.jsonl", root)


@pytest.mark.parametrize("fallback", [False, True], ids=["descriptor-relative", "portable-fallback"])
def test_output_parent_reports_create_only_state_and_rejects_symlinks(tmp_path, monkeypatch, fallback):
    root = _workspace(tmp_path, monkeypatch)
    _select_path_mode(monkeypatch, fallback=fallback)
    output_dir = root / "exports"
    output_dir.mkdir()

    path, parent_fd, name, available = mcp_paths.open_mcp_workspace_output_parent(
        "exports/report.jsonl", root,
    )
    try:
        assert path == output_dir / "report.jsonl"
        assert name == "report.jsonl"
        assert available is True
        assert (parent_fd is None) is fallback
    finally:
        if parent_fd is not None:
            os.close(parent_fd)

    (output_dir / "report.jsonl").write_text("existing", encoding="utf-8")
    _path, existing_fd, _name, available = mcp_paths.open_mcp_workspace_output_parent(
        "exports/report.jsonl", root,
    )
    try:
        assert available is False
    finally:
        if existing_fd is not None:
            os.close(existing_fd)

    secret = tmp_path / "secret-output.jsonl"
    secret.write_text("outside", encoding="utf-8")
    link = output_dir / "linked.jsonl"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")
    with pytest.raises(ValueError):
        mcp_paths.open_mcp_workspace_output_parent("exports/linked.jsonl", root)




def test_descriptor_mode_uses_pinned_directory_scans_for_untrusted_components(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    _select_path_mode(monkeypatch, fallback=False)
    (root / "exports").mkdir()

    real_scandir = mcp_paths.os.scandir
    scanned_descriptors: list[int] = []

    def record_descriptor_scans(path="."):
        if isinstance(path, int):
            scanned_descriptors.append(path)
        return real_scandir(path)

    monkeypatch.setattr(mcp_paths.os, "scandir", record_descriptor_scans)

    resolved, stream = mcp_paths.open_mcp_workspace_file("data/sample.jsonl", root)
    with stream:
        assert stream.read() == b'{"text":"safe"}' + bytes((10,))
    assert resolved == root / "data" / "sample.jsonl"

    _path, parent_fd, name, available = mcp_paths.open_mcp_workspace_output_parent(
        "exports/report.jsonl", root,
    )
    try:
        assert parent_fd is not None
        assert name == "report.jsonl"
        assert available is True
    finally:
        if parent_fd is not None:
            os.close(parent_fd)

    # Relative components are resolved from pinned directory listings.
    assert len(scanned_descriptors) >= 3

def test_pinned_output_parent_survives_directory_swap_without_writing_outside_workspace(tmp_path, monkeypatch):
    import stat

    root = _workspace(tmp_path, monkeypatch)
    if not mcp_paths._descriptor_access_supported() or (
        os.link not in os.supports_dir_fd or os.unlink not in os.supports_dir_fd
    ):
        pytest.skip("Descriptor-relative output creation is unavailable")

    output_dir = root / "exports"
    output_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    parent_path, parent_fd, name, available = mcp_paths.open_mcp_workspace_output_parent(
        "exports/result.jsonl", root,
    )
    assert parent_fd is not None
    assert available is True
    try:
        output_dir.rename(root / "exports-original")
        try:
            output_dir.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Directory symlinks are unavailable: {exc}")

        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("workspace-only")

        assert (root / "exports-original" / name).read_text(encoding="utf-8") == "workspace-only"
        assert not (outside / name).exists()
        assert stat.S_ISDIR(os.fstat(parent_fd).st_mode)
    finally:
        os.close(parent_fd)


def test_descriptor_mode_uses_pinned_directory_scans_for_untrusted_components(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    _select_path_mode(monkeypatch, fallback=False)
    (root / "exports").mkdir()

    real_scandir = mcp_paths.os.scandir
    scanned_descriptors: list[int] = []

    def record_descriptor_scans(path="."):
        if isinstance(path, int):
            scanned_descriptors.append(path)
        return real_scandir(path)

    monkeypatch.setattr(mcp_paths.os, "scandir", record_descriptor_scans)

    resolved, stream = mcp_paths.open_mcp_workspace_file("data/sample.jsonl", root)
    with stream:
        assert stream.read() == b'{"text":"safe"}' + bytes((10,))
    assert resolved == root / "data" / "sample.jsonl"

    _path, parent_fd, name, available = mcp_paths.open_mcp_workspace_output_parent(
        "exports/report.jsonl", root,
    )
    try:
        assert parent_fd is not None
        assert name == "report.jsonl"
        assert available is True
    finally:
        if parent_fd is not None:
            os.close(parent_fd)

    # Relative components are resolved from pinned directory listings.
    assert len(scanned_descriptors) >= 3



def test_mcp_file_opener_validates_workspace_before_opening_regular_file(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    _select_path_mode(monkeypatch, fallback=True)

    resolved, stream = mcp_paths._open_mcp_workspace_file("data/sample.jsonl", root)
    with stream:
        assert stream.read() == b'{"text":"safe"}' + bytes((10,))
    assert resolved == (root / "data" / "sample.jsonl").resolve()
    assert str(resolved).startswith(str(root.resolve()).rstrip(os.sep) + os.sep)

    outside = tmp_path / "outside.jsonl"
    outside.write_text("private", encoding="utf-8")
    escape_dir = root / "escape-dir"
    try:
        escape_dir.symlink_to(outside.parent, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    with pytest.raises(ValueError):
        mcp_paths._open_mcp_workspace_file("escape-dir/outside.jsonl", root)

    link = root / "data" / "linked.jsonl"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")
    with pytest.raises(ValueError):
        mcp_paths._open_mcp_workspace_file("data/linked.jsonl", root)

    with pytest.raises(ValueError):
        mcp_paths._open_mcp_workspace_file("data", root)

    # A sibling prefix is not a child of the workspace: the separator in
    # the containment prefix prevents /workspace-evil from matching /workspace.
    sibling = tmp_path / "workspace-evil"
    sibling.mkdir()
    (sibling / "outside.jsonl").write_text("private sibling", encoding="utf-8")
    monkeypatch.setattr(
        mcp_paths.path_safety,
        "resolve_workspace_output_path",
        lambda _raw, _workspace: sibling / "outside.jsonl",
    )
    with pytest.raises(ValueError):
        mcp_paths._open_mcp_workspace_file("data/outside.jsonl", root)

