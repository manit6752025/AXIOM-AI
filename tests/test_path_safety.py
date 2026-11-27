import pytest

from axiom.path_safety import SafePathError, find_existing_relative_path, open_regular_file, resolve_workspace_output_path


def test_resolves_existing_relative_file(tmp_path):
    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    expected = root / "data" / "train.jsonl"
    expected.write_text('{"text":"hello"}\n', encoding="utf-8")

    assert find_existing_relative_path(root, "data/train.jsonl") == expected.resolve()


@pytest.mark.parametrize(
    "raw",
    [
        "/etc/passwd",
        "../outside",
        "data/../../etc/passwd",
        "data//train.jsonl",
        r"data\train.jsonl",
    ],
)
def test_rejects_unsafe_relative_paths(tmp_path, raw):
    root = tmp_path / "workspace"
    root.mkdir()

    with pytest.raises(SafePathError):
        find_existing_relative_path(root, raw)


def test_rejects_symlinked_file(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not for serving", encoding="utf-8")
    link = root / "leak.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks are not available on this platform")

    with pytest.raises(SafePathError):
        find_existing_relative_path(root, "leak.txt", allow_directories=False)


def test_rejects_symlinked_workspace_root(tmp_path):
    root = tmp_path / "real-workspace"
    root.mkdir()
    (root / "inside.txt").write_text("private", encoding="utf-8")
    alias = tmp_path / "workspace-link"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are not available: {exc}")

    with pytest.raises(SafePathError, match="workspace root"):
        find_existing_relative_path(alias, "inside.txt")


def test_symlinked_intermediate_directory_is_never_walked(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private", encoding="utf-8")
    alias = root / "external"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are not available: {exc}")

    with pytest.raises((SafePathError, FileNotFoundError)):
        find_existing_relative_path(root, "external/secret.txt", allow_directories=False)


def test_open_regular_file_rejects_final_symlink(tmp_path):
    if not getattr(__import__("os"), "O_NOFOLLOW", 0):
        pytest.skip("Final-component O_NOFOLLOW is unavailable on this platform")
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("private", encoding="utf-8")
    link = root / "secret.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks are not available: {exc}")

    with pytest.raises((OSError, SafePathError)):
        open_regular_file(link)


def test_open_regular_file_rejects_symlinked_parent_directory(tmp_path):
    import os

    if os.name != "posix" or not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        pytest.skip("POSIX O_DIRECTORY/O_NOFOLLOW traversal is unavailable")
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    alias = root / "parent"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are not available: {exc}")

    with pytest.raises((OSError, SafePathError)):
        open_regular_file(alias / "secret.txt")


def test_open_regular_file_rejects_parent_components(tmp_path):
    import os

    target = tmp_path / "secret.txt"
    target.write_text("private", encoding="utf-8")
    with pytest.raises(SafePathError, match="parent-directory"):
        open_regular_file(target.parent / ".." / target.name)


def test_rejects_parent_symlink_swapped_after_path_selection(tmp_path):
    import os

    if os.name != "posix" or not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        pytest.skip("POSIX O_DIRECTORY/O_NOFOLLOW traversal is unavailable")
    root = tmp_path / "workspace"
    data = root / "data"
    data.mkdir(parents=True)
    selected_file = data / "record.jsonl"
    selected_file.write_text("workspace data", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "record.jsonl").write_text("outside secret", encoding="utf-8")

    selected = find_existing_relative_path(root, "data/record.jsonl", allow_directories=False)
    data.rename(root / "data-original")
    try:
        data.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are not available: {exc}")

    with pytest.raises((OSError, SafePathError)):
        open_regular_file(selected)



def test_open_regular_file_uses_only_enumerated_names_for_descriptor_opens(tmp_path, monkeypatch):
    import os

    if os.name != "posix" or not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        pytest.skip("POSIX descriptor-relative open is unavailable")
    root = tmp_path / "workspace"
    nested = root / "data"
    nested.mkdir(parents=True)
    target = nested / "record.jsonl"
    target.write_text('{"text":"safe"}\\n', encoding="utf-8")

    real_open = os.open
    descriptor_names = []

    def record_open(path, flags, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            descriptor_names.append(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", record_open)
    monkeypatch.setattr(
        os,
        "supports_dir_fd",
        frozenset((*os.supports_dir_fd, record_open)),
    )
    with open_regular_file(target) as stream:
        assert stream.read() == b'{"text":"safe"}\\n'

    assert descriptor_names[-2:] == ["data", "record.jsonl"]
    assert all(
        isinstance(name, str) and "/" not in name and "\\" not in name
        for name in descriptor_names
    )



def test_codeql_models_the_directory_entry_selector_as_a_path_barrier():
    from pathlib import Path

    model_file = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "codeql"
        / "extensions"
        / "axiom-model-paths"
        / "models"
        / "model-paths.yml"
    )
    model = model_file.read_text(encoding="utf-8")
    assert '["axiom.path_safety", "Member[_selected_entry].ReturnValue", "path-injection"]' in model
    assert '["axiom.path_safety", "Member[find_existing_relative_path].ReturnValue", "path-injection"]' in model


def test_codeql_sarif_workflow_uses_the_checked_in_local_model_pack():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    pack = yaml.safe_load(
        (root / ".github/codeql/extensions/axiom-model-paths/codeql-pack.yml").read_text(encoding="utf-8")
    )
    assert pack["name"] == "axiom/axiom-model-paths"
    assert pack["extensionTargets"]["codeql/python-all"] == "*"

    model = (root / ".github/codeql/extensions/axiom-model-paths/models/model-paths.yml").read_text(encoding="utf-8")
    assert "Member[_selected_entry].ReturnValue" in model
    assert "Member[resolve_workspace_output_path].ReturnValue" in model

    workflow = yaml.load(
        (root / ".github/workflows/codeql-sarif-verification.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    init = next(
        step for step in workflow["jobs"]["sarif"]["steps"]
        if step.get("name") == "Initialize CodeQL for all supported languages"
    )
    assert "config-file" not in init.get("with", {})
    assert "packs" not in init.get("with", {})
    assert "implement/roadmap" in workflow["on"]["push"]["branches"]


def test_workspace_output_resolver_rejects_outside_directory_before_scanning_it(tmp_path, monkeypatch):
    from pathlib import Path
    from contextlib import contextmanager

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "workspace-sibling"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside", encoding="utf-8")
    scanned = []

    class OutsideEntry:
        name = "escaped"
        path = str(outside)

        def stat(self, *, follow_symlinks=True):
            return outside.stat() if follow_symlinks else outside.lstat()

        def is_symlink(self):
            return False

    @contextmanager
    def fake_scandir(path):
        selected_dir = Path(path)
        scanned.append(selected_dir)
        if selected_dir == workspace.resolve():
            yield iter((OutsideEntry(),))
        else:
            raise AssertionError("The resolver scanned outside the trusted workspace.")

    monkeypatch.setattr("axiom.path_safety.os.scandir", fake_scandir)

    with pytest.raises(SafePathError, match="escaped the trusted root"):
        resolve_workspace_output_path("escaped/secret.txt", workspace)

    assert scanned == [workspace.resolve()]
