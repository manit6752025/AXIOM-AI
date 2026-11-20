import pytest

from axiom.path_safety import SafePathError, find_existing_relative_path, open_regular_file


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

    with pytest.raises(OSError):
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

    with pytest.raises(OSError):
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

    with pytest.raises(OSError):
        open_regular_file(selected)
