import pytest

from axiom.path_safety import SafePathError, find_existing_relative_path


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
