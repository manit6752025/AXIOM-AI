from pathlib import Path, PureWindowsPath

DEFAULT_CONFIG = """\
project:
  name: {name}
  version: 0.1.0

model:
  base: null
  source: null

dataset:
  source: null
  format: jsonl

training:
  method: null
  epochs: 1
  learning_rate: null

evaluation:
  enabled: true

deployment:
  target: local
"""


def create_project(name: str, directory: Path | None = None) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Project name cannot be empty.")

    if name != name.strip():
        raise ValueError("Project name cannot start or end with whitespace.")

    if (
        name in {".", ".."}
        or any(ord(character) < 32 for character in name)
        or any(character in name for character in '/\\\x00<>:"|?*')
    ):
        raise ValueError("Project name must be a single safe directory name.")

    windows_name = PureWindowsPath(name)
    if windows_name.anchor or name.rstrip(" .") != name:
        raise ValueError("Project name must be a single safe directory name.")

    root = (directory or Path.cwd()) / name

    if root.exists():
        raise FileExistsError(f"Project already exists: {root}")

    directories = [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "models",
        root / "experiments",
        root / "evaluations",
        root / "outputs",
    ]

    for directory_path in directories:
        directory_path.mkdir(parents=True, exist_ok=True)

    (root / "axiom.yaml").write_text(
        DEFAULT_CONFIG.format(name=name),
        encoding="utf-8",
    )

    (root / "README.md").write_text(
        f"# {name}\n\nAI project created with AXIOM.\n",
        encoding="utf-8",
    )

    return root
