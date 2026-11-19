import pytest
import re


@pytest.fixture(autouse=True)
def _plain_cli_output(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TERMINAL_WIDTH", "200")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PY_COLORS", raising=False)
    try:
        import typer.rich_utils as rich_utils

        monkeypatch.setattr(rich_utils, "MAX_WIDTH", 200, raising=False)
        monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", False, raising=False)
    except ImportError:
        pass


@pytest.fixture
def plain():
    def _plain(text: str) -> str:
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        text = re.sub(r"[│╭╮╰╯─┃┏┓┗┛━]", " ", text)
        return " ".join(text.split())

    return _plain
