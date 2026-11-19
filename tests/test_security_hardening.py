from __future__ import annotations

from pathlib import Path

import pytest

from axiom.daemon import _RequestHandler
from axiom.models.inspector import inspect_model


def test_response_header_rejects_crlf_injection() -> None:
    handler = object.__new__(_RequestHandler)
    for value in ("safe\r\nInjected: yes", "safe\nInjected: yes", "safe\rInjected: yes"):
        with pytest.raises(ValueError, match="CR/LF"):
            handler.send_header("X-Test", value)
    with pytest.raises(ValueError, match="CR/LF"):
        handler.send_response(400, "Bad request\r\nInjected: yes")


def test_model_inspection_rejects_symlinked_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside_config = tmp_path / "outside-config.json"
    outside_config.write_text('{"model_type": "external"}', encoding="utf-8")
    config = model_dir / "config.json"
    try:
        config.symlink_to(outside_config)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks are unavailable on this platform: {exc}")

    with pytest.raises(ValueError, match="symbolic link"):
        inspect_model(str(model_dir))
