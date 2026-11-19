from __future__ import annotations

from axiom.core import runtime_discovery


def test_runtime_discovery_is_passive_and_does_not_reveal_executable_paths(monkeypatch) -> None:
    monkeypatch.setattr(runtime_discovery.shutil, "which", lambda command: "/secret/local/path/ollama" if command == "ollama" else None)
    monkeypatch.setattr(runtime_discovery, "_package_present", lambda _name: False)
    result = runtime_discovery.discover_runtime_connectors()
    ollama = next(item for item in result["connectors"] if item["id"] == "ollama")
    assert result["schema_version"] == 1
    assert result["discovery_mode"] == "passive_local_discovery"
    assert ollama["present"] is True
    assert ollama["status"] == "discovered_not_connected"
    assert ollama["connection_checked"] is False
    assert "/secret/local/path" not in str(result)
    assert "not a connection" in result["scope_note"]


def test_runtime_discovery_handles_nothing_detected(monkeypatch) -> None:
    monkeypatch.setattr(runtime_discovery.shutil, "which", lambda _command: None)
    monkeypatch.setattr(runtime_discovery, "_package_present", lambda _name: False)
    result = runtime_discovery.discover_runtime_connectors()
    assert result["connectors"]
    assert all(item["present"] is False for item in result["connectors"])
    assert all(item["status"] == "not_detected" for item in result["connectors"])
