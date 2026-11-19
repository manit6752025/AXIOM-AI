"""Acceptance tests for the AXIOM developer-tools safety contract.

These tests target the installer and keep-awake boundaries documented in
``docs/developer-tools-contract.md``. All operating-system, network, secret,
and process behavior is injected so the tests remain deterministic.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


def _load(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"The developer-tools contract is not implemented: expected {module_name}",
            pytrace=False,
        )
        raise AssertionError from exc


@pytest.fixture(scope="module")
def installer() -> Any:
    return _load("axiom.tools.installer")


@pytest.fixture(scope="module")
def awake() -> Any:
    return _load("axiom.tools.awake")


def test_availability_does_not_claim_authentication(installer: Any):
    def which(command: str) -> str | None:
        return f"/usr/local/bin/{command}" if command == "npm" else None

    availability = installer.detect_availability(which=which)
    target = installer.InstallTarget(
        name="demo-agent",
        package="demo-agent",
        auth_instructions=(
            "Use DEMO_AGENT_API_KEY from the environment or keychain service com.example.demo-agent.",
        ),
    )
    plan = installer.build_install_plan(
        target,
        system="linux",
        which=which,
        dry_run=True,
    )

    assert availability["npm"] is True
    assert availability["curl"] is False
    assert plan.availability["npm"] is True
    assert "authenticated" not in plan.as_dict()
    auth_text = "\n".join(plan.auth_instructions).lower()
    assert "never accepts" in auth_text
    assert "login" in auth_text or "outside" in auth_text


def test_api_keys_are_references_and_never_serialized(installer: Any):
    secret = "demo-secret-that-must-not-escape"

    with pytest.raises(installer.SecretInputError) as error:
        installer.build_install_plan(api_key=secret)
    assert secret not in str(error.value)

    target = installer.InstallTarget(
        package="demo-agent",
        auth_instructions=(
            "Read DEMO_AGENT_API_KEY from the environment; the keychain service is com.example.demo-agent.",
        ),
    )
    plan = installer.build_install_plan(
        target,
        system="linux",
        which=lambda command: "/usr/local/bin/npm" if command == "npm" else None,
    )
    serialized = json.dumps(plan.as_dict(), sort_keys=True)

    assert "DEMO_AGENT_API_KEY" in serialized
    assert "com.example.demo-agent" in serialized
    assert secret not in serialized


def test_remote_install_is_reviewable_and_requires_explicit_opt_in(
    installer: Any,
    tmp_path: Path,
):
    target = installer.InstallTarget(
        name="demo-agent",
        package="demo-agent",
        package_manager="curl",
        download_url="https://vendor.example/demo-agent",
        download_path=str(tmp_path / "demo-agent"),
    )

    preview = installer.build_install_plan(
        target,
        system="linux",
        which=lambda command: "/usr/bin/curl" if command == "curl" else None,
        dry_run=True,
    )
    executions: list[tuple[list[str], dict[str, Any]]] = []

    def runner(command: list[str], **kwargs: Any) -> None:
        executions.append((command, kwargs))

    assert preview.dry_run is True
    assert preview.command is not None
    assert preview.command[0] == "curl"
    assert "https://vendor.example/demo-agent" in preview.command
    assert "--output" in preview.command
    assert preview.as_dict()["commands"] == [list(preview.command)]

    with pytest.raises(installer.InstallerSafetyError):
        installer.execute_install_plan(
            preview,
            runner=runner,
            root_check=lambda: False,
        )
    assert executions == []

    approved = installer.build_install_plan(
        target,
        system="linux",
        which=lambda command: "/usr/bin/curl" if command == "curl" else None,
        dry_run=False,
    )
    installer.execute_install_plan(
        approved,
        runner=runner,
        root_check=lambda: False,
    )
    assert executions == [(list(approved.command), {"check": True, "shell": False})]


def test_unsupported_os_and_package_manager_return_guidance(installer: Any):
    with pytest.raises(ValueError) as error:
        installer.build_install_plan(
            system="haiku",
            package_manager="apt",
            which=lambda command: None,
        )

    guidance = str(error.value).lower()
    assert "unsupported" in guidance or "not supported" in guidance
    assert "haiku" in guidance
    assert "apt" in guidance
    assert any(word in guidance for word in ("manual", "vendor", "choose", "supported"))


def test_path_update_is_user_scoped_idempotent_and_reversible(
    installer: Any,
    tmp_path: Path,
):
    home = tmp_path / "home"
    home.mkdir()
    profile = home / ".profile"
    original = 'export PATH="/usr/bin:/bin"\n# user content\n'
    profile.write_text(original, encoding="utf-8")
    tool_bin = home / ".local" / "bin"
    tool_bin.mkdir(parents=True)

    change = installer.update_profile_path(
        profile,
        str(tool_bin),
        home_dir=home,
        shell="sh",
        dry_run=False,
    )
    assert change.path == profile
    assert change.changed is True
    assert profile.resolve().is_relative_to(home.resolve())
    assert str(tool_bin) in profile.read_text(encoding="utf-8")
    assert "sudo" not in profile.read_text(encoding="utf-8").lower()

    second = installer.update_profile_path(
        profile,
        str(tool_bin),
        home_dir=home,
        shell="sh",
        dry_run=False,
    )
    assert second.changed is False
    assert callable(getattr(change, "revert", None))
    change.revert()
    assert profile.read_text(encoding="utf-8") == original


class _FakeAwakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.running = True
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminated = True
        self.running = False

    def kill(self) -> None:
        self.killed = True
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


class _FakeAwakeRunner:
    def __init__(self) -> None:
        self.process = _FakeAwakeProcess()
        self.started: list[tuple[str, ...]] = []

    def which(self, executable: str) -> str | None:
        return "/usr/bin/systemd-inhibit" if executable == "systemd-inhibit" else None

    def popen(self, argv: tuple[str, ...]) -> _FakeAwakeProcess:
        self.started.append(argv)
        return self.process


@pytest.mark.parametrize("fail", [False, True])
def test_keep_awake_restores_child_on_every_exit(awake: Any, fail: bool):
    runner = _FakeAwakeRunner()
    session = awake.KeepAwakeSession(
        duration=5,
        system="linux",
        reason="acceptance test",
        runner=runner,
    )

    assert session.available is True
    if fail:
        with pytest.raises(RuntimeError, match="child failed"), session:
            raise RuntimeError("child failed")
    else:
        with session:
            assert session.is_running is True

    assert runner.started
    assert runner.process.terminated is True
    assert runner.process.killed is False
    assert session.process is None
