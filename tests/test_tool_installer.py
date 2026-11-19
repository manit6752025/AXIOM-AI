from __future__ import annotations

from pathlib import Path

import pytest

from axiom.tools.awake import (
    KeepAwakeSession,
    build_awake_command,
)
from axiom.tools.installer import (
    Installer,
    InstallerSafetyError,
    SecretInputError,
    UnsafeProfileError,
    build_install_plan,
    deduplicate_path_entries,
    detect_availability,
    merge_path_entries,
    update_profile_path,
    validate_profile_target,
)


def _which_for(*available: str):
    available_set = set(available)

    def which(command: str) -> str | None:
        return f"/fake/bin/{command}" if command in available_set else None

    return which


def test_install_defaults_to_a_dry_run_and_uses_argv_only():
    calls: list[tuple[str, ...]] = []

    def runner(argv, **kwargs):
        assert kwargs["shell"] is False
        calls.append(tuple(argv))

    installer = Installer(
        package_name="demo-tool",
        system="linux",
        which=_which_for("npm"),
        runner=runner,
        root_check=lambda: False,
    )

    plan = installer.install()

    assert plan.dry_run is True
    assert plan.manager == "npm"
    assert plan.commands == (("npm", "install", "--global", "demo-tool"),)
    assert calls == []
    assert all(isinstance(token, str) for token in plan.commands[0])

    executed = installer.install(dry_run=False)
    assert executed.dry_run is False
    assert calls == [("npm", "install", "--global", "demo-tool")]


def test_detection_covers_all_supported_installer_commands():
    availability = detect_availability(which=_which_for("npm", "curl"))

    assert availability == {
        "npm": True,
        "brew": False,
        "winget": False,
        "choco": False,
        "curl": True,
    }


def test_root_execution_is_refused_without_calling_runner():
    calls: list[tuple[str, ...]] = []
    installer = Installer(
        package_name="demo-tool",
        system="linux",
        which=_which_for("brew"),
        runner=lambda argv, **kwargs: calls.append(tuple(argv)),
        root_check=lambda: True,
    )

    with pytest.raises(InstallerSafetyError, match="root"):
        installer.install(dry_run=False)
    assert calls == []


def test_api_keys_are_rejected_and_never_persisted(tmp_path: Path):
    with pytest.raises(SecretInputError):
        build_install_plan({"package": "demo-tool", "api_key": "sk-test-secret"})

    profile = tmp_path / ".zshrc"
    original = 'export OPENAI_API_KEY="do-not-touch"\n'
    profile.write_text(original, encoding="utf-8")

    with pytest.raises(SecretInputError):
        update_profile_path(profile, ["/tmp/demo-bin"], home_dir=tmp_path)
    assert profile.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("system", "available", "executable"),
    [
        ("Darwin", ("caffeinate",), "caffeinate"),
        ("Linux", ("systemd-inhibit",), "systemd-inhibit"),
        ("Windows", ("powershell",), "powershell"),
    ],
)
def test_platform_command_selection(
    system: str, available: tuple[str, ...], executable: str
):
    selection = build_awake_command(
        system,
        duration=12.5,
        which=_which_for(*available),
    )

    assert selection.available is True
    assert selection.argv[0] == executable
    if system == "Windows":
        assert "13" in selection.argv[-1]
    else:
        assert "13" in selection.argv
    if system == "Windows":
        assert "SetThreadExecutionState" in selection.argv[-1]


def test_linux_session_reports_missing_systemd_inhibit():
    selection = build_awake_command(
        "Linux",
        which=_which_for(),
    )

    assert selection.available is False
    assert selection.argv == ()
    assert "systemd-inhibit" in " ".join(selection.instructions)


def test_path_entries_are_deduplicated_and_profile_updates_are_idempotent(
    tmp_path: Path,
):
    assert deduplicate_path_entries(["/one", "/one", "/two"]) == (
        "/one",
        "/two",
    )
    assert merge_path_entries("/one:/one", ["/two", "/one"]) == (
        "/one",
        "/two",
    )

    profile = tmp_path / ".zshrc"
    profile.write_text('export PATH="/one:/one:$PATH"\n', encoding="utf-8")

    first = update_profile_path(
        profile,
        ["/one", "/two", "/two"],
        home_dir=tmp_path,
    )
    second = update_profile_path(
        profile,
        ["/one", "/two"],
        home_dir=tmp_path,
    )

    content = profile.read_text(encoding="utf-8")
    assert first.added == ("/two",)
    assert second.added == ()
    assert content.count("/one") == 1
    assert content.count("/two") == 1


def test_unsafe_profile_targets_are_refused(tmp_path: Path):
    with pytest.raises(UnsafeProfileError):
        validate_profile_target("/etc/profile", home_dir=tmp_path)
    with pytest.raises(UnsafeProfileError):
        validate_profile_target(tmp_path / "settings.txt", home_dir=tmp_path)
    with pytest.raises(UnsafeProfileError):
        validate_profile_target(tmp_path, home_dir=tmp_path)


class _FakeProcess:
    def __init__(self) -> None:
        self.pid = 1234
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        return self.returncode or 0


class _FakeRunner:
    def __init__(self) -> None:
        self.process = _FakeProcess()
        self.started: list[tuple[str, ...]] = []

    def which(self, executable: str) -> str | None:
        if executable == "systemd-inhibit":
            return "/usr/bin/systemd-inhibit"
        return None

    def popen(self, argv):
        self.started.append(tuple(argv))
        return self.process


def test_keep_awake_child_is_cleaned_up_on_context_exit():
    runner = _FakeRunner()

    with KeepAwakeSession(
        duration=30,
        platform_name="Linux",
        runner=runner,
    ) as session:
        assert session.is_running is True
        assert runner.started[0][:2] == (
            "systemd-inhibit",
            "--what=idle:sleep",
        )

    assert runner.process.terminated is True
    assert runner.process.killed is False
    assert runner.process.poll() == 0
