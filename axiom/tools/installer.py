"""Safety-first, cross-platform installation planning.

The installer in this module is deliberately a planner first.  It produces
argv sequences and instructions by default; it never evaluates a shell string,
collects credentials, or silently escalates privileges.
"""

from __future__ import annotations

import inspect
import os
import platform as platform_module
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

INSTALLER_COMMANDS = ("npm", "brew", "winget", "choco", "curl")

COMMAND_ALLOWLIST: dict[str, frozenset[str]] = {
    "npm": frozenset({"install", "prefix"}),
    "brew": frozenset({"install", "--prefix"}),
    "winget": frozenset({"install"}),
    "choco": frozenset({"install"}),
    "curl": frozenset({"--fail", "--silent", "--show-error", "--location", "--output"}),
}
ALLOWED_COMMANDS = COMMAND_ALLOWLIST

_MANAGER_ORDER = {
    "darwin": ("brew", "npm", "curl"),
    "linux": ("brew", "npm", "curl"),
    "windows": ("winget", "choco", "npm", "curl"),
    "other": ("npm", "curl"),
}
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9@._+:/=-]+$")
_SHELL_META = frozenset('$;|&<>`\n\r\x00"')
_SECRET_FIELD_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|token|secret|password|credential)s?(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:api[_-]?key|access[_-]?token|token|secret|password|private[_-]?key)\s*=",
    re.IGNORECASE,
)
_KNOWN_SECRET_RE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{8,}\b|\bgh[pousr]_[A-Za-z0-9_-]{8,}\b)",
    re.IGNORECASE,
)
_PROFILE_NAMES = frozenset(
    {
        ".bash_profile",
        ".bashrc",
        ".profile",
        ".zprofile",
        ".zshrc",
        "config.fish",
        "profile",
        "profile.ps1",
        "Microsoft.PowerShell_profile.ps1",
    }
)


class InstallerSafetyError(RuntimeError):
    """Raised when an install would violate an explicit safety boundary."""


class SecretInputError(ValueError):
    """Raised when credentials are supplied to an installer API."""


class UnsafeCommandError(ValueError):
    """Raised for commands that are not in the installer allowlist."""


class UnsafeProfileError(ValueError):
    """Raised when a profile target is not a safe user profile file."""


def _normalise_platform(system: str | None) -> str:
    value = (system or platform_module.system()).strip().lower()
    if value in {"darwin", "mac", "macos", "osx"}:
        return "darwin"
    if value in {"windows", "win32", "win"}:
        return "windows"
    if value in {"linux", "gnu/linux"}:
        return "linux"
    return "other"


def _reject_secret_fields(value: object, *, context: str = "input") -> None:
    """Reject credential-shaped fields without inspecting or storing values."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).strip().lower().replace(" ", "_")
            if _SECRET_FIELD_RE.search(key_text):
                raise SecretInputError(
                    f"{context} cannot accept API keys or other credentials."
                )
            _reject_secret_fields(nested, context=context)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for nested in value:
            _reject_secret_fields(nested, context=context)


def _reject_secret_text(value: str, *, context: str) -> None:
    if _SECRET_ASSIGNMENT_RE.search(value) or _KNOWN_SECRET_RE.search(value):
        raise SecretInputError(
            f"{context} cannot contain API keys or other credentials."
        )


def _which(which: Callable[[str], str | None], command: str) -> str | None:
    try:
        result = which(command)
    except (OSError, TypeError):
        return None
    if isinstance(result, str):
        return result or None
    if result:
        return command
    return None


def detect_command_paths(
    *,
    which: Callable[[str], str | None] = shutil.which,
    commands: Iterable[str] = INSTALLER_COMMANDS,
) -> dict[str, str | None]:
    """Return resolved executable paths for the supported tool commands."""

    result: dict[str, str | None] = {}
    for command in commands:
        if command not in INSTALLER_COMMANDS:
            raise ValueError(f"Unsupported installer command: {command}")
        result[command] = _which(which, command)
    return result


def detect_availability(
    *,
    which: Callable[[str], str | None] = shutil.which,
    commands: Iterable[str] = INSTALLER_COMMANDS,
) -> dict[str, bool]:
    """Return a boolean availability map for npm, brew, winget, choco, curl."""

    return {
        command: bool(path)
        for command, path in detect_command_paths(
            which=which,
            commands=commands,
        ).items()
    }


def detect_tools(
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, bool]:
    """Compatibility alias for callers that call package managers "tools"."""

    return detect_availability(which=which)


def _safe_package(value: object, *, label: str = "package") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    result = value.strip()
    _reject_secret_text(result, context=label)
    if any(char in _SHELL_META for char in result) or not _PACKAGE_RE.fullmatch(result):
        raise ValueError(f"{label} contains unsupported command characters")
    return result


def _safe_path_value(value: object, *, label: str = "PATH entry") -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    result = value.strip()
    if any(char in _SHELL_META for char in result):
        raise ValueError(f"{label} contains shell metacharacters")
    _reject_secret_text(result, context=label)
    return result


def _safe_https_url(value: object, *, label: str = "download URL") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty HTTPS URL")
    result = value.strip()
    _reject_secret_text(result, context=label)
    parsed = urlsplit(result)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"{label} must use HTTPS")
    if parsed.username or parsed.password:
        raise SecretInputError(f"{label} cannot contain embedded credentials")
    query_keys = {
        key.lower() for key in re.findall(r"(?:^|&)\s*([^=&]+)", parsed.query)
    }
    if query_keys & {"key", "api_key", "apikey", "token", "secret", "password"}:
        raise SecretInputError(f"{label} cannot contain credential query parameters")
    return result


@dataclass(frozen=True)
class InstallTarget:
    """The non-secret, package-facing portion of an install request."""

    name: str = "AXIOM"
    package: str = "axiom-ai"
    package_manager: str | None = None
    download_url: str | None = None
    download_path: str | None = None
    path_entries: tuple[str, ...] = ()
    auth_instructions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("target name must be a non-empty string")
        _reject_secret_text(self.name, context="target name")
        _safe_package(self.package)

        if self.package_manager is not None:
            manager = self.package_manager.strip().lower()
            if manager not in INSTALLER_COMMANDS:
                raise ValueError(f"Unsupported package manager: {manager}")
            object.__setattr__(self, "package_manager", manager)

        if self.download_url is not None:
            object.__setattr__(
                self,
                "download_url",
                _safe_https_url(self.download_url),
            )
        if self.download_path is not None:
            object.__setattr__(
                self,
                "download_path",
                _safe_path_value(self.download_path, label="download path"),
            )

        raw_entries = (
            (self.path_entries,)
            if isinstance(self.path_entries, (str, os.PathLike))
            else self.path_entries
        )
        entries = tuple(_safe_path_value(entry) for entry in raw_entries)
        object.__setattr__(self, "path_entries", entries)

        instructions = tuple(self.auth_instructions)
        if not all(isinstance(item, str) for item in instructions):
            raise ValueError("auth_instructions must contain strings")
        for instruction in instructions:
            _reject_secret_text(instruction, context="auth instructions")
        object.__setattr__(self, "auth_instructions", instructions)


def _target_from_input(
    target: InstallTarget | Mapping[str, Any] | str | Any | None,
) -> InstallTarget:
    if target is None:
        return InstallTarget()
    if isinstance(target, InstallTarget):
        return target
    if isinstance(target, str):
        return InstallTarget(package=_safe_package(target))

    if isinstance(target, Mapping):
        _reject_secret_fields(target, context="target")
        data = target
        name = data.get("name", "AXIOM")
        package = data.get(
            "package", data.get("package_name", data.get("id", "axiom-ai"))
        )
        manager = data.get("package_manager", data.get("manager"))
        path_entries = data.get("path_entries", data.get("paths", ()))
        if isinstance(path_entries, str):
            path_entries = (path_entries,)
        instructions = data.get("auth_instructions", ())
        if isinstance(instructions, str):
            instructions = (instructions,)
        return InstallTarget(
            name=name,
            package=package,
            package_manager=manager,
            download_url=data.get("download_url", data.get("url")),
            download_path=data.get("download_path"),
            path_entries=tuple(path_entries),
            auth_instructions=tuple(instructions),
        )

    values = getattr(target, "__dict__", {})
    if isinstance(values, Mapping):
        _reject_secret_fields(values, context="target")
    path_entries = getattr(target, "path_entries", getattr(target, "paths", ()))
    if isinstance(path_entries, str):
        path_entries = (path_entries,)
    instructions = getattr(target, "auth_instructions", ())
    if isinstance(instructions, str):
        instructions = (instructions,)
    return InstallTarget(
        name=getattr(target, "name", "AXIOM"),
        package=getattr(
            target,
            "package",
            getattr(target, "package_name", getattr(target, "id", "axiom-ai")),
        ),
        package_manager=getattr(
            target,
            "package_manager",
            getattr(target, "manager", None),
        ),
        download_url=getattr(
            target,
            "download_url",
            getattr(target, "url", None),
        ),
        download_path=getattr(target, "download_path", None),
        path_entries=tuple(path_entries),
        auth_instructions=tuple(instructions),
    )


def _manager_command(manager: str, target: InstallTarget) -> tuple[str, ...]:
    package = target.package
    if manager == "npm":
        return ("npm", "install", "--global", package)
    if manager == "brew":
        return ("brew", "install", package)
    if manager == "winget":
        return ("winget", "install", "--id", package, "--exact")
    if manager == "choco":
        return ("choco", "install", package, "--yes", "--no-progress")
    if manager == "curl":
        if target.download_url is None or target.download_path is None:
            raise ValueError(
                "curl installs require both download_url and download_path"
            )
        return (
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--output",
            target.download_path,
            target.download_url,
        )
    raise ValueError(f"Unsupported package manager: {manager}")


def validate_command(command: Sequence[str]) -> tuple[str, ...]:
    """Validate one explicit installer argv sequence against the allowlist."""

    if not command or not all(isinstance(item, str) and item for item in command):
        raise UnsafeCommandError("installer commands must be non-empty argv sequences")
    argv = tuple(command)
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if executable not in COMMAND_ALLOWLIST:
        raise UnsafeCommandError(f"command is not allowlisted: {executable}")
    if any(
        token.lower() in {"sudo", "doas", "pkexec"}
        or any(char in _SHELL_META for char in token)
        for token in argv
    ):
        raise UnsafeCommandError("shell syntax and privilege escalation are forbidden")

    allowed_words = COMMAND_ALLOWLIST[executable]
    if len(argv) < 2 or argv[1] not in allowed_words:
        raise UnsafeCommandError(f"command form is not allowlisted: {argv!r}")

    if executable == "npm" and argv[1] == "install":
        if len(argv) != 4 or argv[2] != "--global":
            raise UnsafeCommandError("npm installs must be explicit global installs")
        _safe_package(argv[3])
    if executable == "brew" and argv[1] == "install":
        if len(argv) != 3:
            raise UnsafeCommandError("brew install accepts one package")
        _safe_package(argv[2])
    if executable == "winget":
        if len(argv) != 5 or argv[1:3] != ("install", "--id"):
            raise UnsafeCommandError(
                "winget installs must use one explicit exact package id"
            )
        if argv[4] != "--exact":
            raise UnsafeCommandError("winget installs must use an exact package id")
        _safe_package(argv[3])
    if executable == "choco":
        if len(argv) != 5 or argv[3:] != ("--yes", "--no-progress"):
            raise UnsafeCommandError(
                "choco installs must be explicit confirmed installs"
            )
        _safe_package(argv[2])
    if executable == "curl":
        expected_options = (
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--output",
        )
        if len(argv) != 8 or argv[:6] != expected_options:
            raise UnsafeCommandError(
                "curl downloads must use the fixed fail-safe option set"
            )
        _safe_path_value(argv[6], label="curl output path")
        _safe_https_url(argv[7])

    return argv


@dataclass(frozen=True)
class InstallPlan:
    """A serialisable dry-run plan containing only explicit argv commands."""

    target: InstallTarget
    platform: str
    manager: str | None
    availability: Mapping[str, bool]
    commands: tuple[tuple[str, ...], ...]
    path_instructions: tuple[str, ...]
    auth_instructions: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    dry_run: bool = True

    @property
    def command(self) -> tuple[str, ...] | None:
        """Return the single planned command, when there is exactly one."""

        return self.commands[0] if len(self.commands) == 1 else None

    @property
    def argv(self) -> tuple[tuple[str, ...], ...]:
        return self.commands

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.name,
            "package": self.target.package,
            "platform": self.platform,
            "manager": self.manager,
            "availability": dict(self.availability),
            "commands": [list(command) for command in self.commands],
            "path_instructions": list(self.path_instructions),
            "auth_instructions": list(self.auth_instructions),
            "warnings": list(self.warnings),
            "dry_run": self.dry_run,
        }


def _path_instructions(manager: str | None, target: InstallTarget) -> tuple[str, ...]:
    instructions: list[str] = []
    if manager == "npm":
        instructions.append(
            "If npm's global bin directory is missing from PATH, run "
            "`npm prefix -g` and add its `bin` directory to your user PATH."
        )
    elif manager == "brew":
        instructions.append(
            "If Homebrew is not already on PATH, run `brew --prefix` and "
            "add that prefix's `bin` directory to your user PATH."
        )
    elif manager in {"winget", "choco"}:
        instructions.append(
            "Restart the terminal after installation so user PATH changes are visible."
        )
    elif manager == "curl":
        instructions.append(
            "Add the downloaded tool's user bin directory to PATH in your shell profile."
        )

    for entry in target.path_entries:
        instructions.append(f"Ensure this user PATH entry is present: {entry}")
    if not instructions:
        instructions.append(
            "If the installed command is not found, add its user-level bin directory "
            "to PATH and restart the terminal."
        )
    return tuple(instructions)


def _auth_instructions(manager: str | None, target: InstallTarget) -> tuple[str, ...]:
    instructions = [
        "AXIOM never accepts, stores, or writes API keys. Complete any package-manager login in its own interactive tool."
    ]
    if manager == "npm":
        instructions.append(
            "For a private npm package, run `npm login` yourself; no npm token is collected by this engine."
        )
    elif manager == "brew":
        instructions.append(
            "If Homebrew needs access to a private tap, authenticate with Homebrew outside AXIOM."
        )
    elif manager == "winget":
        instructions.append(
            "Review any WinGet source or account prompt interactively before accepting it."
        )
    elif manager == "choco":
        instructions.append(
            "Review any Chocolatey source or account prompt interactively before accepting it."
        )
    elif manager == "curl":
        instructions.append(
            "Review the HTTPS URL and downloaded artifact before running it; never append credentials to the URL."
        )
    instructions.extend(target.auth_instructions)
    return tuple(instructions)


def build_install_plan(
    target: InstallTarget | Mapping[str, Any] | str | Any | None = None,
    *,
    system: str | None = None,
    package_manager: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    dry_run: bool = True,
    profile: str | Path | None = None,
    home_dir: str | Path | None = None,
    api_key: None = None,
) -> InstallPlan:
    """Build an install plan; execution is opt-in and never the default."""

    if api_key is not None:
        raise SecretInputError("API keys are not accepted by the installer")
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be a boolean")

    target_value = _target_from_input(target)
    raw_system = (system or platform_module.system()).strip().lower()
    normalized_system = _normalise_platform(system)
    if system is not None and normalized_system == "other":
        requested = package_manager or target_value.package_manager or "auto"
        raise ValueError(
            f"Unsupported platform '{raw_system}' for package manager "
            f"'{requested}'. Choose a supported platform/manager combination "
            "or follow the manual/vendor installation instructions."
        )
    availability = detect_availability(which=which)

    requested_manager = package_manager or target_value.package_manager
    if requested_manager is not None:
        requested_manager = requested_manager.strip().lower()
        if requested_manager not in INSTALLER_COMMANDS:
            raise ValueError(
                f"Unsupported package manager '{requested_manager}' for platform "
                f"'{raw_system}'. Choose a supported manager or follow the "
                "manual/vendor installation instructions."
            )
        manager_order = (requested_manager,)
    else:
        manager_order = _MANAGER_ORDER[normalized_system]

    manager: str | None = None
    warnings: list[str] = []
    commands: tuple[tuple[str, ...], ...] = ()
    for candidate in manager_order:
        if not availability.get(candidate, False):
            continue
        manager = candidate
        try:
            command = validate_command(_manager_command(candidate, target_value))
        except ValueError as exc:
            warnings.append(str(exc))
            manager = None
            continue
        commands = (command,)
        break

    if requested_manager and not availability.get(requested_manager, False):
        warnings.append(
            f"{requested_manager} is not available; install it or choose an available manager."
        )
    elif manager is None:
        warnings.append(
            "No supported package manager can install this target with the detected tools."
        )

    if profile is not None:
        safe_profile = validate_profile_target(profile, home_dir=home_dir)
        warnings.append(
            f"PATH profile target validated for explicit update: {safe_profile}"
        )

    return InstallPlan(
        target=target_value,
        platform=normalized_system,
        manager=manager,
        availability=availability,
        commands=commands,
        path_instructions=_path_instructions(manager, target_value),
        auth_instructions=_auth_instructions(manager, target_value),
        warnings=tuple(warnings),
        dry_run=dry_run,
    )


def _default_root_check() -> bool:
    geteuid = getattr(os, "geteuid", None)
    return bool(geteuid is not None and geteuid() == 0)


def _invoke_run(runner: Callable[..., Any], command: tuple[str, ...]) -> Any:
    """Call a real runner with shell=False while accommodating small test doubles."""

    try:
        parameters = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        parameters = None

    accepts_kwargs = parameters is None or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    names = parameters if parameters is not None else {}

    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "check" in names:
        kwargs["check"] = True
    if accepts_kwargs or "shell" in names:
        kwargs["shell"] = False
    return runner(list(command), **kwargs)


def execute_install_plan(
    plan: InstallPlan,
    *,
    runner: Callable[..., Any] = subprocess.run,
    root_check: Callable[[], bool] = _default_root_check,
) -> InstallPlan:
    """Execute a non-dry plan using explicit argv and no privilege escalation."""

    if not isinstance(plan, InstallPlan):
        raise TypeError("plan must be an InstallPlan")
    if plan.dry_run:
        raise InstallerSafetyError(
            "Refusing to execute a dry-run plan; build the plan with dry_run=False."
        )
    if root_check():
        raise InstallerSafetyError(
            "Refusing to install as root. Run the user-level plan from your own account."
        )

    for command in plan.commands:
        validate_command(command)
        _invoke_run(runner, command)
    return plan


class Installer:
    """Small facade for detection, planning, and explicit execution."""

    def __init__(
        self,
        target: InstallTarget | Mapping[str, Any] | str | Any | None = None,
        *,
        package_name: str | None = None,
        system: str | None = None,
        which: Callable[[str], str | None] = shutil.which,
        runner: Callable[..., Any] = subprocess.run,
        root_check: Callable[[], bool] = _default_root_check,
        api_key: None = None,
    ) -> None:
        if api_key is not None:
            raise SecretInputError("API keys are not accepted by the installer")
        target_value = _target_from_input(target)
        if package_name is not None:
            target_value = InstallTarget(
                name=target_value.name,
                package=_safe_package(package_name),
                package_manager=target_value.package_manager,
                download_url=target_value.download_url,
                download_path=target_value.download_path,
                path_entries=target_value.path_entries,
                auth_instructions=target_value.auth_instructions,
            )
        self.target = target_value
        self.system = system
        self.which = which
        self.runner = runner
        self.root_check = root_check

    def detect(self) -> dict[str, bool]:
        return detect_availability(which=self.which)

    def plan(
        self,
        *,
        dry_run: bool = True,
        package_manager: str | None = None,
        profile: str | Path | None = None,
        home_dir: str | Path | None = None,
        api_key: None = None,
    ) -> InstallPlan:
        return build_install_plan(
            self.target,
            system=self.system,
            package_manager=package_manager,
            which=self.which,
            dry_run=dry_run,
            profile=profile,
            home_dir=home_dir,
            api_key=api_key,
        )

    def execute(self, plan: InstallPlan) -> InstallPlan:
        return execute_install_plan(
            plan,
            runner=self.runner,
            root_check=self.root_check,
        )

    def install(
        self,
        *,
        dry_run: bool = True,
        package_manager: str | None = None,
        profile: str | Path | None = None,
        home_dir: str | Path | None = None,
        api_key: None = None,
    ) -> InstallPlan:
        plan = self.plan(
            dry_run=dry_run,
            package_manager=package_manager,
            profile=profile,
            home_dir=home_dir,
            api_key=api_key,
        )
        if not dry_run:
            self.execute(plan)
        return plan

    run = install


def deduplicate_path_entries(
    entries: str | Iterable[str],
    *,
    path_separator: str | None = None,
    case_sensitive: bool | None = None,
) -> tuple[str, ...]:
    """Deduplicate PATH entries while preserving their first-seen spelling."""

    if isinstance(entries, (str, os.PathLike)):
        if isinstance(entries, os.PathLike):
            entries = os.fspath(entries)
        separator = path_separator or os.pathsep
        values: Iterable[str] = entries.split(separator)
    else:
        values = entries

    if case_sensitive is None:
        case_sensitive = os.name != "nt"
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        entry = _safe_path_value(value)
        key = os.path.normpath(os.path.expanduser(entry))
        if not case_sensitive:
            key = key.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return tuple(result)


def merge_path_entries(
    current: str | Iterable[str],
    additions: str | Iterable[str],
    *,
    path_separator: str | None = None,
    case_sensitive: bool | None = None,
) -> tuple[str, ...]:
    """Return a stable union of current PATH entries and additions."""

    current_values = (
        current.split(path_separator or os.pathsep)
        if isinstance(current, str)
        else list(current)
    )
    addition_values = (
        additions.split(path_separator or os.pathsep)
        if isinstance(additions, str)
        else list(additions)
    )
    return deduplicate_path_entries(
        [*current_values, *addition_values],
        path_separator=path_separator,
        case_sensitive=case_sensitive,
    )


def validate_profile_target(
    profile: str | Path,
    *,
    home_dir: str | Path | None = None,
) -> Path:
    """Validate a user profile target without allowing system or directory paths."""

    try:
        raw = Path(profile).expanduser()
    except (OSError, TypeError, ValueError) as exc:
        raise UnsafeProfileError("profile target is not a valid path") from exc
    if not raw.is_absolute():
        raise UnsafeProfileError("profile target must be an absolute user path")

    try:
        home = Path(home_dir).expanduser() if home_dir is not None else Path.home()
        home = home.resolve(strict=True)
        resolved = raw.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise UnsafeProfileError("profile target could not be resolved safely") from exc

    if resolved == home:
        raise UnsafeProfileError("refusing to use the home directory as a profile")
    try:
        resolved.relative_to(home)
    except ValueError as exc:
        raise UnsafeProfileError(
            "profile target must stay inside the user home"
        ) from exc

    if resolved.name not in _PROFILE_NAMES:
        raise UnsafeProfileError(
            "profile target must use an allowlisted shell profile filename"
        )
    if raw.exists() and raw.is_symlink():
        raise UnsafeProfileError("refusing to follow a symlinked profile target")
    if raw.exists() and not raw.is_file():
        raise UnsafeProfileError("profile target must be a regular file")
    return resolved


def _normalise_shell(profile: Path, shell: str | None) -> str:
    value = shell or ("powershell" if profile.suffix.lower() == ".ps1" else "posix")
    value = value.strip().lower()
    if value in {"posix", "bash", "zsh", "fish", "sh"}:
        return "posix"
    if value in {"powershell", "pwsh", "windows"}:
        return "powershell"
    raise ValueError(f"unsupported profile shell: {shell}")


def _path_key(value: str, *, case_sensitive: bool) -> str:
    result = os.path.normpath(os.path.expanduser(value.strip()))
    return result if case_sensitive else result.casefold()


def _profile_path_lines(
    content: str,
    additions: tuple[str, ...],
    *,
    shell: str,
) -> tuple[str, tuple[str, ...]]:
    separator = ";" if shell == "powershell" else ":"
    case_sensitive = os.name != "nt"
    seen: set[str] = set()
    variable_seen = False
    rewritten: list[str] = []
    path_line_seen = False

    if shell == "powershell":
        assignment = re.compile(
            r"^(?P<prefix>\s*\$env:Path\s*=\s*)(?P<body>.*)$", re.IGNORECASE
        )
    else:
        assignment = re.compile(
            r"^(?P<prefix>\s*(?:export\s+)?PATH\s*=\s*)(?P<body>.*)$"
        )

    for line in content.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body_line = line[:-1] if newline else line
        match = assignment.match(body_line)
        if not match:
            rewritten.append(line)
            continue

        path_line_seen = True
        body = match.group("body").strip()
        quote = body[0] if body[:1] in {"'", '"'} else ""
        suffix = ""
        if quote:
            if len(body) < 2 or body[-1] != quote:
                raise UnsafeProfileError(
                    "profile contains an unterminated PATH assignment"
                )
            body = body[1:-1]
        elif " #" in body:
            body, suffix = body.split(" #", 1)
            suffix = " #" + suffix

        pieces = [piece.strip() for piece in body.split(separator) if piece.strip()]
        output: list[str] = []
        for piece in pieces:
            variable = piece.lower() in {
                "$path",
                "${path}",
                "$env:path",
                "%path%",
            }
            key = (
                "__PATH_VARIABLE__"
                if variable
                else _path_key(
                    piece,
                    case_sensitive=case_sensitive,
                )
            )
            if variable:
                if variable_seen:
                    continue
                variable_seen = True
            elif key in seen:
                continue
            else:
                seen.add(key)
            output.append(piece)

        rebuilt = separator.join(output)
        if quote:
            rebuilt = quote + rebuilt + quote
        rewritten.append(match.group("prefix") + rebuilt + suffix + newline)

    missing: list[str] = []
    for addition in additions:
        key = _path_key(addition, case_sensitive=case_sensitive)
        if key in seen:
            continue
        seen.add(key)
        missing.append(addition)

    if missing:
        if rewritten and not rewritten[-1].endswith("\n"):
            rewritten[-1] += "\n"
        if path_line_seen and rewritten and rewritten[-1].strip():
            pass
        marker = "# AXIOM managed PATH entries"
        if not rewritten or rewritten[-1].strip() != marker:
            rewritten.append(marker + "\n")
        if shell == "powershell":
            rewritten.extend(f'$env:Path = "$env:Path;{entry}"\n' for entry in missing)
        else:
            rewritten.extend(f'export PATH="{entry}:$PATH"\n' for entry in missing)

    return "".join(rewritten), tuple(missing)


@dataclass(frozen=True)
class ProfilePathUpdate:
    path: Path
    added: tuple[str, ...]
    changed: bool
    content: str
    previous_content: str
    previously_existed: bool
    applied: bool
    home_dir: Path

    def revert(self) -> None:
        """Restore the exact profile contents observed before this update."""

        if not self.applied or not self.changed:
            return
        validate_profile_target(self.path, home_dir=self.home_dir)
        if self.path.exists() and self.path.read_text(encoding="utf-8") != self.content:
            raise InstallerSafetyError(
                "profile changed after the PATH update; refusing to overwrite it"
            )
        if self.previously_existed:
            self.path.write_text(self.previous_content, encoding="utf-8")
        elif self.path.exists():
            self.path.unlink()


def update_profile_path(
    profile: str | Path,
    entries: str | Iterable[str],
    *,
    home_dir: str | Path | None = None,
    shell: str | None = None,
    dry_run: bool = False,
) -> ProfilePathUpdate:
    """Safely deduplicate and add user PATH entries to an allowlisted profile."""

    target = validate_profile_target(profile, home_dir=home_dir)
    additions = deduplicate_path_entries(entries)
    previously_existed = target.exists()
    current = target.read_text(encoding="utf-8") if previously_existed else ""
    if _SECRET_ASSIGNMENT_RE.search(current):
        raise SecretInputError(
            "refusing to rewrite a profile containing credential assignments"
        )
    profile_shell = _normalise_shell(target, shell)
    updated, added = _profile_path_lines(
        current,
        additions,
        shell=profile_shell,
    )
    changed = updated != current
    resolved_home = (
        Path(home_dir).expanduser().resolve()
        if home_dir is not None
        else Path.home().resolve()
    )
    if changed and not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding="utf-8")
    return ProfilePathUpdate(
        path=target,
        added=added,
        changed=changed,
        content=updated,
        previous_content=current,
        previously_existed=previously_existed,
        applied=changed and not dry_run,
        home_dir=resolved_home,
    )


def append_path_to_profile(
    profile: str | Path,
    entries: str | Iterable[str],
    **kwargs: Any,
) -> ProfilePathUpdate:
    """Compatibility alias for :func:`update_profile_path`."""

    return update_profile_path(profile, entries, **kwargs)


def safe_profile_path(
    profile: str | Path, *, home_dir: str | Path | None = None
) -> Path:
    """Compatibility alias for profile target validation."""

    return validate_profile_target(profile, home_dir=home_dir)


__all__ = [
    "ALLOWED_COMMANDS",
    "COMMAND_ALLOWLIST",
    "INSTALLER_COMMANDS",
    "InstallPlan",
    "InstallTarget",
    "Installer",
    "InstallerSafetyError",
    "ProfilePathUpdate",
    "SecretInputError",
    "UnsafeCommandError",
    "UnsafeProfileError",
    "append_path_to_profile",
    "build_install_plan",
    "deduplicate_path_entries",
    "detect_availability",
    "detect_command_paths",
    "detect_tools",
    "execute_install_plan",
    "merge_path_entries",
    "safe_profile_path",
    "update_profile_path",
    "validate_command",
    "validate_profile_target",
]
