from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READ_ONLY_COMMANDS = {
    "list",
    "show",
    "inspect",
    "search",
    "validate",
    "profile",
    "info",
    "status",
    "summary",
    "discover",
    "doctor",
    "check",
    "manifest",
    "preview-clean",
    "snapshot",
    "snapshots",
    "snapshot-verify",
    "documents",
    "providers",
    "models",
    "files",
    "jobs",
    "versions",
    "backends",
    "benchmark-profiles",
    "benchmark-compare",
    "benchmark-requirements",
    "ps",
    "hf-search",
    "remote-models",
    "mcp-list",
    "schemas",
    "handshake",
}


def _registered_command_name(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Resolve an explicit Typer command name or its function-name default."""
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Attribute) and decorator.attr == "command":
            return function.name.replace("_", "-")
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            continue
        if decorator.func.attr != "command":
            continue
        if decorator.args:
            name = decorator.args[0]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                return name.value
        for keyword in decorator.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    return keyword.value.value
        return function.name.replace("_", "-")
    return None


def _is_read_only_command(command_name: str | None) -> bool:
    if command_name is None:
        return False
    normalized = command_name.replace("_", "-").strip("-").casefold()
    if normalized in READ_ONLY_COMMANDS:
        return True
    # Catch default function names and explicitly qualified names such as
    # list_models, model-list, hf-status, or dataset-validate.
    return any(
        normalized.startswith(f"{name}-") or normalized.endswith(f"-{name}")
        for name in READ_ONLY_COMMANDS
    )


def _has_json_option(function: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> bool:
    if not function.body:
        return False
    lines = source.splitlines()
    signature = "\n".join(lines[function.lineno - 1:function.body[0].lineno - 1])
    return "--json" in signature


def _emits_json_by_default(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Attribute) and callee.attr == "print_json":
            return True
        if (
            isinstance(callee, ast.Attribute)
            and callee.attr == "dumps"
            and isinstance(callee.value, ast.Name)
            and callee.value.id == "json"
        ):
            return True
    return False


def _is_external_passthrough(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_external"
        for node in ast.walk(function)
    )


def test_registered_read_only_cli_commands_have_json_contracts() -> None:
    missing: list[str] = []
    for path in sorted((ROOT / "axiom" / "cli").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(path))
        for function in ast.walk(module):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            command_name = _registered_command_name(function)
            if not _is_read_only_command(command_name) or _is_external_passthrough(function):
                continue
            if not _has_json_option(function, source) and not _emits_json_by_default(function):
                missing.append(
                    f"{path.relative_to(ROOT)}:{function.lineno} "
                    f"({command_name} -> {function.name})"
                )

    assert missing == [], (
        "Read-only CLI commands must expose --json or emit JSON by default: "
        + ", ".join(missing)
    )



def test_registered_command_name_audit_covers_default_and_keyword_names() -> None:
    default_command = ast.parse("@app.command()\ndef list_models():\n    pass\n").body[0]
    keyword_command = ast.parse('@app.command(name="inspect")\ndef inspect_model():\n    pass\n').body[0]
    direct_command = ast.parse("@app.command\ndef validate_project():\n    pass\n").body[0]

    assert _registered_command_name(default_command) == "list-models"
    assert _registered_command_name(keyword_command) == "inspect"
    assert _registered_command_name(direct_command) == "validate-project"
    assert _is_read_only_command("list-models")
    assert _is_read_only_command("model-list")
    assert _is_read_only_command("dataset-validate")
    assert not _is_read_only_command("model-remove")
