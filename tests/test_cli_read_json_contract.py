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
}


def _registered_command_name(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            continue
        if decorator.func.attr != "command" or not decorator.args:
            continue
        name = decorator.args[0]
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            return name.value
    return None


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
    for path in sorted((ROOT / "axiom" / "cli").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(path))
        for function in ast.walk(module):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            command_name = _registered_command_name(function)
            if command_name not in READ_ONLY_COMMANDS or _is_external_passthrough(function):
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
