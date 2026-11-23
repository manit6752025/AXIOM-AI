"""Verify that both published SDKs match the live AXIOM OpenAPI surface."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from axiom.api.server import API_CONTRACT_VERSION, create_app

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "sdk" / "openapi" / "axiom-v1-operations.json"
PYTHON_CLIENT = ROOT / "sdk" / "python" / "axiom_sdk" / "client.py"
TYPESCRIPT_CLIENT = ROOT / "sdk" / "typescript" / "src" / "client.ts"


class SDKContractError(RuntimeError):
    """An SDK operation does not match the live API schema."""


def verify_sdk_contract() -> dict[str, Any]:
    """Check operation methods/paths and public client method bindings."""
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        python_source = PYTHON_CLIENT.read_text(encoding="utf-8")
        typescript_source = TYPESCRIPT_CLIENT.read_text(encoding="utf-8")
        openapi = create_app().openapi()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SDKContractError("SDK contract inputs could not be loaded.") from exc

    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"api_contract_version", "description", "operations"}
        or type(manifest.get("api_contract_version")) is not int
        or manifest["api_contract_version"] != API_CONTRACT_VERSION
        or not isinstance(manifest.get("operations"), list)
    ):
        raise SDKContractError("The SDK operation manifest is malformed or targets another API version.")

    paths = openapi.get("paths")
    if not isinstance(paths, dict):
        raise SDKContractError("The live API OpenAPI document contains no paths.")
    failures: list[str] = []
    seen_names: set[str] = set()
    seen_routes: set[tuple[str, str]] = set()
    operations = manifest["operations"]
    if not operations:
        failures.append("The SDK operation manifest must not be empty.")

    for item in operations:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "method", "path", "python", "typescript"}
        ):
            failures.append("An SDK operation entry has an invalid schema.")
            continue
        name = item["name"]
        method = item["method"]
        route = item["path"]
        python_method = item["python"]
        typescript_method = item["typescript"]
        if any(not isinstance(value, str) or not value for value in (name, method, route, python_method, typescript_method)):
            failures.append("An SDK operation entry contains an empty or non-string value.")
            continue
        if name in seen_names or (method.casefold(), route) in seen_routes:
            failures.append(f"Duplicate SDK operation or route: {method} {route} ({name}).")
            continue
        seen_names.add(name)
        seen_routes.add((method.casefold(), route))
        if method.casefold() not in paths.get(route, {}):
            failures.append(f"Live OpenAPI is missing {method} {route} ({name}).")
        if not re.search(rf"^\s+def\s+{re.escape(python_method)}\s*\(", python_source, re.MULTILINE):
            failures.append(f"Python SDK method {python_method} is missing ({name}).")
        if not re.search(rf"^\s+{re.escape(typescript_method)}\s*\(", typescript_source, re.MULTILINE):
            failures.append(f"TypeScript SDK method {typescript_method} is missing ({name}).")

    if len(seen_names) != len(operations):
        failures.append("The SDK operation manifest contains one or more invalid entries.")
    if failures:
        raise SDKContractError("\n".join(failures))

    return {
        "api_contract_version": API_CONTRACT_VERSION,
        "operation_count": len(operations),
        "python_package": "sdk/python",
        "typescript_package": "sdk/typescript",
        "status": "verified",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable verification report.")
    args = parser.parse_args()
    try:
        report = verify_sdk_contract()
    except SDKContractError as exc:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        else:
            print(f"SDK contract verification failed:\n{exc}")
        return 1
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Verified {report['operation_count']} SDK operations against API v{report['api_contract_version']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
