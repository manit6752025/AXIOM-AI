"""Typed, secret-safe examples generated from the live AXIOM OpenAPI contract."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import quote

MAX_OPERATIONS = 512
MAX_SCHEMA_DEPTH = 12
MAX_SCHEMA_NODES = 4096
MAX_EXAMPLES_BYTES = 2 * 1024 * 1024
_SENSITIVE_NAME = re.compile(
    r"(?:^|_)(?:api_key|client_secret|password|passwd|authorization|secret|token|api_token|"
    r"access_token|refresh_token|bearer_token|private_key|credential_value)(?:$|_)",
    re.IGNORECASE,
)
_METHOD_ORDER = {"get": 0, "post": 1, "put": 2, "patch": 3, "delete": 4, "head": 5, "options": 6}


class ApiExamplesError(ValueError):
    """The OpenAPI document cannot be converted into a bounded safe example set."""


def _is_sensitive(name: str | None, schema: dict[str, Any] | None = None) -> bool:
    field = name or ""
    # Environment-variable references are identifiers, not credential values.
    if field.casefold().endswith("_env"):
        return False
    shape = schema or {}
    return bool(_SENSITIVE_NAME.search(field)) or shape.get("writeOnly") is True or str(shape.get("format", "")).casefold() == "password"


def _resolve_schema(schema: Any, components: dict[str, Any], *, stack: tuple[str, ...] = (), depth: int = 0) -> dict[str, Any]:
    """Resolve local JSON Schema references while bounding recursion."""
    if depth > MAX_SCHEMA_DEPTH or not isinstance(schema, dict):
        return {}
    reference = schema.get("$ref")
    base: dict[str, Any] = {}
    if isinstance(reference, str):
        prefix = "#/components/schemas/"
        if not reference.startswith(prefix):
            raise ApiExamplesError("The OpenAPI contract contains a non-local schema reference.")
        name = reference[len(prefix):].replace("~1", "/").replace("~0", "~")
        if name in stack:
            return {"type": "object", "title": name, "description": "Recursive reference omitted from the example."}
        target = components.get(name)
        if not isinstance(target, dict):
            raise ApiExamplesError("The OpenAPI contract contains an unresolved schema reference.")
        base = _resolve_schema(target, components, stack=stack + (name,), depth=depth + 1)
    for key, value in schema.items():
        if key == "$ref":
            continue
        if key == "properties" and isinstance(value, dict):
            base["properties"] = {str(field): _resolve_schema(child, components, stack=stack, depth=depth + 1) for field, child in list(value.items())[:128]}
        elif key == "items" and isinstance(value, dict):
            base["items"] = _resolve_schema(value, components, stack=stack, depth=depth + 1)
        elif key in {"oneOf", "anyOf", "allOf"} and isinstance(value, list):
            base[key] = [_resolve_schema(child, components, stack=stack, depth=depth + 1) for child in value[:16]]
        elif key == "additionalProperties" and isinstance(value, dict):
            base[key] = _resolve_schema(value, components, stack=stack, depth=depth + 1)
        else:
            base[key] = value
    return base


def _safe_schema(schema: dict[str, Any], *, field_name: str | None = None, depth: int = 0, budget: list[int] | None = None) -> dict[str, Any]:
    """Return a compact JSON Schema projection with secret values removed."""
    if budget is None:
        budget = [MAX_SCHEMA_NODES]
    if budget[0] <= 0 or depth > MAX_SCHEMA_DEPTH:
        return {"type": "object", "description": "Schema depth/budget limit reached."}
    budget[0] -= 1
    if _is_sensitive(field_name, schema):
        return {
            "type": "string", "format": "password", "writeOnly": True,
            "x-axiom-redacted": True,
            "description": "Sensitive input omitted; provide it through a secure request channel.",
        }
    result: dict[str, Any] = {}
    for key in ("type", "title", "description", "format", "pattern", "minimum", "maximum",
                "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
                "minItems", "maxItems", "uniqueItems", "nullable", "readOnly", "writeOnly"):
        value = schema.get(key)
        if isinstance(value, (str, int, bool)) or (isinstance(value, float) and value == value and abs(value) != float("inf")):
            result[key] = value
    if isinstance(schema.get("enum"), list):
        result["enum"] = [item for item in schema["enum"][:32] if isinstance(item, (str, int, float, bool)) or item is None]
    if "const" in schema and isinstance(schema["const"], (str, int, float, bool)):
        result["const"] = schema["const"]
    if isinstance(schema.get("required"), list):
        result["required"] = [item for item in schema["required"][:128] if isinstance(item, str)]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        result["properties"] = {str(name): _safe_schema(child, field_name=str(name), depth=depth + 1, budget=budget) for name, child in list(properties.items())[:128]}
    if isinstance(schema.get("items"), dict):
        result["items"] = _safe_schema(schema["items"], depth=depth + 1, budget=budget)
    for key in ("oneOf", "anyOf", "allOf"):
        value = schema.get(key)
        if isinstance(value, list):
            result[key] = [_safe_schema(item, depth=depth + 1, budget=budget) for item in value[:16]]
    additional = schema.get("additionalProperties")
    if isinstance(additional, bool):
        result["additionalProperties"] = additional
    elif isinstance(additional, dict):
        result["additionalProperties"] = _safe_schema(additional, depth=depth + 1, budget=budget)
    return result


def _example_value(schema: dict[str, Any], *, field_name: str | None = None, response: bool = False,
                   path: str = "$", depth: int = 0, budget: list[int] | None = None) -> tuple[Any, list[str], list[str]]:
    """Create synthetic schema-shaped data; never replay OpenAPI defaults or examples."""
    if budget is None:
        budget = [MAX_SCHEMA_NODES]
    redacted: list[str] = []
    templates: list[str] = []
    if depth > MAX_SCHEMA_DEPTH or budget[0] <= 0:
        return {}, redacted, [path]
    budget[0] -= 1
    if _is_sensitive(field_name, schema):
        redacted.append(path)
        return (None if response else "<provide-secret-securely>"), redacted, templates
    for key in ("oneOf", "anyOf"):
        choices = schema.get(key)
        if isinstance(choices, list):
            non_null = [item for item in choices if item.get("type") != "null"]
            return _example_value((non_null or choices or [{}])[0], field_name=field_name, response=response, path=path, depth=depth + 1, budget=budget)
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for part in all_of:
            if isinstance(part.get("properties"), dict):
                merged["properties"].update(part["properties"])
            merged["required"].extend(part.get("required", []))
        return _example_value(merged, field_name=field_name, response=response, path=path, depth=depth + 1, budget=budget)
    if "const" in schema:
        return schema["const"], redacted, templates
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        selected_enum = "user" if field_name == "role" and "user" in enum else enum[0]
        return selected_enum, redacted, templates
    kind = schema.get("type")
    if isinstance(kind, list):
        kinds = [item for item in kind if item != "null"]
        kind = kinds[0] if kinds else "null"
    if kind == "object" or isinstance(schema.get("properties"), dict):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        sample: dict[str, Any] = {}
        for name, child in list(properties.items())[:128]:
            must_include = name in required
            if _is_sensitive(name, child):
                redacted.append(f"{path}.{name}")
                if must_include and not response:
                    sample[name] = "<provide-secret-securely>"
                continue
            if must_include or "enum" in child or "const" in child:
                value, child_redacted, child_templates = _example_value(child, field_name=name, response=response, path=f"{path}.{name}", depth=depth + 1, budget=budget)
                redacted.extend(child_redacted)
                templates.extend(child_templates)
                if value is not None:
                    sample[name] = value
        return sample, redacted, templates
    if kind == "array":
        item_schema = schema.get("items", {})
        minimum = schema.get("minItems", 0)
        count = max(1, min(int(minimum) if isinstance(minimum, int) else 1, 2))
        sample = []
        for index in range(count):
            value, child_redacted, child_templates = _example_value(item_schema if isinstance(item_schema, dict) else {}, response=response, path=f"{path}[{index}]", depth=depth + 1, budget=budget)
            redacted.extend(child_redacted)
            templates.extend(child_templates)
            sample.append(value)
        return sample, redacted, templates
    if kind == "integer":
        value = schema.get("minimum", 1)
        if not isinstance(value, int):
            value = 1
        maximum = schema.get("maximum")
        return min(value, maximum) if isinstance(maximum, int) else value, redacted, templates
    if kind == "number":
        value = schema.get("minimum", 0.5)
        if not isinstance(value, (int, float)):
            value = 0.5
        maximum = schema.get("maximum")
        return min(value, maximum) if isinstance(maximum, (int, float)) else value, redacted, templates
    if kind == "boolean":
        return True, redacted, templates
    if kind == "null":
        return None, redacted, templates
    if kind == "string" or kind is None:
        field = (field_name or "").casefold()
        fmt = str(schema.get("format", "")).casefold()
        examples = {
            "date-time": "2026-01-01T00:00:00Z",
            "date": "2026-01-01",
            "uuid": "00000000-0000-4000-8000-000000000001",
            "email": "operator@example.invalid",
            "uri": "https://example.invalid/resource",
            "url": "https://example.invalid/resource",
        }
        if fmt in examples:
            return examples[fmt], redacted, templates
        if field.endswith("_env"):
            return "AXIOM_PROVIDER_API_KEY", redacted, templates
        if field in {"model_path", "adapter_path", "path", "output_path", "dataset_path"}:
            templates.append(path)
            return "<trusted-local-path>", redacted, templates
        if field in {"prompt", "query", "text", "content"}:
            templates.append(path)
            return "Example request text", redacted, templates
        if field in {"model", "model_id"}:
            return "local-model", redacted, templates
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            if "0-9a-f" in pattern and "{32}" in pattern:
                return "a" * 32, redacted, templates
            if "A-Za-z_" in pattern and "0-9_" in pattern:
                return "AXIOM_PROVIDER_API_KEY", redacted, templates
            if "A-Za-z0-9" in pattern:
                return "example-id", redacted, templates
        value = "example"
        minimum, maximum = schema.get("minLength", 0), schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            value += "x" * min(64, minimum - len(value))
        if isinstance(maximum, int) and maximum >= 0:
            value = value[:maximum]
        return value, redacted, templates
    return None, redacted, templates


def _typescript_type(schema: dict[str, Any], depth: int = 0) -> str:
    if depth > MAX_SCHEMA_DEPTH:
        return "unknown"
    if schema.get("x-axiom-redacted"):
        return "string"
    for key in ("oneOf", "anyOf"):
        alternatives = schema.get(key)
        if isinstance(alternatives, list):
            return " | ".join(dict.fromkeys(_typescript_type(item, depth + 1) for item in alternatives)) or "unknown"
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        return " & ".join(_typescript_type(item, depth + 1) for item in all_of) or "unknown"
    kind = schema.get("type")
    if isinstance(kind, list):
        kinds = [item for item in kind if item != "null"]
        base = _typescript_type({**schema, "type": kinds[0]} if kinds else {"type": "null"}, depth + 1)
        return base + (" | null" if "null" in kind else "")
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return " | ".join(json.dumps(item, ensure_ascii=False) for item in enum[:32])
    if kind == "array":
        return f"Array<{_typescript_type(schema.get('items', {}), depth + 1)}>"
    if kind == "object" or isinstance(schema.get("properties"), dict):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        rows = []
        for name, child in list(properties.items())[:128]:
            safe_name = name if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name) else json.dumps(name)
            suffix = "" if name in required else "?"
            rows.append(f"  {safe_name}{suffix}: {_typescript_type(child, depth + 1)};")
        if schema.get("additionalProperties") is True:
            rows.append("  [key: string]: unknown;")
        return "{\n" + "\n".join(rows) + "\n}"
    if kind in {"integer", "number"}:
        return "number"
    if kind == "boolean":
        return "boolean"
    if kind == "null":
        return "null"
    if kind == "string" or kind is None:
        return "string"
    return "unknown"


def _safe_identifier(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    if not parts:
        return "ApiOperation"
    name = "".join(part[:1].upper() + part[1:] for part in parts)
    return name if not name[0].isdigit() else "Api" + name


def _media_schema(content: Any) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(content, dict):
        return None, None
    for media_type in ("application/json", "application/merge-patch+json", "application/problem+json"):
        media = content.get(media_type)
        if isinstance(media, dict) and isinstance(media.get("schema"), dict):
            return media_type, media["schema"]
    for media_type, media in content.items():
        if isinstance(media, dict) and isinstance(media.get("schema"), dict):
            return str(media_type), media["schema"]
    return None, None


def _operation_example(path: str, method: str, operation: dict[str, Any], path_item: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    operation_id = str(operation.get("operationId") or f"{method}_{path}")
    identifier = _safe_identifier(operation_id)
    parameter_map: dict[tuple[str, str], dict[str, Any]] = {}
    for item in list(path_item.get("parameters", [])) + list(operation.get("parameters", [])):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        if item.get("in") not in {"path", "query"}:
            continue
        parameter_map[(str(item["in"]), str(item["name"]))] = item
    parameters = []
    for (location, name), parameter in sorted(parameter_map.items()):
        resolved = _resolve_schema(parameter.get("schema", {}), components)
        safe = _safe_schema(resolved, field_name=name)
        value, redacted, templates = _example_value(resolved, field_name=name, path=f"$.parameters.{name}")
        parameters.append({
            "name": name, "in": location,
            "required": bool(parameter.get("required", location == "path")),
            "description": str(parameter.get("description") or "")[:256],
            "schema": safe, "example": value,
            "redacted_fields": redacted, "template_fields": templates,
        })

    request_body = None
    raw_body = operation.get("requestBody")
    if isinstance(raw_body, dict):
        content_type, raw_schema = _media_schema(raw_body.get("content"))
        if raw_schema is not None:
            resolved = _resolve_schema(raw_schema, components)
            safe = _safe_schema(resolved)
            value, redacted, templates = _example_value(resolved, path="$.body")
            request_body = {
                "required": bool(raw_body.get("required")), "content_type": content_type,
                "schema": safe, "example": value, "redacted_fields": redacted,
                "template_fields": templates, "typescript_type": _typescript_type(safe),
            }

    successful = [
        (code, response) for code, response in operation.get("responses", {}).items()
        if re.fullmatch(r"2\d\d", str(code)) and isinstance(response, dict)
    ]
    successful.sort(key=lambda item: (int(item[0]), item[0]))
    response_example = None
    if successful:
        code, response = successful[0]
        content_type, raw_schema = _media_schema(response.get("content"))
        if raw_schema is not None:
            resolved = _resolve_schema(raw_schema, components)
            safe = _safe_schema(resolved)
            value, redacted, templates = _example_value(resolved, response=True, path="$.response")
            response_example = {
                "status_code": str(code), "description": str(response.get("description") or "")[:256],
                "content_type": content_type, "schema": safe, "example": value,
                "redacted_fields": redacted, "template_fields": templates,
                "typescript_type": _typescript_type(safe),
            }
        else:
            response_example = {
                "status_code": str(code), "description": str(response.get("description") or "")[:256],
                "content_type": None, "schema": None, "example": None,
                "redacted_fields": [], "template_fields": [], "typescript_type": "void",
            }

    request_type = request_body["typescript_type"] if request_body else "void"
    response_type = response_example["typescript_type"] if response_example else "unknown"
    typescript = f"export type {identifier}Request = {request_type};\nexport type {identifier}Response = {response_type};"
    query = "&".join(f"{quote(item['name'])}={quote(str(item['example']))}" for item in parameters if item["in"] == "query" and item["example"] is not None)
    url = f"http://127.0.0.1:40507{path}" + (f"?{query}" if query else "")
    curl = f"curl -X {method.upper()} {json.dumps(url)}"
    if request_body:
        curl += f" -H 'Content-Type: {request_body['content_type']}' --data '{json.dumps(request_body['example'], ensure_ascii=False)}'"
    curl += " # Add an AXIOM bearer token only when API scope enforcement is enabled; never paste real tokens into documentation."
    return {
        "operation_id": operation_id, "identifier": identifier, "method": method.upper(),
        "path": path, "summary": str(operation.get("summary") or operation_id)[:256],
        "description": str(operation.get("description") or "")[:512],
        "tags": [str(tag)[:64] for tag in operation.get("tags", [])[:12] if isinstance(tag, str)],
        "side_effecting": method.lower() not in {"get", "head", "options"},
        "parameters": parameters, "request": request_body, "response": response_example,
        "typescript": typescript, "curl_template": curl,
        "scope_note": "Generated from OpenAPI. Template values require operator input; examples are not executed.",
    }


def build_api_examples(openapi: dict[str, Any]) -> dict[str, Any]:
    """Build bounded request/response examples and TypeScript types from OpenAPI."""
    if not isinstance(openapi, dict) or not isinstance(openapi.get("paths"), dict):
        raise ApiExamplesError("A valid OpenAPI document with paths is required.")
    components_obj = openapi.get("components", {})
    schemas = components_obj.get("schemas", {}) if isinstance(components_obj, dict) else {}
    if not isinstance(schemas, dict):
        raise ApiExamplesError("The OpenAPI component schemas are malformed.")
    canonical = json.dumps(openapi, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    operations: list[dict[str, Any]] = []
    total = 0
    for path, path_item in sorted(openapi["paths"].items()):
        if not isinstance(path, str) or not isinstance(path_item, dict):
            continue
        for method in sorted(_METHOD_ORDER, key=lambda item: _METHOD_ORDER[item]):
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            total += 1
            if len(operations) < MAX_OPERATIONS:
                operations.append(_operation_example(path, method, operation, path_item, schemas))
    report = {
        "schema_version": 1, "api_contract_version": 1,
        "openapi_version": str(openapi.get("openapi") or ""),
        "application_version": str((openapi.get("info") or {}).get("version") or ""),
        "source_sha256": digest, "operation_count": len(operations),
        "total_operations": total, "truncated": total > len(operations),
        "operations": operations,
        "privacy": {
            "credential_values_included": False, "environment_values_included": False,
            "schema_defaults_replayed": False, "requests_executed": False,
        },
        "scope_note": (
            "Typed examples and synthetic payloads are generated from the supported OpenAPI contract. "
            "They do not make requests. Secret fields are redacted, values are templates rather than real workspace data, "
            "and cloud/mutating operations require their normal consent and API-scope policies when executed separately."
        ),
    }
    encoded = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_EXAMPLES_BYTES:
        raise ApiExamplesError("The generated API examples exceed the 2 MiB response limit.")
    return report


def render_api_examples_markdown(report: dict[str, Any]) -> str:
    """Render the generated operation catalogue as a readable Markdown reference."""
    if not isinstance(report, dict) or not isinstance(report.get("operations"), list):
        raise ApiExamplesError("A valid generated API example report is required.")
    fence = chr(96) * 3
    lines = [
        "# Generated AXIOM API examples", "",
        f"OpenAPI {report.get('openapi_version') or 'unknown'} · application {report.get('application_version') or 'unknown'}", "",
        f"Generated from contract SHA-256: {report.get('source_sha256', '')}", "",
        "> Examples are synthetic templates, not executed requests. Secret values, environment values and workspace data are never included.", "",
    ]
    for operation in report["operations"]:
        lines.extend([
            f"## {operation['method']} {operation['path']}", "",
            str(operation.get("summary") or ""), "",
            f"Operation ID: {operation['operation_id']} · Side-effecting: {'yes' if operation['side_effecting'] else 'no'}", "",
        ])
        request = operation.get("request")
        if request:
            lines.extend([
                "### Request example", "", f"Content-Type: {request.get('content_type')}", "",
                f"{fence}json", json.dumps(request.get("example"), indent=2, ensure_ascii=False), fence, "",
                "### TypeScript contract", "", f"{fence}ts",
                operation["typescript"], fence, "",
            ])
        response = operation.get("response")
        if response and response.get("example") is not None:
            lines.extend([
                f"### Response example ({response.get('status_code')})", "",
                f"{fence}json", json.dumps(response.get("example"), indent=2, ensure_ascii=False), fence, "",
            ])
        lines.extend(["### cURL template", "", f"{fence}bash", operation["curl_template"], fence, ""])
    lines.extend(["## Scope and safety", "", str(report.get("scope_note") or ""), ""])
    return "\n".join(lines)
