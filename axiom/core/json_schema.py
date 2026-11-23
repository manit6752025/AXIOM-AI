"""Bounded JSON Schema subset for local structured-output inference.

The accepted subset intentionally excludes references, remote resolution,
regular expressions, conditional applicators, and arbitrary extensions.
Validation is local, deterministic, recursive-depth bounded, and does not
execute or fetch schema content.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any


MAX_SCHEMA_BYTES = 32 * 1024
MAX_SCHEMA_DEPTH = 8
MAX_SCHEMA_NODES = 128
MAX_OBJECT_PROPERTIES = 64
MAX_ARRAY_ITEMS = 256
MAX_STRING_LENGTH = 16_384
MAX_OUTPUT_CHARS = 65_536
MAX_ERROR_MESSAGES = 8

_ALLOWED_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
_ALLOWED_KEYWORDS = frozenset({
    "$schema", "title", "description", "type", "properties", "required",
    "additionalProperties", "items", "enum", "const", "minLength",
    "maxLength", "minimum", "maximum", "minItems", "maxItems",
})
_SUPPORTED_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_PROPERTY_NAME = re.compile(r"^[^\x00]{1,128}$")


class JsonSchemaError(ValueError):
    """Raised when a schema or generated value is outside the supported contract."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _bounded_integer(node: dict[str, Any], key: str, *, maximum: int) -> None:
    if key not in node:
        return
    value = node[key]
    if not _is_integer(value) or not 0 <= value <= maximum:
        raise JsonSchemaError(f"{key} must be an integer between 0 and {maximum}.")


def _validate_schema_node(schema: Any, *, depth: int, counter: list[int]) -> dict[str, Any]:
    if depth > MAX_SCHEMA_DEPTH:
        raise JsonSchemaError(f"Schema nesting exceeds {MAX_SCHEMA_DEPTH} levels.")
    if not isinstance(schema, dict):
        raise JsonSchemaError("Each schema node must be a JSON object.")
    counter[0] += 1
    if counter[0] > MAX_SCHEMA_NODES:
        raise JsonSchemaError(f"Schema contains more than {MAX_SCHEMA_NODES} nodes.")
    unknown = set(schema) - _ALLOWED_KEYWORDS
    if unknown:
        keyword = sorted(unknown)[0]
        raise JsonSchemaError(f"Unsupported JSON Schema keyword: {keyword}.")
    draft = schema.get("$schema")
    if draft is not None and draft != _SUPPORTED_DRAFT:
        raise JsonSchemaError("Only JSON Schema Draft 2020-12 is supported.")
    kind = schema.get("type")
    if not isinstance(kind, str) or kind not in _ALLOWED_TYPES:
        raise JsonSchemaError("Every schema node must declare one supported type.")
    normalized: dict[str, Any] = {"type": kind}

    for label in ("title", "description"):
        value = schema.get(label)
        if value is not None:
            if not isinstance(value, str) or len(value) > 160:
                raise JsonSchemaError(f"{label} must be a string of at most 160 characters.")
            normalized[label] = value

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not 1 <= len(enum) <= 32:
            raise JsonSchemaError("enum must contain between 1 and 32 JSON values.")
        encoded = json.dumps(enum, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 8_192:
            raise JsonSchemaError("enum exceeds the 8 KiB limit.")
        if len({json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False) for item in enum}) != len(enum):
            raise JsonSchemaError("enum values must be unique.")
        normalized["enum"] = enum
    if "const" in schema:
        encoded = json.dumps(schema["const"], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 8_192:
            raise JsonSchemaError("const exceeds the 8 KiB limit.")
        normalized["const"] = schema["const"]

    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict) or len(properties) > MAX_OBJECT_PROPERTIES:
            raise JsonSchemaError(f"properties must be an object with at most {MAX_OBJECT_PROPERTIES} entries.")
        normalized_properties: dict[str, Any] = {}
        for name, child in properties.items():
            if not isinstance(name, str) or not _PROPERTY_NAME.fullmatch(name):
                raise JsonSchemaError("Property names must be non-empty strings of at most 128 characters.")
            normalized_properties[name] = _validate_schema_node(child, depth=depth + 1, counter=counter)
        required = schema.get("required", [])
        if not isinstance(required, list) or len(required) > MAX_OBJECT_PROPERTIES:
            raise JsonSchemaError(f"required must be an array with at most {MAX_OBJECT_PROPERTIES} names.")
        if any(not isinstance(name, str) or name not in normalized_properties for name in required):
            raise JsonSchemaError("Every required entry must name a declared property.")
        if len(set(required)) != len(required):
            raise JsonSchemaError("required entries must be unique.")
        additional = schema.get("additionalProperties", False)
        if additional is not False:
            raise JsonSchemaError("additionalProperties must be false; structured outputs are default-deny.")
        normalized.update({
            "properties": normalized_properties,
            "required": list(required),
            "additionalProperties": False,
        })
    elif "properties" in schema or "required" in schema or "additionalProperties" in schema:
        raise JsonSchemaError("Object-only keywords may only be used with type=object.")

    if kind == "array":
        if "items" not in schema:
            raise JsonSchemaError("Array schemas must declare items.")
        normalized["items"] = _validate_schema_node(schema["items"], depth=depth + 1, counter=counter)
        _bounded_integer(schema, "minItems", maximum=MAX_ARRAY_ITEMS)
        _bounded_integer(schema, "maxItems", maximum=MAX_ARRAY_ITEMS)
        if schema.get("minItems", 0) > schema.get("maxItems", MAX_ARRAY_ITEMS):
            raise JsonSchemaError("minItems may not exceed maxItems.")
        for key in ("minItems", "maxItems"):
            if key in schema:
                normalized[key] = schema[key]
    elif any(key in schema for key in ("items", "minItems", "maxItems")):
        raise JsonSchemaError("Array-only keywords may only be used with type=array.")

    if kind == "string":
        _bounded_integer(schema, "minLength", maximum=MAX_STRING_LENGTH)
        _bounded_integer(schema, "maxLength", maximum=MAX_STRING_LENGTH)
        if schema.get("minLength", 0) > schema.get("maxLength", MAX_STRING_LENGTH):
            raise JsonSchemaError("minLength may not exceed maxLength.")
        for key in ("minLength", "maxLength"):
            if key in schema:
                normalized[key] = schema[key]
    elif "minLength" in schema or "maxLength" in schema:
        raise JsonSchemaError("String-only keywords may only be used with type=string.")

    if kind in {"integer", "number"}:
        for key in ("minimum", "maximum"):
            if key in schema:
                value = schema[key]
                if not _is_number(value) or abs(value) > 1e100:
                    raise JsonSchemaError(f"{key} must be a finite JSON number with magnitude at most 1e100.")
                normalized[key] = value
        if normalized.get("minimum", -math.inf) > normalized.get("maximum", math.inf):
            raise JsonSchemaError("minimum may not exceed maximum.")
    elif "minimum" in schema or "maximum" in schema:
        raise JsonSchemaError("Numeric-only keywords may only be used with type=integer or type=number.")

    # A schema's own enum/const values must satisfy its type and constraints;
    # otherwise Python's bool/int equality could accept the wrong JSON type.
    candidates = ([normalized["const"]] if "const" in normalized else [])
    candidates.extend(normalized.get("enum", []))
    for candidate in candidates:
        candidate_errors: list[str] = []
        _validate_value(candidate, normalized, "$schema-value", depth=0, counter=[0], errors=candidate_errors)
        if candidate_errors:
            raise JsonSchemaError("const and enum values must satisfy their schema type and constraints.")

    return normalized


def validate_json_schema(schema: Any) -> dict[str, Any]:
    """Validate and normalize the safe bounded JSON Schema subset."""
    if not isinstance(schema, dict):
        raise JsonSchemaError("JSON Schema must be an object.")
    try:
        encoded = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise JsonSchemaError("JSON Schema must contain only finite JSON values within the nesting limit.") from None
    if len(encoded) > MAX_SCHEMA_BYTES:
        raise JsonSchemaError(f"JSON Schema exceeds the {MAX_SCHEMA_BYTES}-byte limit.")
    normalized = _validate_schema_node(schema, depth=0, counter=[0])
    normalized["$schema"] = _SUPPORTED_DRAFT
    return normalized


def _instance_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if _is_integer(value):
        return "integer"
    if _is_number(value):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "invalid"


def _matches_type(value: Any, kind: str) -> bool:
    actual = _instance_type(value)
    if kind == "number":
        return actual in {"integer", "number"}
    return actual == kind


def _validate_value(value: Any, schema: dict[str, Any], path: str, *, depth: int, counter: list[int], errors: list[str]) -> None:
    if len(errors) >= MAX_ERROR_MESSAGES:
        return
    if depth > MAX_SCHEMA_DEPTH:
        errors.append(f"{path}: value nesting exceeds {MAX_SCHEMA_DEPTH} levels.")
        return
    counter[0] += 1
    if counter[0] > MAX_SCHEMA_NODES * 4:
        errors.append("Output contains too many nested values.")
        return
    kind = schema["type"]
    if not _matches_type(value, kind):
        errors.append(f"{path}: expected {kind}, received {_instance_type(value)}.")
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: value does not match const.")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not one of the allowed enum values.")
    if kind == "object":
        if len(value) > MAX_OBJECT_PROPERTIES:
            errors.append(f"{path}: object has more than {MAX_OBJECT_PROPERTIES} properties.")
            return
        if any(not isinstance(name, str) for name in value):
            errors.append(f"{path}: object property names must be strings.")
            return
        properties = schema["properties"]
        unknown = set(value) - set(properties)
        if unknown:
            errors.append(f"{path}: unexpected property {sorted(unknown)[0]!r}.")
        for required in schema["required"]:
            if required not in value:
                errors.append(f"{path}: missing required property {required!r}.")
                if len(errors) >= MAX_ERROR_MESSAGES:
                    return
        for name, item in value.items():
            child = properties.get(name)
            if child is not None:
                _validate_value(item, child, f"{path}.{name}", depth=depth + 1, counter=counter, errors=errors)
                if len(errors) >= MAX_ERROR_MESSAGES:
                    return
    elif kind == "array":
        if len(value) > MAX_ARRAY_ITEMS:
            errors.append(f"{path}: array contains more than {MAX_ARRAY_ITEMS} items.")
            return
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", MAX_ARRAY_ITEMS):
            errors.append(f"{path}: array length is outside the configured bounds.")
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], f"{path}[{index}]", depth=depth + 1, counter=counter, errors=errors)
            if len(errors) >= MAX_ERROR_MESSAGES:
                return
    elif kind == "string":
        if len(value) > MAX_STRING_LENGTH:
            errors.append(f"{path}: string exceeds {MAX_STRING_LENGTH} characters.")
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", MAX_STRING_LENGTH):
            errors.append(f"{path}: string length is outside the configured bounds.")
    elif kind in {"integer", "number"}:
        if not _is_number(value):
            errors.append(f"{path}: numeric value is not finite.")
            return
        if ("minimum" in schema and value < schema["minimum"]) or ("maximum" in schema and value > schema["maximum"]):
            errors.append(f"{path}: number is outside the configured range.")


def validate_json_instance(value: Any, schema: dict[str, Any]) -> None:
    """Raise JsonSchemaError when a parsed JSON value does not satisfy schema."""
    normalized = validate_json_schema(schema)
    errors: list[str] = []
    _validate_value(value, normalized, "$", depth=0, counter=[0], errors=errors)
    if errors:
        suffix = "" if len(errors) < MAX_ERROR_MESSAGES else " Additional errors were omitted."
        raise JsonSchemaError("; ".join(errors[:MAX_ERROR_MESSAGES])[:1_200] + suffix)


def parse_and_validate_json_output(output: str, schema: dict[str, Any]) -> Any:
    """Parse bounded model output as strict JSON and validate it against schema."""
    if not isinstance(output, str) or not output.strip():
        raise JsonSchemaError("The local model did not return JSON output.")
    if len(output) > MAX_OUTPUT_CHARS:
        raise JsonSchemaError(f"Generated JSON exceeds the {MAX_OUTPUT_CHARS}-character limit.")

    def reject_constant(_value: str) -> None:
        raise JsonSchemaError("Generated JSON contains a non-finite numeric value.")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise JsonSchemaError(f"Generated JSON contains duplicate property {key!r}.")
            result[key] = value
        return result

    try:
        value = json.loads(output, parse_constant=reject_constant, object_pairs_hook=unique_pairs)
    except JsonSchemaError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError):
        raise JsonSchemaError("The local model returned invalid JSON; no structured value was accepted.") from None
    validate_json_instance(value, schema)
    return value
