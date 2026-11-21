from __future__ import annotations

import json

import pytest

from axiom.core.json_schema import (
    JsonSchemaError,
    MAX_OUTPUT_CHARS,
    MAX_SCHEMA_BYTES,
    parse_and_validate_json_output,
    validate_json_instance,
    validate_json_schema,
)


def _schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 40},
            "count": {"type": "integer", "minimum": 1, "maximum": 10},
            "enabled": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 16}, "maxItems": 4},
        },
        "required": ["name", "count"],
        "additionalProperties": False,
    }


def test_schema_subset_normalizes_to_supported_draft_and_default_deny_objects():
    normalized = validate_json_schema({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    })
    assert normalized["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert normalized["additionalProperties"] is False
    assert normalized["properties"]["name"]["type"] == "string"


def test_valid_json_instance_satisfies_nested_object_array_and_scalar_contracts():
    schema = _schema()
    value = {"name": "AXIOM", "count": 3, "enabled": True, "tags": ["local", "private"]}
    validate_json_instance(value, schema)
    assert parse_and_validate_json_output(json.dumps(value), schema) == value


@pytest.mark.parametrize("value", [
    {"name": "", "count": 3},
    {"name": "ok", "count": True},
    {"name": "ok", "count": 11},
    {"name": "ok", "count": 3, "unexpected": "denied"},
    {"name": "ok"},
    {"name": "ok", "count": 3, "tags": ["x"] * 5},
])
def test_schema_rejects_values_outside_contract(value):
    with pytest.raises(JsonSchemaError):
        validate_json_instance(value, _schema())


@pytest.mark.parametrize("schema", [
    {"type": "object", "$ref": "https://example.invalid/schema"},
    {"type": "object", "patternProperties": {".*": {"type": "string"}}},
    {"type": "object", "additionalProperties": True},
    {"type": "string", "pattern": "(a+)+"},
    {"$schema": "https://json-schema.org/draft/2019-09/schema", "type": "string"},
    {"type": "object", "properties": {"x": {"type": "function"}}},
    {"type": "array", "items": {"type": "string"}, "maxItems": 257},
    {"type": "string", "maxLength": MAX_OUTPUT_CHARS + 1},
    {"type": "integer", "minimum": True},
    {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["missing"]},
])
def test_schema_rejects_unsupported_or_unbounded_keywords(schema):
    with pytest.raises(JsonSchemaError):
        validate_json_schema(schema)


@pytest.mark.parametrize("schema", [
    {"type": "integer", "const": True},
    {"type": "integer", "enum": [1, True]},
    {"type": "integer", "const": 11, "maximum": 10},
    {"type": "string", "enum": ["ok", 1]},
])
def test_schema_rejects_const_and_enum_values_that_violate_declared_constraints(schema):
    with pytest.raises(JsonSchemaError, match="const and enum values"):
        validate_json_schema(schema)


def test_large_integer_values_are_validated_without_float_overflow():
    value = 10**400
    validate_json_instance(value, {"type": "integer"})
    with pytest.raises(JsonSchemaError, match="outside the configured range"):
        validate_json_instance(value, {"type": "integer", "maximum": 100})


def test_object_validation_rejects_non_string_python_keys_cleanly():
    with pytest.raises(JsonSchemaError, match="property names must be strings"):
        validate_json_instance(
            {"name": "AXIOM", 1: "unexpected"},
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        )


def test_schema_rejects_oversized_input_and_excessive_nesting():
    oversized = {"type": "string", "description": "x" * MAX_SCHEMA_BYTES}
    with pytest.raises(JsonSchemaError, match="byte limit"):
        validate_json_schema(oversized)

    nested = {"type": "string"}
    for _ in range(12):
        nested = {"type": "array", "items": nested}
    with pytest.raises(JsonSchemaError, match="nesting"):
        validate_json_schema(nested)


@pytest.mark.parametrize("output, message", [
    ('{"name":"AXIOM","name":"duplicate","count":3}', "duplicate property"),
    ('{"name":"AXIOM","count":NaN}', "non-finite"),
    ("not JSON", "invalid JSON"),
    ('{"name":"AXIOM","count":true}', "expected integer"),
])
def test_output_parser_rejects_duplicate_non_finite_invalid_or_wrong_type_values(output, message):
    with pytest.raises(JsonSchemaError, match=message):
        parse_and_validate_json_output(output, _schema())


def test_output_parser_enforces_size_limit_before_json_parsing():
    with pytest.raises(JsonSchemaError, match="character limit"):
        parse_and_validate_json_output(" " + "x" * MAX_OUTPUT_CHARS, {"type": "string"})



def test_schema_allows_ordinary_property_names_containing_x_zero_and_backslash():
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "x0": {"type": "string"},
            "path\\name": {"type": "string"},
        },
        "required": ["text", "x0", "path\\name"],
    }
    validate_json_instance(
        {"text": "hello", "x0": "value", "path\\name": "local"},
        schema,
    )


@pytest.mark.parametrize("property_name", ["", chr(0), "n" * 129])
def test_schema_rejects_empty_nul_and_oversized_property_names(property_name):
    with pytest.raises(JsonSchemaError, match="Property names"):
        validate_json_schema({
            "type": "object",
            "properties": {property_name: {"type": "string"}},
        })
