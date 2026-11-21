# Structured local inference

AXIOM can ask an installed local Ollama model for a JSON value constrained by a caller-supplied JSON Schema. Structured output stays on the existing `127.0.0.1:11434` adapter. It does not introduce a cloud provider, download a model, execute tools, or evaluate schema code.

## Safety and output bounds

- JSON Schema Draft 2020-12 subset only. Schema references (`$ref`), remote resolution, regex/pattern features, conditional applicators, custom extensions, and unknown keywords are rejected before model invocation.
- Schemas are limited to 32 KiB, 8 nesting levels, and 128 schema nodes.
- The supported types are object, array, string, integer, number, boolean, and null.
- Objects are default-deny: only declared properties are accepted and `additionalProperties` must be `false`.
- At most 64 object properties and 256 array items are accepted. Strings are limited to 16,384 characters.
- Enum lists are limited to 32 values; enum and const JSON values are limited to 8 KiB.
- Structured generation is capped at 512 requested output tokens. The accepted response is capped at 65,536 characters.
- Output must be one strict JSON value. Duplicate object keys, non-finite numbers, malformed JSON, incorrect types, unknown properties, and constraint violations are rejected. AXIOM returns no accepted structured object unless validation succeeds.
- Up to eight concise validation errors are returned internally; API callers receive a generic model-output failure to avoid exposing raw generated output in error messages.
- The normal runtime trace records only metadata and safe error codes, never the schema, prompt, or generated text.

Supported keywords are `type`, `properties`, `required`, `additionalProperties: false`, `items`, `minItems`, `maxItems`, `minLength`, `maxLength`, `minimum`, `maximum`, `enum`, `const`, `title`, and `description`. Each object node must declare `type`.

## Example schema

Save as `response.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "answer": { "type": "string", "minLength": 1, "maxLength": 500 },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "sources": {
      "type": "array",
      "items": { "type": "string", "maxLength": 240 },
      "maxItems": 8
    }
  },
  "required": ["answer", "confidence", "sources"],
  "additionalProperties": false
}
```

## CLI

Use a regular local UTF-8 JSON file. AXIOM rejects symlinked, non-regular, oversized, malformed, or unsupported schema files.

```bash
axiom --no-banner infer run qwen2.5:0.5b \
  --prompt "Explain why local-first inference is useful." \
  --schema-file ./response.schema.json \
  --num-predict 256 \
  --json
```

The command sends the validated schema to Ollama's structured-output format and validates the returned JSON again before printing. The JSON report includes `structured_output` and `structured_output_validated: true` only after local validation succeeds. `--num-predict` must be 1–512 in this mode.

## API

Use `POST /api/inference/generate/structured`:

```json
{
  "model": "qwen2.5:0.5b",
  "prompt": "Return an answer and confidence.",
  "num_predict": 256,
  "temperature": 0.1,
  "json_schema": {
    "type": "object",
    "properties": {
      "answer": { "type": "string", "minLength": 1 },
      "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
    },
    "required": ["answer", "confidence"],
    "additionalProperties": false
  }
}
```

A successful response returns the ordinary local inference metadata plus:

- `structured_output`: the parsed JSON value.
- `structured_output_validated`: `true` only after schema validation.

Invalid request/schema values return HTTP 422. A model response that is not valid JSON matching the schema returns HTTP 502. A local runtime failure returns HTTP 503. Responses include the request ID in the body and `X-Request-ID` header. Structured generation is a side-effecting model invocation and therefore requires the write API scope when API scopes are enabled.

## Control Center

Open **Runtime → Local model inference**. Expand **Structured JSON output · optional**, paste a supported schema, and run the prompt. When the schema field is empty, the existing streaming free-text generation path remains unchanged. When a schema is supplied, AXIOM uses a bounded non-streaming structured call and displays the parsed object only after validation, together with a **SCHEMA VALIDATED** indicator.

See [Local inference](../Local-Inference.md) for runtime setup and privacy boundaries, and [API reference](../API.md) for the local API contract.
