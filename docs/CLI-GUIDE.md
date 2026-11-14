# AXIOM CLI guide

AXIOM's CLI supports human-readable terminal output and versioned JSON for the
commands that document `--json`. Use `axiom --help` and each group's
`--help` to inspect the command options installed on your machine.

## Workflow inventory pagination

```bash
axiom workflow list --limit 20
axiom workflow list --limit 20 --search training
axiom workflow list --limit 20 --after <next-cursor> --search training
axiom workflow list --limit 20 --page-json
```

`--json` keeps the existing JSON-array output for compatibility. `--page-json` includes `total`, `has_more`, and `next_cursor`. Keep the same search term between cursor requests; start a fresh listing to include newly saved designs.

## Global output controls

These root options are placed before the command name:

```bash
axiom --verbose doctor
axiom --quiet status
axiom --json-errors completion tcsh
axiom --quiet --json-errors completion tcsh
```

- `--verbose` / `-v` enables AXIOM diagnostic logging.
- `--quiet` / `-q` suppresses normal Rich terminal output; errors remain visible.
- `--json-errors` emits a compact schema-versioned JSON error on standard error with a stable `code`, `message`, and `exit_code`.
- `--verbose` and `--quiet` are mutually exclusive.
- Existing command-level `--json` flags continue to control successful command output. `--json-errors` only changes error envelopes.

## Shell completion

The root Typer command exposes shell completion. Install completion in the
current user's shell configuration with:

```bash
axiom --install-completion
```

Preview the generated completion script without changing shell configuration:

```bash
axiom --show-completion bash
axiom --show-completion zsh
axiom --show-completion fish
axiom --show-completion powershell
```

The preview command writes the script to standard output. Redirect it to a
file only when you deliberately want to inspect or manage that generated file.
If the shell was already open, restart it or source the appropriate generated
completion script according to the shell's own conventions.

## Configuration validation and repair guidance

The command `axiom config validate --path axiom.yaml --json` emits a
versioned validation report with a stable `error_code` and safe
`suggestions` when it finds a problem. The suggestions are advisory:
`repair_applied` remains false and the command never rewrites the supplied
file. Symlinks are rejected, file contents are not echoed, and oversized files are
rejected before their contents are read.

## Machine-readable output

Commands that support `--json` emit a versioned payload on standard output.
For example:

```bash
axiom status --json
axiom summary --json
axiom model list --json
axiom dataset inspect ./data/train.jsonl --json
axiom engine discover --json
```

JSON mode does not imply that a requested action succeeded: validation and
operation failures keep a non-zero exit code. Commands without a documented
JSON option retain human-oriented output.


## Unified LLM client

List the available adapters and load a model inventory:

```bash
axiom llm providers
axiom llm providers --json
axiom llm models --provider axiom-local
axiom llm models --provider ollama-local
axiom llm models --provider deepseek
axiom llm chat deepseek-chat "Explain KV caching briefly." --provider deepseek --send-to-provider
```

Cloud commands prompt for an API key with hidden input unless the variable selected by `--api-key-env` is already set (`AXIOM_LLM_API_KEY` by default). Cloud chat requires `--send-to-provider` because the prompt and system instruction are sent to the selected service. `--base-url` is only valid with `--provider openai-compatible`, which requires a public HTTPS API base URL ending in `/v1`. Credentials are not saved by AXIOM. See [Unified LLM client](Unified-LLM-Client.md).

## Mobile-agent bridge

The bridge exposes only AXIOM Native Engine model listing and chat completions with a per-process random bearer token:

```bash
axiom engine discover --json
axiom mobile-bridge
# For a separate device on a trusted private network:
axiom mobile-bridge --listen-lan --port 40509
```

Configure an OpenAI-compatible mobile client with the printed base URL and token as its API key. Choose a native model ID from `axiom engine discover --json`, or an installed Ollama model ID with the `ollama/` prefix (for example `ollama/qwen2.5:0.5b`) from `/v1/models`. LAN mode uses HTTP, so never port-forward it; use it only on a trusted private network or through a VPN/encrypted tunnel.

## Local Control Center

Run `axiom gui` to start the API on loopback and open the bundled Control
Center. The default port is `40507`; set `AXIOM_GUI_PORT` to choose a different
default, or pass `--port` explicitly. The explicit command-line option takes
precedence over the environment variable. Port `0` asks the operating system to
select an available port for that launch.

See [API environment configuration](API.md) and [data retention](DATA-RETENTION.md)
for the associated network and local-storage boundaries.


## Local Ollama lifecycle

```bash
# Inspect the live loopback API and service ownership
axiom --no-banner infer service status --json

# Start or stop only a service managed by AXIOM
axiom --no-banner infer service start
axiom --no-banner infer service stop

# Explicitly download an Ollama library model (network and disk space required)
axiom --no-banner infer pull qwen2.5:0.5b
axiom --no-banner infer ps --json
axiom --no-banner infer inspect qwen2.5:0.5b --json
axiom --no-banner infer unload qwen2.5:0.5b
```

Control Center exposes the same service actions in **Runtime → Local model inference**, including bounded progress reporting and cancellation for model pulls, loaded-model memory inventory, metadata inspection, and memory unload. `axiom infer ps --json` lists point-in-time runtime-reported sizes and VRAM placement without claiming exact allocator telemetry. Inspection returns only bounded metadata summaries; it does not return template/license text or prompt contents. Unload releases the selected model from runtime memory and does not delete downloaded files. AXIOM-managed service startup binds to `127.0.0.1:11434`. AXIOM will not stop an independently managed service.


### Ollama generation controls

```bash
axiom --no-banner infer run qwen2.5:0.5b --prompt "Explain AXIOM" \
  --top-p 0.83 --top-k 36 --repeat-penalty 1.2 --seed 123 \
  --num-ctx 8192 --stop DONE --stop END --json
```

The Runtime GUI exposes the same settings under **Advanced sampling options**. Leave seed and context size blank to use the runtime defaults; repeat `--stop` for additional stop strings.
