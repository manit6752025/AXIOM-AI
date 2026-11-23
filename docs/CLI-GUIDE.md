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

## Security audit

```bash
axiom security audit
axiom security audit --json
axiom security audit --strict --json
```

The audit checks local configuration and MCP policy permissions, active tool
grants, API token-scope configuration, and CORS origins. It never prints token
values. The `--daemon-host` and `--allow-network` options evaluate a proposed
listener profile; they do not inspect or modify a live listener. Non-loopback
daemon profiles are reported as unsafe because that lightweight HTTP service
does not provide authentication.

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

## Progress and cancellation

Local Model Studio downloads and GGUF conversions show byte progress on an
interactive terminal; dataset profiling, manifest hashing, cleaning previews,
and JSONL cleaning show row/byte progress on stderr. Redirected/JSON output
remains clean, and progress is suppressed by `--quiet`.

Press **Ctrl+C once** to request cooperative cancellation at the next file
chunk, row, or subprocess boundary. The CLI exits with status `130`. Partial
staged model downloads and temporary dataset-cleaning output are removed; a
cleaning operation never changes its original source. Model Studio jobs also
retain their API/GUI cancellation endpoint and persistent status history.

Local LoRA training jobs persist the states `queued`, `running`,
`cancelling`, `finalizing`, `completed`, `cancelled`, and `failed`.
Use `axiom train status JOB_ID --json` to inspect the current state and its
cancellation contract, or `axiom train cancel JOB_ID --json` to request a
cooperative stop. A request is not a completed cancellation: poll status until
the job reaches `cancelled`. Cancellation is accepted before adapter
finalization and is acted on before the worker starts or between training
steps; once `finalizing` begins, adapter writing is not interrupted. The
response's `cancellation.requestable` and `unavailable_reason` fields are
authoritative.

## Non-interactive execution

Use the root `--non-interactive` (alias `--no-input`) option before the
command to disable all prompts. AXIOM also refuses to prompt when standard
input is not a terminal, even when the flag is omitted. Supply required values
through command arguments/options or environment variables instead.

```bash
axiom --non-interactive optimize run --profile 2 --model ./models/qwen --target-tps 12
AXIOM_LLM_API_KEY="$MY_PROVIDER_KEY" axiom --non-interactive llm chat qwen "Hello" --provider openai --send-to-provider
axiom --non-interactive model studio remote-models https://provider.example/v1 --api-key-env AXIOM_LLM_API_KEY
```

The Model Studio `wizard` command is explicitly interactive; use the
`search`, `files`, `pull`, `pull-native`, `install-gguf`, or `convert-gguf`
commands for scripted operations. Remote prompt submission requires
`--send-to-provider` in non-interactive mode. API keys should be provided
through environment variables, not command-line arguments.

Destructive confirmations default to **no** when input is unavailable.
Existing `--yes` flags remain the explicit opt-in for deletion/removal.

## Security checks and release artifacts

Run the local posture audit and inspect release packages without extracting
or executing them:

```bash
axiom security audit --json
axiom security artifact inspect ./dist/axiom_all-0.3.0-py3-none-any.whl --json
axiom security artifact inspect ./dist/axiom-all-0.3.0.tar.gz --strict --json
```

The artifact command validates wheel metadata and `RECORD` hashes, or checks
source-archive paths, links, special entries, duplicate names, and size limits.
It reads artifacts as data only. It does not verify publisher signatures or
consult vulnerability databases. Review non-zero exit codes before publishing.


Inspect and review the bounded local security event journal:

```bash
axiom security events list --json
axiom security events verify --json
axiom security events review <event-id> "Security Admin"
axiom security events export > axiom-security-events.json
```

MCP profile creation, permission updates, and profile activation are recorded
using fixed summaries. Review labels are self-attested, not authenticated
identities. The export is redacted JSON; retain it outside the workspace for
independent retention. A full journal refuses new entries rather than silently
rotating history.

## Machine-readable CLI errors

Place `--json-errors` before the command to emit a schema-versioned error
envelope on standard error while preserving the command's non-zero exit code.
The envelope shape is:

```json
{"schema_version":1,"error":{"code":"model_studio_error","message":"...","exit_code":1}}
```

The same emitter is used by root commands and the inference, engine, training,
Model Studio, LLM, workflow, evaluation, tools, agent-profile, support-bundle,
dataset, and workspace command groups. Command-level `--json` remains the
success/report format; global `--json-errors` takes precedence only when a
command fails. Exit codes are unchanged, including exit code 2 for unsafe
support-bundle output paths. Shell/Click argument parsing errors that occur
before the root command callback are still rendered by Click itself.

## CLI exit codes and automation contract

| Exit code | Meaning | Typical cause |
|---:|---|---|
| `0` | Success | The requested command completed. |
| `1` | Operation or validation failure | A local resource is invalid or missing, or an operation could not complete. |
| `2` | Usage or safety-policy rejection | Click/Typer argument parsing failed, or a support-bundle output path was unsafe or already existed. |
| `130` | User cancellation | The user pressed **Ctrl+C** and a cooperative operation stopped at a safe boundary. |

Check the process exit code first. With `--json-errors`, command failures that
reach AXIOM's command error handler also emit this schema-versioned envelope
on standard error:

```json
{"schema_version":1,"error":{"code":"workflow_error","message":"...","exit_code":1}}
```

Treat `error.code` and `error.exit_code` as the machine-readable contract;
`error.message` is explanatory text and may become more detailed. Codes are
stable identifiers, for example `workflow_error`, `model_studio_error`,
`llm_error`, `tool_not_found`, `support_bundle_symlink`,
`support_bundle_not_file`, `support_bundle_exists`, and
`support_bundle_invalid_parent`. Command-specific success `--json` output
is unchanged. Errors raised by Click/Typer argument parsing before the root
command callback remain usage errors (exit code `2`) and are not wrapped in
AXIOM's JSON error envelope.

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
file. Stable codes distinguish missing files, non-regular paths, symlinks,
unreadable files, files over 256 KiB, malformed YAML, non-mapping documents,
and known sections with the wrong shape. The validator opens regular files
without following a final symlink and reads no more than 256 KiB plus one byte
before deciding whether the file is oversized. File values are not echoed.

Configuration validation is intentionally a local CLI operation because it
accepts a user-selected filesystem path; it is not exposed as a network API or
a GUI action. Review and apply any suggested edits yourself, then validate
again.

## Machine-readable output

Commands that support `--json` emit a versioned payload on standard output.
For example:

```bash
axiom status --json
axiom summary --json
axiom model list --json
axiom dataset inspect ./data/train.jsonl --json
axiom dataset profile ./data/train.jsonl --json
axiom ai hf-search qwen --limit 10 --json
axiom ai policy-audit ./models/my-model --json
axiom ai plan --model ./models/my-model --json
axiom project validate --json
axiom check --json
axiom tools list --json
axiom config validate --path ./axiom.yaml --json
axiom tools doctor --json
axiom tools plan claude-code --json
axiom engine discover --json
```

Configuration JSON uses safe YAML parsing, rejects symlinks and files over 256 KiB, and redacts likely credential values by key name. It does not return the original raw YAML document.

Tool inventory, tool doctor, and tool install-plan outputs are schema-versioned. They report catalog metadata, local executable presence, and whether a credential environment variable is present; they never return credential values or claim that vendor authentication has been verified.

The dataset profile payload includes aggregate counts, field/type summaries,
privacy flags, optional opt-in PII match counts, and a schema version; it does
not return source row values. The ai hf-search command returns model metadata
only, ai policy-audit returns local policy indicators, and ai plan returns
an estimate with an explicit non-benchmark scope note.

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


## Updating and refreshing the Control Center

```bash
axiom update --check
axiom update --channel stable --check
axiom update --auto
axiom gui --refresh
```

The `axiom` console script is installed by the `axiom-all` Python distribution.
In a source checkout, activate the intended virtual environment and install the
project once with `python -m pip install -e .`. Then use `axiom gui --help`
instead of `python -m axiom.cli.main gui --help`.

`axiom update` checks GitHub releases (beta includes published prereleases;
stable excludes them). `--auto` / `--yes` downloads the matching PyPI
`axiom-all` wheel, maps beta tags to canonical PEP 440 versions, verifies its PyPI SHA-256, wheel metadata, full RECORD hashes, and archive paths with AXIOM's non-executing artifact inspector, then installs it
using the active Python interpreter. The updater only replaces the application
distribution; it does not target model directories, workspaces, configuration,
experiment history, account databases or browser storage. Restart AXIOM after
installation. `--check` never installs.

`axiom gui --refresh` is for source checkouts. It runs `npm run lint` then
`npm run build`. Existing Control Center tabs poll a no-cache build fingerprint
and reload themselves when the new `index.html` is present. The reload keeps
the same origin and browser tab, so localStorage and sessionStorage (including
the active tab session) remain intact. A healthy GUI API already bound to the
configured port is reused; no second API process or duplicate tab is needed.
Keep the port stable (default 40507) to retain the origin's sign-in state.
Packaged installs without frontend source should use a release update instead.
If the build preflight reports that `lucide-react` declarations are missing from `node_modules`, run `npm ci` at the repository root to restore the exact lockfile contents. Do not add a separate `@types/lucide-react` package; the installed Lucide package supplies its own declarations.


## Local text embeddings

~~~bash
python -m pip install -e '.[embeddings]'
axiom engine embed --model-path ./models/my-encoder --text "First document" --text "Second document" --json
axiom engine embed --model-path ./models/my-encoder --input-file ./data/document.txt --device cpu
axiom engine embed-unload
~~~

The optional adapter uses trusted local safetensors checkpoints, bounded
character chunking, attention-mask mean pooling, and optional L2 normalization.
It does not download models, execute remote code, or persist source text.
The JSON response includes vectors and character offsets; see
[Local Embeddings](Local-Embeddings.md) for device and request limits.


## Local cross-encoder reranking

~~~bash
axiom engine rerank --model-path ./models/my-reranker --query "local AI" --document "Candidate one" --document "Candidate two" --json
axiom engine rerank --model-path ./models/my-reranker --query "local AI" --documents-file ./data/candidates.json
axiom engine rerank-unload
~~~

Use a local sequence-classification checkpoint with safetensors weights and a
compatible tokenizer. One-logit models use sigmoid scores; two-logit models
use the positive-class softmax score. Ties preserve input order. Scores are
ranking signals rather than calibrated relevance probabilities. See
[Local Reranking](Local-Reranking.md).


## Workspace knowledge bases

~~~bash
axiom knowledge base create "Engineering notes"
axiom knowledge base list
axiom knowledge ingest BASE_ID documents/guide.md --json
axiom knowledge documents BASE_ID --json
axiom knowledge remove-document BASE_ID DOCUMENT_ID --yes
axiom knowledge base remove BASE_ID --yes
~~~

Knowledge bases are scoped to the active workspace. Ingestion supports UTF-8
TXT, Markdown, RST, CSV, JSON, and JSONL files (up to 8 MiB each); source
paths must be workspace-relative, symlinks and traversal are rejected, and
the original source file is never changed. Private copies and SHA-256 metadata
live under the workspace's .axiom/knowledge directory. ~~~bash
axiom knowledge search BASE_ID --query "local inference" --json
axiom knowledge search BASE_ID --query "local inference" --format md --source-prefix documents/engineering
axiom knowledge search BASE_ID --query "local inference" --model-path ./models/my-encoder --vector-weight 0.35 --device cpu --json
~~~

Search uses bounded BM25 lexical scoring by default. Supplying a trusted local
embedding model path enables dense-vector fusion. Results carry source paths,
SHA-256 hashes, chunk numbers, and character offsets; no hosted embedding API
is used. See [Workspace Knowledge and Retrieval](Local-Knowledge.md).


## Native local serving profiles

~~~bash
axiom engine profile create workstation --model-path ~/models/qwen2.5-7b --device auto --memory-mode layer_streaming --precision bf16 --context-length 8192 --max-new-tokens 512 --json
axiom engine profile list --json
axiom engine profile probe PROFILE_ID --json
axiom engine profile run PROFILE_ID --prompt "Explain the local inference boundary." --json
axiom engine profile delete PROFILE_ID --yes
~~~

Profiles persist trusted local model configuration, request limits and an optional LoRA adapter. Readiness checks do not load model weights or start a listener. The first profile run uses the existing AXIOM Native Engine; current admission is one active request and zero queued requests, with bounded prompt/output/context limits and cooperative timeout cancellation. See [Local Serving Profiles](Local-Serving-Profiles.md).


## Engine Fabric connector profiles

Engine Fabric connector profiles are separate from native model-serving profiles. They store only a fixed local backend ID and exact model ID; credentials and custom endpoints are not persisted.

```bash
axiom engine fabric-profile list --json
axiom engine fabric-profile save workstation-ollama --backend ollama --model qwen2.5:0.5b --json
axiom engine fabric-profile connect PROFILE_ID --json
```

`axiom engine profile` remains the native AXIOM serving-profile interface (`create`, `list`, `show`, `probe`, `run`, and `delete`).
