# AXIOM CLI Reference

This page is the verified top-level command index and the CLI/API/Control Center
parity map for the current AXIOM workspace. It is paired with the
[CLI guide](../CLI-GUIDE.md) for usage details and the
[Local API reference](../API.md) for request schemas, bounds, and permissions.

## Discover the installed command surface

```bash
axiom --help
axiom model --help
axiom train --help
axiom --version
```

The root command index below is checked against AXIOM's registered Typer
command tree in the test suite. Subcommands and options can differ by
installation extras or platform; use the installed `--help` output as the
final authority for those details.

Prompt template usage, variable types, safety limits, and API examples are in the [Prompt Studio guide](Prompt-Studio.md).
Structured local JSON output, schema limits, and API examples are in the [Structured Output guide](Structured-Output.md).

### Complete top-level command index

| Area | Registered root commands |
|---|---|
| Workspace and diagnostics | `axiom doctor`, `axiom info`, `axiom status`, `axiom summary`, `axiom guide`, `axiom check`, `axiom project`, `axiom config`, `axiom roadmap`, `axiom support-bundle`, `axiom workspace` |
| Model lifecycle and access | `axiom model`, `axiom hf` |
| Data and workflows | `axiom dataset`, `axiom workflow`, `axiom knowledge` |
| Training and evaluation | `axiom train`, `axiom experiment`, `axiom evaluation` |
| Local inference and providers | `axiom infer`, `axiom engine`, `axiom llm`, `axiom ai`, `axiom optimize`, `axiom supercompress` |
| Agent, tools, and integrations | `axiom agent-profile`, `axiom prompt`, `axiom mcp`, `axiom integration`, `axiom tools`, `axiom session`, `axiom headroom`, `axiom mobile-bridge` |
| Host and application lifecycle | `axiom system`, `axiom version`, `axiom update`, `axiom init`, `axiom daemon`, `axiom gui`, `axiom ui`, `axiom completion` |

Useful command examples:

```bash
axiom model list --json
axiom dataset inspect ./data/train.jsonl --json
axiom dataset profile ./data/train.jsonl --json
axiom ai hf-search qwen --limit 10 --json
axiom ai policy-audit ./models/my-model --json
axiom ai plan --model ./models/my-model --json
axiom project validate --json
axiom check --json
axiom tools list --json
axiom tools doctor --json
axiom tools plan claude-code --json
axiom knowledge base create "Engineering notes"
axiom knowledge ingest BASE_ID documents/guide.md
axiom knowledge search BASE_ID --query "local inference" --json
axiom train plan 7 --method qlora --json
axiom workflow list --limit 20 --page-json
axiom experiment print <experiment-id>
axiom experiment diff <baseline-id> <candidate-id> --json
axiom evaluation list --json
axiom evaluation run ./data/eval.jsonl --prediction-field prediction --reference-field reference --experiment-id <experiment-id> --json
axiom evaluation compare-experiments <baseline-experiment-id> <candidate-experiment-id> --metric token_f1_micro --json
axiom engine discover --json
axiom engine profile create workstation --model-path ./models/qwen --device auto --context-length 8192 --max-new-tokens 512 --json
axiom engine fabric-profile list --json
axiom engine fabric-profile save workstation-ollama --backend ollama --model qwen2.5:0.5b --json
axiom engine fabric-profile connect PROFILE_ID --json
axiom update --check
axiom update --auto
axiom gui --refresh
axiom infer service status --json
axiom infer run qwen2.5:0.5b --prompt "Return an answer and confidence." --schema-file ./response.schema.json --num-predict 256 --json
axiom workspace metadata export --output ./axiom-workspace-metadata.zip
axiom config validate --path ./axiom.yaml --json
```

## CLI, API, and Control Center parity

The API paths below are representative entry points used by the typed web
client; they are not an exhaustive list of every route or option. GUI names
refer to Control Center navigation surfaces. Parity is feature-family level,
not a promise that every CLI flag has an identical HTTP parameter or GUI
control. The API remains bounded by its route schemas and permission policy.

| Feature family | CLI entry points | Local API surface | Control Center surface |
|---|---|---|---|
| Workspace and diagnostics | `doctor`, `info`, `status`, `summary`, `guide`, `check`, `project`, `config`, `roadmap`, `support-bundle`, `workspace` | `/api/health`, `/api/diagnostics/workspace`, `/api/diagnostics/roadmap` | Overview, Roadmap, Settings |
| Models and model acquisition | `model`, `hf` | `/api/workspace/models`, `/api/models/studio/remote/models`, `/api/models/studio/jobs` | Models, Model Studio |
| Datasets and data quality | `dataset` | `/api/workspace/datasets`, `/api/workspace/dataset-cleaning-preview`, `/api/workspace/dataset-quality` | Datasets |
| Workspace knowledge and retrieval | `knowledge base`, `knowledge ingest`, `knowledge search` | `/api/workspace/knowledge-bases` and document/search routes | Runtime |
| Training plans and saved workflows | `train`, `workflow`, `experiment` | `/api/fine-tuning/jobs`, `/api/workflows`, `/api/workspace/experiments` | Training, Evaluation |
| Offline evaluation | `evaluation`, `experiment` | `/api/evaluation/runs`, `/api/workspace/experiments` | Evaluation |
| Local inference and engine | `infer`, `engine`, `system`, `headroom` | `/api/inference/generate`, `/api/inference/generate/structured`, `/api/inference/status`, `/api/engine/status`, `/api/engine/telemetry` | Runtime |
| LLM providers and optimization | `llm`, `ai`, `optimize`, `supercompress` | `/api/llm/providers`, `/api/llm/models`, `/api/llm/chat`, `/api/optimizer/plan` | Runtime, Model Studio |
| Agents, permissions, and connectors | `agent-profile`, `mcp`, `security`, `integration`, `tools`, `session`, `mobile-bridge` | `/api/agent-profiles`, `/api/security/audit`, `/api/security/tool-permissions`, `/api/diagnostics/runtime-connectors` | Agent Profiles, Settings, API Playground |
| Prompt engineering | `prompt` | `/api/prompt-templates` | Prompt Studio |
| Structured local inference | `infer run --schema-file` | `/api/inference/generate/structured` | Runtime → Local model inference |

## Update and GUI refresh

`axiom update --check` inspects the beta channel without installing. Use
`--channel stable` to exclude prereleases, or `axiom update --auto` to fetch
and install the exact newer release. The updater maps AXIOM beta tags to
canonical PEP 440 versions, verifies the PyPI SHA-256, and runs AXIOM's
non-executing wheel inspector to check package metadata, archive paths, the
complete RECORD manifest, and each recorded file hash/size before invoking pip
in the active Python environment. It does not operate on model directories,
workspaces, account databases, configuration or browser storage. Restart AXIOM
after installation.

`axiom gui --refresh` is a source-checkout workflow: it runs frontend lint and
production build. Existing tabs detect the new bundle through a no-cache
fingerprint endpoint and reload in place on the same origin. The app shell is
revalidated while hashed assets remain cacheable; localStorage and
sessionStorage (including the current tab session) remain intact. A running API
is reused without a duplicate process or tab. Install the source checkout once
with `python -m pip install -e .` to expose the `axiom` console command.

## Security commands

- `axiom security` reviews workspace permissions, MCP grants, API token-scope presence, CORS origin safety, package integrity, and the configured local network posture.
- `axiom security audit --json` emits the redacted, machine-readable posture report.
- `axiom security artifact inspect ./dist/<artifact> --json` checks a wheel or source archive without extracting or executing its contents.
- `axiom security events list --json`, `verify --json`, `review <event-id> <reviewer-label>`, and `export` list, verify, acknowledge, and export the redacted local security journal.

## Commands without a one-to-one GUI/API action

Several commands operate on the local host or launch a service rather than
representing a workspace resource. They intentionally remain CLI entry
points:

- `completion` prints shell completion scripts; it does not change shell
  configuration unless the user separately invokes Typer's install option.
- `version` reports the installed package version.
- `init` creates a project directory and starter files.
- `config show/validate` reads a user-selected local YAML path. Validation
  only reports advisory suggestions and never writes the file; it is not
  exposed as an arbitrary-path network API.
- `daemon` starts the local API process. `gui` and `ui` start/open the
  Control Center. These are host-process lifecycle commands, not API routes.
- `mobile-bridge` runs a separate OpenAI-compatible local bridge with its
  own explicit listen and access controls.

Conversely, some API and GUI panels expose pagination, status, diagnostics,
or cancellation actions that are not represented by one single CLI command.
Use `docs/API.md` for the exact route contracts and `docs/CLI-GUIDE.md`
for CLI-specific safety, output, cancellation, and exit-code behavior.

## Distribution

Python distribution: `axiom-all`

npm launcher: `@netcore-technologies/axiom-cli`


## Declarative plugin manifests

The root command is `axiom plugin`; it groups the manifest inventory and permission operations below.

- `axiom plugin list` displays registered manifest versions, fingerprints, and permission status.
- `axiom plugin manifest validate ./plugin-manifest.json` validates metadata without registering or granting access.
- `axiom plugin manifest register ./plugin-manifest.json` registers a version; permissions remain denied.
- `axiom plugin grant com.example.research --manifest-sha256 <sha256> --tool search:read` grants explicitly selected declared scopes against the exact current manifest.
- `axiom plugin revoke com.example.research` revokes all grants.

Manifest registration does not load code or execute commands. See [Plugin manifests](../Plugin-Manifests.md) for the schema and permission lifecycle.

## Hardware snapshot history

- `axiom system snapshots enable --max-snapshots 20 --max-age-days 90` explicitly enables capture and sets retention.
- `axiom system snapshots capture --json` captures one snapshot; it refuses to run while capture is disabled.
- `axiom system snapshots list --limit 20 --json` lists retained records without probing hardware.
- `axiom system snapshots disable` prevents future captures but preserves retained records.
- `axiom system snapshots clear --yes` permanently deletes all retained snapshots.

The Dashboard provides the same opt-in, capture, retention and clear operations. Nothing is sampled automatically.


### Generated API contract examples

- `axiom api` prints a Markdown reference derived from the installed OpenAPI schema.
- `axiom api --format json` prints machine-readable request/response schemas and synthetic examples.
- `axiom api --output ./api-examples.md` creates a new file and refuses to overwrite an existing file. Generation never submits requests.


## Engine resource telemetry

- `axiom engine telemetry --json` returns one privacy-bounded point-in-time sample.
- `axiom engine telemetry --watch --interval-seconds 2 --samples 30` explicitly opts into a bounded series of samples.
- Add `--output ./resource-samples.jsonl` to write a new JSONL file; an existing file is never overwritten. `--output` is rejected unless `--watch` is enabled.


## Benchmark resource trends and telemetry export

The Runtime page automatically takes two point-in-time telemetry snapshots only when an operator explicitly starts a native benchmark. It links the snapshots to that benchmark in a bounded 20-run browser-memory trend view. Missing system/driver values are labelled `Not reported`; the charts do not infer unavailable metrics.

The Resource Monitoring panel supports JSON and CSV export. Choose which metric groups are included before exporting; process IDs and prompt/model-path/credential values are not exported. Clear history removes the browser's retained samples and stops active monitoring.
