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

### Complete top-level command index

| Area | Registered root commands |
|---|---|
| Workspace and diagnostics | `axiom doctor`, `axiom info`, `axiom status`, `axiom summary`, `axiom guide`, `axiom check`, `axiom project`, `axiom config`, `axiom roadmap`, `axiom support-bundle`, `axiom workspace` |
| Model lifecycle and access | `axiom model`, `axiom hf` |
| Data and workflows | `axiom dataset`, `axiom workflow` |
| Training and evaluation | `axiom train`, `axiom experiment`, `axiom evaluation` |
| Local inference and providers | `axiom infer`, `axiom engine`, `axiom llm`, `axiom ai`, `axiom optimize`, `axiom supercompress` |
| Agent, tools, and integrations | `axiom agent-profile`, `axiom mcp`, `axiom integration`, `axiom tools`, `axiom session`, `axiom headroom`, `axiom mobile-bridge` |
| Host and application lifecycle | `axiom system`, `axiom version`, `axiom init`, `axiom daemon`, `axiom gui`, `axiom ui`, `axiom completion` |

Useful command examples:

```bash
axiom model list --json
axiom dataset inspect ./data/train.jsonl --json
axiom train plan 7 --method qlora --json
axiom workflow list --limit 20 --page-json
axiom experiment print <experiment-id>
axiom experiment diff <baseline-id> <candidate-id> --json
axiom evaluation list --json
axiom engine discover --json
axiom infer service status --json
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
| Training plans and saved workflows | `train`, `workflow`, `experiment` | `/api/fine-tuning/jobs`, `/api/workflows`, `/api/workspace/experiments` | Training, Evaluation |
| Offline evaluation | `evaluation`, `experiment` | `/api/evaluation/runs`, `/api/workspace/experiments` | Evaluation |
| Local inference and engine | `infer`, `engine`, `system`, `headroom` | `/api/inference/generate`, `/api/inference/status`, `/api/engine/status`, `/api/engine/telemetry` | Runtime |
| LLM providers and optimization | `llm`, `ai`, `optimize`, `supercompress` | `/api/llm/providers`, `/api/llm/models`, `/api/llm/chat`, `/api/optimizer/plan` | Runtime, Model Studio |
| Agents, permissions, and connectors | `agent-profile`, `mcp`, `integration`, `tools`, `session`, `mobile-bridge` | `/api/agent-profiles`, `/api/security/tool-permissions`, `/api/diagnostics/runtime-connectors` | Agent Profiles, Settings, API Playground |

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
