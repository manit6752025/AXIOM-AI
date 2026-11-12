# AXIOM Agent Integrations

AXIOM is designed to sit underneath external AI-agent runtimes as a local
inspection and planning layer.

## Supported integration families

### MCP

Model Context Protocol is the preferred integration path.

It allows an agent runtime to discover AXIOM capabilities as tools.

Examples include OpenClaw and PicoClaw.

Agents can begin with the read-only `axiom_workspace_context` tool for one
structured snapshot of project layout, hardware, external developer-tool
availability, the tools allowed by the active profile, and a suggested next
command. It omits absolute paths, executable paths, and configuration
contents. Suggestions are never executed; agents should get user approval
before running a suggested shell command. Hardware detection may query
`nvidia-smi` locally when available; the tool does not modify workspace files
or access the network. Each MCP tool, including this one, remains denied until
it is granted in the active AXIOM tool profile.

The separately permissioned `axiom_workspace_assets` tool lists up to 200
workspace-relative model directories and JSONL dataset paths, with a 50,000
entry scan budget per category. The response distinguishes reaching the
result cap from a truncated or incomplete scan. It does not read file contents
or follow symlinked entries, and omits the absolute workspace path from its
response.
Both workspace tools use the MCP server process's working directory as the
workspace root. Start the server from the target project or set the client's
working-directory option where supported. Pass asset paths to
`axiom_model_info` or `axiom_dataset_inspect` for details. MCP permissions are
stored in `.axiom/tool_permissions.json` under that same root, so run
`axiom mcp allow` from the project where the server will run.

The separately permissioned `axiom_training_preflight` tool checks a local
model directory, validates dataset text using the trainer's bounded rules,
checks training-package metadata, and includes a heuristic hardware estimate.
It reads dataset text locally but never returns its values, loads model
weights, creates a job, writes files, or accesses the network. Package checks
read presence and version metadata only; they do not import or initialize
training dependencies or verify a device backend. Its verdict is
`blockers_found`, `review_needed`, or `no_known_blockers`; the final status is
not a guarantee that training will load or fit. AXIOM's current local trainer
supports LoRA, not QLoRA, and the generic planner may suggest a QLoRA fallback.
Hardware status `warn` means only the generic estimate predicts a fit; actual
loading was not tested. The estimate also cannot verify Apple MPS or CPU fit.

The separately permissioned `axiom_dataset_clean_preview` reports bounded
keep/remove counts and a proposed output path for a workspace-relative JSONL
file. It returns no row values and writes no file. A partial scan includes its
truncation reason and its counts cover only the scanned portion; overlong rows are reported as unclassified because the
cleaner may still keep them. CR-only line endings and invalid UTF-8 are flagged
because the preview cannot promise matching cleaner results for those inputs.
The preview is advisory because the source can change between calls; re-preview
immediately before the separately permissioned `axiom_dataset_clean` write
action. That action only accepts workspace-relative paths, writes a new
`.jsonl` file, requires an existing parent directory, and refuses to overwrite
an existing output. It repeats the preview and enforces the same byte, row, and
line limits while writing. Review the counts and proposed destination with the
user. Both paths use descriptor-relative access where the OS supports it and
checked workspace-relative paths elsewhere, including Windows. On systems
without descriptor-relative filesystem operations, concurrent changes to a
parent directory can race the path checks; the fallback is scoped against
static traversal and refuses existing outputs, but does not pin directory
handles against another process changing the workspace at the same time.

### CLI

AXIOM can also integrate with local agent binaries.

### HTTP

The current HTTP surface is the local daemon. It binds to loopback by default
and exposes read-only health, project-summary, hardware, action, and tool
information. It is not a hosted API or a remote agent service.

### A2A

Future versions can expose agent-to-agent capabilities for systems that support A2A-style communication.

## Philosophy

AXIOM should make local AI preflight work legible.

External agents decide:

- what the user wants
- what tools to call
- how to communicate

AXIOM currently handles:

- model metadata and local model inspection;
- JSONL dataset inspection, validation, statistics, and cleaning;
- hardware detection and conservative training-plan estimates;
- local project state and developer-tool discovery.

External runtimes handle training, evaluation, inference, and deployment. AXIOM
does not claim that a plan ran or that a detected tool is authenticated.

## Optional Headroom proxy

[Headroom](https://github.com/headroomlabs-ai/headroom) is an external local
context-compression proxy. When it is installed separately, its documented
entry points are:

```bash
headroom doctor
headroom proxy
headroom dashboard
```

AXIOM's role is limited to discovering or guiding a developer toward Headroom;
it does not bundle, install, start, or supervise it. Starting the proxy or
dashboard is always the user's explicit action. AXIOM makes no savings claim for Headroom;
any effect must be measured in the user's own workload and environment.

This separation keeps the architecture modular and prevents AXIOM from becoming tied to a single agent runtime.
