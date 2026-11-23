# Workflow Design and Execution Boundaries

Beta.4 improves workflow design discoverability, templates, versioned saves, duplication, revision history, structural validation, and dry-run estimates.

## Inspect a saved design

    axiom workflow list
    axiom workflow show <workflow-id>
    axiom workflow validate <workflow-id>

Use `axiom workflow <command> --help` to check flags supported by the installed version.

An MCP agent may also save a validated graph with `axiom_workflow_save`. The
tool is separately default-deny and writes the approved design to the local
`.axiom/workflows.json` store. It accepts only the existing design node types
and reports `execution_status: not_executable`; it does not run nodes or create
jobs. The workspace retains up to 200 designs and rejects new saves at the
limit rather than silently removing older designs. The agent should show the
exact graph and get explicit user approval before calling it:

    axiom mcp allow axiom_workflow_save

The tool response includes the saved `workflow.id` and graph. To inspect that
design later, use `axiom workflow show <workflow.id>` or
`axiom workflow validate <workflow.id>` from the same workspace.

MCP agents can also use `axiom_workflow_read` with its own default-deny
permission to list bounded summaries or fetch one validated graph by ID. Pass
the returned `next_cursor` as `after_workflow_id` with the same `limit` to
continue; start a fresh listing to include designs saved after paging began.
Read access does not grant the separate write permission; every response
labels a saved design `execution_status: not_executable`:

    axiom mcp allow axiom_workflow_read

## Schema v2 and typed graph ports

Workflow graphs use schema version 2. Each supported node type receives a
backend-owned port schema with named input/output ports, data types, required
flags, and optional "one of" input requirements. The current design node types
are:

- `dataset.profile`: emits a `dataset` and a generic `artifact`.
- `model.inspect`: emits a `model` and a generic `artifact`.
- `training.plan`: requires both `dataset` and `model`; emits a `plan`.
- `evaluation.run`: requires a `plan`; can also accept a `dataset`; emits an `evaluation`.
- `report.export`: accepts dataset, plan, evaluation, or generic artifact inputs and requires at least one of the first three.

Connections now identify `source_port` and `target_port` as well as their
source/target node IDs. The backend checks that both ports exist, their data
types match, each target input has at most one connection, the graph is
acyclic, and required inputs are connected. Structural/port errors are
rejected; missing required inputs are returned as validation warnings so an
incomplete design can still be saved and edited.

The Models/Workflows graph editor displays node port schemas, automatically
connects compatible typed ports when adding a step, and allows explicit
connections between selected nodes. Branching connections are stored and
autosaved as graph edges; removing a node removes only its attached edges.
Connection rendering is based on the saved graph rather than the order in
which nodes were added.

Schema-v1 workflows migrate when read: the backend assigns authoritative
ports and upgrades untyped legacy edges. When a v1 edge has no matching
specific data type, it is preserved through the generic `artifact` ports.
The migration does not delete old workflow data. Saving an updated migrated
record writes schema version 2 and retains normalized prior revision snapshots.
Schema-v2 node ports are authoritative; client-supplied port definitions must
match the backend-owned schema.

The validation endpoint remains backend-owned:
`POST /api/workflows/validate` returns structural validity, node/connection
counts, and any missing input warnings. A valid graph or dry-run remains a
design/planning result only; nodes are not executed and no training or
evaluation jobs are created.

## Design is not execution

A valid design or successful dry run does not prove a workload ran. These design tools do not themselves start training/fine-tuning, run inference, create checkpoints, provide production model serving, or prove an external framework is installed and connected.

Use a supported external runtime/framework to execute work, and treat a runtime as connected only when it has actually been detected and verified.

## Planning is not benchmarking

Hardware-aware sizing uses detected hardware and assumptions. It is useful for comparing plausible starting configurations, not for claiming measured tokens per second, actual VRAM use, or guaranteed fit.

## Evaluation is not inference

`axiom evaluation run` scores predictions already present in the evaluation data and does not call a model. Separately, the permissioned MCP prediction-job tools can generate a bounded local artifact for that offline evaluator.

## Roadmap integrity

`axiom roadmap audit --json` reports consistency problems; it does not mark items complete merely because a file exists. Only update roadmap status after the implementation, tests, documentation, and applicable platform gates have been reviewed.
