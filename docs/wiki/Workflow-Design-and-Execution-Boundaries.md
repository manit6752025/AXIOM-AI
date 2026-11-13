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

## Design is not execution

A valid design or successful dry run does not prove a workload ran. These design tools do not themselves start training/fine-tuning, run inference, create checkpoints, provide production model serving, or prove an external framework is installed and connected.

Use a supported external runtime/framework to execute work, and treat a runtime as connected only when it has actually been detected and verified.

## Planning is not benchmarking

Hardware-aware sizing uses detected hardware and assumptions. It is useful for comparing plausible starting configurations, not for claiming measured tokens per second, actual VRAM use, or guaranteed fit.

## Evaluation is not inference

`axiom evaluation run` scores predictions already present in the evaluation data and does not call a model. Separately, the permissioned MCP prediction-job tools can generate a bounded local artifact for that offline evaluator.

## Roadmap integrity

`axiom roadmap audit --json` reports consistency problems; it does not mark items complete merely because a file exists. Only update roadmap status after the implementation, tests, documentation, and applicable platform gates have been reviewed.
