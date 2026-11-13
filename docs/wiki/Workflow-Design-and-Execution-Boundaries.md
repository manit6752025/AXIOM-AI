# Workflow Design and Execution Boundaries

Beta.4 improves workflow design discoverability, templates, versioned saves, duplication, revision history, structural validation, and dry-run estimates.

## Inspect a saved design

    axiom workflow list
    axiom workflow show <workflow-id>
    axiom workflow validate <workflow-id>

Use `axiom workflow <command> --help` to check flags supported by the installed version.

## Design is not execution

A valid design or successful dry run does not prove a workload ran. These design tools do not themselves start training/fine-tuning, run inference, create checkpoints, provide production model serving, or prove an external framework is installed and connected.

Use a supported external runtime/framework to execute work, and treat a runtime as connected only when it has actually been detected and verified.

## Planning is not benchmarking

Hardware-aware sizing uses detected hardware and assumptions. It is useful for comparing plausible starting configurations, not for claiming measured tokens per second, actual VRAM use, or guaranteed fit.

## Evaluation is not inference

`axiom evaluation run` scores predictions already present in the evaluation data and does not call a model. Separately, the permissioned MCP prediction-job tools can generate a bounded local artifact for that offline evaluator.

## Roadmap integrity

`axiom roadmap audit --json` reports consistency problems; it does not mark items complete merely because a file exists. Only update roadmap status after the implementation, tests, documentation, and applicable platform gates have been reviewed.
