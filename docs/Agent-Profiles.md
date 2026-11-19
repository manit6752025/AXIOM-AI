# Agent profiles

Agent Profiles is a workspace-local catalogue of versioned **task-intent metadata**. The default profile has no task labels. Profiles are stored at `.axiom/agent-profiles.json` with schema version 1 and are written atomically with restrictive file permissions where the host supports them.

## Supported task labels

- `model.inspect`
- `dataset.profile`
- `training.plan`
- `evaluation.review`
- `runtime.inspect`
- `mcp.inspect`

Each update increments the profile version. The built-in `default` profile cannot be deleted, names and IDs must be unique, the store has a profile-count and file-size limit, and symlinked `.axiom`/store paths are rejected. The CLI's delete command requires `--yes`.

**Important security boundary:** task labels are descriptive metadata only. They do not grant MCP tools, shell execution, network access, or actual job capabilities. MCP access remains controlled by the independent server-side default-deny policy. Runtime/agent execution adapters are not implemented by this metadata store.

## CLI

```bash
axiom agent-profile list
axiom agent-profile list --json
axiom agent-profile create "Dataset review" --capability dataset.profile --json
axiom agent-profile set-capabilities <profile-id> --capability dataset.profile --capability model.inspect
axiom agent-profile delete <profile-id> --yes
```
