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


## MCP tool authorization audit

Every registered MCP handler checks its server-side permission before doing work. The authorization boundary records both allowed and denied decisions in the workspace security-event journal. If the policy cannot be read or the decision cannot be durably audited, access is blocked.

Decision records contain the tool name, allow/deny outcome, active profile, a bounded reason code, and argument **names**. Each argument name is paired with the constant marker `[REDACTED]`; argument values are never copied, serialized, hashed, or stored. This prevents prompts, local paths, tokens, and other user-supplied values from entering the journal. The Diagnostics security-event table and JSON export expose the same redacted metadata.

The journal uses schema version 2 for decision records and reads existing version-1 journals. The first successful append after reading a version-1 journal upgrades it atomically to version 2 without discarding prior events. Events remain append-only, hash-chained, bounded, and are not silently rotated. When the journal is full or unavailable, MCP tool execution fails closed.
