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
