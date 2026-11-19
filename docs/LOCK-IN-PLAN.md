# AXIOM lock-in plan

**Purpose:** make AXIOM a dependable terminal-first companion for local AI
work, then ship improvements in small, reviewable releases.

**Product sentence:** AXIOM helps a developer understand what their model,
data, and machine can do before they spend time or compute.

**Scope lock:** the product is a landing page, a Python CLI, and a local
loopback daemon. There is no hosted control center, fake telemetry, account
system, or dashboard-shaped substitute for useful commands.

Each page below is an execution surface, not a list of vague ideas. A feature
is ready to ship only when it has a clear user, a safe failure mode, tests, and
documentation that says exactly what it does.

---

## Page 1 — Product contract

### The user we serve

The primary user is a developer who has a model directory, a dataset, a local
machine, or an AI coding tool and wants an honest answer to: “What should I do
next?” They should not need to understand AXIOM’s internals before getting
value.

### The promise

AXIOM makes four local facts legible:

1. What is in the model or model directory?
2. Is the dataset structurally usable?
3. What hardware is actually available?
4. Which next command is safe and relevant?

The daemon makes those facts available to local scripts through a free
loopback port. It does not imply that training ran, that a benchmark was
measured, or that a provider account is configured.

### The non-negotiables

- Local by default. Bind to loopback and choose an operating-system port when
  no port is supplied.
- Truthful output. Label estimates, detections, plans, and measured results as
  different evidence classes.
- Terminal first. A browser is useful for the public explanation, not a reason
  to build a control center.
- Safe by default. Never print or persist API-key values; never execute a
  remote installer without explicit user intent.
- Every visible word earns its place. Remove decorative copy before adding
  another card, badge, metric, or section.

### The decision filter

Before accepting a new feature, ask: does it shorten the path from a local AI
question to a verified next action? If not, it belongs in a later experiment or
does not belong in AXIOM.

---

## Page 2 — CLI golden path

### First-run experience

The bare command is the front door:

```text
axiom
```

It shows the startup mark once per terminal session, explains AXIOM in one
sentence, prints the small set of useful commands, starts the daemon on a free
loopback port, and stays attached until Ctrl-C. `axiom --help` remains the
complete reference; `axiom guide` remains the non-blocking explanation.

### Golden commands

```text
axiom guide
axiom model inspect ./models/my-model
axiom dataset validate ./data/train.jsonl
axiom system info
axiom train plan 7 --method qlora
axiom tools doctor
```

The commands must work without an AXIOM account. Network-dependent commands
must identify that boundary before making a request.

### Next CLI improvements

- Keep the non-blocking `axiom summary` command focused on the current project,
  hardware, tool presence, and one recommended next command.
- Add `--json` to local inspection and summary commands where automation has a
  clear use case.
- Make error messages name the missing path, explain why it matters, and give
  one copyable fix.
- Keep command names stable. Add aliases only when they remove confusion; do
  not create multiple names for the same concept without documentation.
- Ensure Ctrl-C, broken pipes, missing paths, and invalid YAML exit cleanly.

### Acceptance test

From an empty temporary directory, a new user can run `axiom guide`, follow its
next command, inspect a fixture model, validate a fixture JSONL file, read
hardware, and generate a plan without network access. The test should assert
the output and exit code, not merely that Python imports succeed.

---

## Page 3 — Local daemon

### Current boundary

The daemon is dependency-free HTTP on a loopback address. Its stable routes
are:

```text
GET /health
GET /api/info
GET /api/summary
GET /api/actions
GET /api/hardware
GET /api/tools
```

These routes are intentionally JSON, small, and inspectable with `curl`. The
root response should link to the routes without pretending to be a web app.

### Current high-value feature

`GET /api/summary` combines the current working directory, whether
`axiom.yaml` exists, missing standard project paths, detected hardware,
available executables, and the next recommended local command. Keep it a
snapshot, not a background watcher.

The response must make its evidence boundaries explicit:

- `detected`: read from the current machine now;
- `available`: found on `PATH`, not authenticated;
- `recommended`: a deterministic suggestion, not an executed action;
- `estimated`: a planning value, not a benchmark.

### Later daemon features

- Add an optional `POST /api/inspect` only after request size, path safety, and
  content-type rules are tested.
- Add a daemon version field and a small capability list so integrations can
  negotiate safely.
- Add graceful shutdown and an explicit “already running” message when a
  caller asks for a fixed port that is occupied.
- Keep network binding opt-in and visibly warn when `--allow-network` is used.

### Acceptance test

Start the daemon with port `0`, read every documented route, verify that the
reported port is actually listening, and stop the process. Test loopback
validation, non-loopback rejection, malformed paths, and a temporary project
with missing directories.

---

## Page 4 — Developer-tool layer

### What belongs here

AXIOM can describe and safely prepare a developer’s terminal for Codex, Claude
Code, Antigravity CLI, GitHub Copilot CLI, Freebuff, Cursor Agent, free-pi,
OpenCode, Gemini, OpenRouter, and z.ai GLM. The catalog is useful only when it
is current, platform-aware, and honest about authentication.

### Safe contract

- `axiom tools list` is data-only and works offline.
- `axiom tools doctor` checks executable presence without running the tool.
- `axiom tools plan <id>` prints a vendor command and source link.
- `axiom tools install <id>` previews by default.
- `--yes` is required for a supported package-manager install.
- Remote shell scripts, API-key setup, and unknown installers are shown for
  review rather than silently executed.
- PATH changes are user-level, reversible, and printed before they are made.

### Best next features

1. Add a `tools doctor --json` report with stable IDs and platform metadata.
2. Add `tools outdated` only if version checks are bounded and vendor APIs do
   not become a hidden network dependency.
3. Add an explicit `tools path` command that explains where AXIOM will look and
   how to repair a missing user-level bin directory.
4. Add provider setup guides that name environment variables but never echo
   their values.

### Acceptance test

Mock `PATH`, platform, package-manager output, and subprocess failures. Prove
that a preview never executes a command, that `--yes` runs only the approved
candidate, and that secret-like environment values never appear in output.

---

## Page 5 — Model, data, and optimization workflows

### Model inspection

The model path should be the most reliable local primitive. Inspect file type,
size, configuration, architecture hints, and detected capabilities without
loading model weights into memory. Fail clearly when the path is missing or
ambiguous.

### Dataset validation

Keep JSONL validation deterministic and useful: malformed records, empty lines,
duplicate records, missing fields, and record counts. Cleaning must write a new
file and preserve the source. Never label a dataset “ready” when AXIOM only
checked syntax.

### Hardware planning

Hardware detection should report OS, architecture, CPU cores, RAM, GPU, VRAM,
and CUDA availability. Training plans should show their assumptions and state
that fit values are conservative estimates. A plan must never be described as
a measured speedup.

### Optimization work

Keep `axiom optimize` separate from inspection. A runtime bundle is an output
artifact; it is not proof of throughput. When a real benchmark exists, record
the model, runtime, device, seed, command, and measurement environment. Until
then, use words such as “recommended,” “estimated,” and “target.”

### Acceptance test

Use fixtures for a small model directory, valid/invalid/duplicate JSONL, and
hardware with and without a GPU. Assert outputs, files written, non-zero
failures, and the absence of network calls in local-only flows.

---

## Page 6 — Landing page and brand system

### The public story

The landing page has one job: make a developer understand AXIOM within one
screen and reach the install commands within one click. The hero should say
what the CLI does in plain language. The workflow section should show real
commands, not a fake product pipeline.

### Visual rules

- Keep the workflow coffee background as the site-wide surface.
- Keep typography black and use Geist/Geist Mono consistently.
- Use the galaxy only in the opening visual.
- Keep the galaxy level and unlabelled; the image does not need product copy
  pasted on top of it.
- Use motion for navigation, reveal, and hover feedback; respect reduced
  motion.
- No gradients, fake counters, eyebrow pills, empty badges, or decorative
  sections.
- Use the AXIOM mark in the favicon and brand lockup at every size.

### Favicon milestone

The mark should be a simple vector symbol that remains legible at 16px, 32px,
and 180px. Test it in the browser tab, header, footer, and a dark browser UI.
If a detail disappears at small size, remove it instead of adding more detail.

### Acceptance test

Playwright checks install anchors, GitHub destinations, favicon loading, no
removed section names, no gradient styling, identical section backgrounds, and
the absence of fake dashboard language. Review a desktop and mobile screenshot
before each public release.

---

## Page 7 — Reliability and security

### Threat model

AXIOM reads local paths, may call package managers when explicitly authorized,
may use Hugging Face when a user asks, and may expose local JSON to a process
that can reach the loopback interface. The design must assume malformed files,
unexpected symlinks, hostile repository content, unavailable commands, and
secrets in the environment.

### Required controls

- Keep file operations within the requested path or an explicit project root.
- Bound subprocess timeouts and capture output safely.
- Never interpolate untrusted paths into shell strings.
- Redact credentials and avoid writing them to project files or logs.
- Keep daemon responses bounded and content types explicit.
- Validate host binding before opening a socket.
- Add security regression tests for every repaired bug.

### Failure language

Errors should answer three questions: what failed, why it matters, and what the
user can try next. Do not hide an unavailable GPU, provider, package manager,
or optional dependency behind a green status word.

### Acceptance test

Run focused tests for path traversal, invalid YAML, command failures, blocked
hosts, oversized inputs, missing executables, and secret redaction. Run the
repository’s CodeQL and dependency checks on every mergeable change.

---

## Page 8 — Test and release system

### Required checks per change

```text
pytest -q
npm run e2e
node --check website/script.js
python -m compileall -q axiom
git diff --check
```

The test matrix must distinguish CLI, daemon, landing page, installer,
developer-tool, MCP, packaging, and deployment coverage. “All tests passed” is
not a substitute for stating which surface was actually exercised.

### Branch and PR discipline

- One user-visible milestone per branch.
- One PR with a title that describes the behavior, not the internal file.
- Review the diff and the rendered result before merging.
- Do not create empty PRs or deploys to inflate a number.
- Keep the main branch deployable.

### Deployment cadence

For a meaningful website change, use at most two deployments: the automatic
push deployment and one manual rerun against the exact same SHA if a second
verification is useful. Verify the workflow run, SHA, HTTP status, and a live
content marker. Stop when the evidence is complete.

### Release evidence

Record the commit, PR, workflow run, live URL, and tests in the handoff. A
successful old workflow does not prove that a newer commit is live.

---

## Page 9 — Product feedback without surveillance

### What to measure

For local development, prefer explicit evidence over hidden telemetry:

- Does a new user reach a useful command after `axiom`?
- Does the first daemon start on a free port?
- Can a script consume `/api/info` and `/api/summary`?
- Do model and dataset paths fail clearly?
- Can a developer find and safely install a tool?
- Does the website answer “what does AXIOM do?” without scrolling through
  filler?

Use test fixtures, release checklists, issue labels, and opt-in user reports.
Do not add hosted analytics merely to create a dashboard or a growth number.

### Release scorecard

Every milestone should report:

```text
User problem:
Changed behavior:
Evidence class:
Tests:
Security impact:
Docs:
Commit / PR:
Deployment run:
Live verification:
```

### Feedback loop

Keep a short issue template with reproduction command, OS, Python version,
AXIOM version, expected output, actual output, and whether the daemon was
loopback-only. Convert repeated support questions into a command explanation or
a test before adding a new UI surface.

### Definition of learning

The product is improving when users ask fewer “what is this?” questions and
when failures become shorter, more specific, and easier to recover from—not
when the site has more sections or the repository has more deploy records.

---

## Page 10 — 30-day execution schedule

### Baseline already landed

- The static landing page, refined brand mark, local summary boundary, and
  non-blocking `axiom summary` path are in the current product baseline.
- README examples and daemon route documentation should stay aligned with the
  command implementation.

### Next 30 days

### Week 2: make local inspection sharper

- Improve model error messages and fixture coverage.
- Add dataset schema hints without claiming semantic quality.
- Add JSON output to the safest local commands.
- Exercise macOS, Linux, and Windows path behavior in CI or deterministic
  platform mocks.

### Week 3: make tools safer and more useful

- Add `axiom tools doctor --json` and `axiom tools path`.
- Review every catalog candidate against its current first-party source.
- Add preview/execute regression tests for package installs and PATH edits.
- Keep provider keys out of output, logs, and generated config.

### Week 4: release hardening

- Run the full command and daemon matrix in a clean environment.
- Review the landing page at desktop and mobile widths.
- Confirm the exact merge SHA is live on Pages.
- Publish a short changelog with shipped behavior and known boundaries.
- Only then select the next milestone from the backlog.

### Immediate next action

Choose the smallest unfinished item that shortens a real local workflow, add a
focused test and documentation example, and ship it as one reviewable change.
Do not expand the product surface merely to create another page, dashboard, or
deployment record.
