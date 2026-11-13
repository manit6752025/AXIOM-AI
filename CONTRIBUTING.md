# Contributing to AXIOM

## AXIOM — NetCore Technologies

**Build AI. Own AI.**

Thank you for considering contributing to AXIOM.

AXIOM is an open-source AI engineering platform built around local-first workflows, observability, performance and practical engineering. Contributions of code, documentation, testing, bug reports and ideas all help move the project forward.

---

## 1. Before you contribute

Before starting a change:

1. Read the project README.
2. Check existing issues and pull requests.
3. Read the relevant documentation.
4. Check whether the work is already in progress.
5. Keep the scope of your change clear.

Repository:

https://github.com/manit6752025/AXIOM-AI

---

## 2. Areas you can contribute to

AXIOM covers several engineering areas.

### AI engineering

- Models
- Datasets
- Training
- Evaluation
- Runtime
- AI tooling

### Platform engineering

- CLI
- Local daemon
- Authentication
- Diagnostics
- Telemetry
- Configuration

### Infrastructure

- Linux
- Windows
- Release packaging
- CI/CD
- Build systems
- Deployment tooling

### MCP

- Servers
- Tools
- Resources
- Prompts
- Schemas
- Requests
- Testing and diagnostics

### Documentation

- Wiki pages
- README improvements
- Guides
- Examples
- Architecture documentation
- Troubleshooting documentation

---

## 3. Development environment

AXIOM combines a Python package/CLI with a static landing page and a small
loopback daemon.

Typical setup:

```bash
git clone https://github.com/manit6752025/AXIOM-AI.git
cd AXIOM-AI

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

For the landing page browser checks:

```bash
npm install
npm run e2e
```

Use the current repository documentation and package scripts for version-specific setup.

---

## 4. Create a branch

Do not normally develop directly on `main`.

Create a focused branch:

```bash
git checkout -b feature/my-change
```

Examples:

```text
feature/improve-models
feature/mcp-inspector
fix/authentication-timeout
fix/windows-installer
docs/wiki-runtime
```

Keep a branch focused on one logical change.

---

## 5. Make your changes

Prefer small, understandable changes over large unrelated rewrites.

Good changes should be:

- Focused
- Reviewable
- Testable
- Documented where necessary
- Compatible with the existing architecture

Avoid changing unrelated files simply because they are nearby.

---

## 6. Code style

Follow the style already used by the relevant part of the repository.

Before committing, check for accidental whitespace problems:

```bash
git diff --check
```

For Python, JavaScript/TypeScript and other project areas, use the repository's current linting and formatting commands.

Do not introduce unnecessary formatting changes across unrelated files.

---

## 7. Testing

Test the behaviour you changed.

Useful areas include:

- CLI commands
- Configuration
- Dataset validation
- Training planning
- Evaluation
- Runtime
- MCP
- Authentication
- Website landing page
- Local daemon
- Release packaging

For UI changes, verify the actual user experience rather than relying only on compilation.

For release changes, verify the resulting artifact names and packaging outputs.

---

## 8. Authentication changes

Authentication changes require extra care.

When changing:

- First-boot setup
- Login
- Password validation
- Password changes
- Session timeout
- User profile behaviour

test the complete flow from first boot through logout.

Never commit real passwords, tokens or credentials.

---

## 9. Documentation changes

Documentation should match the implementation.

When behaviour changes, update the relevant:

- README
- Wiki page
- CLI documentation
- Troubleshooting page
- Release notes

Do not document planned functionality as though it has already shipped.

Use explicit wording such as:

```text
Planned
Proposed
Experimental
Currently implemented
```

when the status matters.

---

## 10. Commit messages

Use clear commit messages that describe the change.

Examples:

```text
feat: add runtime telemetry
fix: correct Windows MSI packaging
docs: expand MCP documentation
refactor: simplify model registry
test: add dataset validation coverage
chore: update release workflow
```

Keep commits logically focused.

---

## 11. Pull requests

Push your branch:

```bash
git push -u origin feature/my-change
```

Then open a pull request against `main`.

A useful pull request description should explain:

### What changed

Describe the implementation.

### Why

Explain the problem or goal.

### Testing

List the commands and manual tests performed.

### Known limitations

Clearly identify anything unfinished or platform-specific.

---

## 12. Pull request checklist

Before opening a pull request:

```text
[ ] Change is focused
[ ] Existing code style followed
[ ] Relevant tests completed
[ ] UI manually checked when applicable
[ ] Documentation updated when needed
[ ] git diff --check passes
[ ] No secrets or credentials included
[ ] Commit history is understandable
```

---

## 13. Issues and bug reports

A useful bug report should include:

- AXIOM version
- Operating system
- Relevant hardware/runtime information
- What you expected
- What actually happened
- Steps to reproduce
- Relevant logs or errors

Avoid posting sensitive information.

Issue tracker:

https://github.com/manit6752025/AXIOM-AI/issues

---

## 14. Feature requests

Feature requests are welcome.

Explain:

1. What problem the feature solves.
2. Who would benefit.
3. How you expect it to work.
4. Any alternatives you considered.

For larger architectural proposals, include diagrams, examples or a small prototype when useful.

---

## 15. Security

Do not publicly disclose exploitable vulnerabilities, credentials or other sensitive security information.

For security vulnerabilities, follow:

https://github.com/manit6752025/AXIOM-AI/blob/main/SECURITY.md

Never commit:

```text
passwords
API keys
access tokens
private GPG keys
private certificates
production credentials
private datasets
```

---

## 16. Release engineering

AXIOM uses automated GitHub Actions for releases.

Release work may involve:

```text
Windows
  .exe
  .msi
  .ps1
  .bat

Linux
  .deb
  .AppImage
  .sh
  .elf
```

Changes to release automation should be tested carefully because one workflow change can affect multiple platforms.

---

## 17. Architecture principles

AXIOM development generally favours:

### Local-first

Keep important workflows close to the operator.

### Observable

Make system state, health and failures visible.

### Performance-focused

Avoid unnecessary overhead and respect the underlying hardware.

### Modular

Keep platform areas understandable and separable.

### Open

Prefer inspectable, documented engineering decisions.

---

## 18. Code review

Reviewers should focus on:

- Correctness
- Security
- Maintainability
- Performance
- Testing
- User experience
- Compatibility
- Project direction

Feedback should be specific and constructive.

Contributors are encouraged to explain trade-offs and technical reasoning when proposing a different approach.

---

## 19. Community behaviour

All contributors are expected to follow the project's Code of Conduct.

See:

https://github.com/manit6752025/AXIOM-AI/blob/main/CODE_OF_CONDUCT.md

Technical disagreement is welcome.

Personal attacks, harassment, discrimination, malicious behaviour and deliberate disruption are not.

---

## 20. License

By contributing to AXIOM, you agree that your contributions will be handled under the project's applicable license and repository contribution terms.

See the repository LICENSE file for the authoritative licensing terms.

---

## 21. Questions

For development questions, use the repository's GitHub discussions/issues where appropriate and provide enough technical context for others to reproduce the problem.

---

<div align="center">

**AXIOM — Build AI. Own AI.**

*NetCore Technologies*

</div>
