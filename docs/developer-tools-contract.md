# AXIOM developer-tools contract

This is the user-visible safety contract for setting up optional third-party
developer tools. The implementation may change, but these boundaries must not.

## Setup behavior

- Availability is observational. AXIOM may report an executable, version, path,
  or missing dependency. Finding an executable does not mean the user is
  authenticated. Authentication is `unverified` until the vendor's own check
  succeeds, and AXIOM must show that distinction.
- Credentials are references, never values. Accept an environment-variable
  name and/or keychain service/account reference. Do not print, log, serialize,
  persist, or include raw API-key values in errors, previews, or setup state.
- Remote installers are reviewable before they run. Show the source URL,
  package-manager choice, and exact script/command first. Preview has no network
  or subprocess side effect. Execution is denied unless the user explicitly
  opts in for that specific plan.
- OS and package-manager support is explicit. An unsupported combination must
  return actionable guidance naming the combination and a manual or vendor
  path; it must not silently fall back to a privileged or unrelated installer.
- PATH changes are user-scoped, idempotent, and reversible. Preserve unrelated
  user content, never edit machine/system PATH, and make undo restore the exact
  prior user configuration.
- A keep-awake session is bounded by its helper child. On normal completion,
  error, or interruption it must stop that child and restore the prior power
  state, releasing its wake hold.

AXIOM reports setup state; it does not become the vendor. Third-party tool
behavior, pricing, quotas, account limits, and authentication remain with the
respective vendors. AXIOM must not imply that its availability check grants
access or changes a vendor's billing or usage policy.

## Acceptance surface

`tests/test_tool_security.py` uses injected lookup, runner, filesystem, and
process fakes. It expects the installer boundary `axiom.tools.installer` and
the keep-awake boundary `axiom.tools.awake` to expose the small operations
named in that test. The tests intentionally do not contact a vendor, access a
real keychain, execute a shell, edit the real PATH, or keep the machine awake.

## References

https://keyring.readthedocs.io/en/stable/

https://docs.python.org/3/library/shutil.html#shutil.which

https://docs.python.org/3/library/subprocess.html

https://specifications.freedesktop.org/basedir/

https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables

https://github.com/manit6752025/AXIOM-AI
