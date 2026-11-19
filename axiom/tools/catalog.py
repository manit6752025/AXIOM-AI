"""Secret-free, data-only registry of developer tools and capabilities.

The catalog records current first-party installation candidates and links. It
does not probe hosts, install software, read credentials, or make network
requests. API-only providers deliberately have no fabricated executable name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Platform: TypeAlias = Literal["macos", "linux", "windows"]
ToolKind: TypeAlias = Literal["cli", "api", "cli-and-api"]
InstallKind: TypeAlias = Literal["package", "script", "sdk"]

ALL_PLATFORMS: tuple[Platform, ...] = ("macos", "linux", "windows")


@dataclass(frozen=True, slots=True)
class InstallCandidate:
    """A non-executed, first-party install or SDK command for one platform."""

    platform: Platform
    command: str
    install_kind: InstallKind
    source_url: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Descriptive metadata for one external developer tool or API provider."""

    id: str
    name: str
    kind: ToolKind
    executable_names: tuple[str, ...]
    purpose: str
    auth_notes: str
    api_key_env_vars: tuple[str, ...]
    homepage_url: str
    source_url: str
    supported_platforms: tuple[Platform, ...]
    install_candidates: tuple[InstallCandidate, ...]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """A future AXIOM capability that is not itself an installable tool."""

    id: str
    name: str
    purpose: str
    command: str
    reference_url: str
    supported_platforms: tuple[Platform, ...]
    reference_supported_platforms: tuple[Platform, ...] = ("macos",)
    notes: str = ""


TOOL_CATALOG: tuple[ToolSpec, ...] = (
    ToolSpec(
        id="codex",
        name="Codex",
        kind="cli",
        executable_names=("codex",),
        purpose="Inspect, edit, run, review, and automate work in a local repository from the terminal.",
        auth_notes=(
            "Sign in with ChatGPT for plan-based access, or use the supported API-key flow. "
            "Keep any credential in the CLI's secure flow or environment, never in this catalog."
        ),
        api_key_env_vars=("OPENAI_API_KEY", "CODEX_API_KEY"),
        homepage_url="https://openai.com/codex/",
        source_url="https://github.com/openai/codex",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="curl -fsSL https://chatgpt.com/codex/install.sh | sh",
                install_kind="script",
                source_url="https://github.com/openai/codex/blob/main/README.md",
                notes="Official standalone installer for macOS.",
            ),
            InstallCandidate(
                platform="linux",
                command="curl -fsSL https://chatgpt.com/codex/install.sh | sh",
                install_kind="script",
                source_url="https://github.com/openai/codex/blob/main/README.md",
                notes="Official standalone installer for Linux.",
            ),
            InstallCandidate(
                platform="windows",
                command=(
                    "powershell -ExecutionPolicy ByPass -c "
                    '"irm https://chatgpt.com/codex/install.ps1 | iex"'
                ),
                install_kind="script",
                source_url="https://github.com/openai/codex/blob/main/README.md",
                notes="Official standalone PowerShell installer for Windows.",
            ),
        ),
        notes="The CLI README also lists npm and Homebrew candidates; no installer is run by AXIOM.",
    ),
    ToolSpec(
        id="claude-code",
        name="Claude Code",
        kind="cli",
        executable_names=("claude",),
        purpose="Work with a repository through an agentic terminal workflow that can explain, edit, test, and manage Git tasks.",
        auth_notes=(
            "Authenticate with Anthropic or Claude account OAuth, or configure an Anthropic API, "
            "Amazon Bedrock, or Google Vertex AI provider. Store API credentials in secure credential "
            "storage or environment variables, not in this catalog."
        ),
        api_key_env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        homepage_url="https://www.anthropic.com/claude-code",
        source_url="https://docs.anthropic.com/en/docs/claude-code/overview",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="npm install -g @anthropic-ai/claude-code",
                install_kind="package",
                source_url="https://code.claude.com/docs/en/getting-started",
                notes="Current official npm candidate; Node.js 22 or later is required by the package.",
            ),
            InstallCandidate(
                platform="linux",
                command="npm install -g @anthropic-ai/claude-code",
                install_kind="package",
                source_url="https://code.claude.com/docs/en/getting-started",
                notes="Current official npm candidate; signed apt, dnf, and apk channels are also documented.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm install -g @anthropic-ai/claude-code",
                install_kind="package",
                source_url="https://code.claude.com/docs/en/getting-started",
                notes="Current official npm candidate; native Windows x64 and arm64 packages are documented.",
            ),
        ),
        notes="Windows support depends on the selected native, WSL, or Git Bash setup; check the current Anthropic setup guide.",
    ),
    ToolSpec(
        id="antigravity-cli",
        name="Antigravity CLI",
        kind="cli",
        executable_names=("agy",),
        purpose="Use Google's Antigravity agent harness from a keyboard-first terminal interface, including remote SSH sessions.",
        auth_notes=(
            "The CLI uses the system keyring and falls back to Google Sign-In. Headless use can "
            "select the Gemini provider and supply GEMINI_API_KEY; no key value is placed in this catalog."
        ),
        api_key_env_vars=("GEMINI_API_KEY",),
        homepage_url="https://antigravity.google/product/antigravity-cli",
        source_url="https://github.com/google-antigravity/antigravity-cli",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="curl -fsSL https://antigravity.google/cli/install.sh | bash",
                install_kind="script",
                source_url="https://antigravity.google/docs/cli/install/",
                notes="Official macOS installer.",
            ),
            InstallCandidate(
                platform="linux",
                command="curl -fsSL https://antigravity.google/cli/install.sh | bash",
                install_kind="script",
                source_url="https://antigravity.google/docs/cli/install/",
                notes="Official Linux installer.",
            ),
            InstallCandidate(
                platform="windows",
                command="irm https://antigravity.google/cli/install.ps1 | iex",
                install_kind="script",
                source_url="https://antigravity.google/docs/cli/install/",
                notes="Official Windows PowerShell installer.",
            ),
        ),
    ),
    ToolSpec(
        id="github-copilot-cli",
        name="GitHub Copilot CLI",
        kind="cli",
        executable_names=("copilot",),
        purpose="Use GitHub's terminal-native coding agent for planning, editing, GitHub workflows, and parallel agent work.",
        auth_notes=(
            "Authenticate with an eligible GitHub Copilot account; organization or enterprise policy can "
            "disable access. The normal sign-in flow does not require a provider API key."
        ),
        api_key_env_vars=(),
        homepage_url="https://github.com/features/copilot/cli",
        source_url="https://github.com/github/copilot-cli",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="curl -fsSL https://gh.io/copilot-install | bash",
                install_kind="script",
                source_url="https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli",
                notes="Official macOS install script.",
            ),
            InstallCandidate(
                platform="linux",
                command="curl -fsSL https://gh.io/copilot-install | bash",
                install_kind="script",
                source_url="https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli",
                notes="Official Linux install script.",
            ),
            InstallCandidate(
                platform="windows",
                command="winget install GitHub.Copilot",
                install_kind="package",
                source_url="https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli",
                notes="Official Windows WinGet candidate.",
            ),
        ),
        notes="The official quickstart also documents npm installation with Node.js 22 or later.",
    ),
    ToolSpec(
        id="freebuff",
        name="Freebuff",
        kind="cli",
        executable_names=("freebuff",),
        purpose="Run an ad-supported coding agent from the terminal, with companion desktop, web, and GitHub workflows.",
        auth_notes=(
            "The official project says no subscription, credits, or API key are required. A connected "
            "GitHub account can be used for repository workflows; keep account credentials outside the catalog."
        ),
        api_key_env_vars=(),
        homepage_url="https://freebuff.com/",
        source_url="https://github.com/CodebuffAI/freebuff",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="npm install -g freebuff",
                install_kind="package",
                source_url="https://freebuff.com/cli",
                notes="Official npm CLI package; requires a supported Node/npm runtime.",
            ),
            InstallCandidate(
                platform="linux",
                command="npm install -g freebuff",
                install_kind="package",
                source_url="https://freebuff.com/cli",
                notes="Official npm CLI package; requires a supported Node/npm runtime.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm install -g freebuff",
                install_kind="package",
                source_url="https://freebuff.com/cli",
                notes="Official npm CLI package; requires a supported Node/npm runtime.",
            ),
        ),
        notes="The official site separately lists desktop builds for macOS, Windows, and Linux.",
    ),
    ToolSpec(
        id="cursor-agent",
        name="Cursor Agent",
        kind="cli",
        executable_names=("cursor-agent", "agent"),
        purpose="Run Cursor's coding agent interactively or headlessly from a terminal for code changes, review, and automation.",
        auth_notes=(
            "Browser login is recommended. Headless scripts can use CURSOR_API_KEY; never put that key "
            "in a command recorded in the catalog or in source control."
        ),
        api_key_env_vars=("CURSOR_API_KEY",),
        homepage_url="https://cursor.com/",
        source_url="https://docs.cursor.com/en/cli/overview",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="curl https://cursor.com/install -fsS | bash",
                install_kind="script",
                source_url="https://docs.cursor.com/en/cli/installation",
                notes="Official CLI installer for macOS.",
            ),
            InstallCandidate(
                platform="linux",
                command="curl https://cursor.com/install -fsS | bash",
                install_kind="script",
                source_url="https://docs.cursor.com/en/cli/installation",
                notes="Official CLI installer for Linux.",
            ),
            InstallCandidate(
                platform="windows",
                command="irm 'https://cursor.com/install?win32=true' | iex",
                install_kind="script",
                source_url="https://prod.cursor.com/docs/enterprise/deployment-patterns",
                notes="Current official native Windows PowerShell candidate; WSL uses the POSIX command.",
            ),
        ),
        notes=(
            "The POSIX and WSL docs use cursor-agent; current native Windows deployment docs use agent. "
            "Review command approvals before headless write access."
        ),
    ),
    ToolSpec(
        id="free-pi",
        name="free-pi",
        kind="cli",
        executable_names=("free-pi-cli",),
        purpose="Run a free, ad-supported Pi-based coding agent in a terminal through the free-pi service proxy.",
        auth_notes=(
            "First use completes a GitHub device-code login; the proxy supplies model access, so users "
            "do not handle a model API key. Review the service's consent, privacy, and training terms before sending source code."
        ),
        api_key_env_vars=(),
        homepage_url="https://www.freepi.ai/",
        source_url="https://github.com/dennisonbertram/free-pi-cli",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="npx free-pi-cli",
                install_kind="package",
                source_url="https://www.freepi.ai/",
                notes="Official zero-install npm/npx entry point.",
            ),
            InstallCandidate(
                platform="linux",
                command="npx free-pi-cli",
                install_kind="package",
                source_url="https://www.freepi.ai/",
                notes="Official zero-install npm/npx entry point.",
            ),
            InstallCandidate(
                platform="windows",
                command="npx free-pi-cli",
                install_kind="package",
                source_url="https://www.freepi.ai/",
                notes="Official zero-install npm/npx entry point.",
            ),
        ),
        notes="The official npm-based project does not publish a narrower OS matrix; use a supported Node/npm runtime.",
    ),
    ToolSpec(
        id="opencode",
        name="OpenCode",
        kind="cli",
        executable_names=("opencode",),
        purpose="Use an open-source terminal coding agent with multi-session workflows, provider choice, and editor integrations.",
        auth_notes=(
            "Connect a provider API key, or use a supported provider sign-in such as GitHub Copilot or "
            "ChatGPT where offered. Provider credentials are external to this catalog and must stay secret."
        ),
        api_key_env_vars=(),
        homepage_url="https://opencode.ai/",
        source_url="https://github.com/anomalyco/opencode",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="brew install anomalyco/tap/opencode",
                install_kind="package",
                source_url="https://opencode.ai/docs/",
                notes="Officially documented Homebrew tap; the install script is another documented option.",
            ),
            InstallCandidate(
                platform="linux",
                command="curl -fsSL https://opencode.ai/install | bash",
                install_kind="script",
                source_url="https://opencode.ai/docs/",
                notes="Official Linux install script.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm i -g opencode-ai",
                install_kind="package",
                source_url="https://opencode.ai/docs/",
                notes="Official package candidate; the official docs recommend WSL for the best Windows experience.",
            ),
        ),
        notes="OpenCode is not a sandbox; use its permission prompts and an isolated environment for untrusted work.",
    ),
    ToolSpec(
        id="gemini-cli",
        name="Gemini CLI / API harness",
        kind="cli",
        executable_names=("gemini",),
        purpose="Use Google's open-source Gemini CLI for terminal coding tasks and the Google GenAI SDK for API-backed harnesses.",
        auth_notes=(
            "The CLI can use Google sign-in, a Gemini API key, or Vertex AI credentials depending on the "
            "selected flow. The API harness uses a Google AI Studio or Vertex credential; keep keys in secure storage."
        ),
        api_key_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        homepage_url="https://geminicli.com/",
        source_url="https://github.com/google-gemini/gemini-cli",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="npm install -g @google/gemini-cli",
                install_kind="package",
                source_url="https://github.com/google-gemini/gemini-cli/blob/main/docs/get-started/index.md",
                notes="Official CLI package; Node.js 20 or later is required.",
            ),
            InstallCandidate(
                platform="linux",
                command="npm install -g @google/gemini-cli",
                install_kind="package",
                source_url="https://github.com/google-gemini/gemini-cli/blob/main/docs/get-started/index.md",
                notes="Official CLI package; Node.js 20 or later is required.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm install -g @google/gemini-cli",
                install_kind="package",
                source_url="https://github.com/google-gemini/gemini-cli/blob/main/docs/get-started/index.md",
                notes="Official CLI package; Node.js 20 or later is required.",
            ),
            InstallCandidate(
                platform="macos",
                command="pip install google-genai",
                install_kind="sdk",
                source_url="https://ai.google.dev/gemini-api/docs/libraries",
                notes="Official Python Google GenAI SDK for API harnesses.",
            ),
            InstallCandidate(
                platform="linux",
                command="pip install google-genai",
                install_kind="sdk",
                source_url="https://ai.google.dev/gemini-api/docs/libraries",
                notes="Official Python Google GenAI SDK for API harnesses.",
            ),
            InstallCandidate(
                platform="windows",
                command="pip install google-genai",
                install_kind="sdk",
                source_url="https://ai.google.dev/gemini-api/docs/libraries",
                notes="Official Python Google GenAI SDK for API harnesses.",
            ),
            InstallCandidate(
                platform="macos",
                command="npm install @google/genai",
                install_kind="sdk",
                source_url="https://ai.google.dev/gemini-api/docs/libraries",
                notes="Official JavaScript/TypeScript Google GenAI SDK for API harnesses.",
            ),
            InstallCandidate(
                platform="linux",
                command="npm install @google/genai",
                install_kind="sdk",
                source_url="https://ai.google.dev/gemini-api/docs/libraries",
                notes="Official JavaScript/TypeScript Google GenAI SDK for API harnesses.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm install @google/genai",
                install_kind="sdk",
                source_url="https://ai.google.dev/gemini-api/docs/libraries",
                notes="Official JavaScript/TypeScript Google GenAI SDK for API harnesses.",
            ),
        ),
        notes="The CLI and API harness share the Gemini ecosystem but are separate surfaces; the API SDK does not add a second executable.",
    ),
    ToolSpec(
        id="openrouter",
        name="OpenRouter",
        kind="cli-and-api",
        executable_names=("openrouter",),
        purpose="Route requests to many model providers through one OpenAI-compatible API, with official SDK, agent-SDK, and DevTools CLI options.",
        auth_notes=(
            "Create an OpenRouter API key and provide it through the official SDK or the OPENROUTER_API_KEY "
            "environment variable. The key is required for requests and must never be stored in this catalog."
        ),
        api_key_env_vars=("OPENROUTER_API_KEY",),
        homepage_url="https://openrouter.ai/",
        source_url="https://github.com/OpenRouterTeam/docs",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="npm install -g @openrouter/cli",
                install_kind="package",
                source_url="https://openrouter.ai/developers",
                notes="Official OpenRouter DevTools CLI; it is not a standalone coding agent.",
            ),
            InstallCandidate(
                platform="linux",
                command="npm install -g @openrouter/cli",
                install_kind="package",
                source_url="https://openrouter.ai/developers",
                notes="Official OpenRouter DevTools CLI; it is not a standalone coding agent.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm install -g @openrouter/cli",
                install_kind="package",
                source_url="https://openrouter.ai/developers",
                notes="Official OpenRouter DevTools CLI; it is not a standalone coding agent.",
            ),
            InstallCandidate(
                platform="macos",
                command="npm install @openrouter/sdk",
                install_kind="sdk",
                source_url="https://openrouter.ai/docs/client-sdks/overview",
                notes="Official TypeScript SDK candidate.",
            ),
            InstallCandidate(
                platform="linux",
                command="npm install @openrouter/sdk",
                install_kind="sdk",
                source_url="https://openrouter.ai/docs/client-sdks/overview",
                notes="Official TypeScript SDK candidate.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm install @openrouter/sdk",
                install_kind="sdk",
                source_url="https://openrouter.ai/docs/client-sdks/overview",
                notes="Official TypeScript SDK candidate.",
            ),
            InstallCandidate(
                platform="macos",
                command="pip install openrouter",
                install_kind="sdk",
                source_url="https://openrouter.ai/docs/client-sdks/overview",
                notes="Official Python SDK candidate.",
            ),
            InstallCandidate(
                platform="linux",
                command="pip install openrouter",
                install_kind="sdk",
                source_url="https://openrouter.ai/docs/client-sdks/overview",
                notes="Official Python SDK candidate.",
            ),
            InstallCandidate(
                platform="windows",
                command="pip install openrouter",
                install_kind="sdk",
                source_url="https://openrouter.ai/docs/client-sdks/overview",
                notes="Official Python SDK candidate.",
            ),
        ),
        notes="The API is the model-access surface; the official openrouter executable is a DevTools viewer/statusline utility.",
    ),
    ToolSpec(
        id="zai-glm",
        name="z.ai GLM",
        kind="cli-and-api",
        executable_names=("zai-cli",),
        purpose="Access Z.AI's GLM models through its official terminal toolkit, HTTP API, SDK, OpenAI-compatible clients, or supported coding plans.",
        auth_notes=(
            "Create a Z.AI API key or use a Z.AI GLM Coding Plan account. The official quick start passes "
            "the key to the SDK/client; the official CLI also supports ZAI_API_KEY and region selection. Keep the value in secure storage."
        ),
        api_key_env_vars=("ZAI_API_KEY",),
        homepage_url="https://z.ai/",
        source_url="https://www.npmjs.com/package/@z_ai/zai-cli",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command="npm install -g @z_ai/zai-cli",
                install_kind="package",
                source_url="https://www.npmjs.com/package/@z_ai/zai-cli",
                notes="Official Z.ai terminal toolkit; requires Node.js 18 or later.",
            ),
            InstallCandidate(
                platform="linux",
                command="npm install -g @z_ai/zai-cli",
                install_kind="package",
                source_url="https://www.npmjs.com/package/@z_ai/zai-cli",
                notes="Official Z.ai terminal toolkit; requires Node.js 18 or later.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm install -g @z_ai/zai-cli",
                install_kind="package",
                source_url="https://www.npmjs.com/package/@z_ai/zai-cli",
                notes="Official Z.ai terminal toolkit; requires Node.js 18 or later.",
            ),
            InstallCandidate(
                platform="macos",
                command="pip install zai-sdk",
                install_kind="sdk",
                source_url="https://docs.z.ai/guides/overview/quick-start",
                notes="Official Python SDK candidate.",
            ),
            InstallCandidate(
                platform="linux",
                command="pip install zai-sdk",
                install_kind="sdk",
                source_url="https://docs.z.ai/guides/overview/quick-start",
                notes="Official Python SDK candidate.",
            ),
            InstallCandidate(
                platform="windows",
                command="pip install zai-sdk",
                install_kind="sdk",
                source_url="https://docs.z.ai/guides/overview/quick-start",
                notes="Official Python SDK candidate.",
            ),
            InstallCandidate(
                platform="macos",
                command="npm install openai",
                install_kind="sdk",
                source_url="https://docs.z.ai/guides/overview/quick-start",
                notes="Official OpenAI-compatible JavaScript client path; configure Z.AI's base URL separately.",
            ),
            InstallCandidate(
                platform="linux",
                command="npm install openai",
                install_kind="sdk",
                source_url="https://docs.z.ai/guides/overview/quick-start",
                notes="Official OpenAI-compatible JavaScript client path; configure Z.AI's base URL separately.",
            ),
            InstallCandidate(
                platform="windows",
                command="npm install openai",
                install_kind="sdk",
                source_url="https://docs.z.ai/guides/overview/quick-start",
                notes="Official OpenAI-compatible JavaScript client path; configure Z.AI's base URL separately.",
            ),
        ),
        notes="The official CLI and provider API are separate access surfaces; no key value is stored here.",
    ),
    ToolSpec(
        id="headroom",
        name="Headroom",
        kind="cli",
        executable_names=("headroom",),
        purpose=(
            "Optional local proxy for routing AI-client traffic through Headroom; "
            "AXIOM only reports whether the external CLI is available."
        ),
        auth_notes=(
            "Headroom and each provider own their setup and authentication. AXIOM does not "
            "read, store, or forward Headroom credentials."
        ),
        api_key_env_vars=(),
        homepage_url="https://docs.headroomlabs.ai/docs",
        source_url="https://github.com/headroomlabs-ai/headroom",
        supported_platforms=ALL_PLATFORMS,
        install_candidates=(
            InstallCandidate(
                platform="macos",
                command='uv tool install --python 3.13 "headroom-ai[all]"',
                install_kind="package",
                source_url="https://github.com/headroomlabs-ai/headroom",
                notes=(
                    "Optional vendor-documented CLI setup; AXIOM only displays this candidate "
                    "and never runs it from axiom headroom."
                ),
            ),
            InstallCandidate(
                platform="linux",
                command='uv tool install --python 3.13 "headroom-ai[all]"',
                install_kind="package",
                source_url="https://github.com/headroomlabs-ai/headroom",
                notes=(
                    "Optional vendor-documented CLI setup; AXIOM only displays this candidate "
                    "and never runs it from axiom headroom."
                ),
            ),
            InstallCandidate(
                platform="windows",
                command='uv tool install --python 3.13 "headroom-ai[all]"',
                install_kind="package",
                source_url="https://github.com/headroomlabs-ai/headroom",
                notes=(
                    "Optional vendor-documented CLI setup; AXIOM only displays this candidate "
                    "and never runs it from axiom headroom."
                ),
            ),
        ),
        notes=(
            "Optional external local proxy only. Use the documented commands headroom doctor, "
            "headroom proxy, and headroom dashboard; AXIOM does not perform or measure savings."
        ),
    ),
)

# Short aliases keep the public data surface easy to discover without adding
# lookup functions or runtime behavior to this module.
TOOLS = TOOL_CATALOG


KEEP_AWAKE_CAPABILITY = CapabilitySpec(
    id="insomnia-keep-awake",
    name="Insomnia / keep-awake session capability",
    purpose="Keep a long-running AXIOM developer-tool session active across supported desktop platforms.",
    command="axiom session --keep-awake",
    reference_url="https://github.com/krishhgg/Insomnia",
    supported_platforms=ALL_PLATFORMS,
    reference_supported_platforms=("macos",),
    notes=(
        "Future AXIOM capability only: the command is not implemented by this catalog. "
        "The referenced krishhgg/Insomnia app is macOS-only; its app is not claimed to run on Linux or Windows. "
        "The cross-platform claim applies to a future AXIOM session command, not to that app."
    ),
)

CAPABILITY_CATALOG: tuple[CapabilitySpec, ...] = (KEEP_AWAKE_CAPABILITY,)
CAPABILITIES = CAPABILITY_CATALOG
