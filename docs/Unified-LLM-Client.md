# Unified LLM client and mobile-agent bridge

AXIOM exposes one text-chat interface for its native local engine and selected cloud providers. The React Control Center uses the same API as the CLI; the OpenAI-compatible bridge is a separate, narrowly mounted app for clients that understand the standard chat-completions protocol.

## Supported providers

- **AXIOM Native Engine (`axiom-local`)** — local Llama, Mistral, Qwen2, and Qwen3 floating-point safetensors checkpoints found under the trusted native-model roots.
- **Ollama (`ollama-local`)** — models already installed in Ollama, including Ollama-supported GGUF models, through the fixed loopback-only Ollama chat API.
- **OpenAI**, **Anthropic**, **Google Gemini**, **DeepSeek**, **Moonshot AI/Kimi**, and **OpenRouter** — provider-specific HTTPS API adapters.
- **Custom OpenAI-compatible provider** — an explicitly supplied public HTTPS endpoint ending in `/v1`.

This client currently handles text conversations. It does not add multimodal/image payloads, function/tool calling, model downloads, persistent chat history, or provider-side fallback. The native provider does not make GGUF, Ollama, or unsupported checkpoints work through the AXIOM Native Engine; choose `ollama-local` for models already installed in Ollama.

## Control Center

Open **Runtime → Unified LLM client**. Choose a provider and load its model catalogue, or enter the exact model ID yourself. Supply a system instruction and prompt, then set the output-token budget and temperature.

Cloud requests are blocked until the user confirms that the selected provider receives the conversation. Provider credentials are sent in the HTTPS provider authorization header and are not written to AXIOM workspace files or settings by this client. The browser form keeps the current key in component memory while the panel is mounted; clear it or leave the page when finished. A cloud provider processes data under its own policies, so never send secrets or private data without reviewing those terms.

## CLI

```bash
axiom llm providers
axiom llm providers --json
axiom llm models --provider axiom-local
axiom llm models --provider ollama-local
axiom llm models --provider deepseek
axiom llm chat deepseek-chat "Explain KV caching in two sentences." --provider deepseek --send-to-provider
axiom llm chat MODEL_ID "Summarize this local note." --provider axiom-local --device auto --memory-mode auto
```

Cloud commands prompt for the API key with hidden input unless the environment variable named by `--api-key-env` is set. The default variable is `AXIOM_LLM_API_KEY`. The default is not to remember or persist credentials. `--send-to-provider` is required for cloud chat to make the user-visible intent explicit. `--base-url` is accepted only with `--provider openai-compatible`.

## HTTP API

- `GET /api/llm/providers` — provider capabilities without credentials or network requests.
- `POST /api/llm/models` — list local trusted models or request a cloud provider catalogue.
- `POST /api/llm/chat` — generate a single text completion with bounded input/output limits.

Cloud requests include `consent: true`; otherwise the API rejects the request before invoking the provider. API keys are accepted as secret request fields and are not returned in responses. The app does not retain conversation content in runtime traces. See the OpenAPI contract at `/api/openapi.json` for request and response schemas.

## Mobile and third-party clients

The `axiom mobile-bridge` command exposes only `/v1/models` and `/v1/chat/completions` for AXIOM Native Engine models and locally installed Ollama models, so an OpenAI-compatible mobile client can use AXIOM's computer as its inference host:

```bash
axiom engine discover --json
axiom mobile-bridge
# To make the bridge reachable from another device on the private LAN:
axiom mobile-bridge --listen-lan --port 40509
```

The command prints the exact base URL and a random bearer token. Configure the client with base URL `http://<AXIOM-PC-LAN-IP>:40509/v1` and use the printed token as its API key. Native model IDs are listed by `axiom engine discover --json`; installed Ollama models are listed at `/v1/models` with an explicit `ollama/` prefix, for example `ollama/qwen2.5:0.5b`. The bridge routes those namespaced IDs only to the local Ollama endpoint; it does not forward prompts to a cloud provider. The bearer token is active only for that process and is cleared on exit.

**LAN security:** the bearer token authenticates requests, but the bridge uses plain HTTP, not TLS. Run it only on a trusted private network, never port-forward it to the public internet, and prefer a VPN or encrypted tunnel across untrusted networks. Stop the bridge when it is no longer needed. Only the mobile-bridge app is bound to the LAN; it does not mount the full AXIOM workspace API.

PrivateLM's [cross-platform-llm-client](https://github.com/orailnoor/cross-platform-llm-client) and [PrivateAgent](https://github.com/orailnoor/private-agent) are useful external compatibility references for local/cloud model switching and Android automation. For PrivateAgent, set the custom base URL to the bridge address, paste the one-time bearer token as its API key, and choose an `/v1/models` ID (using `ollama/` for Ollama models). AXIOM does not vendor their Flutter code. Direct interoperability with every release of those apps is not guaranteed; configure them as an OpenAI-compatible provider and test with a local model on a trusted network before enabling automation.

## Current limits

- Text only, one completion per chat request (`n=1` for the compatibility bridge).
- Maximum output is 2,048 tokens; message and combined-character limits are enforced.
- No persistent credential vault, chat storage, function calling, or tool execution in this interface.
- Native model compatibility depends on AXIOM Native Engine architecture, tensor, and device support; a listed model is not a promise that it fits available memory.
- Cloud provider model IDs and capabilities can change independently of AXIOM.
