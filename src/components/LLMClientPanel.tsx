import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Activity, KeyRound, Play, RefreshCcw, ShieldCheck } from "lucide-react";
import { axiomApi, type LLMProvider, type LLMProviderInfo, type LLMChatResult } from "../api/client";

const providerFallback: LLMProviderInfo[] = [
  { id: "axiom-local", label: "AXIOM Native Engine", kind: "local", requires_api_key: false, supports_model_listing: true, privacy: "local-only", note: "Local trusted safetensors checkpoints." },
  { id: "ollama-local", label: "Ollama (local)", kind: "local", requires_api_key: false, supports_model_listing: true, privacy: "local-only", note: "Uses locally installed Ollama models, including GGUF through Ollama." },
  { id: "openai", label: "OpenAI", kind: "cloud", requires_api_key: true, supports_model_listing: true, privacy: "remote-consent-required", note: "Credentials are request-only." },
  { id: "anthropic", label: "Anthropic", kind: "cloud", requires_api_key: true, supports_model_listing: true, privacy: "remote-consent-required", note: "Credentials are request-only." },
  { id: "gemini", label: "Google Gemini", kind: "cloud", requires_api_key: true, supports_model_listing: true, privacy: "remote-consent-required", note: "Credentials are request-only." },
  { id: "deepseek", label: "DeepSeek", kind: "cloud", requires_api_key: true, supports_model_listing: true, privacy: "remote-consent-required", note: "Credentials are request-only." },
  { id: "moonshot", label: "Moonshot AI (Kimi)", kind: "cloud", requires_api_key: true, supports_model_listing: true, privacy: "remote-consent-required", note: "Credentials are request-only." },
  { id: "openrouter", label: "OpenRouter", kind: "cloud", requires_api_key: true, supports_model_listing: true, privacy: "remote-consent-required", note: "Routes supported models through OpenRouter; credentials are request-only." },
  { id: "openai-compatible", label: "Custom OpenAI-compatible", kind: "cloud", requires_api_key: true, supports_model_listing: true, privacy: "remote-consent-required", note: "Public HTTPS /v1 endpoints only." },
];

export default function LLMClientPanel() {
  const [providers, setProviders] = useState<LLMProviderInfo[]>(providerFallback);
  const [provider, setProvider] = useState<LLMProvider>("axiom-local");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful, precise assistant.");
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState("256");
  const [temperature, setTemperature] = useState("0.7");
  const [topP, setTopP] = useState("0.9");
  const [topK, setTopK] = useState("0");
  const [minP, setMinP] = useState("0");
  const [repetitionPenalty, setRepetitionPenalty] = useState("1.0");
  const [seed, setSeed] = useState("");
  const [stopSequencesText, setStopSequencesText] = useState("");
  const [precision, setPrecision] = useState<"auto" | "fp32" | "fp16" | "bf16">("auto");
  const [consent, setConsent] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<LLMChatResult | null>(null);
  const [conversation, setConversation] = useState<Array<{ role: "user" | "assistant"; content: string }>>([]);

  const local = provider === "axiom-local" || provider === "ollama-local";
  const selected = providers.find((item) => item.id === provider);
  const loadModels = async (targetProvider = provider) => {
    setLoadingModels(true);
    setError("");
    try {
      const next = await axiomApi.llmModels({
        provider: targetProvider,
        api_key: targetProvider === "axiom-local" || targetProvider === "ollama-local" ? undefined : apiKey,
        base_url: targetProvider === "openai-compatible" ? baseUrl : undefined,
      });
      setModels(next.models);
      setModel((current) => current && next.models.includes(current) ? current : next.models[0] || current);
      if (next.models.length === 0) setError(next.note || "No models were returned. Check provider access or enter a model ID manually.");
    } catch (cause) {
      setModels([]);
      setError(cause instanceof Error ? cause.message : "The model catalogue could not be loaded.");
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    axiomApi.llmProviders().then((value) => setProviders(value.providers)).catch(() => setProviders(providerFallback));
    void axiomApi.llmModels({ provider: "axiom-local" }).then((next) => {
      setModels(next.models);
      setModel(next.models[0] || "");
    }).catch((cause: unknown) => {
      setError(cause instanceof Error ? cause.message : "Local AXIOM model inventory is unavailable.");
    });
  }, []);

  const changeProvider = (value: string) => {
    setProvider(value as LLMProvider);
    setApiKey("");
    setConsent(false);
    setModels([]);
    setModel("");
    setResult(null);
    setConversation([]);
    setPrompt("");
    setError("");
  };

  const run = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setResult(null);
    const tokenLimit = Number(maxTokens);
    const temp = Number(temperature);
    const topPValue = Number(topP);
    const topKValue = Number(topK);
    const minPValue = Number(minP);
    const repeatPenaltyValue = Number(repetitionPenalty);
    const seedValue = seed.trim() === "" ? null : Number(seed);
    const stopSequences = stopSequencesText.split(/\r?\n/).filter((value) => value.trim().length > 0);
    if (!model.trim()) { setError("Choose a model or enter an exact provider model ID."); return; }
    if (!prompt.trim()) { setError("Enter a prompt before generating."); return; }
    if (!Number.isInteger(tokenLimit) || tokenLimit < 1 || tokenLimit > 2048) { setError("Maximum output tokens must be between 1 and 2048."); return; }
    if (!Number.isFinite(temp) || temp < 0 || temp > 2) { setError("Temperature must be between 0 and 2."); return; }
    if (local && (!Number.isFinite(topPValue) || topPValue <= 0 || topPValue > 1)) { setError("Top-p must be greater than 0 and at most 1."); return; }
    if (local && (!Number.isInteger(topKValue) || topKValue < 0 || topKValue > 100000)) { setError("Top-k must be an integer from 0 to 100,000; 0 disables the limit."); return; }
    if (local && (!Number.isFinite(minPValue) || minPValue < 0 || minPValue > 1)) { setError("Min-p must be between 0 and 1; 0 disables the cutoff."); return; }
    if (local && (!Number.isFinite(repeatPenaltyValue) || repeatPenaltyValue < 0.1 || repeatPenaltyValue > 5)) { setError("Repetition penalty must be between 0.1 and 5."); return; }
    if (local && seedValue !== null && (!Number.isSafeInteger(seedValue) || seedValue < 0)) { setError("Seed must be a non-negative safe integer, or leave it blank."); return; }
    if (local && (stopSequences.length > 16 || stopSequences.some((stop) => stop.length > 256))) { setError("Use at most 16 stop sequences, each up to 256 characters."); return; }
    if (!local && !consent) { setError("Confirm that the selected cloud provider may receive this conversation."); return; }
    setRunning(true);
    try {
      const currentUserMessage = { role: "user" as const, content: prompt.trim() };
      const systemMessage = systemPrompt.trim();
      const priorConversation = [...conversation];
      const conversationBudget = 24_000 - currentUserMessage.content.length - systemMessage.length;
      const retainedCharacters = () => priorConversation.reduce((total, item) => total + item.content.length, 0);
      // Drop oldest complete user/assistant turns until request character and message limits are met.
      while (priorConversation.length && (retainedCharacters() > Math.max(0, conversationBudget) || priorConversation.length > 60)) {
        priorConversation.splice(0, Math.min(2, priorConversation.length));
      }
      const messages = [
        ...(systemMessage ? [{ role: "system" as const, content: systemMessage }] : []),
        ...priorConversation,
        currentUserMessage,
      ];
      const nextResult = await axiomApi.llmChat({
        provider,
        api_key: local ? undefined : apiKey,
        base_url: provider === "openai-compatible" ? baseUrl : undefined,
        model: model.trim(),
        messages,
        max_tokens: tokenLimit,
        temperature: temp,
        consent: local || consent,
        ...(local ? {
          top_p: topPValue,
          top_k: topKValue,
          min_p: minPValue,
          repetition_penalty: repeatPenaltyValue,
          seed: seedValue,
          stop_sequences: stopSequences,
        } : {}),
        precision: provider === "axiom-local" ? precision : undefined,
      });
      setResult(nextResult);
      setConversation([...priorConversation, currentUserMessage, { role: "assistant" as const, content: nextResult.response }].slice(-62));
      setPrompt("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The selected model could not generate a response.");
    } finally {
      setRunning(false);
    }
  };

  return <section className="surface local-inference-panel llm-client-panel">
    <div className="surface-head">
      <div>
        <p className="surface-eyebrow">ONE CLIENT · LOCAL OR CLOUD</p>
        <h2>Unified LLM client</h2>
      </div>
      <button type="button" className="button secondary-button" onClick={() => void loadModels()} disabled={loadingModels || running}>
        <RefreshCcw size={14} /> {loadingModels ? "Loading" : "Load models"}
      </button>
    </div>
    <p>Use AXIOM Native Engine or local Ollama models (including GGUF via Ollama), or connect to OpenAI, Anthropic, Google Gemini, DeepSeek, Moonshot/Kimi, OpenRouter, or a public OpenAI-compatible endpoint. Local prompts stay on this computer; cloud requests are sent only after confirmation.</p>
    <div className="status-strip" aria-label="Unified LLM client status">
      <div className="status-strip-item">
        <span className={"status-pill " + (local ? "success" : "warning")}><i />{local ? "LOCAL EXECUTION" : "CLOUD PROVIDER"}</span>
        <div><b>{selected?.label || provider}</b><span>{local ? "No API key or cloud service required for local models." : "The provider receives your prompt after you confirm consent."}</span></div>
      </div>
    </div>
    <form onSubmit={(event) => void run(event)}>
      <div className="two-column-grid">
        <label className="field">
          <span className="field-label-row"><span>Provider</span></span>
          <select aria-label="LLM provider" value={provider} onChange={(event) => changeProvider(event.target.value)}>
            {providers.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
          <small>{selected?.note || "Choose the inference backend for this conversation."}</small>
        </label>
        <label className="field">
          <span className="field-label-row"><span>Model ID</span></span>
          <input list="axiom-unified-llm-models" value={model} onChange={(event) => setModel(event.target.value)} maxLength={256} required autoComplete="off" placeholder="Load models or enter an exact ID" />
          <datalist id="axiom-unified-llm-models">{models.map((name) => <option key={name} value={name} />)}</datalist>
          <small>{models.length} model IDs returned by the selected provider; model IDs can also be entered manually.</small>
        </label>
      </div>
      {provider === "openai-compatible" && <label className="field">
        <span className="field-label-row"><span>Public HTTPS API base URL</span></span>
        <input type="url" value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} required maxLength={512} placeholder="https://provider.example/api/v1" autoComplete="url" />
        <small>Private, loopback, non-HTTPS, and redirecting endpoints are blocked.</small>
      </label>}
      {!local && <label className="field">
        <span className="field-label-row"><span>Provider API key</span><KeyRound size={14} /></span>
        <input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} maxLength={512} required autoComplete="new-password" placeholder="Used only for requests in this page session" />
        <small>Not written to localStorage, workspace files, settings, or runtime traces.</small>
      </label>}
      {provider === "axiom-local" && <label className="field">
        <span className="field-label-row"><span>Native compute precision</span></span>
        <select aria-label="Native compute precision" value={precision} onChange={(event) => setPrecision(event.target.value as typeof precision)}>
          <option value="auto">Auto · preserve checkpoint dtype</option>
          <option value="fp32">FP32 · highest compatibility, highest memory</option>
          <option value="fp16">FP16 · lower memory for suitable hardware</option>
          <option value="bf16">BF16 · wider dynamic range where supported</option>
        </select>
        <small>This only affects AXIOM Native Engine. It is floating-point precision, not quantization.</small>
      </label>}
      <label className="field">
        <span className="field-label-row"><span>System instruction</span><span>{systemPrompt.length.toLocaleString()} / 12,000</span></span>
        <textarea value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} maxLength={12000} rows={2} placeholder="Optional instructions for the model." />
      </label>
      <label className="field">
        <span className="field-label-row"><span>Prompt</span><span>{prompt.length.toLocaleString()} / 12,000</span></span>
        <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} maxLength={12000} rows={5} required placeholder="Ask your selected local or cloud model…" />
      </label>
      <div className="two-column-grid">
        <label className="field"><span className="field-label-row"><span>Maximum output tokens</span></span><input type="number" min={1} max={2048} step={1} value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} required /></label>
        <label className="field"><span className="field-label-row"><span>Temperature</span></span><input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(event) => setTemperature(event.target.value)} required /></label>
      </div>
      {local && <div className="panel-surface" style={{ padding: 14 }}>
        <p className="surface-eyebrow">ADVANCED LOCAL SAMPLING</p>
        <div className="two-column-grid">
          <label className="field"><span className="field-label-row"><span>Top-p</span></span><input type="number" value={topP} onChange={(event) => setTopP(event.target.value)} min={0.000001} max={1} step="any" required /><small>Cumulative probability cutoff.</small></label>
          <label className="field"><span className="field-label-row"><span>Top-k</span></span><input type="number" value={topK} onChange={(event) => setTopK(event.target.value)} min={0} max={100000} step={1} required /><small>0 disables the top-k cutoff.</small></label>
          <label className="field"><span className="field-label-row"><span>Min-p relative probability</span></span><input aria-label="Min-p" type="number" value={minP} onChange={(event) => setMinP(event.target.value)} min={0} max={1} step={0.01} required /><small>0 disables min-p; otherwise keep tokens close enough to the top token's probability.</small></label>
          <label className="field"><span className="field-label-row"><span>Repetition penalty</span></span><input type="number" value={repetitionPenalty} onChange={(event) => setRepetitionPenalty(event.target.value)} min={0.1} max={5} step={0.05} required /><small>1.0 disables the penalty.</small></label>
          <label className="field"><span className="field-label-row"><span>Seed (optional)</span></span><input type="number" value={seed} onChange={(event) => setSeed(event.target.value)} min={0} step={1} placeholder="Runtime default" /><small>Repeatability may differ between devices and runtimes.</small></label>
        </div>
        <label className="field"><span className="field-label-row"><span>Stop sequences</span><span>One per line</span></span><textarea value={stopSequencesText} onChange={(event) => setStopSequencesText(event.target.value)} rows={2} placeholder={"\n---\nEND"} /><small>At most 16 strings of up to 256 characters each. These options apply only to AXIOM Native Engine and local Ollama, not cloud providers.</small></label>
      </div>}
      {!local && <label className="consent-check">
        <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
        <span>I understand this conversation will be sent to {selected?.label || "the selected provider"} and processed under its terms.</span>
      </label>}
      <div className="integration-form-actions">
        <button type="submit" className="button primary-button" disabled={running || loadingModels || !model.trim() || !prompt.trim() || (!local && (!apiKey.trim() || !consent || (provider === "openai-compatible" && !baseUrl.trim())))}><Play size={15} /> {running ? "Generating…" : local ? "Generate locally" : "Send and generate"}</button>
        <span className="quiet-note"><ShieldCheck size={14} /> {local ? "Local model only" : "Explicit cloud consent"}</span>
      </div>
    </form>
    {error && <div className="inline-alert warning" role="alert">{error}</div>}
    {result && <section className="local-inference-result" aria-label="Unified LLM response">
      <div className="surface-head"><div><p className="surface-eyebrow">{result.execution === "local" ? "GENERATED LOCALLY" : "GENERATED BY CLOUD PROVIDER"}</p><h3>{result.model}</h3><small className="quiet-note">{result.provider} · {result.prompt_tokens ?? "?"} input tokens · {result.completion_tokens ?? "?"} output tokens</small>
          {result.execution === "local" && result.provider === "axiom-local" && result.precision && <small className="quiet-note">Compute precision: {result.precision_selection?.selected ?? result.precision} (requested {result.precision_selection?.requested ?? precision})</small>}</div><button type="button" className="button secondary-button" onClick={() => { setConversation([]); setResult(null); }} disabled={running}>Clear conversation</button></div>
      <div className="llm-conversation" aria-label="Current in-memory conversation">
        {conversation.map((message, index) => <div className={"llm-message " + (message.role === "assistant" ? "assistant-message" : "user-message")} key={message.role + "-" + index}>
          <span className="llm-message-role">{message.role === "assistant" ? "ASSISTANT" : "YOU"}</span>
          <pre className="local-inference-output">{message.content}</pre>
        </div>)}
      </div>
      {result.execution === "local" && result.context_management?.chat_history_supplied && <p className="quiet-note">Context budget: {result.context_management.original_message_count} message(s) · {result.context_management.messages_dropped_to_fit_context} older turn(s) dropped · {result.context_management.prompt_tokens_used.toLocaleString()} / {result.context_management.prompt_tokens_before_token_truncation.toLocaleString()} prompt token(s){result.context_management.token_truncation_applied ? " after token-level truncation" : ""}{result.context_management.system_prompt_may_be_truncated ? " · system instructions may have been truncated by the hard context limit" : ""}</p>}
    </section>}
    <div className="agent-bridge-note">
      <div className="surface-eyebrow"><Activity size={13} /> ANDROID AGENT COMPATIBILITY</div>
      <p>PrivateAgent-style clients can connect to AXIOM's token-protected OpenAI-compatible endpoint when you intentionally expose it on your trusted LAN:</p>
      <code>axiom mobile-bridge --listen-lan</code>
      <p>Use http://&lt;AXIOM-PC-LAN-IP&gt;:40509/v1 as the base URL, the one-time token printed by the command as the API key, and a model ID from axiom engine discover --json. Keep the bridge on a trusted network; do not port-forward it.</p>
    </div>
  </section>;
}
