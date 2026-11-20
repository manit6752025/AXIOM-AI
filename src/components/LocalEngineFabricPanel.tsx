import { useEffect, useState, type FormEvent } from "react";
import { Activity, BarChart3, Gauge, RefreshCcw, Rocket, Trash2 } from "lucide-react";
import { axiomApi, type EngineFabricProfile, type EngineFabricReport, type EngineFabricChatResult, type EngineFabricBenchmarkPreflight, type EngineFabricBenchmarkResult, type EngineFabricSavedProfile } from "../api/client";

const formatStatus = (status: string) => status.replaceAll("_", " ").toUpperCase();

export default function LocalEngineFabricPanel() {
  const [profile, setProfile] = useState<EngineFabricProfile>("balanced");
  const [routingMode, setRoutingMode] = useState<"auto" | "selected">("auto");
  const [report, setReport] = useState<EngineFabricReport | null>(null);
  const [selectedBackend, setSelectedBackend] = useState("ollama");
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState("256");
  const [temperature, setTemperature] = useState("0.7");
  const [topP, setTopP] = useState("0.9");
  const [topK, setTopK] = useState("0");
  const [minP, setMinP] = useState("0");
  const [repetitionPenalty, setRepetitionPenalty] = useState("1.0");
  const [seed, setSeed] = useState("");
  const [stopSequencesText, setStopSequencesText] = useState("");
  const [result, setResult] = useState<EngineFabricChatResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [benchmarking, setBenchmarking] = useState(false);
  const [benchmarkWarmups, setBenchmarkWarmups] = useState("1");
  const [benchmarkIterations, setBenchmarkIterations] = useState("3");
  const [benchmarkTokens, setBenchmarkTokens] = useState("64");
  const [benchmarkResults, setBenchmarkResults] = useState<EngineFabricBenchmarkResult[]>([]);
  const [benchmarkPreflight, setBenchmarkPreflight] = useState<EngineFabricBenchmarkPreflight | null>(null);
  const [checkingRequirements, setCheckingRequirements] = useState(false);
  const [savedProfiles, setSavedProfiles] = useState<EngineFabricSavedProfile[]>([]);
  const [profileName, setProfileName] = useState("");
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [profileChecking, setProfileChecking] = useState(false);
  const [connectionVerified, setConnectionVerified] = useState(false);

  const refresh = async (nextProfile = profile) => {
    setConnectionVerified(false);
    setLoading(true);
    setError("");
    try {
      const next = await axiomApi.engineBackends(nextProfile);
      setReport(next);
      setBenchmarkPreflight(null);
      setSelectedBackend((current) => {
        const existing = next.backends.find((item) => item.id === current && item.supports_chat && item.status === "ready");
        if (existing) return current;
        const recommended = next.backends.find((item) => item.id === next.recommendation.backend_id && item.supports_chat && item.status === "ready");
        const firstReady = next.backends.find((item) => item.supports_chat && item.status === "ready");
        return recommended?.id || firstReady?.id || "ollama";
      });
      const backend = next.backends.find((item) => item.id === selectedBackend && item.supports_chat && item.status === "ready");
      const preferred = next.backends.find((item) => item.id === next.recommendation.backend_id && item.supports_chat && item.status === "ready");
      const firstReady = next.backends.find((item) => item.supports_chat && item.status === "ready");
      const active = backend || preferred || firstReady;
      if (active && active.models.length > 0) {
        setModel((current) => active.models.includes(current) ? current : active.models[0]);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Local engine discovery could not be completed.");
      setReport(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh("balanced");
    void axiomApi.engineBackendProfiles().then((result) => setSavedProfiles(result.profiles)).catch(() => {
      // Engine Fabric profile controls remain usable through the API/CLI if the workspace API is unavailable.
    });
  }, []);

  const selected = report?.backends.find((item) => item.id === selectedBackend) ?? null;
  const choices = report?.backends.filter((item) => item.supports_chat) ?? [];
  const modelOptions = routingMode === "auto"
    ? [...new Set((report?.backends ?? []).filter((item) => item.supports_chat && item.status === "ready").flatMap((item) => item.models))]
    : selected?.models ?? [];

  const run = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setResult(null);
    if (routingMode === "selected" && (!selected || selected.status !== "ready")) {
      setError("Select a local backend whose loopback API is ready.");
      return;
    }
    if (!model.trim()) {
      setError("Enter a model ID reported by this server or the exact model name configured in it.");
      return;
    }
    if (!prompt.trim()) {
      setError("Enter a prompt first.");
      return;
    }
    const tokenLimit = Number(maxTokens);
    const temperatureValue = Number(temperature);
    const topPValue = Number(topP);
    const topKValue = Number(topK);
    const minPValue = Number(minP);
    const repetitionPenaltyValue = Number(repetitionPenalty);
    const seedValue = seed.trim() === "" ? null : Number(seed);
    const stopSequences = stopSequencesText.split(/\r?\n/).filter((value) => value.trim().length > 0);
    if (!Number.isInteger(tokenLimit) || tokenLimit < 1 || tokenLimit > 2048) {
      setError("Maximum output tokens must be between 1 and 2,048.");
      return;
    }
    if (!Number.isFinite(temperatureValue) || temperatureValue < 0 || temperatureValue > 2) {
      setError("Temperature must be between 0 and 2.");
      return;
    }
    if (!Number.isFinite(topPValue) || topPValue <= 0 || topPValue > 1) {
      setError("Top-p must be greater than 0 and at most 1.");
      return;
    }
    if (!Number.isInteger(topKValue) || topKValue < 0 || topKValue > 100000) {
      setError("Top-k must be an integer from 0 to 100,000; 0 disables the limit.");
      return;
    }
    if (!Number.isFinite(minPValue) || minPValue < 0 || minPValue > 1) {
      setError("Min-p must be between 0 and 1; 0 disables the cutoff.");
      return;
    }
    if (!Number.isFinite(repetitionPenaltyValue) || repetitionPenaltyValue < 0.1 || repetitionPenaltyValue > 5) {
      setError("Repetition penalty must be between 0.1 and 5.");
      return;
    }
    if (seedValue !== null && (!Number.isSafeInteger(seedValue) || seedValue < 0)) {
      setError("Seed must be a non-negative safe integer, or leave it blank for the runtime default.");
      return;
    }
    if (stopSequences.length > 16 || stopSequences.some((stop) => stop.length > 256)) {
      setError("Use at most 16 stop sequences, each up to 256 characters. Enter one per line.");
      return;
    }
    const sampling = {
      top_p: topPValue,
      top_k: topKValue,
      min_p: minPValue,
      repetition_penalty: repetitionPenaltyValue,
      seed: seedValue,
      stop_sequences: stopSequences,
    };
    setRunning(true);
    try {
      const input = {
        model: model.trim(),
        messages: [{ role: "user" as const, content: prompt }],
        max_new_tokens: tokenLimit,
        temperature: temperatureValue,
        ...sampling,
      };
      const response = routingMode === "auto"
        ? await axiomApi.engineBackendAutoChat({ ...input, profile })
        : await axiomApi.engineBackendChat({
            ...input,
            backend_id: selected!.id as EngineFabricChatResult["backend_id"],
          });
      setResult(response);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Local backend generation failed.");
    } finally {
      setRunning(false);
    }
  };


  const saveConnectorProfile = async () => {
    const name = profileName.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9 _.-]{0,47}$/.test(name)) {
      setError("Connector profile names must be 1–48 safe characters and start with a letter or number.");
      return;
    }
    if (!selected || !model.trim()) {
      setError("Select a local backend and exact model ID before saving a connector profile.");
      return;
    }
    try {
      const saved = await axiomApi.saveEngineBackendProfile({
        name,
        backend_id: selected.id as EngineFabricChatResult["backend_id"],
        model: model.trim(),
      });
      setSavedProfiles((current) => [
        saved.profile,
        ...current.filter((item) => item.id !== saved.profile.id && item.name.toLowerCase() !== name.toLowerCase()),
      ]);
      setSelectedProfileId(saved.profile.id);
      setConnectionVerified(false);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The workspace connector profile could not be saved.");
    }
  };

  const applyConnectorProfile = () => {
    const item = savedProfiles.find((profileItem) => profileItem.id === selectedProfileId);
    if (!item) return;
    setSelectedBackend(item.backend_id);
    setModel(item.model);
    setRoutingMode("selected");
    setResult(null);
    setConnectionVerified(false);
    setBenchmarkPreflight(null);
    setError("");
  };

  const connectConnectorProfile = async () => {
    if (!selectedProfileId) return;
    setProfileChecking(true);
    setError("");
    try {
      const verification = await axiomApi.connectEngineBackendProfile(selectedProfileId);
      const item = savedProfiles.find((profileItem) => profileItem.id === verification.profile_id);
      if (!item) throw new Error("The verified profile is no longer in the saved profile list.");
      setSelectedBackend(item.backend_id);
      setModel(item.model);
      setRoutingMode("selected");
      setConnectionVerified(true);
      setResult(null);
      setBenchmarkPreflight(null);
    } catch (cause) {
      setConnectionVerified(false);
      setError(cause instanceof Error ? cause.message : "The local connector could not be verified.");
    } finally {
      setProfileChecking(false);
    }
  };

  const disconnectConnectorProfile = () => {
    setConnectionVerified(false);
    setResult(null);
    setPrompt("");
    setModel("");
    setBenchmarkPreflight(null);
    setBenchmarkResults([]);
    setError("Disconnected from this AXIOM session. The external runtime remains running and is not stopped. Your saved profile is retained for a later reconnect.");
  };

  const deleteConnectorProfile = async () => {
    if (!selectedProfileId) return;
    try {
      await axiomApi.deleteEngineBackendProfile(selectedProfileId);
      setSavedProfiles((current) => current.filter((item) => item.id !== selectedProfileId));
      setSelectedProfileId("");
      setProfileName("");
      setConnectionVerified(false);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The connector profile could not be deleted.");
    }
  };

  const checkBenchmarkRequirements = async () => {
    setError("");
    if (!selected || !model.trim()) { setError("Select a local adapter and an exact discovered model ID first."); return; }
    setCheckingRequirements(true);
    try {
      const checked = await axiomApi.engineBackendBenchmarkRequirements(selected.id, model.trim());
      setBenchmarkPreflight(checked);
      if (!checked.can_benchmark) setError(checked.preflight.reason ?? "Benchmark requirements are not satisfied.");
    } catch (cause) {
      setBenchmarkPreflight(null);
      setError(cause instanceof Error ? cause.message : "Benchmark requirements could not be checked.");
    } finally { setCheckingRequirements(false); }
  };

  const benchmarkSelected = async () => {
    setError("");
    if (!selected || selected.status !== "ready") {
      setError("Select a local backend whose API is ready before benchmarking.");
      return;
    }
    if (!model.trim() || !prompt.trim()) {
      setError("Choose a model and enter a representative prompt before benchmarking.");
      return;
    }
    const iterations = Number(benchmarkIterations);
    const warmups = Number(benchmarkWarmups);
    const tokens = Number(benchmarkTokens);
    const minPValue = Number(minP);
    if (!Number.isFinite(minPValue) || minPValue < 0 || minPValue > 1) {
      setError("Min-p must be between 0 and 1; 0 disables the cutoff.");
      return;
    }
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 5 || !Number.isInteger(warmups) || warmups < 0 || warmups > 1 || !Number.isInteger(tokens) || tokens < 1 || tokens > 256) {
      setError("Benchmark limits are 1–5 measured requests, 0–1 warm-up and 1–256 output tokens.");
      return;
    }
    setBenchmarking(true);
    try {
      const checked = await axiomApi.engineBackendBenchmarkRequirements(selected.id, model.trim());
      setBenchmarkPreflight(checked);
      if (!checked.can_benchmark) throw new Error(checked.preflight.reason ?? "The selected runtime/model does not meet benchmark requirements.");
      const measured = await axiomApi.engineBackendBenchmark({
        backend_id: selected.id as EngineFabricChatResult["backend_id"],
        model: model.trim(),
        prompt,
        max_new_tokens: tokens,
        warmup_runs: warmups,
        iterations,
        temperature: 0.2,
        top_p: Number(topP),
        top_k: Number(topK),
        min_p: minPValue,
        repetition_penalty: Number(repetitionPenalty),
        seed: seed.trim() === "" ? null : Number(seed),
        stop_sequences: stopSequencesText.split(/\r?\n/).filter((value) => value.trim().length > 0),
      });
      setBenchmarkResults((current) => [
        measured,
        ...current.filter((item) => !(item.backend_id === measured.backend_id && item.model === measured.model)),
      ].slice(0, 8));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The local engine benchmark failed.");
    } finally {
      setBenchmarking(false);
    }
  };

  return <section className="surface local-inference-panel engine-fabric-panel" aria-labelledby="engine-fabric-title">
    <div className="surface-head">
      <div>
        <p className="surface-eyebrow">AXIOM ENGINE FABRIC · LOCAL ROUTING</p>
        <h2 id="engine-fabric-title">Choose the right local engine</h2>
      </div>
      <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading || running}>
        <RefreshCcw size={14} /> {loading ? "Checking" : "Refresh engines"}
      </button>
    </div>
    <p>AXIOM Native Engine is a first-class in-process backend; Fabric compares it with Ollama, llama.cpp, LM Studio, vLLM and llama-cpp-python, then routes local chat requests using your workload profile and exact model ID.</p>
    <section className="surface" aria-label="Engine Fabric connector profiles" style={{ marginTop: 16, padding: 16, border: "1px solid var(--line)", borderRadius: 2 }}>
      <div className="surface-head">
        <div><p className="surface-eyebrow">WORKSPACE-LOCAL · SECRET-FREE</p><h3>Saved local connector profiles</h3></div>
        <span className={"status-pill " + (connectionVerified ? "success" : "neutral")}><i />{connectionVerified ? "VERIFIED THIS SESSION" : "NOT CONNECTED"}</span>
      </div>
      <div className="two-column-grid">
        <label className="field">
          <span className="field-label-row"><span>Saved Engine Fabric profile</span></span>
          <select aria-label="Saved Engine Fabric profile" value={selectedProfileId} onChange={(event) => { setSelectedProfileId(event.target.value); setConnectionVerified(false); setResult(null); }}>
            <option value="">Choose a saved local profile</option>
            {savedProfiles.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.backend_id} · {item.model}</option>)}
          </select>
        </label>
        <div className="integration-form-actions">
          <button type="button" className="button secondary-button" onClick={applyConnectorProfile} disabled={!selectedProfileId}>Apply</button>
          <button type="button" className="button secondary-button" onClick={connectConnectorProfile} disabled={!selectedProfileId || profileChecking || running || benchmarking}>
            {profileChecking ? "Verifying…" : "Connect"}
          </button>
          <button type="button" className="button secondary-button" onClick={disconnectConnectorProfile} disabled={!connectionVerified || profileChecking}>Disconnect</button>
          <button type="button" className="button secondary-button" onClick={() => void deleteConnectorProfile()} disabled={!selectedProfileId || profileChecking}>Delete</button>
        </div>
      </div>
      <div className="two-column-grid">
        <label className="field">
          <span className="field-label-row"><span>Profile name</span></span>
          <input aria-label="Engine Fabric profile name" value={profileName} onChange={(event) => setProfileName(event.target.value)} maxLength={48} placeholder="e.g. Office llama.cpp" />
        </label>
        <div className="integration-form-actions">
          <button type="button" className="button secondary-button" onClick={() => void saveConnectorProfile()} disabled={!profileName.trim() || !selected || !model.trim()}>Save selected backend + model</button>
        </div>
      </div>
      <p className="quiet-note">Profiles store only a fixed backend ID and exact model ID in the active workspace. API keys, endpoint URLs, prompts and chat history are never saved. Connect performs a fresh loopback health/model check; it does not retain a socket or launch a process. Disconnect clears this AXIOM view only and does not stop an external runtime.</p>
    </section>
    <div className="two-column-grid">
      <label className="field">
        <span className="field-label-row"><span>Optimization profile</span></span>
        <select value={profile} onChange={(event) => { const next = event.target.value as EngineFabricProfile; setProfile(next); void refresh(next); }} disabled={loading || running}>
          {(report?.profiles ?? [{ id: "balanced", label: "Balanced" }, { id: "low-memory", label: "Low memory" }, { id: "throughput", label: "High throughput" }, { id: "easy-setup", label: "Easy setup" }, { id: "portable", label: "Portable / quantized" }, { id: "minimal-daemons", label: "Minimal background services" }]).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
      </label>
      <div className="panel-surface" style={{ padding: 14, alignSelf: "end" }}>
        <p className="surface-eyebrow"><Gauge size={13} style={{ verticalAlign: "middle", marginRight: 5 }} />AXIOM RECOMMENDATION</p>
        <strong>{report?.recommendation.name ?? "Checking local engines…"}</strong>
        <p className="quiet-note">{report?.recommendation.reason ?? "The recommendation is based on local runtime readiness and your selected profile, not synthetic speed claims."}</p>
        {report?.recommendation.status === "setup_required" && <span className="status-pill warning"><i />SETUP REQUIRED</span>}
      </div>
    </div>
    {error && <div className="inline-alert warning" role="alert">{error}</div>}
    <div className="info-grid" aria-label="Local engine inventory">
      {(report?.backends ?? []).map((backend) => <article className="panel-surface" key={backend.id} style={{ padding: 14, minWidth: 0 }}>
        <div className="surface-head">
          <div><p className="surface-eyebrow">{backend.kind === "native" ? "AXIOM-OWNED RUNTIME" : "OPTIONAL LOCAL ADAPTER"}</p><h3>{backend.name}</h3></div>
          <span className={"status-pill " + (backend.status === "ready" ? "success" : "warning")}><i />{formatStatus(backend.status)}</span>
        </div>
        <p className="quiet-note">{backend.endpoint ? backend.endpoint : "In-process · trusted local model roots"}</p>
        <p className="quiet-note">{backend.model_count} model{backend.model_count === 1 ? "" : "s"} discovered{backend.installed ? " · runtime installed" : ""}</p>
        <p><strong>Strengths</strong></p>
        <ul>{backend.strengths.slice(0, 3).map((strength) => <li key={strength}>{strength}</li>)}</ul>
        <p><strong>Trade-offs</strong></p>
        <ul>{backend.tradeoffs.slice(0, 2).map((tradeoff) => <li key={tradeoff}>{tradeoff}</li>)}</ul>
        <p><strong>Generation contract</strong></p>
        <p className="quiet-note">{backend.generation.note}</p>
        {backend.generation.extension_sampling_controls.length > 0 && <p className="quiet-note">Compatibility extensions: {backend.generation.extension_sampling_controls.join(", ")}. Acceptance can vary by runtime version.</p>}
        {backend.error && <p className="quiet-note">{backend.error}</p>}
        {backend.id === report?.recommendation.backend_id && <p className="quiet-note"><Rocket size={13} style={{ verticalAlign: "middle", marginRight: 5 }} />Recommended for this profile</p>}
      </article>)}
    </div>
    <section className="surface" aria-labelledby="fabric-chat-title" style={{ marginTop: 18, padding: 18, border: "1px solid var(--line-strong)", borderRadius: 2, background: "var(--surface)" }}>
      <div className="surface-head"><div><p className="surface-eyebrow">LOCAL EXECUTION · NO CLOUD ROUTING</p><h3 id="fabric-chat-title">Run a local chat</h3></div><span className="quiet-note"><Activity size={14} style={{ verticalAlign: "middle", marginRight: 5 }} />Bounded request</span></div>
      <p className="quiet-note">External runtimes use fixed loopback APIs; AXIOM Native Engine runs in-process from trusted local model roots. No API keys, remote hosts, service launches or model downloads are used during dispatch.</p>
      <form onSubmit={(event) => void run(event)}>
        <div className="two-column-grid">
          <label className="field"><span className="field-label-row"><span>Execution strategy</span></span>
            <select aria-label="Execution strategy" value={routingMode} onChange={(event) => { setRoutingMode(event.target.value as "auto" | "selected"); setResult(null); }} disabled={loading || running}>
              <option value="auto">Automatic profile routing + safe failover</option>
              <option value="selected">Choose a specific local runtime</option>
            </select>
            <small>Automatic routing stays local and only fails over to ready services advertising the exact same model ID.</small>
          </label>
          {routingMode === "selected" && <label className="field"><span className="field-label-row"><span>Local engine</span></span>
            <select aria-label="Local engine" value={selectedBackend} onChange={(event) => { const id = event.target.value; setSelectedBackend(id); setConnectionVerified(false); setResult(null); const target = report?.backends.find((item) => item.id === id); setModel(target?.models[0] ?? ""); }} disabled={loading || running}>
              {choices.map((item) => <option key={item.id} value={item.id}>{item.name} — {formatStatus(item.status)}</option>)}
            </select>
          </label>}
          <label className="field"><span className="field-label-row"><span>Model ID</span></span>
            <input list="axiom-fabric-model-options" value={model} onChange={(event) => { setModel(event.target.value); setConnectionVerified(false); }} maxLength={120} required placeholder={routingMode === "auto" ? "Exact model ID shared by matching local services" : "Model ID reported by the selected local server"} disabled={running || (routingMode === "selected" && (!selected || selected.status !== "ready"))} />
            <datalist id="axiom-fabric-model-options">{modelOptions.map((name) => <option key={name} value={name} />)}</datalist>
            <small>{routingMode === "auto" ? "Pick an ID shared by a ready local service; model names do not guarantee identical weights across engines." : selected?.status === "ready" ? "Choose an ID from this service or enter its exact configured alias." : "Start the selected service and refresh discovery to enable local chat."}</small>
          </label>
        </div>
        <label className="field"><span className="field-label-row"><span>Prompt</span><span>{prompt.length.toLocaleString()} / 12,000 characters</span></span>
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} maxLength={12000} rows={4} required placeholder="Ask the selected local model to do something…" />
        </label>
        <div className="two-column-grid">
          <label className="field"><span className="field-label-row"><span>Maximum generated tokens</span></span><input type="number" value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} min={1} max={2048} step={1} required /></label>
          <label className="field"><span className="field-label-row"><span>Temperature</span></span><input type="number" value={temperature} onChange={(event) => setTemperature(event.target.value)} min={0} max={2} step={0.1} required /></label>
        </div>
        <p className="surface-eyebrow">ADVANCED SAMPLING</p>
        <div className="two-column-grid">
          <label className="field"><span className="field-label-row"><span>Top-p</span></span><input type="number" value={topP} onChange={(event) => setTopP(event.target.value)} min={0.000001} max={1} step="any" required /><small>Cumulative probability threshold.</small></label>
          <label className="field"><span className="field-label-row"><span>Top-k</span></span><input type="number" value={topK} onChange={(event) => setTopK(event.target.value)} min={0} max={100000} step={1} required /><small>0 disables the top-k cutoff.</small></label>
          <label className="field"><span className="field-label-row"><span>Min-p relative probability</span></span><input aria-label="Min-p" type="number" value={minP} onChange={(event) => setMinP(event.target.value)} min={0} max={1} step={0.01} required /><small>0 disables min-p. Nonzero values require AXIOM Native Engine or Ollama; auto-routing filters to those compatible backends.</small></label>
          <label className="field"><span className="field-label-row"><span>Repetition penalty</span></span><input type="number" value={repetitionPenalty} onChange={(event) => setRepetitionPenalty(event.target.value)} min={0.1} max={5} step={0.05} required /><small>1.0 leaves repetition unpenalized.</small></label>
          <label className="field"><span className="field-label-row"><span>Seed (optional)</span></span><input type="number" value={seed} onChange={(event) => setSeed(event.target.value)} min={0} step={1} placeholder="Random / runtime default" /><small>Fixed seeds improve repeatability but cannot guarantee identical output across different runtimes.</small></label>
        </div>
        <label className="field"><span className="field-label-row"><span>Stop sequences (optional)</span><span>One exact string per line</span></span><textarea value={stopSequencesText} onChange={(event) => setStopSequencesText(event.target.value)} rows={2} placeholder={"\n---\nEND"} /><small>Up to 16 sequences, each at most 256 characters. Stop strings are passed to the selected local runtime.</small></label>
        <div className="integration-form-actions">
          <button type="submit" className="button primary-button" disabled={loading || running || (routingMode === "selected" && (!selected || selected.status !== "ready")) || !model.trim() || !prompt.trim()}><PlayIcon />{running ? "Generating locally…" : routingMode === "auto" ? "Auto-route local request" : "Run on selected engine"}</button>
          {result && <span className="quiet-note">{result.eval_count ?? "—"} completion tokens · {result.total_duration_ms} ms · {result.backend_name}</span>}
        </div>
      </form>
      {result && <section className="local-inference-result" aria-label="Engine Fabric response"><div className="surface-head"><div><p className="surface-eyebrow">GENERATED LOCALLY</p><h3>{result.model}</h3><small className="quiet-note">{result.backend_name} · {result.generated_at}</small></div><span className="status-pill success"><i />COMPLETED</span></div><pre className="local-inference-output">{result.response || "(The model returned an empty response.)"}</pre></section>}
    </section>

    <section className="surface" aria-labelledby="fabric-benchmark-title" style={{ marginTop: 18, padding: 18, border: "1px solid var(--line-strong)", borderRadius: 2, background: "var(--surface)" }}>
      <div className="surface-head">
        <div><p className="surface-eyebrow">MEASURED ON THIS MACHINE · SESSION-ONLY</p><h3 id="fabric-benchmark-title"><BarChart3 size={17} style={{ verticalAlign: "middle", marginRight: 8 }} />Local engine shootout</h3></div>
        <button type="button" className="button primary-button" onClick={() => void benchmarkSelected()} disabled={loading || benchmarking || running || routingMode === "auto" || !selected || selected.status !== "ready" || !model.trim() || !prompt.trim()}>
          <Activity size={15} /> {benchmarking ? "Measuring…" : "Benchmark selected engine"}
        </button>
      </div>
      <p className="quiet-note">Runs repeated requests against one selected local runtime, including AXIOM Native Engine when ready. Switch Execution strategy to “Choose a specific local runtime” before benchmarking; automatic failover is deliberately excluded so every sample measures the same backend. Generated text is discarded by the API, and samples stay in this screen session only.</p>
      {selected?.benchmark_requirements && <div className="panel-surface" style={{ padding: 12, marginBottom: 12 }}>
        <p className="surface-eyebrow">ADAPTER REQUIREMENTS · {selected.benchmark_requirements.adapter_id}</p>
        <p className="quiet-note"><strong>Runtime:</strong> {selected.benchmark_requirements.runtime.component ?? "AXIOM Native Engine"} · {selected.benchmark_requirements.runtime.readiness_evidence}</p>
        <p className="quiet-note"><strong>Model:</strong> exact advertised ID from {selected.benchmark_requirements.model.inventory_source}</p>
        <p className="quiet-note"><strong>Formats:</strong> {selected.benchmark_requirements.model.supported_formats.join(", ")} · sequential requests, max 5 measured / 1 warm-up / 256 output tokens</p>
        <button type="button" className="button secondary-button" onClick={() => void checkBenchmarkRequirements()} disabled={checkingRequirements || benchmarking || !model.trim()}>{checkingRequirements ? "Checking requirements…" : "Check benchmark requirements"}</button>
        {benchmarkPreflight?.backend_id === selected.id && benchmarkPreflight.preflight.model_id === model.trim() && <p className="quiet-note" role="status">Preflight: {benchmarkPreflight.preflight.status.toUpperCase()}{benchmarkPreflight.preflight.reason ? " · " + benchmarkPreflight.preflight.reason : " · Runtime and exact model are ready."}</p>}
      </div>}
      <div className="two-column-grid">
        <label className="field"><span className="field-label-row"><span>Warm-up requests</span></span><select value={benchmarkWarmups} onChange={(event) => setBenchmarkWarmups(event.target.value)} disabled={benchmarking}><option value="0">0</option><option value="1">1</option></select></label>
        <label className="field"><span className="field-label-row"><span>Measured requests</span></span><select value={benchmarkIterations} onChange={(event) => setBenchmarkIterations(event.target.value)} disabled={benchmarking}>{[1,2,3,4,5].map((count) => <option key={count} value={String(count)}>{count}</option>)}</select></label>
        <label className="field"><span className="field-label-row"><span>Output token cap</span></span><select value={benchmarkTokens} onChange={(event) => setBenchmarkTokens(event.target.value)} disabled={benchmarking}>{[16,32,64,128,256].map((count) => <option key={count} value={String(count)}>{count}</option>)}</select></label>
      </div>
      {benchmarkResults.length > 0 && <div style={{ overflowX: "auto", marginTop: 14 }}>
        <div className="surface-head"><div><p className="surface-eyebrow">CURRENT SCREEN SESSION</p><h4>Measured backend comparison</h4></div><button type="button" className="button secondary-button" onClick={() => setBenchmarkResults([])} disabled={benchmarking}><Trash2 size={14} /> Clear comparison</button></div>
        <table className="data-table" aria-label="Measured local engine comparison">
          <thead><tr><th>Backend / model</th><th>Median</th><th>P95</th><th>Completion throughput</th><th>Samples</th></tr></thead>
          <tbody>{benchmarkResults.map((item) => <tr key={item.backend_id + ":" + item.model}><td>{item.backend_name}<br /><span className="quiet-note">{item.model}</span></td><td>{item.measurements.latency_ms.median} ms</td><td>{item.measurements.latency_ms.p95} ms</td><td>{item.measurements.completion_tokens_per_second === null ? "Not reported" : item.measurements.completion_tokens_per_second + " tokens/s"}</td><td>{item.measurements.completion_tokens_known}/{item.configuration.measured_requests} token counts</td></tr>)}</tbody>
        </table>
        <p className="quiet-note">Compare results carefully: model IDs, quantization, context, hardware, cache state and runtime settings can differ. These measurements are observational, not a universal ranking.</p>
      </div>}
    </section>
    {result?.routing && <div className="panel-surface" aria-label="Automatic routing attempts" style={{ padding: 14, marginTop: 12 }}>
      <p className="surface-eyebrow">AUTOMATIC ROUTING · {result.routing.profile}</p>
      <p>{result.routing.fallback_used ? "A fallback was used after an earlier backend failed." : "The first matching ready backend completed the request."}</p>
      <ul>{result.routing.attempts.map((attempt, index) => <li key={attempt.backend_id + ":" + index}>
        <strong>{attempt.backend_name}</strong>: {attempt.outcome === "selected" ? "selected" : "failed"}
        {attempt.reason ? " — " + attempt.reason : ""}
      </li>)}</ul>
      <p className="quiet-note">{result.routing.scope_note}</p>
    </div>}
    {report?.scope_note && <p className="quiet-note" style={{ marginTop: 12 }}>{report.scope_note}</p>}
  </section>;
}

function PlayIcon() {
  return <Rocket size={15} aria-hidden="true" />;
}
