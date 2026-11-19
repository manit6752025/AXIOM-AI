import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Activity, BarChart3, Download, Info, Play, Power, RefreshCcw, Square } from "lucide-react";
import { axiomApi, type LocalInferenceBenchmark, type LocalInferenceResult, type LocalInferenceStatus, type OllamaPullJob, type OllamaModelDetails, type OllamaRunningModels, type LocalInferenceStreamEvent } from "../api/client";

export default function LocalInferencePanel() {
  const [status, setStatus] = useState<LocalInferenceStatus | null>(null);
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [numPredict, setNumPredict] = useState("256");
  const [temperature, setTemperature] = useState("0.7");
  const [topP, setTopP] = useState("0.9");
  const [topK, setTopK] = useState("40");
  const [minP, setMinP] = useState("0");
  const [repetitionPenalty, setRepetitionPenalty] = useState("1.1");
  const [seedValue, setSeedValue] = useState("");
  const [contextWindow, setContextWindow] = useState("");
  const [stopSequences, setStopSequences] = useState("");
  const [result, setResult] = useState<LocalInferenceResult | null>(null);
  const [benchmark, setBenchmark] = useState<LocalInferenceBenchmark | null>(null);
  const [warmupRuns, setWarmupRuns] = useState("1");
  const [benchmarkIterations, setBenchmarkIterations] = useState("3");
  const [concurrency, setConcurrency] = useState("1");
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState("");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [streamController, setStreamController] = useState<AbortController | null>(null);
  const [error, setError] = useState("");
  const [serviceBusy, setServiceBusy] = useState(false);
  const [serviceError, setServiceError] = useState("");
  const [pullModelName, setPullModelName] = useState("qwen2.5:0.5b");
  const [pullJob, setPullJob] = useState<OllamaPullJob | null>(null);
  const [pullBusy, setPullBusy] = useState(false);
  const [pullError, setPullError] = useState("");
  const [modelDetails, setModelDetails] = useState<OllamaModelDetails | null>(null);
  const [inspectBusy, setInspectBusy] = useState(false);
  const [inspectError, setInspectError] = useState("");
  const [unloadBusy, setUnloadBusy] = useState(false);
  const [unloadingModel, setUnloadingModel] = useState<string | null>(null);
  const [unloadMessage, setUnloadMessage] = useState("");
  const [removeBusy, setRemoveBusy] = useState(false);
  const [removeMessage, setRemoveMessage] = useState("");
  const [runningModels, setRunningModels] = useState<OllamaRunningModels | null>(null);
  const [runningBusy, setRunningBusy] = useState(false);
  const [runningError, setRunningError] = useState("");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const next = await axiomApi.inferenceStatus();
      setStatus(next);
      setModel((current) => current && next.models.includes(current) ? current : next.models[0] || current);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Local inference status is unavailable.");
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const manageService = async (action: "start" | "stop") => {
    setServiceError(""); setServiceBusy(true);
    try { if (action === "start") await axiomApi.startLocalOllama(); else await axiomApi.stopLocalOllama(); await refresh(); }
    catch (cause) { setServiceError(cause instanceof Error ? cause.message : "Ollama service management failed."); await refresh(); }
    finally { setServiceBusy(false); }
  };

  const pullModel = async () => {
    setPullError(""); setPullJob(null);
    const name = pullModelName.trim();
    if (!name) { setPullError("Enter an Ollama model tag, such as qwen2.5:0.5b."); return; }
    if (!online) { setPullError("Start Ollama before downloading a model."); return; }
    setPullBusy(true);
    try {
      let job = await axiomApi.startOllamaModelPull(name); setPullJob(job);
      while (job.status === "queued" || job.status === "running") {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 900));
        job = await axiomApi.ollamaModelPullStatus(job.id); setPullJob(job);
      }
      if (job.status === "completed") { setPullModelName(""); await refresh(); }
      else if (job.status === "failed") setPullError(job.error || "The model download failed.");
    } catch (cause) { setPullError(cause instanceof Error ? cause.message : "The local model download failed."); }
    finally { setPullBusy(false); }
  };

  const cancelPull = async () => {
    if (!pullJob || !["queued", "running"].includes(pullJob.status)) return;
    try { setPullJob(await axiomApi.cancelOllamaModelPull(pullJob.id)); }
    catch (cause) { setPullError(cause instanceof Error ? cause.message : "Could not cancel the model download."); }
  };

  const refreshRunningModels = async () => {
    setRunningError(""); setRunningBusy(true);
    try { setRunningModels(await axiomApi.runningOllamaModels()); }
    catch (cause) { setRunningError(cause instanceof Error ? cause.message : "Loaded model inventory is unavailable."); }
    finally { setRunningBusy(false); }
  };

  const inspectModel = async () => {
    setInspectError(""); setModelDetails(null);
    if (!online || !model.trim()) { setInspectError("Choose an installed local model while Ollama is online."); return; }
    setInspectBusy(true);
    try { setModelDetails(await axiomApi.inspectOllamaModel(model.trim())); }
    catch (cause) { setInspectError(cause instanceof Error ? cause.message : "Could not inspect the selected model."); }
    finally { setInspectBusy(false); }
  };

  const unloadModelByName = async (modelName: string) => {
    setUnloadMessage("");
    if (!online || !modelName.trim()) { setUnloadMessage("Choose an installed local model while Ollama is online."); return; }
    setUnloadBusy(true);
    setUnloadingModel(modelName);
    try {
      const result = await axiomApi.unloadOllamaModel(modelName.trim());
      setUnloadMessage(result.scope_note);
      if (runningModels) await refreshRunningModels();
    } catch (cause) { setUnloadMessage(cause instanceof Error ? cause.message : "Could not unload this model."); }
    finally { setUnloadBusy(false); setUnloadingModel(null); }
  };

  const removeModel = async () => {
    setRemoveMessage("");
    const modelName = model.trim();
    if (!online || !modelName || !status?.models.includes(modelName)) { setRemoveMessage("Choose an exact installed model tag before removing it."); return; }
    const confirmed = window.confirm(`Permanently remove the downloaded Ollama model tag “${modelName}”? This cannot be undone from AXIOM. Model blobs shared with other tags may remain.`);
    if (!confirmed) return;
    setRemoveBusy(true);
    try {
      const result = await axiomApi.removeOllamaModel(modelName);
      setRemoveMessage(result.removed ? `${modelName} was removed from the local Ollama inventory. ${result.scope_note}` : `Ollama did not confirm removal of ${modelName}.`);
      if (result.removed) {
        setModel(""); setResult(null); setModelDetails(null);
        await refresh();
        if (runningModels) await refreshRunningModels();
      }
    } catch (cause) { setRemoveMessage(cause instanceof Error ? cause.message : "Could not remove the selected local model."); }
    finally { setRemoveBusy(false); }
  };

  const unloadModel = async () => {
    if (!model.trim()) { setUnloadMessage("Choose an installed local model while Ollama is online."); return; }
    await unloadModelByName(model.trim());
  };

  const run = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setResult(null);
    if (!model.trim()) { setError("Enter the name of a model that is already available in Ollama."); return; }
    if (!prompt.trim()) { setError("Enter a prompt first."); return; }
    const tokenLimit = Number(numPredict);
    const temperatureValue = Number(temperature);
    if (!Number.isInteger(tokenLimit) || tokenLimit < 1 || tokenLimit > 2048) { setError("Maximum generated tokens must be between 1 and 2048."); return; }
    if (!Number.isFinite(temperatureValue) || temperatureValue < 0 || temperatureValue > 2) { setError("Temperature must be between 0 and 2."); return; }
    const topPValue = Number(topP);
    const topKValue = Number(topK);
    const minPValue = Number(minP);
    const repetitionPenaltyValue = Number(repetitionPenalty);
    const seedText = seedValue.trim();
    const contextText = contextWindow.trim();
    const seedNumber = seedText ? Number(seedText) : null;
    const contextNumber = contextText ? Number(contextText) : null;
    const stops = stopSequences.split(/\r?\n/).map((value) => value.trim()).filter(Boolean);
    if (!Number.isFinite(topPValue) || topPValue <= 0 || topPValue > 1) { setError("Top-p must be greater than 0 and at most 1."); return; }
    if (!Number.isInteger(topKValue) || topKValue < 0 || topKValue > 100000) { setError("Top-k must be between 0 and 100000."); return; }
    if (!Number.isFinite(minPValue) || minPValue < 0 || minPValue > 1) { setError("Min-p must be between 0 and 1; 0 disables min-p."); return; }
    if (!Number.isFinite(repetitionPenaltyValue) || repetitionPenaltyValue < 0.1 || repetitionPenaltyValue > 5) { setError("Repetition penalty must be between 0.1 and 5."); return; }
    if (seedText && (!Number.isSafeInteger(seedNumber) || seedNumber! < 0)) { setError("Seed must be a non-negative safe integer."); return; }
    if (contextText && (!Number.isInteger(contextNumber) || contextNumber! < 512 || contextNumber! > 131072)) { setError("Context window must be between 512 and 131072 tokens."); return; }
    if (stops.length > 16 || stops.some((value) => value.length > 256)) { setError("Use at most 16 stop sequences, each no longer than 256 characters."); return; }
    setRunning(true);
    const controller = new AbortController();
    setStreamController(controller);
    setResult({ model: model.trim(), response: "", done: false, prompt_eval_count: null, eval_count: null, total_duration_ms: null, generated_at: new Date().toISOString() });
    try {
      const finalResult = await axiomApi.streamLocalInference(
        { model: model.trim(), prompt, num_predict: tokenLimit, temperature: temperatureValue, top_p: topPValue, top_k: topKValue, min_p: minPValue, repetition_penalty: repetitionPenaltyValue, ...(seedNumber !== null ? { seed: seedNumber } : {}), ...(contextNumber !== null ? { num_ctx: contextNumber } : {}), ...(stops.length ? { stop_sequences: stops } : {}) },
        (event: LocalInferenceStreamEvent) => {
          if (event.type === "token") {
            setResult((current) => current ? { ...current, response: current.response + event.token, done: false } : current);
          }
        },
        controller.signal,
      );
      setResult(finalResult);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") {
        setError("Generation stopped. Partial output is retained.");
      } else {
        setError(cause instanceof Error ? cause.message : "Local generation failed.");
      }
    } finally {
      setRunning(false);
      setStreamController(null);
    }
  };

  const benchmarkLocalModel = async () => {
    setBenchmarkError("");
    setBenchmark(null);
    if (!online) { setBenchmarkError("Start the local Ollama runtime and check its status before benchmarking."); return; }
    if (!model.trim()) { setBenchmarkError("Choose an installed local Ollama model."); return; }
    if (!prompt.trim()) { setBenchmarkError("Enter a prompt before benchmarking."); return; }
    const tokenLimit = Number(numPredict);
    const temperatureValue = Number(temperature);
    const warmupCount = Number(warmupRuns);
    const repetitions = Number(benchmarkIterations);
    const parallel = Number(concurrency);
    if (!Number.isInteger(tokenLimit) || tokenLimit < 1 || tokenLimit > 512) {
      setBenchmarkError("Benchmark token limit must be between 1 and 512. Lower the existing generation token setting first.");
      return;
    }
    if (!Number.isFinite(temperatureValue) || temperatureValue < 0 || temperatureValue > 2) {
      setBenchmarkError("Temperature must be between 0 and 2.");
      return;
    }
    if (!Number.isInteger(warmupCount) || warmupCount < 0 || warmupCount > 2) {
      setBenchmarkError("Warm-up generations must be between 0 and 2.");
      return;
    }
    if (!Number.isInteger(repetitions) || repetitions < 1 || repetitions > 10) {
      setBenchmarkError("Measured iterations must be between 1 and 10.");
      return;
    }
    if (!Number.isInteger(parallel) || parallel < 1 || parallel > 4 || repetitions * parallel > 10) {
      setBenchmarkError("Concurrency must be 1–4 and iterations × concurrency must not exceed 10 requests.");
      return;
    }
    setBenchmarkBusy(true);
    try {
      setBenchmark(await axiomApi.runLocalInferenceBenchmark({
        model: model.trim(),
        prompt,
        num_predict: tokenLimit,
        temperature: temperatureValue,
        warmup_runs: warmupCount,
        iterations: repetitions,
        concurrency: parallel,
      }));
    } catch (cause) {
      setBenchmarkError(cause instanceof Error ? cause.message : "The local inference benchmark failed.");
    } finally {
      setBenchmarkBusy(false);
    }
  };

  const online = status?.status === "online";
  return <section className="surface local-inference-panel ollama-inference-panel">
    <div className="surface-head">
      <div>
        <p className="surface-eyebrow">REAL GENERATION · LOOPBACK ONLY</p>
        <h2>Local model inference</h2>
      </div>
      <div className="integration-form-actions">
        <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading || running || serviceBusy}><RefreshCcw size={14} /> {loading ? "Checking" : "Check runtime"}</button>
        <button type="button" className="button primary-button" onClick={() => void manageService("start")} disabled={loading || serviceBusy || online || status?.service_can_start === false || !status?.runtime_installed}><Power size={14} /> {serviceBusy ? "Managing…" : "Start Ollama"}</button>
        <button type="button" className="button secondary-button" onClick={() => void manageService("stop")} disabled={loading || serviceBusy || status?.service_can_stop !== true}><Square size={13} /> Stop AXIOM service</button>
      </div>
    </div>
    <p>Generate text with a model already installed in Ollama. AXIOM sends requests only to <code>127.0.0.1:11434</code>; it does not download models or call a cloud inference API.</p>
    <div className="status-strip" aria-label="Local inference status">
      <div className="status-strip-item">
        <span className={`status-pill ${online ? "success" : "warning"}`}><i />{loading ? "CHECKING" : online ? "OLLAMA ONLINE" : "OLLAMA OFFLINE"}</span>
        <div><b>{online ? `${status?.models.length ?? 0} local models` : "Runtime unavailable"}</b><span>{status?.runtime_installed ? "Ollama executable detected" : "Ollama executable not detected"}</span></div>
      </div>
    </div>
    <p className="quiet-note">{status?.service_message || "Service management is restricted to Ollama processes started and verified by AXIOM."} AXIOM-managed services bind to <code>127.0.0.1:11434</code>.</p>
    {serviceError && <div className="inline-alert warning" role="alert">{serviceError}</div>}
    {!loading && !online && <div className="inline-alert warning" role="status">{status?.error || error || "Ollama is offline."} {status?.runtime_installed ? "Start Ollama to run or download a model." : "Install Ollama for your operating system, then refresh status."}</div>}
    <section className="surface" aria-labelledby="ollama-pull-title" style={{ marginTop: 16, padding: 16, border: "1px solid var(--line-strong)", borderRadius: 2, background: "var(--surface)" }}>
      <div className="surface-head"><div><p className="surface-eyebrow">EXPLICIT DOWNLOAD · LOCAL MODEL STORE</p><h3 id="ollama-pull-title"><Download size={16} style={{ verticalAlign: "middle", marginRight: 8 }} />Install an Ollama model</h3></div></div>
      <p className="quiet-note">Downloading a model uses the Ollama library and may consume substantial bandwidth and disk space. It starts only when you press Pull model. Downloads are limited to two concurrent jobs and retain only bounded in-memory history.</p>
      <form onSubmit={(event) => { event.preventDefault(); void pullModel(); }}>
        <label className="field"><span className="field-label-row"><span>Ollama model tag</span></span><input value={pullModelName} onChange={(event) => setPullModelName(event.target.value)} maxLength={120} placeholder="e.g. qwen2.5:0.5b" autoComplete="off" required disabled={pullBusy} /><small>Use a model tag from the Ollama library, for example <code>qwen2.5:0.5b</code>.</small></label>
        <div className="integration-form-actions"><button type="submit" className="button primary-button" disabled={!online || loading || pullBusy || !pullModelName.trim()}><Download size={14} /> {pullBusy ? "Downloading…" : "Pull model"}</button>{pullBusy && pullJob && <button type="button" className="button secondary-button" onClick={() => void cancelPull()}><Square size={12} /> Cancel download</button>}</div>
      </form>
      {pullError && <div className="inline-alert warning" role="alert">{pullError}</div>}
      {pullJob && <div className="panel-surface" role="status" aria-label="Model download progress" style={{ marginTop: 14, padding: 14 }}>
        <div className="surface-head"><div><b>{pullJob.model}</b><small className="quiet-note" style={{ display: "block" }}>{pullJob.progress}</small></div><span className={`status-pill ${pullJob.status === "completed" ? "success" : pullJob.status === "failed" ? "warning" : "success"}`}><i />{pullJob.status.toUpperCase()}</span></div>
        {pullJob.percent !== null && <div style={{ margin: "10px 0", height: 6, background: "var(--line-strong)", overflow: "hidden" }}><div style={{ width: `${pullJob.percent}%`, height: "100%", background: "var(--accent)", transition: "width 160ms ease" }} /></div>}
        <ul className="quiet-note" style={{ paddingLeft: 18, margin: "8px 0" }}>{pullJob.logs.slice(-5).map((line, index) => <li key={index}>{line}</li>)}</ul>{pullJob.error && <div className="inline-alert warning" role="alert">{pullJob.error}</div>}
      </div>}
    </section>
    <form onSubmit={(event) => void run(event)}>
      <label className="field">
        <span className="field-label-row"><span>Model name</span></span>
        <input list="axiom-local-model-options" value={model} onChange={(event) => setModel(event.target.value)} maxLength={120} placeholder="e.g. qwen2.5:0.5b" autoComplete="off" />
        <datalist id="axiom-local-model-options">{(status?.models || []).map((name) => <option key={name} value={name} />)}</datalist>
        <small>{status?.models.length ? "Suggestions come from the local model inventory." : "Enter an exact model name that has already been pulled locally."}</small>
      </label>
      <div className="integration-form-actions">
        <button type="button" className="button secondary-button" onClick={() => void refreshRunningModels()} disabled={!online || loading || runningBusy}><Activity size={14} /> {runningBusy ? "Checking memory…" : "Show loaded models"}</button>
        <button type="button" className="button secondary-button" onClick={() => void inspectModel()} disabled={!online || loading || inspectBusy || running || !model.trim()}><Info size={14} /> {inspectBusy ? "Inspecting…" : "Inspect model"}</button>
        <button type="button" className="button secondary-button" onClick={() => void removeModel()} disabled={!online || loading || removeBusy || running || !model.trim() || !status?.models.includes(model.trim())}><Square size={12} /> {removeBusy ? "Removing model…" : "Remove downloaded model"}</button>
        <button type="button" className="button secondary-button" onClick={() => void unloadModel()} disabled={!online || loading || unloadBusy || running || !model.trim()}><Square size={12} /> {unloadBusy ? "Unloading…" : "Unload from memory"}</button>
      </div>
      {inspectError && <div className="inline-alert warning" role="alert">{inspectError}</div>}
      {unloadMessage && <div className="inline-alert" role="status">{unloadMessage}</div>}
      {removeMessage && <div className="inline-alert" role="status">{removeMessage}</div>}
      {runningError && <div className="inline-alert warning" role="alert">{runningError}</div>}
      {runningModels && <section className="panel-surface" aria-label="Loaded Ollama models" style={{ marginTop: 12, padding: 16 }}>
        <div className="surface-head"><div><p className="surface-eyebrow">LIVE PROCESS INVENTORY</p><h3>Loaded models · {runningModels.count}</h3></div><button type="button" className="button secondary-button" onClick={() => void refreshRunningModels()} disabled={runningBusy}><RefreshCcw size={13} /> Refresh</button></div>
        {runningModels.models.length === 0 ? <p className="quiet-note">No models are currently loaded in Ollama memory. Installed model files are unaffected.</p> : <div className="info-grid">
          {runningModels.models.map((item) => <div className="panel-surface" key={item.name} style={{ padding: 14, minWidth: 0 }}>
            <strong style={{ display: "block", overflowWrap: "anywhere" }}>{item.name}</strong>
            <span className="status-pill success" style={{ marginTop: 8, display: "inline-flex" }}><i />{item.memory_placement === "gpu" ? "GPU/VRAM" : item.memory_placement === "mixed" ? "CPU + GPU" : "CPU/RAM or unreported"}</span>
            <p className="quiet-note" style={{ marginTop: 10 }}>Total loaded size: {item.size_bytes === null ? "unknown" : `${(item.size_bytes / 1_073_741_824).toFixed(2)} GiB`}</p>
            <p className="quiet-note">GPU memory: {item.size_vram_bytes === null ? "not reported" : `${(item.size_vram_bytes / 1_073_741_824).toFixed(2)} GiB`}</p>
            {item.size_ram_bytes !== null && <p className="quiet-note">Approx. non-VRAM size: {(item.size_ram_bytes / 1_073_741_824).toFixed(2)} GiB</p>}
            <p className="quiet-note">{[item.family, item.parameter_size, item.quantization_level].filter(Boolean).join(" · ") || "Model details not reported"}</p>
            {item.expires_at && <p className="quiet-note">Runtime expiry: {item.expires_at}</p>}
            <button type="button" className="button secondary-button" aria-label={`Unload ${item.name} from memory`} onClick={() => void unloadModelByName(item.name)} disabled={!online || unloadBusy} style={{ marginTop: 10 }}><Square size={12} />{unloadingModel === item.name ? "Unloading…" : "Unload from memory"}</button>
          </div>)}
        </div>}
        <p className="quiet-note">{runningModels.scope_note}</p>
      </section>}
      {modelDetails && <section className="panel-surface" aria-label="Local model metadata" style={{ marginTop: 12, padding: 16 }}>
        <div className="surface-head"><div><p className="surface-eyebrow">LOCAL MODEL METADATA</p><h3>{modelDetails.model}</h3></div><span className="status-pill success"><i />INSPECTED</span></div>
        <div className="info-grid">
          {[
            { label: "Format", value: modelDetails.format || "Unknown" },
            { label: "Family", value: modelDetails.family || modelDetails.families.join(", ") || "Unknown" },
            { label: "Parameter size", value: modelDetails.parameter_size || (modelDetails.parameter_count ? modelDetails.parameter_count.toLocaleString() : "Unknown") },
            { label: "Quantization", value: modelDetails.quantization_level || "Not reported" },
            { label: "Context length", value: modelDetails.context_length?.toLocaleString() || "Not reported" },
            { label: "Disk size", value: modelDetails.size_bytes === null ? "Unknown" : `${(modelDetails.size_bytes / 1_073_741_824).toFixed(2)} GiB` },
          ].map((item) => <div className="panel-surface" key={item.label} style={{ padding: 12, minWidth: 0 }}><p className="surface-eyebrow">{item.label}</p><strong style={{ display: "block", marginTop: 6, overflowWrap: "anywhere" }}>{item.value}</strong></div>)}
        </div>
        <p className="quiet-note" style={{ marginTop: 12 }}>Capabilities: {modelDetails.capabilities.length ? modelDetails.capabilities.join(", ") : "Not reported"}. {modelDetails.template_configured ? "A model template is configured." : "No template was reported."} {modelDetails.license_present ? "License metadata is present." : "License metadata was not reported."}</p>
        <p className="quiet-note">{modelDetails.scope_note}</p>
      </section>}
      <label className="field">
        <span className="field-label-row"><span>Prompt</span><span>{prompt.length.toLocaleString()} / 12,000 characters</span></span>
        <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} maxLength={12000} rows={6} required placeholder="Ask your local model to do something…" />
      </label>
      <div className="two-column-grid">
        <label className="field"><span className="field-label-row"><span>Maximum generated tokens</span></span><input type="number" min={1} max={2048} step={1} value={numPredict} onChange={(event) => setNumPredict(event.target.value)} required /></label>
        <label className="field"><span className="field-label-row"><span>Temperature</span></span><input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(event) => setTemperature(event.target.value)} required /></label>
      </div>
      <details className="surface" style={{ marginTop: 14, padding: 14, border: "1px solid var(--line-strong)", borderRadius: 2, background: "var(--surface)" }}>
        <summary style={{ cursor: "pointer", fontWeight: 600 }}>Advanced sampling options</summary>
        <p className="quiet-note" style={{ marginTop: 10 }}>These controls are sent to Ollama as local runtime options. Leave seed and context size blank to use the model/runtime defaults.</p>
        <div className="two-column-grid">
          <label className="field"><span className="field-label-row"><span>Top-p</span></span><input type="number" min={0.001} max={1} step="any" value={topP} onChange={(event) => setTopP(event.target.value)} /></label>
          <label className="field"><span className="field-label-row"><span>Top-k</span></span><input type="number" min={0} max={100000} step={1} value={topK} onChange={(event) => setTopK(event.target.value)} /></label>
          <label className="field"><span className="field-label-row"><span>Min-p relative probability</span></span><input aria-label="Min-p" type="number" min={0} max={1} step={0.01} value={minP} onChange={(event) => setMinP(event.target.value)} /><small>0 disables min-p; otherwise keep tokens close enough to the highest-probability token.</small></label>
          <label className="field"><span className="field-label-row"><span>Repetition penalty</span></span><input type="number" min={0.1} max={5} step={0.1} value={repetitionPenalty} onChange={(event) => setRepetitionPenalty(event.target.value)} /></label>
          <label className="field"><span className="field-label-row"><span>Random seed (optional)</span></span><input type="number" min={0} step={1} value={seedValue} onChange={(event) => setSeedValue(event.target.value)} placeholder="Use runtime default" /></label>
          <label className="field"><span className="field-label-row"><span>Context window (optional)</span></span><input type="number" min={512} max={131072} step={256} value={contextWindow} onChange={(event) => setContextWindow(event.target.value)} placeholder="Use model default" /></label>
        </div>
        <label className="field"><span className="field-label-row"><span>Stop sequences (one per line, optional)</span><span>{stopSequences.split(/\r?\n/).filter((value) => value.trim()).length} / 16</span></span><textarea value={stopSequences} onChange={(event) => setStopSequences(event.target.value)} rows={3} placeholder="Example: END_OF_RESPONSE\n<|end|>" /></label>
      </details>
      <div className="integration-form-actions">
        <button type="submit" className="button primary-button" disabled={!online || loading || running || !model.trim() || !prompt.trim()}><Play size={15} /> {running ? "Generating locally…" : "Run locally"}</button>
        {running && <button type="button" className="button secondary-button" onClick={() => streamController?.abort()} disabled={!streamController}><Square size={13} /> Stop generation</button>}
        {result && <span className="quiet-note"><Activity size={14} /> {result.eval_count ?? "—"} generated tokens{result.total_duration_ms !== null ? ` · ${result.total_duration_ms} ms` : ""}</span>}
      </div>
    </form>
    {error && <div className="inline-alert warning" role="alert">{error}</div>}
    {result && <section className="local-inference-result" aria-label="Generated response">
      <div className="surface-head"><div><p className="surface-eyebrow">GENERATED ON THIS MACHINE</p><h3>{result.model}</h3><small className="quiet-note">Request ID: {result.request_id || "unavailable"}</small></div><span className={`status-pill ${running ? "warning" : result.done ? "success" : "warning"}`}><i />{running ? "GENERATING" : result.done ? "COMPLETED" : "STOPPED"}</span></div>
      <pre data-testid="local-inference-output" className="local-inference-output">{result.response || "(The model returned an empty response.)"}</pre>
    </section>}
    <section className="surface" aria-labelledby="inference-benchmark-title" style={{ marginTop: 18, padding: 18, border: "1px solid var(--line-strong)", borderRadius: 2, background: "var(--surface)" }}>
      <div className="surface-head">
        <div><p className="surface-eyebrow">MEASURED RUNS · LOCAL OLLAMA</p><h3 id="inference-benchmark-title"><BarChart3 size={17} style={{ verticalAlign: "middle", marginRight: 8 }} />Inference benchmark</h3></div>
        <button type="button" className="button primary-button" onClick={() => void benchmarkLocalModel()} disabled={!online || loading || running || benchmarkBusy || !model.trim() || !prompt.trim()}>
          <Activity size={15} /> {benchmarkBusy ? "Measuring local runtime…" : "Benchmark this prompt"}
        </button>
      </div>
      <p className="quiet-note">Uses your selected local model, prompt, output-token limit and temperature. Warm-ups are excluded from reported timings. No result text is returned or saved; runs are bounded to 10 measured generations.</p>
      <div className="two-column-grid">
        <label className="field"><span className="field-label-row"><span>Warm-up generations</span></span><select aria-label="Benchmark warm-up generations" value={warmupRuns} onChange={(event) => setWarmupRuns(event.target.value)} disabled={benchmarkBusy}><option value="0">0</option><option value="1">1</option><option value="2">2</option></select></label>
        <label className="field"><span className="field-label-row"><span>Measured iterations</span></span><select aria-label="Benchmark iterations" value={benchmarkIterations} onChange={(event) => setBenchmarkIterations(event.target.value)} disabled={benchmarkBusy}>{[1,2,3,4,5,6,7,8,9,10].map((count) => <option key={count} value={count}>{count}</option>)}</select></label>
        <label className="field"><span className="field-label-row"><span>Concurrency</span></span><select aria-label="Benchmark concurrency" value={concurrency} onChange={(event) => setConcurrency(event.target.value)} disabled={benchmarkBusy}>{[1,2,3,4].map((count) => <option key={count} value={count}>{count} simultaneous request{count === 1 ? "" : "s"}</option>)}</select></label>
      </div>
      {Number(benchmarkIterations) * Number(concurrency) > 10 && <div className="inline-alert warning" role="status">Reduce iterations or concurrency: the benchmark permits at most 10 measured requests per run.</div>}
      {benchmarkError && <div className="inline-alert warning" role="alert">{benchmarkError}</div>}
      {benchmark && <div className="local-inference-benchmark-result" role="region" aria-label="Inference benchmark result" style={{ marginTop: 18 }}>
        <div className="status-strip" aria-label="Benchmark status">
          <div className="status-strip-item"><span className="status-pill success"><i />MEASURED</span><div><b>{benchmark.model}</b><span>{benchmark.configuration.measured_requests} requests · concurrency {benchmark.configuration.concurrency} · {benchmark.configuration.num_predict} max tokens</span><span>Request ID: {benchmark.request_id}</span></div></div>
        </div>
        <div className="info-grid">
          {[
            { label: "Completion throughput", value: benchmark.measurements.completion_tokens_per_second === null ? "Unavailable" : `${benchmark.measurements.completion_tokens_per_second} tokens/s`, detail: `${benchmark.measurements.total_completion_tokens ?? "Unknown"} completion tokens measured` },
            { label: "Median request latency", value: benchmark.measurements.latency_ms.median === null ? "Unavailable" : `${benchmark.measurements.latency_ms.median} ms`, detail: `P95 ${benchmark.measurements.latency_ms.p95 ?? "—"} ms` },
            { label: "Elapsed wall time", value: `${benchmark.measurements.wall_time_ms} ms`, detail: `${benchmark.measurements.requests_per_second} completed requests/s` },
            { label: "Token-count coverage", value: `${benchmark.measurements.completion_tokens_known}/${benchmark.configuration.measured_requests}`, detail: "Responses with runtime token counts" },
          ].map((metric) => <div className="panel-surface" key={metric.label} style={{ padding: 14, minWidth: 0 }}><p className="surface-eyebrow">{metric.label}</p><strong style={{ display: "block", fontSize: 21, margin: "8px 0", overflowWrap: "anywhere" }}>{metric.value}</strong><p className="quiet-note">{metric.detail}</p></div>)}
        </div>
        <div style={{ overflowX: "auto", marginTop: 14 }}>
          <table className="data-table" aria-label="Per-request benchmark timings">
            <thead><tr><th>Request</th><th>Latency</th><th>Prompt tokens</th><th>Completion tokens</th><th>Finished</th></tr></thead>
            <tbody>{benchmark.measurements.requests.map((request, index) => <tr key={index}><td>{index + 1}</td><td>{request.latency_ms} ms</td><td>{request.prompt_tokens ?? "—"}</td><td>{request.completion_tokens ?? "—"}</td><td>{request.done ? "Yes" : "No"}</td></tr>)}</tbody>
          </table>
        </div>
        <p className="quiet-note" style={{ marginTop: 12 }}>{benchmark.scope_note}</p>
      </div>}
    </section>
  </section>;
}
