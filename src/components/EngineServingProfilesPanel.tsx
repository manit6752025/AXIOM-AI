import { useEffect, useState } from "react";
import { Activity, Play, Plus, RefreshCcw, ShieldCheck, Trash2 } from "lucide-react";
import {
  axiomApi,
  type AxiomEngineServingProfile,
  type AxiomEngineServingReadiness,
} from "../api/client";

export default function EngineServingProfilesPanel() {
  const [profiles, setProfiles] = useState<AxiomEngineServingProfile[]>([]);
  const [models, setModels] = useState<Array<{ path: string; name: string; model_type: string }>>([]);
  const [adapters, setAdapters] = useState<Array<{ path: string; name: string }>>([]);
  const [name, setName] = useState("Local serving");
  const [modelPath, setModelPath] = useState("");
  const [adapterPath, setAdapterPath] = useState("");
  const [device, setDevice] = useState<AxiomEngineServingProfile["device_preference"]>("auto");
  const [memoryMode, setMemoryMode] = useState<AxiomEngineServingProfile["memory_mode"]>("auto");
  const [precision, setPrecision] = useState<AxiomEngineServingProfile["precision"]>("auto");
  const [contextLength, setContextLength] = useState("4096");
  const [maxTokens, setMaxTokens] = useState("256");
  const [timeout, setTimeout] = useState("300");
  const [maxQueuedRequests, setMaxQueuedRequests] = useState("2");
  const [prompt, setPrompt] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [readiness, setReadiness] = useState<Record<string, AxiomEngineServingReadiness>>({});
  const [output, setOutput] = useState("");
  const [admission, setAdmission] = useState<{
    queue_position: number;
    queued: boolean;
    waited_ms: number;
    concurrency_limit: 1;
    queue_limit: number;
  } | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const refresh = async () => {
    setBusy("refresh");
    setError("");
    try {
      const [profileReport, assets] = await Promise.all([
        axiomApi.axiomEngineServingProfiles(),
        axiomApi.axiomEngineModels(),
      ]);
      setProfiles(profileReport.profiles);
      setModels(assets.models);
      setAdapters(assets.adapters);
      setModelPath((current) => current || assets.models[0]?.path || "");
      setSelectedId((current) => current || profileReport.profiles[0]?.id || "");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Serving profile inventory is unavailable.");
    } finally {
      setBusy("");
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const selectedProfile = profiles.find((profile) => profile.id === selectedId) ?? null;

  const createProfile = async () => {
    if (!name.trim() || !modelPath) return;
    setBusy("create");
    setError("");
    setOutput("");
    try {
      const result = await axiomApi.createAxiomEngineServingProfile({
        name: name.trim(),
        model_path: modelPath,
        adapter_path: adapterPath || null,
        device_preference: device,
        memory_mode: memoryMode,
        precision,
        context_length: Number(contextLength),
        max_new_tokens: Number(maxTokens),
        max_prompt_chars: 12000,
        max_concurrent_requests: 1,
        max_queued_requests: Number(maxQueuedRequests),
        request_timeout_seconds: Number(timeout),
      });
      setSelectedId(result.profile.id);
      setReadiness((current) => {
        const next = { ...current };
        delete next[result.profile.id];
        return next;
      });
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Serving profile could not be saved.");
    } finally {
      setBusy("");
    }
  };

  const probe = async (profile: AxiomEngineServingProfile) => {
    setBusy("probe:" + profile.id);
    setError("");
    try {
      const report = await axiomApi.probeAxiomEngineServingProfile(profile.id);
      setReadiness((current) => ({ ...current, [profile.id]: report }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Serving profile readiness probe failed.");
    } finally {
      setBusy("");
    }
  };

  const generate = async () => {
    if (!selectedProfile || !prompt.trim()) return;
    setBusy("generate");
    setError("");
    setOutput("");
    setAdmission(null);
    try {
      const report = await axiomApi.generateAxiomEngineServingProfile(selectedProfile.id, { prompt });
      setOutput(report.result.response);
      setAdmission(report.admission);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Profile generation failed.");
    } finally {
      setBusy("");
    }
  };

  const remove = async (profile: AxiomEngineServingProfile) => {
    setBusy("delete:" + profile.id);
    setError("");
    try {
      await axiomApi.deleteAxiomEngineServingProfile(profile.id);
      setProfiles((current) => current.filter((item) => item.id !== profile.id));
      setReadiness((current) => {
        const next = { ...current };
        delete next[profile.id];
        return next;
      });
      if (selectedId === profile.id) setSelectedId("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Serving profile could not be deleted.");
    } finally {
      setBusy("");
    }
  };

  return (
    <section className="surface" aria-label="Native model serving profiles">
      <div className="surface-head">
        <div>
          <p className="surface-eyebrow"><Activity size={14} /> NATIVE SERVING / LOCAL PROFILES</p>
          <h3>Serving profiles</h3>
          <p className="quiet-note">Save model, context, device and request limits once. Readiness probes inspect the checkpoint without loading weights; the first profile request loads the model locally.</p>
        </div>
        <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={busy !== ""}>
          <RefreshCcw size={15} /> Refresh
        </button>
      </div>
      {error && <p className="inline-alert warning" role="alert">{error}</p>}
      <div className="two-column-grid">
        <div>
          <label className="field">
            <span className="field-label-row"><span>Profile name</span></span>
            <input aria-label="Serving profile name" value={name} onChange={(event) => setName(event.target.value)} maxLength={48} />
          </label>
          <label className="field">
            <span className="field-label-row"><span>Local model checkpoint</span></span>
            <select aria-label="Serving model checkpoint" value={modelPath} onChange={(event) => setModelPath(event.target.value)}>
              <option value="">Select a discovered local model</option>
              {models.map((model) => <option value={model.path} key={model.path}>{model.name} · {model.model_type}</option>)}
            </select>
          </label>
          <label className="field">
            <span className="field-label-row"><span>LoRA adapter (optional)</span></span>
            <select aria-label="Serving LoRA adapter" value={adapterPath} onChange={(event) => setAdapterPath(event.target.value)}>
              <option value="">No adapter</option>
              {adapters.map((adapter) => <option value={adapter.path} key={adapter.path}>{adapter.name}</option>)}
            </select>
          </label>
          <div className="two-column-grid">
            <label className="field">
              <span className="field-label-row"><span>Execution device</span></span>
              <select aria-label="Serving device" value={device} onChange={(event) => setDevice(event.target.value as typeof device)}>
                <option value="auto">Automatic</option><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="rocm">ROCm</option><option value="mps">Metal / MPS</option><option value="xpu">Intel XPU</option>
              </select>
            </label>
            <label className="field">
              <span className="field-label-row"><span>Memory mode</span></span>
              <select aria-label="Serving memory mode" value={memoryMode} onChange={(event) => setMemoryMode(event.target.value as typeof memoryMode)}>
                <option value="auto">Automatic</option><option value="full">Full</option><option value="layer_streaming">Layer streaming</option><option value="disk_streaming">Disk streaming</option>
              </select>
            </label>
          </div>
          <label className="field">
            <span className="field-label-row"><span>Compute precision</span></span>
            <select aria-label="Serving precision" value={precision} onChange={(event) => setPrecision(event.target.value as typeof precision)}>
              <option value="auto">Automatic</option><option value="fp32">FP32</option><option value="fp16">FP16</option><option value="bf16">BF16</option>
            </select>
          </label>
        </div>
        <div>
          <div className="two-column-grid">
            <label className="field"><span className="field-label-row"><span>Context tokens</span></span><input aria-label="Serving context length" type="number" min="1" max="131072" value={contextLength} onChange={(event) => setContextLength(event.target.value)} /></label>
            <label className="field"><span className="field-label-row"><span>Maximum output tokens</span></span><input aria-label="Serving maximum output tokens" type="number" min="1" max="2048" value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} /></label>
          </div>
          <label className="field"><span className="field-label-row"><span>Request timeout (seconds)</span></span><input aria-label="Serving request timeout" type="number" min="5" max="900" value={timeout} onChange={(event) => setTimeout(event.target.value)} /></label>
          <label className="field"><span className="field-label-row"><span>Maximum queued requests</span></span><input aria-label="Serving profile queue depth" type="number" min="0" max="32" step="1" value={maxQueuedRequests} onChange={(event) => setMaxQueuedRequests(event.target.value)} /><small>FIFO queue while the single native worker is busy. Set 0 to reject immediately.</small></label>
          <p className="quiet-note">Native execution remains single-worker. Queue depth is bounded to 32, queue waits count toward the profile timeout, and disconnects remove waiting requests.</p>
          <button type="button" className="button primary-button" onClick={() => void createProfile()} disabled={busy !== "" || !modelPath || !name.trim()}>
            <Plus size={15} /> {busy === "create" ? "Saving profile…" : "Save serving profile"}
          </button>
        </div>
      </div>
      <div className="surface-divider" />
      <div className="surface-head">
        <div><p className="surface-eyebrow">SAVED CONFIGURATIONS</p><h4>{profiles.length} profile(s)</h4></div>
      </div>
      {profiles.length ? <div className="module-list">
        {profiles.map((profile) => {
          const report = readiness[profile.id];
          return <div className="module-row" key={profile.id}>
            <span className="module-icon"><ShieldCheck size={16} /></span>
            <span className="module-copy">
              <b>{profile.name}</b>
              <small>{profile.model_path} · {profile.device_preference} · context {profile.context_length.toLocaleString()} · max output {profile.max_new_tokens}</small>
              {report && <small>{report.ready ? "Ready for a local generation attempt" : report.problems.join(" · ")} · weights loaded: {report.weights_loaded ? "yes" : "no"}</small>}
            </span>
            <span className={"status-pill " + (report ? (report.ready ? "success" : "warning") : "neutral")}><i />{report ? (report.ready ? "READY" : "NOT READY") : "UNPROBED"}</span>
            <button type="button" className="button secondary-button" onClick={() => { setSelectedId(profile.id); void probe(profile); }} disabled={busy !== ""}>{busy === "probe:" + profile.id ? "Probing…" : "Probe"}</button>
            <button type="button" className="icon-button danger-icon-button" aria-label={"Delete serving profile " + profile.name} onClick={() => void remove(profile)} disabled={busy !== ""}><Trash2 size={15} /></button>
          </div>;
        })}
      </div> : <p className="quiet-note">No serving profiles saved. Discover a local model above to create the first profile.</p>}
      <div className="surface-divider" />
      <div className="surface-head"><div><p className="surface-eyebrow">PROFILE-BOUNDED INFERENCE</p><h4>Generate locally</h4></div></div>
      <label className="field">
        <span className="field-label-row"><span>Serving profile</span></span>
        <select aria-label="Profile for generation" value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
          <option value="">Select a saved profile</option>
          {profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.name}</option>)}
        </select>
      </label>
      <label className="field">
        <span className="field-label-row"><span>Prompt</span></span>
        <textarea aria-label="Serving profile prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} maxLength={12000} rows={3} placeholder="Ask the selected local model…" />
      </label>
      <button type="button" className="button primary-button" onClick={() => void generate()} disabled={busy !== "" || !selectedProfile || !prompt.trim()}>
        <Play size={15} /> {busy === "generate" ? "Generating locally…" : "Generate through profile"}
      </button>
      {admission && <p className="quiet-note" role="status">
        {admission.queued
          ? `Admitted from FIFO queue position ${admission.queue_position}`
          : "Admitted immediately"}
        · waited {(admission.waited_ms / 1000).toFixed(2)} s
        · one active worker
        · per-profile queue cap {admission.queue_limit}
      </p>}
      {output && <pre className="local-inference-output" aria-label="Serving profile generated response">{output}</pre>}
      <p className="quiet-note">Profiles persist only configuration. They do not save credentials, custom endpoints, prompts or responses. Readiness is a preflight, not a guarantee of runtime fit or a separate HTTP listener.</p>
    </section>
  );
}
