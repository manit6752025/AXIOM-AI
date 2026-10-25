import { useEffect, useState } from "react";
import { ArrowDownToLine, BookOpen, Cpu, ExternalLink, FileCog, Globe, HardDrive, Search, Sparkles } from "lucide-react";
import { axiomApi, type ModelStudioFile, type ModelStudioJob, type ModelStudioModel, type ModelStudioStatus } from "../api/client";

const activeStates = new Set(["queued", "running"]);
const tasks = [
  { value: "general", label: "General assistant", query: "instruct GGUF" },
  { value: "coding", label: "Coding", query: "coder GGUF" },
  { value: "reasoning", label: "Reasoning", query: "reasoning GGUF" },
  { value: "multilingual", label: "Multilingual", query: "multilingual instruct GGUF" },
  { value: "low-memory", label: "Small / low memory", query: "1B instruct GGUF Q4_K_M" },
];

function formatBytes(value: number | null | undefined): string {
  if (typeof value !== "number" || value <= 0) return "Size not supplied";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = value, unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
  return `${amount.toFixed(unit >= 3 ? 2 : 1)} ${units[unit]}`;
}

export default function ModelStudioPanel() {
  const [status, setStatus] = useState<ModelStudioStatus | null>(null);
  const [task, setTask] = useState("general");
  const [ramBand, setRamBand] = useState("16 GB");
  const [query, setQuery] = useState("instruct GGUF");
  const [results, setResults] = useState<ModelStudioModel[]>([]);
  const [repo, setRepo] = useState("");
  const [files, setFiles] = useState<ModelStudioFile[]>([]);
  const [filename, setFilename] = useState("");
  const [modelName, setModelName] = useState("axiom-local-model");
  const [registerOllama, setRegisterOllama] = useState(true);
  const [installNative, setInstallNative] = useState(false);
  const [ollamaTag, setOllamaTag] = useState("");
  const [nativeRepo, setNativeRepo] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [sourceDir, setSourceDir] = useState("");
  const [llamaCppDir, setLlamaCppDir] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [quantization, setQuantization] = useState("Q4_K_M");
  const [job, setJob] = useState<ModelStudioJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");
  const [remoteBaseUrl, setRemoteBaseUrl] = useState("");
  const [remoteApiKey, setRemoteApiKey] = useState("");
  const [remoteModels, setRemoteModels] = useState<string[]>([]);
  const [remoteModel, setRemoteModel] = useState("");
  const [remotePrompt, setRemotePrompt] = useState("Explain what GGUF quantization does in simple terms.");
  const [remoteResponse, setRemoteResponse] = useState("");
  const [remoteAcknowledged, setRemoteAcknowledged] = useState(false);
  const [remoteBusy, setRemoteBusy] = useState(false);
  const [remoteError, setRemoteError] = useState("");

  const refreshStatus = async () => {
    try {
      const result = await axiomApi.modelStudioStatus();
      setStatus(result);
      if (!job) {
        const recent = await axiomApi.modelStudioJobs(10);
        if (recent.jobs.length) setJob(recent.jobs[0]);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Model Studio status is unavailable.");
    }
  };
  useEffect(() => { void refreshStatus(); }, []);
  useEffect(() => {
    if (!job || !activeStates.has(job.status)) return;
    const timer = window.setInterval(() => {
      axiomApi.modelStudioJob(job.id).then(setJob).catch((cause: unknown) => setError(cause instanceof Error ? cause.message : "Could not refresh model job status."));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  const selectTask = (value: string) => {
    setTask(value);
    setQuery(tasks.find((item) => item.value === value)?.query || "instruct GGUF");
    setResults([]); setRepo(""); setFiles([]); setFilename(""); setError("");
  };
  const search = async () => {
    setSearching(true); setError(""); setRepo(""); setFiles([]); setFilename("");
    try { setResults((await axiomApi.modelStudioSearch(query.trim(), 12)).results); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "GGUF catalogue search failed."); }
    finally { setSearching(false); }
  };
  const browseFiles = async (model: ModelStudioModel) => {
    setRepo(model.id); setFiles([]); setFilename(""); setError("");
    setModelName(`axiom-${model.id.split("/").at(-1)?.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").slice(0, 48) || "local-model"}`);
    try {
      const inventory = await axiomApi.modelStudioFiles(model.id);
      setFiles(inventory.files);
      const preferredOrder = ramBand === "4 GB"
        ? ["Q2_K", "Q3_K_M", "Q4_K_M", "Q4_K"]
        : ramBand === "8 GB"
          ? ["Q4_K_M", "Q4_K", "Q3_K_M", "Q5_K_M"]
          : ["32 GB", "64 GB+"].includes(ramBand)
          ? ["Q5_K_M", "Q6_K", "Q4_K_M", "Q8_0", "Q4_K"]
          : ["Q4_K_M", "Q4_K", "Q5_K_M", "Q3_K_M", "Q2_K"];
      const preferred = preferredOrder.map((quant) => inventory.files.find((item) => item.quantization === quant)).find(Boolean) || inventory.files[0];
      if (preferred) setFilename(preferred.filename);
      if (!inventory.files.length) setError("This repository did not expose a single-file GGUF in its file list.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not read GGUF files for this repository."); }
  };
  const startJob = async (config: Record<string, unknown>) => {
    setBusy(true); setError("");
    try { const started = await axiomApi.startModelStudioJob(config as never); setJob(started); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Model Studio job could not start."); }
    finally { setBusy(false); }
  };
  const testRemoteConnection = async () => {
    setRemoteBusy(true); setRemoteError(""); setRemoteResponse("");
    try {
      const result = await axiomApi.testRemoteModelConnection({ base_url: remoteBaseUrl.trim(), api_key: remoteApiKey });
      setRemoteModels(result.models);
      if (!remoteModel && result.models.length) setRemoteModel(result.models[0]);
      if (!result.models.length) setRemoteError("The endpoint connected, but its /models response did not contain model IDs. You can enter the provider's model ID manually.");
    } catch (cause) { setRemoteError(cause instanceof Error ? cause.message : "Remote endpoint connection failed."); setRemoteModels([]); }
    finally { setRemoteBusy(false); }
  };
  const sendRemotePrompt = async () => {
    setRemoteBusy(true); setRemoteError(""); setRemoteResponse("");
    try {
      const result = await axiomApi.remoteModelChat({ base_url: remoteBaseUrl.trim(), api_key: remoteApiKey, model: remoteModel.trim(), prompt: remotePrompt, max_tokens: 256, temperature: 0.7 });
      setRemoteResponse(result.response);
    } catch (cause) { setRemoteError(cause instanceof Error ? cause.message : "Remote model request failed."); }
    finally { setRemoteBusy(false); }
  };
  const activeJob = Boolean(job && activeStates.has(job.status));
  const savedPath = typeof job?.result?.path === "string" ? job.result.path : "";
  const savedModelName = typeof job?.result?.model_name === "string" ? job.result.model_name : "";
  const isRegisteredWithOllama = job?.result?.registered_with_ollama === true;

  return <section className="surface model-studio-panel" aria-label="Model Studio">
    <div className="surface-head"><div><p className="surface-eyebrow">MODEL CATALOGUE · LOCAL INSTALLATION</p><h2>Model Studio</h2></div><button type="button" className="button secondary-button" onClick={() => void refreshStatus()}><Sparkles size={14} /> Refresh status</button></div>
    <p>Search and install GGUF models, pull compatible public Hugging Face safetensors checkpoints directly into AXIOM Native Engine, pull an Ollama library tag, or convert local Transformers weights into GGUF. Native pulls validate the architecture and file inventory before downloading weights.</p>
    <div className="model-studio-status-grid">
      <div><span><Cpu size={15} /> Runtime</span><b>{status?.ollama_installed ? "Ollama found" : "Ollama not found"}</b><small>{status?.ollama_endpoint || "Checking local runtime"}</small></div>
      <div><span><HardDrive size={15} /> Download limit</span><b>20 GiB per file</b><small>Free-space checked before download when size metadata is available.</small></div>
      <div><span><Globe size={15} /> Model sources</span><b>Hugging Face + Ollama</b><small>GGUF search and native safetensors pulls are explicit actions.</small></div>
    </div>
    <div className="two-column-grid model-studio-layout">
      <div className="model-studio-column">
        <div className="model-studio-section-head"><span className="model-studio-step">01</span><div><h3>Recommendation questionnaire</h3><p>Choose what you want to do; use the result list to inspect model cards before installation.</p></div></div>
        <label className="field"><span className="field-label-row"><span>Primary use</span></span><select aria-label="Primary use" value={task} onChange={(event) => selectTask(event.target.value)}>{tasks.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        <label className="field"><span className="field-label-row"><span>Available system RAM</span></span><select aria-label="Available system RAM" value={ramBand} onChange={(event) => setRamBand(event.target.value)}><option>4 GB</option><option>8 GB</option><option>16 GB</option><option>32 GB</option><option>64 GB+</option></select><small>This guides the preferred quantization: small-memory systems prioritize smaller Q2/Q3/Q4 files, while larger systems can prefer Q5/Q6. It is not a guaranteed fit estimate.</small></label>
        <label className="field"><span className="field-label-row"><span>Search query</span></span><input aria-label="Model search" value={query} onChange={(event) => setQuery(event.target.value)} maxLength={120} placeholder="e.g. Qwen instruct GGUF" /></label>
        <button type="button" className="button primary-button" onClick={() => void search()} disabled={searching || !query.trim()}><Search size={15} /> {searching ? "Searching public catalogue…" : "Search GGUF catalogue"}</button>
        {results.length > 0 && <div className="model-studio-results" aria-label="Model search results">{results.map((model) => <article className="model-studio-result" key={model.id}>
          <div className="model-studio-result-copy"><b>{model.id}</b><small>{model.downloads.toLocaleString()} downloads · {model.likes.toLocaleString()} likes{model.last_modified ? ` · updated ${model.last_modified.slice(0, 10)}` : ""}</small><small>{model.pipeline_tag}</small></div>
          <div className="model-studio-result-actions"><a href={model.url} target="_blank" rel="noreferrer"><BookOpen size={13} /> Model card <ExternalLink size={12} /></a><button type="button" className="button secondary-button small-button" onClick={() => void browseFiles(model)}>{repo === model.id && files.length ? "Files loaded" : "Choose GGUF file"}</button></div>
        </article>)}</div>}
        {results.length === 0 && !searching && <p className="quiet-note">Try “Qwen GGUF”, “coder GGUF”, or choose a different use case above.</p>}
      </div>
      <div className="model-studio-column">
        <div className="model-studio-section-head"><span className="model-studio-step">02</span><div><h3>Install a selected GGUF</h3><p>Download a public GGUF into AXIOM's native model root or register it with local Ollama. AXIOM-native inference dequantizes GGUF tensors to floating-point; source Q4/Q5/Q8 quantization does not mean low-bit execution or low-bit memory usage.</p></div></div>
        {repo ? <div className="model-studio-selected"><b>{repo}</b><label className="field"><span className="field-label-row"><span>GGUF file / quantization</span></span><select aria-label="GGUF file" value={filename} onChange={(event) => setFilename(event.target.value)}>{files.map((file) => <option key={file.filename} value={file.filename}>{file.filename} · {file.quantization} · {formatBytes(file.size_bytes)}</option>)}</select></label><label className="field"><span className="field-label-row"><span>Local model name</span></span><input aria-label="Local model name" value={modelName} onChange={(event) => setModelName(event.target.value)} maxLength={120} /></label><label className="model-studio-checkbox"><input type="checkbox" checked={registerOllama} onChange={(event) => { const checked = event.target.checked; setRegisterOllama(checked); if (checked) setInstallNative(false); }} /><span><b>Register with Ollama after download</b><small>Creates a local Ollama model from the GGUF file so Ollama commands can use it.</small></span></label><label className="model-studio-checkbox"><input type="checkbox" aria-label="Install directly into AXIOM Native Engine" checked={installNative} onChange={(event) => { const checked = event.target.checked; setInstallNative(checked); if (checked) setRegisterOllama(false); }} /><span><b>Install directly into AXIOM Native Engine</b><small>Places the GGUF in AXIOM's trusted native-model root so it appears in the Runtime panel without Ollama. AXIOM validates architecture/tokenizer support when loading.</small></span></label><button type="button" className="button primary-button" disabled={busy || activeJob || !filename || !modelName.trim()} onClick={() => void startJob({operation:"install_gguf",repo_id:repo,filename,model_name:modelName.trim(),register_ollama:registerOllama && !installNative,install_native:installNative})}><ArrowDownToLine size={15} /> {installNative ? "Install for AXIOM Native Engine" : registerOllama ? "Download GGUF and register" : "Download GGUF file"}</button></div> : <div className="model-studio-placeholder"><HardDrive size={20} /><b>No GGUF selected yet</b><span>Search the catalogue and select “Choose GGUF file” to see available quantizations and file sizes.</span></div>}
        <div className="model-studio-divider" />
        <div className="model-studio-section-head"><span className="model-studio-step">03</span><div><h3>Pull from Ollama Library</h3><p>Use a model tag from the <a href="https://ollama.com/search" target="_blank" rel="noreferrer">official Ollama model library <ExternalLink size={12} /></a>.</p></div></div>
        <label className="field"><span className="field-label-row"><span>Ollama model tag</span></span><input aria-label="Ollama model tag" value={ollamaTag} onChange={(event) => setOllamaTag(event.target.value)} placeholder="e.g. qwen3.5:0.8b" maxLength={120} /></label>
        <button type="button" className="button secondary-button" disabled={busy || activeJob || !ollamaTag.trim()} onClick={() => void startJob({operation:"pull",model_name:ollamaTag.trim()})}><ArrowDownToLine size={15} /> Pull Ollama model</button>
        <div className="model-studio-divider" />
        <div className="model-studio-section-head"><span className="model-studio-step">04</span><div><h3>Pull a model for AXIOM Native Engine</h3><p>Downloads compatible Llama, Mistral, or Qwen2 text checkpoints in safetensors format. AXIOM validates config.json and the Hub file inventory before downloading weights; model Python code is never fetched or executed.</p></div></div>
        <label className="field"><span className="field-label-row"><span>Public Hugging Face repository ID</span></span><input aria-label="Native Hugging Face repository ID" value={nativeRepo} onChange={(event) => setNativeRepo(event.target.value)} maxLength={192} placeholder="Qwen/Qwen2.5-0.5B-Instruct" /><small>Example: Qwen/Qwen2.5-0.5B-Instruct. Gated/private repos and quantized GGUF/EXL2/GPTQ/AWQ files are not accepted by this native path.</small></label>
        <button type="button" className="button primary-button" disabled={busy || activeJob || !nativeRepo.trim()} onClick={() => void startJob({operation:"pull_native",repo_id:nativeRepo.trim()})}><ArrowDownToLine size={15} /> Pull native checkpoint</button>
      </div>
    </div>
    <div className="model-studio-converter">
      <div className="model-studio-section-head"><span className="model-studio-step">05</span><div><h3>Create a GGUF from local weights</h3><p>Conversion requires an existing Transformers model directory and a local llama.cpp checkout. This does not download source weights.</p></div></div>
      <div className="two-column-grid"><label className="field"><span className="field-label-row"><span>Local Transformers model directory</span></span><input aria-label="Source Transformers model directory" value={sourceDir} onChange={(event) => setSourceDir(event.target.value)} placeholder="/home/user/models/base-model" /></label><label className="field"><span className="field-label-row"><span>Local llama.cpp directory</span></span><input aria-label="llama.cpp directory" value={llamaCppDir} onChange={(event) => setLlamaCppDir(event.target.value)} placeholder="/home/user/src/llama.cpp" /></label><label className="field"><span className="field-label-row"><span>New output .gguf path</span></span><input aria-label="GGUF output path" value={outputPath} onChange={(event) => setOutputPath(event.target.value)} placeholder="/home/user/models/model-Q4_K_M.gguf" /></label><label className="field"><span className="field-label-row"><span>Quantization</span></span><select aria-label="GGUF quantization" value={quantization} onChange={(event) => setQuantization(event.target.value)}><option value="Q4_K_M">Q4_K_M · balanced size/quality</option><option value="Q5_K_M">Q5_K_M · higher quality, larger</option><option value="Q6_K">Q6_K · larger, higher fidelity</option><option value="Q8_0">Q8_0 · near high precision, large</option><option value="Q3_K_M">Q3_K_M · smaller, lower fidelity</option><option value="Q2_K">Q2_K · smallest, quality trade-off</option><option value="none">No quantization · F16 GGUF</option></select></label></div>
      <button type="button" className="button secondary-button" disabled={busy || activeJob || !sourceDir.trim() || !llamaCppDir.trim() || !outputPath.trim()} onClick={() => void startJob({operation:"convert_gguf",source_dir:sourceDir.trim(),llama_cpp_dir:llamaCppDir.trim(),output_path:outputPath.trim(),outtype:"f16",quantization})}><FileCog size={15} /> Convert and quantize GGUF</button>
    </div>
    {error && <div className="inline-alert warning" role="alert">{error}</div>}
    {job && <section className="model-studio-job" aria-label="Model Studio job status" role="region"><div className="surface-head"><div><p className="surface-eyebrow">JOB {job.id.slice(0, 12)}</p><h3>{job.operation.replaceAll("_", " ")}</h3></div><span className={`status-pill ${job.status === "completed" ? "success" : job.status === "failed" ? "warning" : "neutral"}`}><i />{job.status.toUpperCase()}</span></div>
      {job.total_bytes && job.total_bytes > 0 && <div className="model-studio-progress"><span style={{ width: `${Math.min(100, (job.progress_bytes / job.total_bytes) * 100)}%` }} /></div>}
      {job.error && <div className="inline-alert warning" role="alert">{job.error}</div>}{savedPath && <p className="quiet-note"><HardDrive size={14} /> Saved locally: <code>{savedPath}</code></p>}{savedModelName && isRegisteredWithOllama && <p className="quiet-note">Use it for local inference: <code>axiom infer run {savedModelName} --prompt "Hello"</code></p>}
      <pre className="model-studio-log" data-testid="model-studio-log">{job.logs.join("\n")}</pre>
    </section>}
    <div className="model-studio-remote">
      <div className="model-studio-section-head"><span className="model-studio-step">06</span><div><h3>Connect to an online model provider</h3><p>For OpenAI-compatible public HTTPS endpoints (for example a hosted provider or gateway). This is separate from local inference.</p></div></div>
      <div className="two-column-grid"><label className="field"><span className="field-label-row"><span>Provider API base URL</span></span><input type="url" aria-label="Remote provider base URL" value={remoteBaseUrl} onChange={(event) => { setRemoteBaseUrl(event.target.value); setRemoteModels([]); }} placeholder="https://provider.example/api/v1" /><small>Use the provider's documented HTTPS base URL ending in /v1.</small></label><label className="field"><span className="field-label-row"><span>Provider API key</span></span><input type="password" aria-label="Remote provider API key" value={remoteApiKey} onChange={(event) => { setRemoteApiKey(event.target.value); setRemoteModels([]); }} autoComplete="off" placeholder="Paste key for this session only" /><small>Held in this page's memory only. AXIOM does not save it in the workspace or local storage.</small></label></div>
      <div className="integration-form-actions"><button type="button" className="button secondary-button" disabled={remoteBusy || !remoteBaseUrl.trim() || !remoteApiKey.trim()} onClick={() => void testRemoteConnection()}><Globe size={15} /> {remoteBusy ? "Connecting…" : "Test connection and list models"}</button></div>
      {remoteModels.length > 0 && <><label className="field"><span className="field-label-row"><span>Remote model ID</span></span><select aria-label="Remote model ID" value={remoteModel} onChange={(event) => setRemoteModel(event.target.value)}>{remoteModels.map((id) => <option key={id} value={id}>{id}</option>)}</select></label></>}
      {remoteModels.length === 0 && <label className="field"><span className="field-label-row"><span>Remote model ID (optional/manual)</span></span><input aria-label="Manual remote model ID" value={remoteModel} onChange={(event) => setRemoteModel(event.target.value)} maxLength={256} placeholder="Model ID from your provider" /></label>}
      <label className="field"><span className="field-label-row"><span>Prompt to send</span></span><textarea aria-label="Remote model prompt" value={remotePrompt} onChange={(event) => setRemotePrompt(event.target.value)} rows={3} maxLength={12000} /></label>
      <label className="model-studio-checkbox"><input type="checkbox" checked={remoteAcknowledged} onChange={(event) => setRemoteAcknowledged(event.target.checked)} /><span><b>I understand this prompt will leave my computer</b><small>The selected remote provider receives the prompt and may process it according to its own terms/privacy policy. Do not include secrets or private training data.</small></span></label>
      <button type="button" className="button primary-button" disabled={remoteBusy || !remoteAcknowledged || !remoteBaseUrl.trim() || !remoteApiKey.trim() || !remoteModel.trim() || !remotePrompt.trim()} onClick={() => void sendRemotePrompt()}><Globe size={15} /> Send prompt to online model</button>
      {remoteError && <div className="inline-alert warning" role="alert">{remoteError}</div>}
      {remoteResponse && <div className="model-studio-remote-response"><b>Remote response</b><pre>{remoteResponse}</pre><small>Response was returned to this page; AXIOM did not save it as a local model or add it to training history.</small></div>}
    </div>
    <p className="quiet-note">External access is used only after explicit search/download or remote-model actions. Local inference remains loopback-only. Remote prompts are sent only after the explicit consent checkbox and are not persisted by this feature. GGUF downloads are limited to 20 GiB per file. Native safetensors pulls are limited to 20 GiB per file and 160 GiB per checkpoint; always review repository licenses and model cards.</p>
  </section>;
}
