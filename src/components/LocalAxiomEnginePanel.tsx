import { useEffect, useRef, useState, type FormEvent } from "react";
import { Activity, Cpu, Play, RefreshCcw, Unplug } from "lucide-react";
const formatMemoryBytes = (bytes: number | null | undefined): string => {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return "not exposed";
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(2) + " GiB";
  if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(1) + " MiB";
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KiB";
  return Math.round(bytes) + " B";
};
const formatEstimatedMemory = (gib: number): string => {
  if (!Number.isFinite(gib) || gib < 0) return "n/a";
  if (gib >= 0.01) return `${gib.toFixed(2)} GiB`;
  if (gib >= 0.001) return `${gib.toFixed(4)} GiB`;
  if (gib >= 1 / (1024 * 1024)) return `${(gib * 1024 * 1024).toFixed(2)} KiB`;
  return `${(gib * 1024 ** 3).toFixed(0)} bytes`;
};

import { axiomApi, type AxiomEngineAdapterCompatibility, type AxiomEngineAssets, type AxiomEngineBatchResult, type AxiomEngineBenchmark, type AxiomEngineLiveTelemetry, type AxiomEngineResult, type AxiomEngineStatus, type AxiomEngineStreamEvent, type AxiomEngineSystemTelemetry, type AxiomQuantizationEstimate } from "../api/client";

export default function LocalAxiomEnginePanel() {
  const [status, setStatus] = useState<AxiomEngineStatus | null>(null);
  const [systemTelemetry, setSystemTelemetry] = useState<AxiomEngineSystemTelemetry | null>(null);
  const [assets, setAssets] = useState<AxiomEngineAssets | null>(null);
  const [adapterCompatibility, setAdapterCompatibility] = useState<AxiomEngineAdapterCompatibility | null>(null);
  const [checkingAdapter, setCheckingAdapter] = useState(false);
  const [modelPath, setModelPath] = useState("");
  const [devicePreference, setDevicePreference] = useState<"auto" | "cpu" | "cuda" | "rocm" | "mps" | "xpu">("auto");
  const [memoryMode, setMemoryMode] = useState<"auto" | "full" | "layer_streaming" | "disk_streaming">("auto");
  const [precision, setPrecision] = useState<"auto" | "fp32" | "fp16" | "bf16">("auto");
  const [selectedModelPath, setSelectedModelPath] = useState("");
  const [adapterPath, setAdapterPath] = useState("");
  const [selectedAdapterPath, setSelectedAdapterPath] = useState("");
  const [prompt, setPrompt] = useState("");
  const [batchPromptsText, setBatchPromptsText] = useState("");
  const [batchResult, setBatchResult] = useState<AxiomEngineBatchResult | null>(null);
  const [maxNewTokens, setMaxNewTokens] = useState("256");
  const [temperature, setTemperature] = useState("0.7");
  const [topP, setTopP] = useState("0.9");
  const [topK, setTopK] = useState("0");
  const [minP, setMinP] = useState("0");
  const [repetitionPenalty, setRepetitionPenalty] = useState("1.0");
  const [seed, setSeed] = useState("");
  const [stopSequencesText, setStopSequencesText] = useState("");
  const [result, setResult] = useState<AxiomEngineResult | null>(null);
  const [liveTelemetry, setLiveTelemetry] = useState<AxiomEngineLiveTelemetry | null>(null);
  const [streamResponse, setStreamResponse] = useState("");
  const [streamTokenCount, setStreamTokenCount] = useState(0);
  const activeStreamRef = useRef<AbortController | null>(null);
  const benchmarkAbortRef = useRef<AbortController | null>(null);
  const streamTokenCountRef = useRef(0);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [unloading, setUnloading] = useState(false);
  const [lifecycleNotice, setLifecycleNotice] = useState("");
  const [error, setError] = useState("");
  const [quantContextLength, setQuantContextLength] = useState("4096");
  const [quantBatchSize, setQuantBatchSize] = useState("1");
  const [ramBudgetGiB, setRamBudgetGiB] = useState("16");
  const [vramBudgetGiB, setVramBudgetGiB] = useState("");
  const [quantizationPlan, setQuantizationPlan] = useState<AxiomQuantizationEstimate | null>(null);
  const [planningQuantization, setPlanningQuantization] = useState(false);
  const [quantizationError, setQuantizationError] = useState("");
  const [benchmarkWarmupRuns, setBenchmarkWarmupRuns] = useState("1");
  const [benchmarkIterations, setBenchmarkIterations] = useState("3");
  const [benchmarking, setBenchmarking] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<AxiomEngineBenchmark | null>(null);
  const [benchmarkError, setBenchmarkError] = useState("");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      setStatus(await axiomApi.axiomEngineStatus());
      try {
        setSystemTelemetry(await axiomApi.axiomEngineTelemetry());
      } catch {
        setSystemTelemetry(null);
      }
    } catch (cause) {
      setStatus(null);
      setError(cause instanceof Error ? cause.message : "AXIOM Native Engine status is unavailable.");
    }
    try {
      setAssets(await axiomApi.axiomEngineModels());
    } catch {
      // Manual trusted-root path entry remains available if discovery is unavailable.
      setAssets(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    return () => activeStreamRef.current?.abort();
  }, []);

  const unload = async () => {
    setUnloading(true);
    setError("");
    setLifecycleNotice("");
    try {
      const outcome = await axiomApi.unloadAxiomEngine();
      setResult(null);
      setLifecycleNotice(outcome.unloaded
        ? "AXIOM released its loaded model and tokenizer references. Framework or OS allocators may retain reserved memory."
        : "No AXIOM model was loaded; the engine state is already clear.");
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The local model could not be unloaded.");
    } finally {
      setUnloading(false);
    }
  };

  const checkAdapterCompatibility = async () => {
    if (!modelPath.trim() || !adapterPath.trim()) {
      setError("Select both a native base model and a LoRA adapter before checking compatibility.");
      return;
    }
    setCheckingAdapter(true);
    setError("");
    setAdapterCompatibility(null);
    try {
      setAdapterCompatibility(await axiomApi.checkAxiomEngineAdapterCompatibility({
        model_path: modelPath.trim(),
        adapter_path: adapterPath.trim(),
      }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Adapter compatibility preflight failed.");
    } finally {
      setCheckingAdapter(false);
    }
  };

  const estimateQuantization = async () => {
    setQuantizationError("");
    setQuantizationPlan(null);
    if (!modelPath.trim()) {
      setQuantizationError("Choose a discovered model before estimating memory.");
      return;
    }
    const contextLength = Number(quantContextLength);
    const batchSize = Number(quantBatchSize);
    const ram = ramBudgetGiB.trim() === "" ? null : Number(ramBudgetGiB);
    const vram = vramBudgetGiB.trim() === "" ? null : Number(vramBudgetGiB);
    if (!Number.isInteger(contextLength) || contextLength < 1 || contextLength > 131072) {
      setQuantizationError("Context length must be an integer from 1 through 131072.");
      return;
    }
    if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 16) {
      setQuantizationError("Batch size must be an integer from 1 through 16.");
      return;
    }
    if (ram === null && vram === null) {
      setQuantizationError("Enter a RAM budget, a VRAM budget, or both.");
      return;
    }
    for (const [label, budget] of [["RAM", ram], ["VRAM", vram]] as const) {
      if (budget !== null && (!Number.isFinite(budget) || budget < 0.1 || budget > 2048)) {
        setQuantizationError(label + " budget must be between 0.1 and 2048 GiB.");
        return;
      }
    }
    setPlanningQuantization(true);
    try {
      const plan = await axiomApi.estimateAxiomQuantization({
        model_path: modelPath.trim(),
        context_length: contextLength,
        batch_size: batchSize,
        ram_budget_gib: ram,
        vram_budget_gib: vram,
      });
      setQuantizationPlan(plan);
    } catch (cause) {
      setQuantizationError(cause instanceof Error ? cause.message : "Memory planning failed.");
    } finally {
      setPlanningQuantization(false);
    }
  };

  const run = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setLifecycleNotice("");
    setResult(null);
    setBatchResult(null);
    setStreamResponse("");
    setStreamTokenCount(0);
    streamTokenCountRef.current = 0;
    const tokenLimit = Number(maxNewTokens);
    const temp = Number(temperature);
    const p = Number(topP);
    const k = Number(topK);
    const minPValue = Number(minP);
    const repeatPenalty = Number(repetitionPenalty);
    const seedValue = seed.trim() === "" ? null : Number(seed);
    const stopSequences = stopSequencesText.split("\n").map((value) => value.replace(/\r$/, "")).filter((value) => value.length > 0);
    if (!modelPath.trim()) { setError("Choose a discovered native model or enter a trusted local Llama/Mistral/Qwen2 model path."); return; }
    if (!prompt.trim()) { setError("Enter a prompt first."); return; }
    if (!Number.isInteger(tokenLimit) || tokenLimit < 1 || tokenLimit > 2048) { setError("Maximum generated tokens must be between 1 and 2048."); return; }
    if (!Number.isFinite(temp) || temp < 0 || temp > 2) { setError("Temperature must be between 0 and 2."); return; }
    if (!Number.isFinite(p) || p <= 0 || p > 1) { setError("Top-p must be greater than 0 and at most 1."); return; }
    if (!Number.isInteger(k) || k < 0 || k > 100000) { setError("Top-k must be an integer from 0 through 100000; 0 disables top-k."); return; }
    if (!Number.isFinite(minPValue) || minPValue < 0 || minPValue > 1) { setError("Min-p must be between 0 and 1; 0 disables min-p."); return; }
    if (!Number.isFinite(repeatPenalty) || repeatPenalty < 0.1 || repeatPenalty > 5) { setError("Repetition penalty must be between 0.1 and 5."); return; }
    if (seed.trim() !== "" && (!Number.isSafeInteger(seedValue) || seedValue! < 0 || seedValue! > Number.MAX_SAFE_INTEGER)) {
      setError("Seed must be a whole number from 0 through 9007199254740991, or blank for a non-fixed seed.");
      return;
    }
    if (stopSequences.length > 16) { setError("At most 16 stop sequences may be configured."); return; }
    if (stopSequences.some((value) => !value.trim() || value.length > 256)) {
      setError("Each stop sequence must contain non-whitespace text and be no longer than 256 characters.");
      return;
    }
    const controller = new AbortController();
    activeStreamRef.current = controller;
    setLiveTelemetry(null);
    setRunning(true);
    try {
      const completed = await axiomApi.streamAxiomEngine({
        model_path: modelPath.trim(),
        adapter_path: adapterPath.trim() || undefined,
        device: devicePreference,
        memory_mode: memoryMode,
        precision,
        prompt,
        max_new_tokens: tokenLimit,
        temperature: temp,
        top_p: p,
        top_k: k,
        min_p: minPValue,
        repetition_penalty: repeatPenalty,
        seed: seedValue,
        stop_sequences: stopSequences,
      }, (event: AxiomEngineStreamEvent) => {
        if (event.type === "token") {
          streamTokenCountRef.current = event.completion_tokens;
          setStreamTokenCount(event.completion_tokens);
          setStreamResponse(event.response);
        } else if (event.type === "telemetry") {
          setLiveTelemetry(event);
        } else if (event.type === "complete") {
          streamTokenCountRef.current = event.result.completion_tokens;
          setStreamTokenCount(event.result.completion_tokens);
          setStreamResponse(event.result.response);
        }
      }, controller.signal);
      setResult(completed);
      void refresh();
    } catch (cause) {
      if (controller.signal.aborted) {
        setError(`Generation cancelled after ${streamTokenCountRef.current} token(s). Partial output remains visible.`);
      } else {
        setError(cause instanceof Error ? cause.message : "AXIOM Native Engine generation failed.");
      }
    } finally {
      if (activeStreamRef.current === controller) activeStreamRef.current = null;
      setRunning(false);
    }
  };

  const runBatch = async () => {
    setError("");
    setLifecycleNotice("");
    setResult(null);
    setBatchResult(null);
    setStreamResponse("");
    setStreamTokenCount(0);
    streamTokenCountRef.current = 0;

    const prompts = batchPromptsText.split(/\r?\n/).map((value) => value.trim()).filter(Boolean);
    if (!modelPath.trim()) { setError("Choose a discovered native model or enter a trusted local model path."); return; }
    if (prompts.length < 1) { setError("Enter at least one batch prompt, one prompt per line."); return; }
    if (prompts.length > 128) { setError("Batch generation supports at most 128 prompts per request."); return; }
    if (prompts.some((value) => value.length > 12000)) { setError("Each batch prompt is limited to 12,000 characters."); return; }
    if (prompts.reduce((total, value) => total + value.length, 0) > 24000) {
      setError("Combined batch prompts exceed the 24,000-character limit.");
      return;
    }

    const tokenLimit = Number(maxNewTokens);
    const temp = Number(temperature);
    const p = Number(topP);
    const k = Number(topK);
    const minPValue = Number(minP);
    const repeatPenalty = Number(repetitionPenalty);
    const seedValue = seed.trim() === "" ? null : Number(seed);
    const stopSequences = stopSequencesText.split("\n").map((value) => value.replace(/\r$/, "")).filter((value) => value.length > 0);
    if (!Number.isInteger(tokenLimit) || tokenLimit < 1 || tokenLimit > 2048) { setError("Maximum generated tokens must be between 1 and 2048."); return; }
    if (!Number.isFinite(temp) || temp < 0 || temp > 2) { setError("Temperature must be between 0 and 2."); return; }
    if (!Number.isFinite(p) || p <= 0 || p > 1) { setError("Top-p must be greater than 0 and at most 1."); return; }
    if (!Number.isInteger(k) || k < 0 || k > 100000) { setError("Top-k must be an integer from 0 through 100000; 0 disables top-k."); return; }
    if (!Number.isFinite(minPValue) || minPValue < 0 || minPValue > 1) { setError("Min-p must be between 0 and 1; 0 disables min-p."); return; }
    if (!Number.isFinite(repeatPenalty) || repeatPenalty < 0.1 || repeatPenalty > 5) { setError("Repetition penalty must be between 0.1 and 5."); return; }
    if (seed.trim() !== "" && (!Number.isSafeInteger(seedValue) || seedValue! < 0 || seedValue! > Number.MAX_SAFE_INTEGER)) {
      setError("Seed must be a whole number from 0 through 9007199254740991, or blank for a non-fixed seed.");
      return;
    }
    if (stopSequences.length > 16) { setError("At most 16 stop sequences may be configured."); return; }
    if (stopSequences.some((value) => !value.trim() || value.length > 256)) {
      setError("Each stop sequence must contain non-whitespace text and be no longer than 256 characters.");
      return;
    }

    const controller = new AbortController();
    activeStreamRef.current = controller;
    setLiveTelemetry(null);
    setRunning(true);
    try {
      const completed = await axiomApi.runAxiomEngineBatch({
        model_path: modelPath.trim(),
        adapter_path: adapterPath.trim() || undefined,
        prompts,
        device: devicePreference,
        memory_mode: memoryMode,
        precision,
        max_new_tokens: tokenLimit,
        temperature: temp,
        top_p: p,
        top_k: k,
        min_p: minPValue,
        repetition_penalty: repeatPenalty,
        seed: seedValue,
        stop_sequences: stopSequences,
      }, controller.signal);
      setBatchResult(completed);
      void refresh();
    } catch (cause) {
      setError(controller.signal.aborted
        ? "Batch generation cancelled."
        : cause instanceof Error ? cause.message : "AXIOM native batch generation failed.");
    } finally {
      if (activeStreamRef.current === controller) activeStreamRef.current = null;
      setRunning(false);
    }
  };

  const runBenchmark = async () => {
    setBenchmarkError("");
    setBenchmarkResult(null);
    if (!modelPath.trim()) { setBenchmarkError("Choose a discovered native model before benchmarking."); return; }
    if (!prompt.trim()) { setBenchmarkError("Enter a prompt before benchmarking."); return; }
    const warmups = Number(benchmarkWarmupRuns);
    const iterations = Number(benchmarkIterations);
    const requestedTokens = Number(maxNewTokens);
    const tokenLimit = Math.min(requestedTokens, 128);
    const temp = Number(temperature);
    const p = Number(topP);
    const k = Number(topK);
    const minPValue = Number(minP);
    const repeatPenalty = Number(repetitionPenalty);
    const seedValue = seed.trim() === "" ? null : Number(seed);
    const stopSequences = stopSequencesText.split("\n").map((value) => value.replace(/\r$/, "")).filter((value) => value.length > 0);
    if (!Number.isInteger(warmups) || warmups < 0 || warmups > 2) { setBenchmarkError("Warm-up runs must be an integer from 0 through 2."); return; }
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 5) { setBenchmarkError("Measured iterations must be an integer from 1 through 5."); return; }
    if (!Number.isInteger(requestedTokens) || requestedTokens < 1 || requestedTokens > 2048) { setBenchmarkError("Maximum generated tokens must be between 1 and 2048."); return; }
    if (!Number.isFinite(temp) || temp < 0 || temp > 2) { setBenchmarkError("Temperature must be between 0 and 2."); return; }
    if (!Number.isFinite(p) || p <= 0 || p > 1) { setBenchmarkError("Top-p must be greater than 0 and at most 1."); return; }
    if (!Number.isInteger(k) || k < 0 || k > 100000) { setBenchmarkError("Top-k must be an integer from 0 through 100000."); return; }
    if (!Number.isFinite(minPValue) || minPValue < 0 || minPValue > 1) { setBenchmarkError("Min-p must be between 0 and 1."); return; }
    if (!Number.isFinite(repeatPenalty) || repeatPenalty < 0.1 || repeatPenalty > 5) { setBenchmarkError("Repetition penalty must be between 0.1 and 5."); return; }
    if (seed.trim() !== "" && (!Number.isSafeInteger(seedValue) || seedValue! < 0 || seedValue! > Number.MAX_SAFE_INTEGER)) {
      setBenchmarkError("Seed must be a whole number from 0 through 9007199254740991, or blank.");
      return;
    }
    if (stopSequences.length > 16 || stopSequences.some((value) => !value.trim() || value.length > 256)) {
      setBenchmarkError("Use at most 16 non-blank stop sequences, each no longer than 256 characters.");
      return;
    }

    const controller = new AbortController();
    benchmarkAbortRef.current = controller;
    setBenchmarking(true);
    try {
      const report = await axiomApi.benchmarkAxiomEngine({
        model_path: modelPath.trim(),
        adapter_path: adapterPath.trim() || undefined,
        prompt,
        warmup_runs: warmups,
        iterations,
        max_new_tokens: tokenLimit,
        temperature: temp,
        top_p: p,
        top_k: k,
        min_p: minPValue,
        repetition_penalty: repeatPenalty,
        seed: seedValue,
        stop_sequences: stopSequences,
        device: devicePreference,
        memory_mode: memoryMode,
        precision,
      }, controller.signal);
      setBenchmarkResult(report);
      void refresh();
    } catch (cause) {
      setBenchmarkError(controller.signal.aborted
        ? "Benchmark cancelled; AXIOM has been asked to stop the native generation loop."
        : cause instanceof Error ? cause.message : "AXIOM Native Engine benchmark failed.");
    } finally {
      if (benchmarkAbortRef.current === controller) benchmarkAbortRef.current = null;
      setBenchmarking(false);
    }
  };

  const telemetryMetrics = result ? {
    stage: "complete",
    prompt_tokens: result.prompt_tokens,
    completion_tokens: result.completion_tokens,
    prefill_ms: result.prefill_ms,
    decode_ms: result.decode_ms,
    time_to_first_token_ms: result.time_to_first_token_ms,
    generation_tokens_per_second: result.generation_tokens_per_second,
    elapsed_ms: result.elapsed_ms,
    stop_reason: result.stop_reason,
  } : liveTelemetry?.metrics;
  const deviceMemory = result?.device_memory ?? liveTelemetry?.device_memory;
  const memorySelection = result?.memory_selection ?? liveTelemetry?.memory_selection;
  const memoryStats = result?.memory_stats ?? liveTelemetry?.memory_stats;
  const telemetryMemoryMode = result?.memory_mode ?? liveTelemetry?.memory_mode ?? memoryMode;
  return <section className="surface axiom-engine-panel" aria-label="AXIOM local engine">
    <div className="surface-head">
      <div><p className="surface-eyebrow">AXIOM ENGINE · DIRECT LOCAL INFERENCE</p><h2>AXIOM Local Engine</h2></div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading || running || benchmarking || unloading}><RefreshCcw size={14} /> {loading ? "Checking" : "Check engine"}</button>
        {status?.loaded && <button type="button" className="button secondary-button" onClick={() => void unload()} disabled={loading || running || benchmarking || unloading}><Unplug size={14} /> {unloading ? "Unloading…" : "Unload model"}</button>}
      </div>
    </div>
    <p>AXIOM Native Engine implements its own PyTorch Llama/Mistral/Qwen2/Qwen3 forward pass, KV-cache decode, attention, sampling, and supported LoRA application. It loads floating-point safetensors checkpoints and can ingest single-file or complete split-shard dense Llama/Mistral/Qwen2/Qwen3 GGUF checkpoints through gguf-py, dequantizing GGUF tensors to floating-point for AXIOM's own runtime. That means GGUF source quantization does not provide low-bit execution or low-bit memory usage. Model Studio can pull compatible public Hugging Face safetensors checkpoints directly into AXIOM. Malformed or incomplete GGUF shard sets, EXL2/GPTQ/AWQ low-bit kernels, multimodal architectures, and vLLM-style continuous-batching serving remain unsupported.</p>
    <div className="status-strip" aria-label="AXIOM Engine status">
      <div className="status-strip-item">
        <span className={`status-pill ${status?.ready ? "success" : "warning"}`}><i />{loading ? "CHECKING" : status?.ready ? "ENGINE READY" : "SETUP REQUIRED"}</span>
        <div><b>{status?.loaded ? `Loaded: ${status.architecture || "local model"} · ${status.loaded_device || "device reported by runtime"}` : "AXIOM-native decoder ready"}</b><span>{status?.setup_note || error || "Checking local PyTorch and native inference dependencies."}</span></div>
      </div>
    </div>
    {status && !status.ready && <div className="inline-alert warning" role="status">{status.setup_note}</div>}
    {status?.devices && <section className="two-column-grid" aria-label="Native execution device capabilities">
      {status.devices.map((device) => <article className="panel-surface" key={device.id} style={{ padding: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <strong>{device.name}</strong>
          <span className={"status-pill " + (device.available ? "success" : "neutral")}>{device.available ? "AVAILABLE" : "UNAVAILABLE"}</span>
        </div>
        {device.memory_total_gib !== null && <p className="quiet-note">Memory: {device.memory_free_gib ?? "n/a"} GiB free / {device.memory_total_gib} GiB total</p>}
        <p className="quiet-note">{device.reason}</p>
      </article>)}
    </section>}
    {systemTelemetry && <section className="panel-surface" aria-label="System resource telemetry" style={{ padding: 12, marginBottom: 14 }}>
      <div className="surface-head"><div><p className="surface-eyebrow"><Activity size={14} /> SYSTEM TELEMETRY</p><h3>Point-in-time resource snapshot</h3></div><small className="quiet-note">Sampled {new Date(systemTelemetry.sampled_at).toLocaleTimeString()}</small></div>
      <div className="two-column-grid">
        <article className="surface"><p className="surface-eyebrow">System CPU</p><h3>{systemTelemetry.cpu.system_percent === null ? "not exposed" : systemTelemetry.cpu.system_percent.toFixed(1) + "%"}</h3><p className="quiet-note">{systemTelemetry.cpu.logical_cores ?? "?"} logical cores</p></article>
        <article className="surface"><p className="surface-eyebrow">System RAM</p><h3>{formatMemoryBytes(systemTelemetry.memory.available_bytes)} free</h3><p className="quiet-note">{formatMemoryBytes(systemTelemetry.memory.used_bytes)} used / {formatMemoryBytes(systemTelemetry.memory.total_bytes)} total{systemTelemetry.memory.percent === null ? "" : " · " + systemTelemetry.memory.percent.toFixed(1) + "% used"}</p></article>
        <article className="surface"><p className="surface-eyebrow">AXIOM process</p><h3>{formatMemoryBytes(systemTelemetry.process.rss_bytes)} RSS</h3><p className="quiet-note">Peak RSS: {formatMemoryBytes(systemTelemetry.process.peak_rss_bytes)} · {systemTelemetry.cpu.process_percent === null ? "CPU usage not exposed" : systemTelemetry.cpu.process_percent.toFixed(1) + "% CPU"} · {systemTelemetry.process.threads ?? "?"} threads</p></article>
        <article className="surface"><p className="surface-eyebrow">Trusted model storage</p><h3>{systemTelemetry.disks.length} root(s)</h3><p className="quiet-note">{systemTelemetry.disks.length ? systemTelemetry.disks.map(disk => disk.label + ": " + formatMemoryBytes(disk.free_bytes) + " free").join(" · ") : "No trusted model roots are currently available."}</p></article>
      </div>
      {systemTelemetry.loaded_runtime.device_allocator && <p className="quiet-note">Loaded-device allocator: {formatMemoryBytes(systemTelemetry.loaded_runtime.device_allocator.current_allocated_bytes)} allocated · {formatMemoryBytes(systemTelemetry.loaded_runtime.device_allocator.current_reserved_bytes)} reserved · {formatMemoryBytes(systemTelemetry.loaded_runtime.device_allocator.peak_allocated_bytes)} peak allocated</p>}
      <p className="quiet-note">{systemTelemetry.scope_note} Raw paths, prompts, generated text, environment values and credentials are omitted.</p>
    </section>}
    {lifecycleNotice && <div className="inline-alert success" role="status">{lifecycleNotice}</div>}
    <form onSubmit={(event) => void run(event)}>
      <label className="field">
        <span className="field-label-row"><span>Discovered native model</span><span>{assets?.models.length ?? 0} found</span></span>
        <select aria-label="Select discovered native model" value={selectedModelPath} onChange={(event) => { setSelectedModelPath(event.target.value); if (event.target.value) setModelPath(event.target.value); setAdapterCompatibility(null); }}>
          <option value="">{assets?.models.length ? "Choose a discovered model or enter a path below" : "No supported local models discovered"}</option>
          {(assets?.models ?? []).map((model) => <option key={model.path} value={model.path}>{model.name} · {model.model_type} · {model.weight_format}{model.effective_max_position_embeddings ? ` · ${model.effective_max_position_embeddings.toLocaleString()} context` : ""}{model.rope_scaling_type && model.rope_scaling_type !== "default" ? ` · RoPE ${model.rope_scaling_type}` : ""}</option>)}
        </select>
        <small>{assets?.scope_note ?? "Model discovery is unavailable; the API still accepts validated paths under trusted roots."}</small>
      </label>
      <label className="field">
        <span className="field-label-row"><span>Local native model directory</span></span>
        <input value={modelPath} onChange={(event) => { setModelPath(event.target.value); if (event.target.value !== selectedModelPath) setSelectedModelPath(""); setAdapterCompatibility(null); }} required maxLength={4096} placeholder="/home/user/models/my-llama-model" autoComplete="off" />
        <small>Use a supported Llama, Mistral, Qwen2 or Qwen3 text checkpoint with config.json, tokenizer.json and model.safetensors (or its shard index), or a complete single-file / metadata-validated split GGUF checkpoint. Discovery shows the effective native context limit and any configured RoPE scaling. GGUF tensors are dequantized to floating-point; source Q4/Q5/Q8 quantization does not mean low-bit execution or memory usage. Malformed/incomplete GGUF shards, EXL2/GPTQ/AWQ kernels and pickle .bin weights are not supported. Paths must match a directory discovered under a trusted root; set AXIOM_MODEL_ROOTS for additional locations.</small>
      </label>
      <label className="field">
        <span className="field-label-row"><span>Discovered LoRA adapter (optional)</span><span>{assets?.adapters.length ?? 0} found</span></span>
        <select aria-label="Select discovered LoRA adapter" value={selectedAdapterPath} onChange={(event) => { setSelectedAdapterPath(event.target.value); if (event.target.value) setAdapterPath(event.target.value); setAdapterCompatibility(null); }}>
          <option value="">No adapter</option>
          {(assets?.adapters ?? []).map((adapter) => <option key={adapter.path} value={adapter.path}>{adapter.name} · {adapter.weight_format}</option>)}
        </select>
      </label>
      <label className="field">
        <span className="field-label-row"><span>Optional LoRA adapter directory</span></span>
        <input value={adapterPath} onChange={(event) => { setAdapterPath(event.target.value); if (event.target.value !== selectedAdapterPath) setSelectedAdapterPath(""); setAdapterCompatibility(null); }} maxLength={4096} placeholder="/home/user/runs/my-lora-adapter" autoComplete="off" />
        <small>Use a standard bias-free CAUSAL_LM adapter with adapter_config.json and adapter_model.safetensors. Compatibility is checked during model loading; unsupported adapter options are rejected.</small>
      </label>
      <div className="integration-form-actions">
        <button type="button" className="button secondary-button" onClick={() => void checkAdapterCompatibility()} disabled={checkingAdapter || running || !modelPath.trim() || !adapterPath.trim()}>
          {checkingAdapter ? "Checking adapter…" : "Check adapter compatibility"}
        </button>
      </div>
      {adapterCompatibility && <section className="panel-surface" aria-label="LoRA device compatibility report" style={{ padding: 14, marginBottom: 14 }}>
        <div className="surface-head">
          <div><p className="surface-eyebrow">NATIVE ADAPTER PREFLIGHT</p><h3>{adapterCompatibility.structurally_compatible ? "Adapter structure supported" : "Adapter incompatible"}</h3></div>
          <span className={"status-pill " + (adapterCompatibility.structurally_compatible ? "success" : "warning")}>{adapterCompatibility.structurally_compatible ? "STRUCTURE OK" : "ISSUES FOUND"}</span>
        </div>
        <p className="quiet-note">{adapterCompatibility.adapter_tensor_count} adapter tensors · {adapterCompatibility.checked_layers} model layers checked · rank {adapterCompatibility.adapter.rank ?? "unknown"} · alpha {adapterCompatibility.adapter.alpha ?? "unknown"}</p>
        <div className="two-column-grid">
          {adapterCompatibility.devices.map((device) => <div key={device.name} className="panel-surface" style={{ padding: 10 }}>
            <strong>{device.name}</strong>
            <span className={"status-pill " + (device.compatible ? "success" : device.available ? "warning" : "neutral")}>{device.status.toUpperCase()}</span>
            <p className="quiet-note">{device.reason}</p>
          </div>)}
        </div>
        {adapterCompatibility.issues.length > 0 && <div className="inline-alert warning" role="alert"><strong>Compatibility blockers</strong><ul>{adapterCompatibility.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul></div>}
        <p className="quiet-note">{adapterCompatibility.scope_note}</p>
      </section>}
      <label className="field">
        <span className="field-label-row"><span>Execution device</span><span>Recommended: {status?.recommended_device ?? "auto"}</span></span>
        <select aria-label="Execution device" value={devicePreference} onChange={(event) => setDevicePreference(event.target.value as typeof devicePreference)}>
          <option value="auto">Auto-select available backend</option>
          {(status?.devices ?? []).map((device) => <option key={device.id} value={device.id} disabled={!device.available}>{device.name}{device.available ? " · available" : " · unavailable"}</option>)}
        </select>
        <small>Auto prefers CUDA/ROCm, then Apple Metal or Intel XPU, and finally CPU. An explicit unavailable device fails with a diagnostic instead of silently changing the selected backend.</small>
      </label>
      <label className="field">
        <span className="field-label-row"><span>Compute precision</span></span>
        <select aria-label="Compute precision" value={precision} onChange={(event) => setPrecision(event.target.value as typeof precision)}>
          <option value="auto">Auto · preserve checkpoint dtype</option>
          <option value="fp32">FP32 · highest compatibility, highest memory</option>
          <option value="fp16">FP16 · half-precision weights and activations</option>
          <option value="bf16">BF16 · wider dynamic range where supported</option>
        </select>
        <small>Floating-point precision is not INT8/INT4 quantization. Hardware/kernel support varies; an unsupported dtype returns a diagnostic instead of silently switching precision.</small>
      </label>
      <label className="field">
        <span className="field-label-row"><span>Memory strategy</span></span>
        <select aria-label="Memory strategy" value={memoryMode} onChange={(event) => setMemoryMode(event.target.value as typeof memoryMode)}>
          <option value="auto">Auto-fit from model size, context and available memory · recommended</option>
          <option value="full">Full weights on execution device · faster, higher VRAM</option>
          <option value="layer_streaming">Layer streaming + CPU KV cache · lower VRAM, slower</option>
          <option value="disk_streaming">Disk-resident layer streaming + CPU KV cache · lowest RAM/VRAM, slowest</option>
        </select>
        <small>Auto-fit estimates checkpoint weight bytes, KV-cache and workspace demand for the configured context, and observed free VRAM/system RAM to select full weights, in-RAM layer streaming, or disk-resident layer streaming. Estimates are advisory, not a guarantee against device OOM. Layer streaming keeps all transformer weights in system RAM and stages one layer at a time; disk streaming reads layers from local safetensors shards or dequantizes local GGUF layer tensors on demand. Both streaming modes keep the KV cache in system RAM and trade throughput for lower RAM/VRAM weight residency. Embeddings, output projection, LoRA adapter tensors, activations, and the active layer still need memory.</small>
      </label>
      <label className="field">
        <span className="field-label-row"><span>AXIOM Native Engine prompt</span><span>{prompt.length.toLocaleString()} / 12,000 characters</span></span>
        <textarea aria-label="AXIOM Native Engine prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} maxLength={12000} rows={5} required placeholder="Ask the local model to do something…" />
      </label>
      <label className="field">
        <span className="field-label-row"><span>Batch prompts (optional)</span><span>{batchPromptsText.split(/\r?\n/).filter((value) => value.trim().length > 0).length} / 128 prompts · 24,000 characters total</span></span>
        <textarea aria-label="AXIOM Native batch prompts" value={batchPromptsText} onChange={(event) => setBatchPromptsText(event.target.value)} maxLength={24000} rows={4} placeholder={"One prompt per line for length-bucketed native batch generation…\nSummarize this sample.\nClassify this second sample."} />
        <small>Each non-blank line becomes an independent prompt. AXIOM groups prompts by token length, completes each response independently, and returns per-prompt token counts and stop reasons.</small>
      </label>
      <div className="two-column-grid">
        <label className="field"><span className="field-label-row"><span>Maximum generated tokens</span></span><input aria-label="Maximum generated tokens" type="number" min={1} max={2048} step={1} value={maxNewTokens} onChange={(event) => setMaxNewTokens(event.target.value)} required /></label>
        <label className="field"><span className="field-label-row"><span>Temperature (0 = deterministic)</span></span><input aria-label="Temperature" type="number" min={0} max={2} step={0.1} value={temperature} onChange={(event) => setTemperature(event.target.value)} required /></label>
        <label className="field"><span className="field-label-row"><span>Top-p nucleus sampling</span></span><input aria-label="Top-p" type="number" min={0.000001} max={1} step="any" value={topP} onChange={(event) => setTopP(event.target.value)} required /><small>Keep the smallest high-probability token set reaching this cumulative probability (0, 1].</small></label>
        <label className="field"><span className="field-label-row"><span>Top-k sampling</span></span><input aria-label="Top-k" type="number" min={0} max={100000} step={1} value={topK} onChange={(event) => setTopK(event.target.value)} required /><small>0 disables top-k. A non-zero value must not exceed the selected model's vocabulary size.</small></label>
        <label className="field"><span className="field-label-row"><span>Min-p relative probability</span></span><input aria-label="Min-p" type="number" min={0} max={1} step={0.01} value={minP} onChange={(event) => setMinP(event.target.value)} required /><small>0 disables min-p; otherwise discard tokens below this fraction of the top token's probability.</small></label>
        <label className="field"><span className="field-label-row"><span>Repetition penalty</span></span><input aria-label="Repetition penalty" type="number" min={0.1} max={5} step={0.05} value={repetitionPenalty} onChange={(event) => setRepetitionPenalty(event.target.value)} required /><small>1.0 disables repetition adjustment; other values penalize or boost previously seen token scores.</small></label>
        <label className="field"><span className="field-label-row"><span>Sampling seed (optional)</span></span><input aria-label="Sampling seed" type="number" min={0} max={Number.MAX_SAFE_INTEGER} step={1} value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="Leave blank to vary sampling" /><small>A seed makes token sampling reproducible on the same model, device and runtime. This does not guarantee cross-device bitwise reproducibility.</small></label>
      </div>
      <label className="field">
        <span className="field-label-row"><span>Stop sequences (optional)</span><span>{stopSequencesText.split("\n").filter((value) => value.length > 0).length} / 16</span></span>
        <textarea aria-label="Stop sequences" value={stopSequencesText} onChange={(event) => setStopSequencesText(event.target.value)} rows={3} maxLength={4112} placeholder={"One exact stop string per line, e.g.\n<END>\n###"} />
        <small>AXIOM stops at the first full match, holds back partial matches across tokens, and excludes the stop text from the answer. Maximum 16 strings, 256 characters each.</small>
      </label>
      <section className="panel-surface" aria-label="Native benchmark controls" style={{ padding: 12, marginBottom: 14 }}>
        <div className="surface-head">
          <div><p className="surface-eyebrow">AXIOM ENGINE · MEASURED PERFORMANCE</p><h3>Native inference benchmark</h3></div>
        </div>
        <p className="quiet-note">Measure real local inference on this model and device. Warm-ups are excluded from the measured statistics; the benchmark uses the current prompt and sampling settings, caps each run at 128 generated tokens, and never returns or stores the generated text.</p>
        <div className="two-column-grid">
          <label className="field"><span className="field-label-row"><span>Benchmark warm-up runs</span></span><input aria-label="Benchmark warm-up runs" type="number" min={0} max={2} step={1} value={benchmarkWarmupRuns} onChange={(event) => setBenchmarkWarmupRuns(event.target.value)} /><small>0–2 runs excluded from the results.</small></label>
          <label className="field"><span className="field-label-row"><span>Measured benchmark runs</span></span><input aria-label="Measured benchmark runs" type="number" min={1} max={5} step={1} value={benchmarkIterations} onChange={(event) => setBenchmarkIterations(event.target.value)} /><small>1–5 measured generations; this is bounded to protect local resources.</small></label>
        </div>
        <div className="integration-form-actions">
          <button type="button" className="button secondary-button" onClick={() => void runBenchmark()} disabled={!status?.ready || loading || running || benchmarking || unloading || !modelPath.trim() || !prompt.trim() || Boolean(adapterCompatibility && !adapterCompatibility.structurally_compatible)}>
            <Activity size={14} /> {benchmarking ? "Benchmarking locally…" : "Benchmark native engine"}
          </button>
          {benchmarking && <button type="button" className="button secondary-button" onClick={() => benchmarkAbortRef.current?.abort()}><Unplug size={14} /> Stop benchmark</button>}
        </div>
        {benchmarkError && <div className="inline-alert warning" role="alert">{benchmarkError}</div>}
      </section>
      <div className="integration-form-actions">
        <button type="submit" className="button primary-button" disabled={!status?.ready || loading || running || benchmarking || !modelPath.trim() || !prompt.trim() || Boolean(adapterCompatibility && !adapterCompatibility.structurally_compatible)}><Play size={15} /> {running ? "Generating locally…" : "Run with AXIOM Engine"}</button>
        <button type="button" className="button secondary-button" onClick={() => void runBatch()} disabled={!status?.ready || loading || running || benchmarking || !modelPath.trim() || !batchPromptsText.trim() || Boolean(adapterCompatibility && !adapterCompatibility.structurally_compatible)}>{running ? "Batch running…" : "Run native batch"}</button>
        {running && <button type="button" className="button secondary-button" onClick={() => activeStreamRef.current?.abort()}><Unplug size={14} /> Stop generation</button>}
        {result && <span className="quiet-note"><Activity size={14} /> {result.completion_tokens} tokens · {result.elapsed_ms} ms wall time · {result.prefill_ms} ms prefill · {result.decode_ms} ms decode · {result.generation_tokens_per_second} tok/s</span>}
        {!result && streamTokenCount > 0 && <span className="quiet-note"><Activity size={14} /> {streamTokenCount} token(s) streamed</span>}
      </div>
    </form>
    {batchResult && <section className="panel-surface" aria-label="Native batch results" style={{ padding: 14, marginTop: 14 }}>
      <div className="surface-head">
        <div><p className="surface-eyebrow">AXIOM ENGINE · NATIVE BATCH</p><h3>{batchResult.batch_size} prompt results</h3></div>
        <span className="status-pill">COMPLETE</span>
      </div>
      <p className="quiet-note">{batchResult.prompt_tokens.toLocaleString()} prompt tokens · {batchResult.completion_tokens.toLocaleString()} completion tokens · {batchResult.elapsed_ms.toFixed(1)} ms wall time · {batchResult.memory_mode.replaceAll("_", " ")} · local-only execution</p>
      <div style={{ display: "grid", gap: 12 }}>
        {batchResult.results.map((item) => <article key={item.index} className="panel-surface" style={{ padding: 12 }}>
          <h4>Prompt {item.index}</h4>
          <p className="quiet-note">{item.prompt_tokens} input token(s) · {item.completion_tokens} output token(s) · finish: {item.stop_reason}{item.context_management.token_truncation_applied ? " · input truncated to fit context" : ""}</p>
          <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", font: "inherit", margin: 0 }}>{item.response}</pre>
        </article>)}
      </div>
    </section>}
    {benchmarkResult && <section className="panel-surface" aria-label="Native benchmark results" style={{ padding: 14, marginTop: 14 }}>
      <div className="surface-head">
        <div><p className="surface-eyebrow">AXIOM ENGINE · MEASURED RUNS</p><h3>{benchmarkResult.model} · {benchmarkResult.architecture ?? "Native model"}</h3></div>
        <span className="status-pill">COMPLETE</span>
      </div>
      <p className="quiet-note">{benchmarkResult.iterations_completed} measured run(s) · {benchmarkResult.warmup_runs_completed} warm-up(s) excluded · {benchmarkResult.device ?? "device not reported"} · {benchmarkResult.precision ?? "precision not reported"} · {benchmarkResult.memory_mode.replaceAll("_", " ")}</p>
      <div className="two-column-grid">
        <article className="surface"><p className="surface-eyebrow">Median latency</p><h3>{benchmarkResult.metrics.median_elapsed_ms.toFixed(2)} ms</h3><p className="quiet-note">p95: {benchmarkResult.metrics.p95_elapsed_ms.toFixed(2)} ms · measured wall time: {benchmarkResult.metrics.measured_wall_time_ms.toFixed(2)} ms</p></article>
        <article className="surface"><p className="surface-eyebrow">Generation throughput</p><h3>{benchmarkResult.metrics.aggregate_tokens_per_second.toFixed(2)} tokens/s</h3><p className="quiet-note">Median run: {benchmarkResult.metrics.median_generation_tokens_per_second.toFixed(2)} tokens/s · {benchmarkResult.metrics.completion_tokens.toLocaleString()} measured output tokens</p></article>
        <article className="surface"><p className="surface-eyebrow">Prefill / time to first token</p><h3>{benchmarkResult.metrics.median_prefill_ms.toFixed(2)} ms</h3><p className="quiet-note">Median time to first token: {benchmarkResult.metrics.median_time_to_first_token_ms.toFixed(2)} ms</p></article>
        <article className="surface"><p className="surface-eyebrow">Cached decode</p><h3>{benchmarkResult.metrics.median_decode_ms.toFixed(2)} ms</h3><p className="quiet-note">{benchmarkResult.metrics.prompt_tokens.toLocaleString()} total prompt tokens measured</p></article>
      </div>
      <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
        {benchmarkResult.runs.map((item) => <article key={item.iteration} className="panel-surface" style={{ padding: 10 }}>
          <strong>Run {item.iteration}</strong>
          <p className="quiet-note">{item.elapsed_ms.toFixed(2)} ms wall · {item.prefill_ms.toFixed(2)} ms prefill · {item.decode_ms.toFixed(2)} ms decode · {item.completion_tokens} token(s) · {item.generation_tokens_per_second.toFixed(2)} tokens/s · {item.stop_reason}</p>
        </article>)}
      </div>
      {(benchmarkResult.memory_stats.layer_weight_bytes_staged > 0
        || benchmarkResult.memory_stats.disk_layer_weight_bytes_read > 0
        || benchmarkResult.memory_stats.kv_cache_bytes_offloaded > 0) && <section className="surface" aria-label="Layer streaming and KV transfer telemetry" style={{ padding: 12, marginTop: 12 }}>
        <p className="surface-eyebrow">MEMORY TRANSFER TELEMETRY</p>
        <div className="two-column-grid">
          <article><p className="quiet-note">Layer weights staged</p><strong>{formatMemoryBytes(benchmarkResult.memory_stats.layer_weight_bytes_staged)}</strong><p className="quiet-note">{benchmarkResult.memory_stats.layer_weight_tensors_staged.toLocaleString()} tensor(s)</p></article>
          <article><p className="quiet-note">Disk layer reads</p><strong>{benchmarkResult.memory_stats.disk_layer_weight_loads.toLocaleString()}</strong><p className="quiet-note">{formatMemoryBytes(benchmarkResult.memory_stats.disk_layer_weight_bytes_read)} read from disk</p></article>
          <article><p className="quiet-note">KV cache offloaded</p><strong>{formatMemoryBytes(benchmarkResult.memory_stats.kv_cache_bytes_offloaded)}</strong><p className="quiet-note">{benchmarkResult.memory_stats.kv_cache_layer_offloads.toLocaleString()} layer offload(s)</p></article>
          <article><p className="quiet-note">Layer weights released</p><strong>{formatMemoryBytes(benchmarkResult.memory_stats.layer_weight_bytes_released)}</strong><p className="quiet-note">{formatMemoryBytes(benchmarkResult.memory_stats.layer_weight_bytes_returned_to_cpu)} returned to CPU</p></article>
        </div>
        <p className="quiet-note">Counters are summed across measured runs only; excluded warm-ups do not contribute.</p>
      </section>}
      <p className="quiet-note">{benchmarkResult.scope_note} No prompts or generated responses are displayed or retained.</p>
    </section>}
    <section className="surface axiom-quantization-planner" aria-label="Memory and quantization planner">
      <div className="surface-head">
        <div>
          <p className="surface-eyebrow">OPTIMIZER · MEMORY PLANNING</p>
          <h3>Quantization and memory fit</h3>
          <p className="quiet-note">Config-based estimate only. INT4/INT8 candidates are sizing comparisons, not formats the current native engine can load.</p>
        </div>
      </div>
      <div className="two-column-grid">
        <label className="field">
          <span className="field-label-row"><span>Planner context length</span></span>
          <input aria-label="Planner context length" type="number" min={1} max={131072} step={1} value={quantContextLength} onChange={(event) => setQuantContextLength(event.target.value)} />
        </label>
        <label className="field">
          <span className="field-label-row"><span>Planner batch size</span></span>
          <input aria-label="Planner batch size" type="number" min={1} max={16} step={1} value={quantBatchSize} onChange={(event) => setQuantBatchSize(event.target.value)} />
        </label>
        <label className="field">
          <span className="field-label-row"><span>RAM budget (GiB)</span></span>
          <input aria-label="RAM budget GiB" type="number" min={0.1} max={2048} step={0.5} value={ramBudgetGiB} onChange={(event) => setRamBudgetGiB(event.target.value)} placeholder="Leave blank to skip RAM" />
        </label>
        <label className="field">
          <span className="field-label-row"><span>VRAM budget (GiB)</span></span>
          <input aria-label="VRAM budget GiB" type="number" min={0.1} max={2048} step={0.5} value={vramBudgetGiB} onChange={(event) => setVramBudgetGiB(event.target.value)} placeholder="Optional GPU memory budget" />
        </label>
      </div>
      <button type="button" className="button secondary-button" onClick={() => void estimateQuantization()} disabled={planningQuantization || !modelPath.trim()}>
        {planningQuantization ? "Estimating memory…" : "Estimate memory fit"}
      </button>
      {quantizationError && <div className="inline-alert warning" role="alert">{quantizationError}</div>}
      {quantizationPlan && <section aria-label="Quantization estimate results">
        <div className="surface-head">
          <div>
            <h4>{quantizationPlan.model.name} · {quantizationPlan.model.architecture}</h4>
            <p className="quiet-note">Estimated {quantizationPlan.model.estimated_parameter_count.toLocaleString()} parameters · Context {quantizationPlan.request.context_length.toLocaleString()} · Batch {quantizationPlan.request.batch_size}</p>
          </div>
        </div>
        <p className="quiet-note">{quantizationPlan.scope_note}</p>
        {typeof quantizationPlan.assumptions.kv_cache_context_length === "number" && typeof quantizationPlan.assumptions.kv_prefill_temporary_context_length === "number" && <p className="quiet-note">KV-cache estimate uses {quantizationPlan.assumptions.kv_cache_context_length.toLocaleString()} retained-context tokens. Additional per-layer prefill temporary memory uses {quantizationPlan.assumptions.kv_prefill_temporary_context_length.toLocaleString()} tokens where a sliding-window model needs a full-context prefill; ordinary non-sliding models do not pay this extra cache charge.</p>}
        <p className="quiet-note">
          <strong>{quantizationPlan.recommendation.format
            ? `Budget-based planning recommendation: ${quantizationPlan.recommendation.format.toUpperCase()}.`
            : "No native precision is estimated to fit all supplied budgets."}</strong>
          {" "}{quantizationPlan.recommendation.note}
        </p>
        <div className="two-column-grid">
          {quantizationPlan.candidates.map((candidate) => <article className="surface" key={candidate.format} aria-label={candidate.format + " memory candidate"}>
            <h4>{candidate.format.toUpperCase()}</h4>
            <p><strong>{formatEstimatedMemory(candidate.estimated_total_gib)} estimated</strong></p>
            <p className="quiet-note">Uncertainty range {formatEstimatedMemory(candidate.estimated_range_gib[0])}–{formatEstimatedMemory(candidate.estimated_range_gib[1])}</p>
            <p className="quiet-note">Weights {formatEstimatedMemory(candidate.estimated_weight_gib)} · KV cache {formatEstimatedMemory(candidate.estimated_kv_cache_gib)} · Prefill temporary {formatEstimatedMemory(candidate.estimated_kv_prefill_temporary_gib)} · Workspace {formatEstimatedMemory(candidate.estimated_workspace_gib)}</p>
            {candidate.fits_ram_budget !== null && <p>RAM budget: {candidate.fits_ram_budget ? "Estimated fit" : "Over estimated budget"}</p>}
            {candidate.fits_vram_budget !== null && <p>VRAM budget: {candidate.fits_vram_budget ? "Estimated fit" : "Over estimated budget"}</p>}
            <p>{candidate.native_checkpoint_supported ? "Floating-point format supported by native loader (actual load still required)." : "Sizing only — native loader does not support this quantized format."}</p>
          </article>)}
        </div>
        <p className="quiet-note">Estimated fit is not measured memory, quality, or speed. Confirm quality with evaluation and speed/memory with an actual run.</p>
      </section>}
    </section>
    {error && <div className="inline-alert warning" role="alert">{error}</div>}
    {(result || streamResponse) && <section className="local-inference-result" aria-label="AXIOM Engine generated response">
      <div className="surface-head"><div>
        <p className="surface-eyebrow">{result ? `GENERATED ON THIS MACHINE · ${result.architecture}` : running ? "STREAMING TOKEN OUTPUT · LOCAL ONLY" : "PARTIAL LOCAL OUTPUT"}</p>
        <h3>{result ? `${result.model}${result.adapter ? ` + ${result.adapter}` : ""}` : (assets?.models.find((model) => model.path === modelPath)?.name || "AXIOM Native Engine")}</h3>
        {result?.device_selection && <p className="quiet-note">Execution device: {result.device_selection.selected}. {result.device_selection.reason}</p>}
        {result?.device_fallback && <p className="quiet-note">Automatic memory fallback: {result.device_fallback.fallback_from.toUpperCase()} → {result.device_fallback.selected.toUpperCase()}. {result.device_fallback.reason}</p>}
        {result?.precision && <p className="quiet-note">Compute precision: requested {result.precision_selection?.requested ?? precision} · selected {result.precision}</p>}
        {result?.memory_mode && <p className="quiet-note">Memory strategy: {result.memory_mode.replaceAll("_", " ")}{result.memory_stats ? (result.memory_mode === "disk_streaming" ? " · Disk layer tensors read: " + result.memory_stats.disk_layer_weight_loads + " · Bytes read: " + (result.memory_stats.disk_layer_weight_bytes_read / (1024 ** 2)).toFixed(1) + " MiB · Layer bytes released: " + (result.memory_stats.layer_weight_bytes_released / (1024 ** 2)).toFixed(1) + " MiB" : " · Layer weight bytes staged: " + (result.memory_stats.layer_weight_bytes_staged / (1024 ** 2)).toFixed(1) + " MiB · Returned to CPU: " + (result.memory_stats.layer_weight_bytes_returned_to_cpu / (1024 ** 2)).toFixed(1) + " MiB") + " · KV-cache bytes offloaded: " + (result.memory_stats.kv_cache_bytes_offloaded / (1024 ** 2)).toFixed(1) + " MiB" : ""}</p>}
        {result?.prompt_format && <p className="quiet-note">Prompt formatting: {result.prompt_format === "tokenizer-config-template" ? "checkpoint chat template" : result.prompt_format === "qwen2-chatml-fallback" ? "Qwen2 ChatML fallback" : result.prompt_format === "qwen3-chatml-fallback" ? "Qwen3 ChatML fallback" : "raw completion"}</p>}
        {result?.context_management?.chat_history_supplied && <p className="quiet-note">Context budget: {result.context_management.original_message_count} message(s) · {result.context_management.messages_dropped_to_fit_context} older turn(s) dropped to fit · {result.context_management.prompt_tokens_used.toLocaleString()} / {result.context_management.prompt_tokens_before_token_truncation.toLocaleString()} prompt token(s) used{result.context_management.token_truncation_applied ? " after token-level truncation" : ""}{result.context_management.system_prompt_may_be_truncated ? " · Warning: system instructions may have been truncated by the hard context limit." : ""}</p>}
        {result && <p className="quiet-note">Finish: {result.stop_reason.replaceAll("_", " ")} · Stop sequences: {result.stop_sequences.length ? result.stop_sequences.length : "off"} · Attention: {result.attention_backend} · Sampling: temperature {result.sampling.temperature} · top-p {result.sampling.top_p} · top-k {result.sampling.top_k || "off"} · min-p {result.sampling.min_p || "off"} · repetition penalty {result.sampling.repetition_penalty} · seed {result.sampling.seed ?? "random"}</p>}
      </div><span className={`status-pill ${result ? "success" : running ? "warning" : "neutral"}`}><i />{result ? "COMPLETED" : running ? "GENERATING" : "CANCELLED"}</span></div>
      <pre data-testid={result ? "axiom-engine-output" : "axiom-engine-live-output"} className="local-inference-output">{(result ? result.response : streamResponse) || "(The model has not emitted printable text yet.)"}</pre>
      {!result && <p className="quiet-note">Cumulative text snapshots update as the native decoder emits tokens. {streamTokenCount} token(s) received.</p>}
      {telemetryMetrics && <section className="surface" aria-label={result ? "Inference telemetry" : "Live inference telemetry"}>
        <div className="surface-head"><div><p className="surface-eyebrow"><Activity size={14} /> {result ? "INFERENCE TELEMETRY" : "LIVE INFERENCE TELEMETRY"}</p><h4>{result ? "Run metrics" : "Generation · " + telemetryMetrics.stage}</h4></div>
          <span className="status-pill"><i />{telemetryMetrics.stage}</span></div>
        <div className="two-column-grid">
          <article className="surface"><p className="surface-eyebrow">Tokens</p><h3>{telemetryMetrics.completion_tokens.toLocaleString()}</h3><p className="quiet-note">{telemetryMetrics.prompt_tokens.toLocaleString()} prompt tokens</p></article>
          <article className="surface"><p className="surface-eyebrow">Throughput</p><h3>{telemetryMetrics.generation_tokens_per_second.toFixed(2)} tok/s</h3><p className="quiet-note">Native decoder estimate based on prefill + decode compute time</p></article>
          <article className="surface"><p className="surface-eyebrow">Prefill · TTFT</p><h3>{telemetryMetrics.prefill_ms.toFixed(1)} ms</h3><p className="quiet-note">Time to first token: {telemetryMetrics.time_to_first_token_ms.toFixed(1)} ms</p></article>
          <article className="surface"><p className="surface-eyebrow">Decode · wall time</p><h3>{telemetryMetrics.decode_ms.toFixed(1)} ms</h3><p className="quiet-note">Elapsed: {telemetryMetrics.elapsed_ms.toFixed(1)} ms</p></article>
        </div>
        {memorySelection && <p className="quiet-note">Memory planner: {memorySelection.reason} {memorySelection.estimates_are_advisory ? "(estimate, not a fit guarantee)" : ""}</p>}
        <p className="quiet-note">Memory mode: {telemetryMemoryMode.replaceAll("_", " ")} · Process RSS: {formatMemoryBytes(deviceMemory?.process_rss_bytes)}{typeof deviceMemory?.process_rss_delta_bytes === "number" ? " (" + (deviceMemory.process_rss_delta_bytes >= 0 ? "+" : "") + formatMemoryBytes(Math.abs(deviceMemory.process_rss_delta_bytes)) + " since generation start)" : ""} · Peak RSS: {formatMemoryBytes(deviceMemory?.process_peak_rss_bytes)}{typeof deviceMemory?.process_peak_rss_delta_bytes === "number" && deviceMemory.process_peak_rss_delta_bytes > 0 ? " (+" + formatMemoryBytes(deviceMemory.process_peak_rss_delta_bytes) + " peak increase)" : ""} · Device: {deviceMemory?.backend ?? result?.device_selection?.selected ?? "unknown"}</p>
        <p className="quiet-note">Device allocator current: {formatMemoryBytes(deviceMemory?.current_allocated_bytes)} · reserved: {formatMemoryBytes(deviceMemory?.current_reserved_bytes)} · peak allocated: {formatMemoryBytes(deviceMemory?.peak_allocated_bytes)} · peak reserved: {formatMemoryBytes(deviceMemory?.peak_reserved_bytes)}{typeof deviceMemory?.peak_increment_bytes === "number" ? " · peak increase: " + formatMemoryBytes(deviceMemory.peak_increment_bytes) : ""}</p>
        {memoryStats && <p className="quiet-note">Layer staging: {formatMemoryBytes(memoryStats.layer_weight_bytes_staged)} · Returned to CPU: {formatMemoryBytes(memoryStats.layer_weight_bytes_returned_to_cpu)} · Disk reads: {memoryStats.disk_layer_weight_loads} tensor(s) / {formatMemoryBytes(memoryStats.disk_layer_weight_bytes_read)} · KV cache copied to CPU: {formatMemoryBytes(memoryStats.kv_cache_bytes_offloaded)}</p>}
        {!deviceMemory?.device_allocator_metrics_supported && <p className="quiet-note">This backend does not expose allocator peak metrics through the installed PyTorch runtime. Null metrics are intentionally shown as "not exposed" rather than estimated.</p>}
      </section>}
    </section>}
    <p className="quiet-note"><Cpu size={14} /> AXIOM owns this runtime and the LoRA training loop; the underlying pretrained base weights retain their original model licence.</p>
  </section>;
}
