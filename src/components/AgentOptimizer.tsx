import { useEffect, useMemo, useState } from "react";
import { BrainCircuit, Cpu, Database, Gauge, HardDrive, Play, Sparkles, Zap } from "lucide-react";

type Goal = "coding" | "cybersecurity" | "assistant" | "reasoning" | "research" | "agentic" | "creative" | "custom";
type Priority = "speed" | "balanced" | "quality";
type Target = "5" | "10" | "20" | "max";
type ApiResult = Record<string, unknown>;

const goals: Array<{ id: Goal; label: string }> = [
  { id: "coding", label: "Coding" },
  { id: "cybersecurity", label: "Cybersecurity" },
  { id: "assistant", label: "General Assistant" },
  { id: "reasoning", label: "Reasoning" },
  { id: "research", label: "Research" },
  { id: "agentic", label: "Tool / Agentic AI" },
  { id: "creative", label: "Creative / Writing" },
  { id: "custom", label: "Custom" },
];

function text(value: unknown, fallback = "Unknown") {
  return typeof value === "string" && value.length ? value : fallback;
}

export default function AgentOptimizer() {
  const [goal, setGoal] = useState<Goal>("assistant");
  const [priority, setPriority] = useState<Priority>("balanced");
  const [target, setTarget] = useState<Target>("10");
  const [context, setContext] = useState("8192");
  const [model, setModel] = useState("");
  const [dataset, setDataset] = useState("");
  const [busy, setBusy] = useState(false);
  const [hardware, setHardware] = useState<ApiResult | null>(null);
  const [plan, setPlan] = useState<ApiResult | null>(null);
  const [error, setError] = useState("");

  const targetLabel = useMemo(() => target === "max" ? "Maximum possible" : `${target} tokens/sec`, [target]);

  useEffect(() => {
    let active = true;
    fetch("/api/optimization/hardware", { credentials: "include" })
      .then((response) => {
        if (!response.ok) throw new Error("Hardware request failed");
        return response.json() as Promise<ApiResult>;
      })
      .then((data) => {
        if (active) setHardware(data);
      })
      .catch(() => {
        if (active) setHardware(null);
      });
    return () => {
      active = false;
    };
  }, []);

  async function runOptimizer() {
    setBusy(true);
    setError("");
    setPlan(null);
    try {
      const [hardwareResponse, planResponse] = await Promise.all([
        fetch("/api/optimization/hardware", { credentials: "include" }),
        fetch("/api/optimization/plan", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: model.trim(), goal, priority, target_tps: target === "max" ? null : Number(target), context_length: Number(context), dataset_path: dataset.trim() || null }),
        }),
      ]);
      if (!hardwareResponse.ok || !planResponse.ok) throw new Error("The AXIOM optimization API is not available or rejected the request.");
      setHardware((await hardwareResponse.json()) as ApiResult);
      setPlan((await planResponse.json()) as ApiResult);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Optimization request failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="optimizer-page">
      <div className="optimizer-hero">
        <div>
          <p className="eyebrow">BETA 6 / AGENT MODEL OPTIMIZER</p>
          <h1>Make the model fit the machine.</h1>
          <p>Tell AXIOM what you are building and what performance matters. AXIOM inspects the model and hardware, selects an appropriate optimization path, and leaves throughput unverified until a real benchmark runs.</p>
        </div>
        <div className="optimizer-hero-stat"><Zap size={18} /><span>{targetLabel}</span></div>
      </div>

      <div className="optimizer-grid">
        <section className="surface optimizer-panel">
          <div className="surface-heading"><div><span className="surface-eyebrow">01 / MODEL + GOAL</span><h2>Optimization questionnaire</h2></div><Sparkles size={18} /></div>

          <label className="optimizer-field"><span>Model ID or local model path</span><input value={model} onChange={(event) => setModel(event.target.value)} placeholder="Qwen/Qwen3-8B or /models/model" /></label>

          <div className="optimizer-label">What are you building?</div>
          <div className="optimizer-goals">{goals.map((item) => <button key={item.id} type="button" className={`optimizer-choice ${goal === item.id ? "active" : ""}`} onClick={() => setGoal(item.id)}>{item.label}</button>)}</div>

          <div className="optimizer-form-grid">
            <label className="optimizer-field"><span>Priority</span><select value={priority} onChange={(event) => setPriority(event.target.value as Priority)}><option value="speed">Maximum speed</option><option value="balanced">Balanced</option><option value="quality">Maximum quality</option></select></label>
            <label className="optimizer-field"><span>Target throughput</span><select value={target} onChange={(event) => setTarget(event.target.value as Target)}><option value="5">5 TPS</option><option value="10">10 TPS</option><option value="20">20 TPS</option><option value="max">Maximum possible</option></select></label>
            <label className="optimizer-field"><span>Context length</span><input inputMode="numeric" value={context} onChange={(event) => setContext(event.target.value.replace(/[^0-9]/g, ""))} /></label>
            <label className="optimizer-field"><span>Optional training dataset</span><input value={dataset} onChange={(event) => setDataset(event.target.value)} placeholder="/data/train.jsonl" /></label>
          </div>

          <button type="button" className="button primary-button optimizer-run" onClick={runOptimizer} disabled={busy || !model.trim()}><Play size={16} />{busy ? "Analyzing hardware + model..." : "Analyze & optimize"}</button>
          {error && <div className="optimizer-error">{error}</div>}
        </section>

        <section className="surface optimizer-panel">
          <div className="surface-heading"><div><span className="surface-eyebrow">02 / HARDWARE</span><h2>Detected environment</h2></div><Cpu size={18} /></div>
          <div className="optimizer-metrics">
            <div><Cpu size={16} /><span>CPU</span><b>{text(hardware?.cpu_name)}</b></div>
            <div><HardDrive size={16} /><span>RAM</span><b>{hardware?.ram_gb ? `${hardware.ram_gb} GB` : "Unknown"}</b></div>
            <div><Gauge size={16} /><span>GPU</span><b>{text(hardware?.gpu_name, "No GPU reported")}</b></div>
            <div><Database size={16} /><span>VRAM</span><b>{hardware?.vram_gb ? `${hardware.vram_gb} GB` : "Unknown"}</b></div>
          </div>

          <div className="optimizer-result">
            <div className="surface-eyebrow">03 / PLAN</div>
            {!plan ? <p>Submit the questionnaire to generate the hardware-aware optimization plan.</p> : (
              <>
                <div className="optimizer-result-row"><span>Quantization</span><strong>{text(plan.quantization, "Auto")}</strong></div>
                <div className="optimizer-result-row"><span>Context</span><strong>{text(plan.context_length ?? plan.recommended_context, context)}</strong></div>
                <div className="optimizer-result-row"><span>Dataset action</span><strong>{text(plan.dataset_action, dataset ? "Analyze relevance before retraining" : "Not requested")}</strong></div>
                <div className="optimizer-result-row"><span>Throughput</span><strong>Benchmark required</strong></div>
              </>
            )}
          </div>
          <div className="optimizer-note"><BrainCircuit size={15} /> AXIOM never presents estimated TPS as a measured benchmark.</div>
        </section>
      </div>
    </div>
  );
}
