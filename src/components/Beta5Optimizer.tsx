import { useState } from "react";
import { Cpu, Gauge, HardDrive, Rocket, Sparkles, Zap } from "lucide-react";
type Task =
  | "coding"
  | "cybersecurity"
  | "general"
  | "reasoning"
  | "research"
  | "agentic"
  | "creative"
  | "custom";
type Priority = "speed" | "balanced" | "quality";
type Plan = {
  hardware?: {
    machine: string;
    cpu_count: number;
    ram_gb: number;
    gpu: string | null;
    vram_gb: number | null;
  };
  model?: {
    name: string;
    size_gb: number | null;
    estimated_params_b: number | null;
  };
  optimization?: {
    recommended_quantization: string;
    memory_budget_gb: number;
    target_tps: number | null;
    context_length: number;
    next_step: string;
  };
  dataset?: {
    requested: boolean;
    kept_candidates?: number;
    duplicates?: number;
    invalid?: number;
  };
};
const TASKS: Array<{ id: Task; label: string }> = [
  { id: "coding", label: "Coding" },
  { id: "cybersecurity", label: "Cybersecurity" },
  { id: "general", label: "General Assistant" },
  { id: "reasoning", label: "Reasoning" },
  { id: "research", label: "Research" },
  { id: "agentic", label: "Tool / Agentic AI" },
  { id: "creative", label: "Creative / Writing" },
  { id: "custom", label: "Custom" },
];
const API_BASE: string = import.meta.env.VITE_AXIOM_API_URL ?? "";
export default function Beta5Optimizer() {
  const [task, setTask] = useState<Task>("general");
  const [priority, setPriority] = useState<Priority>("balanced");
  const [target, setTarget] = useState("10");
  const [context, setContext] = useState("8192");
  const [model, setModel] = useState("");
  const [dataset, setDataset] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  async function optimize() {
    setBusy(true);
    setError("");
    setPlan(null);
    try {
      const r = await fetch(`${API_BASE}/api/optimizer/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model.trim(),
          task,
          priority,
          target_tps: target === "max" ? null : Number(target),
          context_length: Number(context),
          dataset_path: dataset.trim() || null,
        }),
      });
      const p: unknown = await r.json();
      if (!r.ok) {
        const d =
          typeof p === "object" && p !== null && "detail" in p
            ? String(p.detail)
            : "Optimization request failed.";
        throw new Error(d);
      }
      setPlan(p as Plan);
    } catch (e: unknown) {
      setError(
        e instanceof Error
          ? e.message
          : "Unable to reach the AXIOM optimizer API.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="beta5-optimizer-page">
      <div className="beta5-hero">
        <div>
          <p className="eyebrow">BETA 6 / AGENT MODEL OPTIMIZER</p>
          <h1>Fit a model to the machine you actually have.</h1>
          <p>
            Choose the workload, target and model. AXIOM inspects hardware,
            recommends quantization and reviews optional training data before a
            real benchmark.
          </p>
        </div>
        <div className="beta5-hero-badge">
          <Zap size={18} />
          REAL PLAN
        </div>
      </div>
      <div className="beta5-layout">
        <div className="beta5-panel">
          <div className="beta5-panel-head">
            <div>
              <span>01</span>
              <h2>Agent profile</h2>
            </div>
            <Sparkles size={18} />
          </div>
          <div className="beta5-agent-grid">
            {TASKS.map((x) => (
              <button
                key={x.id}
                type="button"
                className={`beta5-agent ${task === x.id ? "active" : ""}`}
                onClick={() => setTask(x.id)}
              >
                {x.label}
              </button>
            ))}
          </div>
          <label className="beta5-field">
            <span>Model ID or local model path</span>
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="Qwen/Qwen3-8B or models/my-model"
            />
          </label>
          <label className="beta5-field">
            <span>Training dataset (optional)</span>
            <input
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              placeholder="datasets/train.jsonl"
            />
          </label>
          <div className="beta5-two-fields">
            <label className="beta5-field">
              <span>Priority</span>
              <select
                value={priority}
                onChange={(e) => setPriority(e.target.value as Priority)}
              >
                <option value="speed">Maximum speed</option>
                <option value="balanced">Balanced</option>
                <option value="quality">Maximum quality</option>
              </select>
            </label>
            <label className="beta5-field">
              <span>Target</span>
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
              >
                <option value="5">5 tokens/sec</option>
                <option value="10">10 tokens/sec</option>
                <option value="20">20 tokens/sec</option>
                <option value="max">Maximum possible</option>
              </select>
            </label>
          </div>
          <label className="beta5-field">
            <span>Context length</span>
            <input
              type="number"
              min="256"
              max="131072"
              value={context}
              onChange={(e) => setContext(e.target.value)}
            />
          </label>
          {error && <div className="beta5-error">{error}</div>}
          <button
            type="button"
            className="button primary-button beta5-optimize-button"
            disabled={!model.trim() || busy}
            onClick={optimize}
          >
            <Rocket size={17} />
            {busy ? "Analyzing..." : "Generate optimized plan"}
          </button>
        </div>
        <div className="beta5-panel beta5-result-panel">
          <div className="beta5-panel-head">
            <div>
              <span>02</span>
              <h2>Recommendation</h2>
            </div>
            <Gauge size={18} />
          </div>
          {!plan ? (
            <div className="beta5-empty">
              <Cpu size={28} />
              <h3>Waiting for analysis</h3>
              <p>
                AXIOM will inspect CPU, RAM, GPU/VRAM, model size and optional
                dataset quality.
              </p>
            </div>
          ) : (
            <>
              <div className="beta5-hardware">
                <div>
                  <HardDrive size={16} />
                  <b>{plan.hardware?.machine ?? "Unknown hardware"}</b>
                  <span>
                    {plan.hardware?.cpu_count ?? "?"} CPU threads ·{" "}
                    {plan.hardware?.ram_gb ?? "?"} GB RAM
                  </span>
                </div>
                <div>
                  <Cpu size={16} />
                  <b>{plan.hardware?.gpu ?? "CPU / no NVIDIA GPU detected"}</b>
                  <span>
                    {plan.hardware?.vram_gb
                      ? `${plan.hardware.vram_gb} GB VRAM`
                      : "VRAM not reported"}
                  </span>
                </div>
              </div>
              <div className="beta5-result">
                <span>Quantization</span>
                <strong>
                  {plan.optimization?.recommended_quantization ?? "—"}
                </strong>
              </div>
              <div className="beta5-result">
                <span>Memory budget</span>
                <strong>{plan.optimization?.memory_budget_gb ?? "—"} GB</strong>
              </div>
              <div className="beta5-result">
                <span>Model</span>
                <strong>{plan.model?.name ?? "—"}</strong>
              </div>
              {plan.dataset?.requested && (
                <div className="beta5-dataset">
                  <b>Dataset analysis</b>
                  <p>
                    {plan.dataset.kept_candidates ?? 0} unique ·{" "}
                    {plan.dataset.duplicates ?? 0} duplicates ·{" "}
                    {plan.dataset.invalid ?? 0} invalid
                  </p>
                </div>
              )}
              <div className="beta5-status">
                <span>REAL BENCHMARK REQUIRED</span>
                <b>
                  Measured tokens/sec is shown only after a real runtime
                  benchmark.
                </b>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
