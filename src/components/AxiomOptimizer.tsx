import React, { useState } from "react";
import { axiomApi, type OptimizationPlan, type OptimizerProfile } from "../api/client";

type Profile = OptimizerProfile;
type Plan = OptimizationPlan;

export default function AxiomOptimizer() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profile, setProfile] = useState(1);
  const [model, setModel] = useState("");
  const [target, setTarget] = useState(10);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function loadProfiles() {
    try {
      setProfiles(await axiomApi.optimizerProfiles());
    } catch {
      setProfiles([]);
    }
  }

  React.useEffect(() => {
    void loadProfiles();
  }, []);

  async function optimize() {
    if (!model.trim()) return;

    setBusy(true);
    setError("");
    setPlan(null);

    try {
      setPlan(await axiomApi.optimizationPlan(model, profile, target));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Optimizer failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="axiom-optimizer">
      <div className="axiom-optimizer-card">
        <div className="axiom-optimizer-eyebrow">
          AXIOM / AGENT OPTIMIZER
        </div>

        <h2>Make a large model fit your device</h2>

        <p>
          Pick an agent workload, choose a model, set a throughput target,
          and AXIOM calculates a device-aware runtime configuration.
        </p>

        <div className="axiom-optimizer-grid">
          <label>
            Agent profile
            <select
              value={profile}
              onChange={(e) => setProfile(Number(e.target.value))}
            >
              {profiles.length ? (
                profiles.map((item) => (
                  <option key={item.number} value={item.number}>
                    {item.number}. {item.name}
                  </option>
                ))
              ) : (
                <>
                  <option value={1}>1. Coding Agent</option>
                  <option value={2}>2. Reasoning Agent</option>
                  <option value={3}>3. Research Agent</option>
                  <option value={4}>4. General Assistant</option>
                  <option value={5}>5. Automation Agent</option>
                  <option value={6}>6. Math Agent</option>
                  <option value={7}>7. Writing Agent</option>
                  <option value={8}>8. Multilingual Agent</option>
                </>
              )}
            </select>
          </label>

          <label>
            Hugging Face model / local path
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="Qwen/..."
            />
          </label>

          <label>
            Target tokens/sec
            <input
              type="number"
              min={1}
              max={1000}
              value={target}
              onChange={(e) => setTarget(Number(e.target.value))}
            />
          </label>
        </div>

        <button
          className="axiom-optimizer-primary"
          onClick={optimize}
          disabled={busy || !model.trim()}
        >
          {busy ? "ANALYZING..." : "OPTIMIZE MODEL"}
        </button>

        {plan && (
          <div className="axiom-optimizer-result">
            <div className="axiom-optimizer-title">
              {plan.profile.name}
            </div>

            <div className="axiom-optimizer-stats">
              <span>Model</span>
              <b>{plan.model.source}</b>

              <span>Parameters</span>
              <b>{plan.model.parameter_billions === null ? "Unknown" : `${plan.model.parameter_billions}B`}</b>

              <span>Parameter basis</span>
              <b>{plan.model.parameter_source}</b>

              <span>FP16 estimate</span>
              <b>{plan.model.estimated_fp16_gb === null ? "Unknown" : `${plan.model.estimated_fp16_gb} GB`}</b>

              <span>Recommended quantization</span>
              <b>{plan.optimization.recommended_quantization}</b>

              <span>RAM</span>
              <b>{plan.system.ram_gb === null ? "Unknown" : `${plan.system.ram_gb} GB`}</b>

              <span>VRAM</span>
              <b>{plan.system.vram_gb === null ? "Unknown" : `${plan.system.vram_gb} GB`}</b>

              <span>Target</span>
              <b>{plan.optimization.target_tokens_per_second} tok/s</b>
            </div>

            <div className="axiom-optimizer-note">
              AXIOM prunes non-runtime repository artifacts and retains
              inference-critical files and model weights. Throughput must
              be benchmarked on the actual device.
            </div>
          </div>
        )}

        {error && (
          <div className="axiom-optimizer-error">
            {error}
          </div>
        )}
      </div>
    </section>
  );
}
