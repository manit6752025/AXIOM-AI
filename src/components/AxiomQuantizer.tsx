import React, { useEffect, useMemo, useState } from "react";
import { axiomApi, type OptimizerProfile, type OptimizationPlan } from "../api/client";

type Profile = OptimizerProfile;
type Plan = OptimizationPlan;

export default function AxiomQuantizer() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profile, setProfile] = useState(1);
  const [model, setModel] = useState("");
  const [target, setTarget] = useState(10);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    axiomApi.optimizerProfiles()
      .then(setProfiles)
      .catch(() => setProfiles([]));
  }, []);

  const selected = useMemo(
    () => profiles.find((item) => item.number === profile),
    [profiles, profile],
  );
  const estimatedFp16 = plan?.model.estimated_fp16_gb ?? null;

  async function optimize() {
    if (!model.trim()) return;

    setBusy(true);
    setError("");
    setPlan(null);

    try {
      setPlan(await axiomApi.optimizationPlan(model, profile, target));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Optimizer failed",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="axiom-quantizer">
      <div className="axiom-quantizer-header">
        <div>
          <div className="axiom-quantizer-eyebrow">
            AXIOM / QUANTIZATION LAB
          </div>
          <h2>Make a large model fit your device</h2>
          <p>
            Choose what the agent needs to do, select a model, and
            AXIOM calculates a device-aware optimization plan.
          </p>
        </div>

        <div className="axiom-quantizer-target">
          <span>TARGET</span>
          <strong>{target} TOK/S</strong>
        </div>
      </div>

      <div className="axiom-quantizer-grid">
        <label>
          Agent profile
          <select
            value={profile}
            onChange={(event) =>
              setProfile(Number(event.target.value))
            }
          >
            {profiles.length > 0 ? (
              profiles.map((item) => (
                <option
                  key={item.number}
                  value={item.number}
                >
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
          Hugging Face model / local model
          <input
            value={model}
            onChange={(event) =>
              setModel(event.target.value)
            }
            placeholder="Qwen/Qwen3-8B"
          />
        </label>

        <label>
          Performance target
          <select
            value={target}
            onChange={(event) =>
              setTarget(Number(event.target.value))
            }
          >
            <option value={5}>5 tok/s</option>
            <option value={10}>10 tok/s</option>
            <option value={15}>15 tok/s</option>
            <option value={20}>20 tok/s</option>
          </select>
        </label>
      </div>

      {selected && (
        <div className="axiom-quantizer-profile">
          <strong>{selected.name}</strong>
          <span>{selected.description}</span>
        </div>
      )}

      <button
        className="axiom-quantizer-button"
        onClick={optimize}
        disabled={busy || !model.trim()}
      >
        {busy ? "ANALYZING MODEL..." : "ANALYZE & OPTIMIZE"}
      </button>

      {plan && (
        <div className="axiom-quantizer-result">
          <div className="axiom-quantizer-result-title">
            Optimization result
          </div>

          <div className="axiom-quantizer-stats">
            <span>Model</span>
            <b>{plan.model.source}</b>

            <span>Parameters</span>
            <b>{plan.model.parameter_billions === null ? "Unknown" : `${plan.model.parameter_billions}B`}</b>

            <span>Parameter basis</span>
            <b>{plan.model.parameter_source}</b>

            <span>Original FP16 estimate</span>
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

          <div className="axiom-quantizer-compare">
            <h3>Estimated weight sizes</h3>
            <table>
              <thead>
                <tr>
                  <th>Format</th>
                  <th>Size</th>
                  <th>Fit</th>
                </tr>
              </thead>
              <tbody>
                {estimatedFp16 === null ? (
                  <tr><td colSpan={3}>Model parameter estimate unavailable.</td></tr>
                ) : [
                  { label: "FP16", factor: 1, key: "fp16" },
                  { label: "INT8", factor: 0.5, key: "int8" },
                  { label: "INT4", factor: 0.25, key: "int4" },
                ].map((row) => {
                  const size = estimatedFp16 * row.factor;
                  const budget = plan.system.vram_gb ?? plan.system.ram_gb;
                  const fit = budget === null
                    ? "unknown"
                    : size <= budget * 0.7 ? "yes" : size <= budget ? "tight" : "no";
                  const recommended = plan.optimization.recommended_quantization
                    .toLowerCase()
                    .includes(row.key);
                  return (
                    <tr key={row.key} className={recommended ? "recommended" : undefined}>
                      <td>{row.label}</td>
                      <td>{size.toFixed(1)} GB</td>
                      <td className={`fit-${fit}`}>
                        {fit === "unknown" ? "unknown" : fit === "yes" ? "fits" : fit === "tight" ? "tight" : "too large"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="axiom-quantizer-disclaimer">
            AXIOM can recommend and prepare a runtime representation.
            A real benchmark is required to verify actual throughput.
          </div>
        </div>
      )}

      {error && (
        <div className="axiom-quantizer-error">
          {error}
        </div>
      )}
    </section>
  );
}
