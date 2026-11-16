import { useEffect, useState } from "react";
import { Activity, Download, Pause, Play, Trash2 } from "lucide-react";
import { axiomApi, type AxiomEngineSystemTelemetry } from "../api/client";

const MAX_SAMPLES = 120;
const INTERVAL_OPTIONS = [1, 2, 5, 10] as const;

function formatBytes(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "not exposed";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatPercent(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "not exposed" : `${value.toFixed(1)}%`;
}

function formatSampleTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString();
}

export default function EngineTelemetryMonitor() {
  const [samples, setSamples] = useState<AxiomEngineSystemTelemetry[]>([]);
  const [recording, setRecording] = useState(false);
  const [intervalSeconds, setIntervalSeconds] = useState<number>(2);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    if (!recording) return undefined;
    let disposed = false;
    let inFlight = false;

    const collect = async () => {
      if (inFlight || disposed) return;
      inFlight = true;
      try {
        const sample = await axiomApi.axiomEngineTelemetry();
        if (!disposed) {
          setSamples((current) => [...current, sample].slice(-MAX_SAMPLES));
          setError("");
        }
      } catch (cause) {
        if (!disposed) {
          setError(cause instanceof Error ? cause.message : "Resource telemetry sample failed.");
        }
      } finally {
        inFlight = false;
      }
    };

    void collect();
    const timer = window.setInterval(() => { void collect(); }, intervalSeconds * 1000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [recording, intervalSeconds]);

  const exportHistory = () => {
    if (samples.length === 0) return;
    const payload = JSON.stringify({
      exported_at: new Date().toISOString(),
      schema_version: 1,
      sample_count: samples.length,
      interval_seconds: intervalSeconds,
      retention: { mode: "browser-memory-only", maximum_samples: MAX_SAMPLES },
      privacy: {
        prompt_content_included: false,
        generated_content_included: false,
        model_paths_included: false,
        environment_values_included: false,
        credentials_included: false,
        network_access_performed: false,
      },
      samples,
    }, null, 2);
    const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "axiom-resource-telemetry.json";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Exported the current in-memory resource samples.");
  };

  const clearHistory = () => {
    setRecording(false);
    setSamples([]);
    setError("");
    setNotice("Cleared the browser's in-memory resource history.");
  };

  const latest = samples.length > 0 ? samples[samples.length - 1] : undefined;
  const cpuSamples = samples
    .map((sample) => sample.cpu.system_percent)
    .filter((value): value is number => value !== null && Number.isFinite(value));
  const cpuPoints = cpuSamples.map((value, index) => {
    const x = cpuSamples.length === 1 ? 320 : 20 + (index * 600) / (cpuSamples.length - 1);
    const y = 76 - (Math.max(0, Math.min(100, value)) / 100) * 60;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  return (
    <section className="surface" aria-labelledby="engine-telemetry-title" style={{ marginTop: 18, padding: 18, border: "1px solid var(--line-strong)", borderRadius: 2, background: "var(--surface)" }}>
      <div className="surface-head">
        <div>
          <p className="surface-eyebrow">OPT-IN · BOUNDED · BROWSER MEMORY</p>
          <h3 id="engine-telemetry-title"><Activity size={17} style={{ verticalAlign: "middle", marginRight: 8 }} />Resource monitoring</h3>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <label className="field" style={{ minWidth: 130, margin: 0 }}>
            <span className="field-label-row"><span>Sample interval</span></span>
            <select aria-label="Telemetry sample interval" value={intervalSeconds} onChange={(event) => setIntervalSeconds(Number(event.target.value))} disabled={recording}>
              {INTERVAL_OPTIONS.map((seconds) => <option key={seconds} value={seconds}>{seconds} second{seconds === 1 ? "" : "s"}</option>)}
            </select>
          </label>
          <button type="button" className="button primary-button" onClick={() => { setError(""); setNotice(""); setRecording((value) => !value); }}>
            {recording ? <><Pause size={14} />Stop monitoring</> : <><Play size={14} />Start monitoring</>}
          </button>
          <button type="button" className="button secondary-button" onClick={exportHistory} disabled={samples.length === 0}><Download size={14} />Export JSON</button>
          <button type="button" className="button danger-button" onClick={clearHistory} disabled={samples.length === 0 && !recording}><Trash2 size={14} />Clear history</button>
        </div>
      </div>
      <p className="quiet-note">Monitoring starts only when you select Start. The browser retains at most {MAX_SAMPLES} recent samples while this page is open; stopping or leaving the page ends sampling. No server-side history is created. Prompt text, generated content, model paths, environment values, and credentials are excluded by the telemetry API.</p>
      {error && <div className="inline-alert warning" role="alert">{error}</div>}
      {notice && <div className="inline-alert success" role="status">{notice}</div>}
      <div className="info-grid">
        <div className="panel-surface" style={{ padding: 14 }}>
          <p className="surface-eyebrow">SYSTEM CPU</p>
          <strong style={{ fontSize: 22 }}>{formatPercent(latest?.cpu.system_percent ?? null)}</strong>
          <p className="quiet-note">Latest reported sample</p>
        </div>
        <div className="panel-surface" style={{ padding: 14 }}>
          <p className="surface-eyebrow">AVAILABLE RAM</p>
          <strong style={{ fontSize: 22 }}>{formatBytes(latest?.memory.available_bytes ?? null)}</strong>
          <p className="quiet-note">{latest ? formatPercent(latest.memory.percent) + " used" : "No sample yet"}</p>
        </div>
        <div className="panel-surface" style={{ padding: 14 }}>
          <p className="surface-eyebrow">PROCESS RSS</p>
          <strong style={{ fontSize: 22 }}>{formatBytes(latest?.process.rss_bytes ?? null)}</strong>
          <p className="quiet-note">Current process resident memory</p>
        </div>
        <div className="panel-surface" style={{ padding: 14 }}>
          <p className="surface-eyebrow">RETAINED SAMPLES</p>
          <strong style={{ fontSize: 22 }}>{samples.length}/{MAX_SAMPLES}</strong>
          <p className="quiet-note">{recording ? "Sampling is active" : "Sampling is stopped"}</p>
        </div>
      </div>
      {samples.length > 0 ? <>
        <div className="panel-surface" style={{ marginTop: 14, padding: 14 }}>
          <p className="surface-eyebrow">SYSTEM CPU · RECENT SAMPLES</p>
          {cpuSamples.length > 1
            ? <svg viewBox="0 0 640 90" role="img" aria-label="System CPU utilisation over retained telemetry samples" style={{ width: "100%", height: 110 }}>
                <line x1="20" y1="16" x2="620" y2="16" stroke="var(--line)" strokeDasharray="3 4" />
                <line x1="20" y1="46" x2="620" y2="46" stroke="var(--line)" strokeDasharray="3 4" />
                <line x1="20" y1="76" x2="620" y2="76" stroke="var(--line)" strokeDasharray="3 4" />
                <polyline points={cpuPoints} fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
              </svg>
            : <p className="quiet-note">At least two CPU samples are required to draw a trend line; missing CPU values are not estimated.</p>}
        </div>
        <div style={{ overflowX: "auto", marginTop: 14 }}>
          <table className="data-table" aria-label="Resource telemetry samples">
            <thead><tr><th>Sampled at</th><th>System CPU</th><th>Available RAM</th><th>Process RSS</th><th>Active device</th></tr></thead>
            <tbody>{samples.slice(-8).reverse().map((sample, index) => <tr key={`${sample.sampled_at}-${index}`}>
              <td>{formatSampleTime(sample.sampled_at)}</td>
              <td>{formatPercent(sample.cpu.system_percent)}</td>
              <td>{formatBytes(sample.memory.available_bytes)}</td>
              <td>{formatBytes(sample.process.rss_bytes)}</td>
              <td>{sample.loaded_runtime.device || "none"}</td>
            </tr>)}</tbody>
          </table>
        </div>
      </> : <div className="empty-state"><Activity size={22} /><h4>No resource samples collected</h4><p>Start monitoring to collect a bounded local resource history. Sampling remains off until you opt in.</p></div>}
    </section>
  );
}
