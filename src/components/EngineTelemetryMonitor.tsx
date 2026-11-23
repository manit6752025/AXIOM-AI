import { useEffect, useState } from "react";
import { Activity, Download, Pause, Play, Trash2 } from "lucide-react";
import { axiomApi, type AxiomEngineResourceTelemetryMetrics, type AxiomEngineResourceTelemetryRecord, type AxiomEngineSystemTelemetry } from "../api/client";

const MAX_BROWSER_SAMPLES = 120;
const INTERVAL_OPTIONS = [1, 2, 5, 10] as const;

type ResourceMonitorSample = AxiomEngineResourceTelemetryMetrics & {
  id?: string;
  fingerprint?: string;
};

function fromRuntimeSample(sample: AxiomEngineSystemTelemetry): ResourceMonitorSample {
  return {
    sampled_at: sample.sampled_at,
    sample_type: sample.sample_type,
    cpu: sample.cpu,
    memory: sample.memory,
    process: {
      rss_bytes: sample.process.rss_bytes,
      peak_rss_bytes: sample.process.peak_rss_bytes,
      threads: sample.process.threads,
    },
    devices: sample.devices.map((device) => ({
      available: device.available,
      memory_free_gib: device.memory_free_gib,
      memory_total_gib: device.memory_total_gib,
    })),
    loaded_runtime: {
      loaded: sample.loaded_runtime.loaded,
      device: sample.loaded_runtime.device,
      memory_mode: sample.loaded_runtime.memory_mode,
      precision: sample.loaded_runtime.precision,
      device_allocator: sample.loaded_runtime.device_allocator,
    },
  };
}

function fromHistoryRecord(record: AxiomEngineResourceTelemetryRecord): ResourceMonitorSample {
  return { ...record.metrics, id: record.id, fingerprint: record.fingerprint };
}

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
  const [samples, setSamples] = useState<ResourceMonitorSample[]>([]);
  const [recording, setRecording] = useState(false);
  const [intervalSeconds, setIntervalSeconds] = useState<number>(2);
  const [persistentHistoryEnabled, setPersistentHistoryEnabled] = useState(false);
  const [maxSamples, setMaxSamples] = useState(120);
  const [maxAgeDays, setMaxAgeDays] = useState(30);
  const [historyReady, setHistoryReady] = useState(false);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [includeCpu, setIncludeCpu] = useState(true);
  const [includeMemory, setIncludeMemory] = useState(true);
  const [includeProcess, setIncludeProcess] = useState(true);
  const [includeDevice, setIncludeDevice] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let active = true;
    void axiomApi.axiomEngineTelemetryHistory().then((history) => {
      if (!active) return;
      setPersistentHistoryEnabled(history.settings.enabled);
      setMaxSamples(history.settings.max_samples);
      setMaxAgeDays(history.settings.max_age_days);
      setSamples(history.samples.map(fromHistoryRecord));
    }).catch((cause) => {
      if (active) setError(cause instanceof Error ? cause.message : "Persistent resource history could not be loaded.");
    }).finally(() => {
      if (active) setHistoryReady(true);
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!recording) return undefined;
    let disposed = false;
    let inFlight = false;

    const collect = async () => {
      if (inFlight || disposed) return;
      inFlight = true;
      try {
        const sample = persistentHistoryEnabled
          ? (await axiomApi.recordAxiomEngineTelemetryHistory()).sample
          : null;
        const value = sample
          ? fromHistoryRecord(sample)
          : fromRuntimeSample(await axiomApi.axiomEngineTelemetry());
        if (!disposed) {
          const retentionLimit = persistentHistoryEnabled ? maxSamples : MAX_BROWSER_SAMPLES;
          setSamples((current) => [...current, value].slice(-retentionLimit));
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
  }, [recording, intervalSeconds, persistentHistoryEnabled, maxSamples]);

  const exportHistory = () => {
    if (samples.length === 0) return;
    const exportedSamples = samples.map((sample) => ({
      id: sample.id ?? null,
      fingerprint: sample.fingerprint ?? null,
      sampled_at: sample.sampled_at,
      sample_type: sample.sample_type,
      cpu: includeCpu ? sample.cpu : null,
      memory: includeMemory ? sample.memory : null,
      // Deliberately do not export the process PID.
      process: includeProcess ? {
        rss_bytes: sample.process.rss_bytes,
        peak_rss_bytes: sample.process.peak_rss_bytes,
        threads: sample.process.threads,
      } : null,
      devices: includeDevice ? sample.devices.map((device) => ({
        available: device.available,
        memory_free_gib: device.memory_free_gib,
        memory_total_gib: device.memory_total_gib,
      })) : null,
      loaded_runtime: {
        loaded: sample.loaded_runtime.loaded,
        device: includeDevice ? sample.loaded_runtime.device : null,
        memory_mode: includeDevice ? sample.loaded_runtime.memory_mode : null,
        precision: includeDevice ? sample.loaded_runtime.precision : null,
        device_allocator: includeDevice && sample.loaded_runtime.device_allocator ? {
          backend: sample.loaded_runtime.device_allocator.backend,
          current_allocated_bytes: sample.loaded_runtime.device_allocator.current_allocated_bytes,
          current_reserved_bytes: sample.loaded_runtime.device_allocator.current_reserved_bytes,
          peak_allocated_bytes: sample.loaded_runtime.device_allocator.peak_allocated_bytes,
          peak_reserved_bytes: sample.loaded_runtime.device_allocator.peak_reserved_bytes,
          peak_increment_bytes: sample.loaded_runtime.device_allocator.peak_increment_bytes,
          peak_reset_supported: sample.loaded_runtime.device_allocator.peak_reset_supported,
          device_allocator_metrics_supported: sample.loaded_runtime.device_allocator.device_allocator_metrics_supported,
          ...(includeProcess ? {
            process_rss_bytes: sample.loaded_runtime.device_allocator.process_rss_bytes,
            process_rss_delta_bytes: sample.loaded_runtime.device_allocator.process_rss_delta_bytes,
            process_peak_rss_bytes: sample.loaded_runtime.device_allocator.process_peak_rss_bytes,
            process_peak_rss_delta_bytes: sample.loaded_runtime.device_allocator.process_peak_rss_delta_bytes,
          } : {}),
        } : null,
      },
    }));
    const payload = JSON.stringify({
      exported_at: new Date().toISOString(),
      schema_version: 1,
      sample_count: samples.length,
      interval_seconds: intervalSeconds,
      retention: {
        mode: samples.some((sample) => sample.id) ? "workspace-persistent" : "browser-memory-only",
        enabled: persistentHistoryEnabled,
        maximum_samples: persistentHistoryEnabled ? maxSamples : MAX_BROWSER_SAMPLES,
        max_age_days: persistentHistoryEnabled ? maxAgeDays : null,
      },
      privacy: {
        prompt_content_included: false,
        generated_content_included: false,
        model_paths_included: false,
        environment_values_included: false,
        credentials_included: false,
        network_access_performed: false,
        cpu_metrics_included: includeCpu,
        memory_metrics_included: includeMemory,
        process_metrics_included: includeProcess,
        device_metrics_included: includeDevice,
        process_ids_included: false,
      },
      samples: exportedSamples,
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

  const exportCsv = () => {
    if (samples.length === 0) return;
    const columns: Array<{ name: string; value: (sample: ResourceMonitorSample) => string | number | null }> = [
      { name: "sampled_at", value: (sample) => sample.sampled_at },
      ...(includeCpu ? [
        { name: "system_cpu_percent", value: (sample: ResourceMonitorSample) => sample.cpu.system_percent },
        { name: "process_cpu_percent", value: (sample: ResourceMonitorSample) => sample.cpu.process_percent },
      ] : []),
      ...(includeMemory ? [
        { name: "system_ram_total_bytes", value: (sample: ResourceMonitorSample) => sample.memory.total_bytes },
        { name: "system_ram_available_bytes", value: (sample: ResourceMonitorSample) => sample.memory.available_bytes },
        { name: "system_ram_used_bytes", value: (sample: ResourceMonitorSample) => sample.memory.used_bytes },
        { name: "system_ram_used_percent", value: (sample: ResourceMonitorSample) => sample.memory.percent },
      ] : []),
      ...(includeProcess ? [
        { name: "process_rss_bytes", value: (sample: ResourceMonitorSample) => sample.process.rss_bytes },
        { name: "process_peak_rss_bytes", value: (sample: ResourceMonitorSample) => sample.process.peak_rss_bytes },
        { name: "process_threads", value: (sample: ResourceMonitorSample) => sample.process.threads },
      ] : []),
      ...(includeDevice ? [
        { name: "device_allocator_current_bytes", value: (sample: ResourceMonitorSample) => sample.loaded_runtime.device_allocator?.current_allocated_bytes ?? null },
        { name: "device_allocator_reserved_bytes", value: (sample: ResourceMonitorSample) => sample.loaded_runtime.device_allocator?.current_reserved_bytes ?? null },
      ] : []),
    ];
    const quote = (value: string | number | null) => `"${String(value ?? "").replace(/"/g, '""')}"`;
    const contents = [
      columns.map((column) => quote(column.name)).join(","),
      ...samples.map((sample) => columns.map((column) => quote(column.value(sample))).join(",")),
    ].join("\n") + "\n";
    const url = URL.createObjectURL(new Blob([contents], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "axiom-resource-telemetry.csv";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Exported the selected resource metrics as CSV.");
  };

  const saveHistorySettings = async () => {
    setHistoryBusy(true);
    setError("");
    try {
      const history = await axiomApi.configureAxiomEngineTelemetryHistory({
        enabled: persistentHistoryEnabled,
        max_samples: maxSamples,
        max_age_days: maxAgeDays,
      });
      setPersistentHistoryEnabled(history.settings.enabled);
      setSamples(history.samples.map(fromHistoryRecord));
      setNotice(history.settings.enabled
        ? "Persistent resource history is enabled for this workspace."
        : "Persistent resource history is disabled; existing samples are retained until cleared or expired.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Persistent resource history settings could not be saved.");
    } finally {
      setHistoryBusy(false);
    }
  };

  const clearHistory = async () => {
    setRecording(false);
    setHistoryBusy(true);
    setError("");
    try {
      if (persistentHistoryEnabled || samples.some((sample) => sample.id)) {
        await axiomApi.clearAxiomEngineTelemetryHistory();
        setNotice("Cleared persistent workspace resource history.");
      } else {
        setNotice("Cleared the browser's in-memory resource history.");
      }
      setSamples([]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Resource history could not be cleared.");
    } finally {
      setHistoryBusy(false);
    }
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
          <button type="button" className="button primary-button" onClick={() => { setError(""); setNotice(""); setRecording((value) => !value); }} disabled={!historyReady || historyBusy}>
            {recording ? <><Pause size={14} />Stop monitoring</> : <><Play size={14} />Start monitoring</>}
          </button>
          <button type="button" className="button secondary-button" onClick={exportHistory} disabled={samples.length === 0}><Download size={14} />Export JSON</button>
          <button type="button" className="button secondary-button" onClick={exportCsv} disabled={samples.length === 0}><Download size={14} />Export CSV</button>
          <button type="button" className="button danger-button" onClick={clearHistory} disabled={samples.length === 0 && !recording || historyBusy}><Trash2 size={14} />Clear history</button>
        </div>
      </div>
      <p className="quiet-note">Monitoring starts only when you select Start. Without persistent history, the browser retains at most {MAX_BROWSER_SAMPLES} recent samples while this page is open. Enabling workspace history stores only allowlisted resource metrics with sample-count and age retention; stopping or leaving the page still ends sampling. Prompt text, generated content, model names and paths, process IDs, environment values, and credentials are excluded.</p>
      <div className="telemetry-export-privacy" role="group" aria-label="Persistent resource history settings">
        <label><input type="checkbox" aria-label="Persist resource history in workspace" checked={persistentHistoryEnabled} onChange={(event) => setPersistentHistoryEnabled(event.target.checked)} disabled={recording || historyBusy || !historyReady} /> Enable persistent resource history for this workspace</label>
        <label>Maximum stored samples
          <select aria-label="Persistent history sample limit" value={maxSamples} onChange={(event) => setMaxSamples(Number(event.target.value))} disabled={recording || historyBusy || !historyReady}>
            {[120, 250, 500].map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label>Maximum history age
          <select aria-label="Persistent history retention days" value={maxAgeDays} onChange={(event) => setMaxAgeDays(Number(event.target.value))} disabled={recording || historyBusy || !historyReady}>
            {[7, 30, 90, 365].map((value) => <option key={value} value={value}>{value} days</option>)}
          </select>
        </label>
        <button type="button" className="button secondary-button" onClick={() => void saveHistorySettings()} disabled={!historyReady || historyBusy || recording}>Save persistent history settings</button>
        {historyBusy && <span role="status">Updating workspace history…</span>}
      </div>
      <div className="telemetry-export-privacy" role="group" aria-label="Telemetry export privacy controls">
        <strong>Export field controls</strong>
        <label><input type="checkbox" checked={includeCpu} onChange={(event) => setIncludeCpu(event.target.checked)} /> Include CPU metrics</label>
        <label><input type="checkbox" checked={includeMemory} onChange={(event) => setIncludeMemory(event.target.checked)} /> Include system RAM metrics</label>
        <label><input type="checkbox" checked={includeProcess} onChange={(event) => setIncludeProcess(event.target.checked)} /> Include process RSS/thread metrics</label>
        <label><input type="checkbox" checked={includeDevice} onChange={(event) => setIncludeDevice(event.target.checked)} /> Include device/allocator metrics</label>
      </div>
      <p className="quiet-note">Export controls apply to JSON and CSV files. Process IDs are never exported. Disabling a category omits it from the export; it does not change the live API snapshot. Prompt/response content, model paths, environment values and credentials are always excluded.</p>
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
          <strong style={{ fontSize: 22 }}>{samples.length}/{persistentHistoryEnabled ? maxSamples : MAX_BROWSER_SAMPLES}</strong>
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
