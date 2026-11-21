import { useEffect, useState } from "react";
import { axiomApi, type HardwareSnapshotHistory } from "../api/client";

function memoryLabel(bytes: number | null): string {
  return bytes === null ? "Not reported" : `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
}

export default function HardwareSnapshotPanel() {
  const [history, setHistory] = useState<HardwareSnapshotHistory | null>(null);
  const [maxSnapshots, setMaxSnapshots] = useState(20);
  const [maxAgeDays, setMaxAgeDays] = useState(90);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const refresh = async () => {
    setBusy(true);
    setError("");
    try {
      const result = await axiomApi.hardwareSnapshots();
      setHistory(result);
      setMaxSnapshots(result.settings.max_snapshots);
      setMaxAgeDays(result.settings.max_age_days);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Hardware snapshot history is unavailable.");
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    let active = true;
    void axiomApi.hardwareSnapshots()
      .then((result) => {
        if (!active) return;
        setHistory(result);
        setMaxSnapshots(result.settings.max_snapshots);
        setMaxAgeDays(result.settings.max_age_days);
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Hardware snapshot history is unavailable.");
      });
    return () => { active = false; };
  }, []);

  const saveSettings = async (enabled: boolean) => {
    setBusy(true); setError(""); setNotice("");
    try {
      const result = await axiomApi.updateHardwareSnapshotSettings({ enabled, max_snapshots: maxSnapshots, max_age_days: maxAgeDays });
      setHistory(result);
      setNotice(enabled ? "Hardware snapshot history enabled." : "Hardware snapshot capture disabled. Existing records were preserved.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Hardware snapshot settings could not be saved.");
    } finally { setBusy(false); }
  };

  const capture = async () => {
    setBusy(true); setError(""); setNotice("");
    try {
      const result = await axiomApi.captureHardwareSnapshot();
      setHistory(result.history);
      setNotice("Hardware snapshot captured and retained locally.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Hardware snapshot capture failed.");
    } finally { setBusy(false); }
  };

  const clear = async () => {
    if (!window.confirm("Delete all retained hardware snapshots from this workspace?")) return;
    setBusy(true); setError(""); setNotice("");
    try {
      const result = await axiomApi.clearHardwareSnapshots();
      setHistory(result);
      setNotice("Hardware snapshot history cleared.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Hardware snapshot history could not be cleared.");
    } finally { setBusy(false); }
  };

  const enabled = history?.settings.enabled === true;
  return (
    <section className="surface hardware-snapshot-panel" role="region" aria-label="Hardware snapshot history">
      <div className="surface-head">
        <div><p className="surface-eyebrow">OPT-IN · WORKSPACE-LOCAL</p><h2>Hardware snapshot history</h2></div>
        <span className={`status-pill ${enabled ? "success" : "neutral"}`}>{enabled ? "CAPTURE ENABLED" : "CAPTURE OFF"}</span>
      </div>
      <p className="quiet-note">Record an explicit point-in-time inventory of OS, CPU, RAM and detected GPU hardware. No automatic sampling, host identity, paths, credentials, network identifiers or workload data.</p>
      <div className="hardware-snapshot-settings">
        <label className="field"><span className="field-label-row"><span>Maximum retained snapshots</span></span>
          <input aria-label="Maximum retained snapshots" type="number" min={1} max={100} value={maxSnapshots} onChange={(event) => setMaxSnapshots(Number(event.target.value))} />
        </label>
        <label className="field"><span className="field-label-row"><span>Maximum snapshot age (days)</span></span>
          <input aria-label="Maximum snapshot age (days)" type="number" min={1} max={3650} value={maxAgeDays} onChange={(event) => setMaxAgeDays(Number(event.target.value))} />
        </label>
      </div>
      <div className="hardware-snapshot-actions">
        <button type="button" className="button secondary-button" onClick={() => void saveSettings(!enabled)} disabled={busy || !history}>{enabled ? "Disable history" : "Enable history"}</button>
        <button type="button" className="button secondary-button" onClick={() => void saveSettings(enabled)} disabled={busy || !history}>Save retention</button>
        <button type="button" className="button primary-button" onClick={() => void capture()} disabled={busy || !enabled}>Capture snapshot</button>
        <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={busy}>Refresh</button>
        <button type="button" className="button danger-button" onClick={() => void clear()} disabled={busy || !history?.total}>Clear history</button>
      </div>
      {error && <div className="inline-alert danger" role="alert">{error}</div>}
      {notice && <div className="inline-alert success" role="status">{notice}</div>}
      {!history && !error && <p className="quiet-note" role="status">Loading local snapshot history…</p>}
      {history && <>
        <div className="inline-meta hardware-snapshot-meta"><span>{history.total} retained snapshots</span><span>Maximum {history.settings.max_snapshots} entries</span><span>{history.settings.max_age_days} day retention</span></div>
        {history.snapshots.length === 0 ? <p className="quiet-note">No hardware snapshots have been captured. Enable history, then capture when you choose.</p> :
          <div className="hardware-snapshot-list">{history.snapshots.map((snapshot) => (
            <article className="hardware-snapshot-row" key={snapshot.id}>
              <div><b>{new Date(snapshot.captured_at).toLocaleString()}</b>
                <small>{snapshot.hardware.os_name} · {snapshot.hardware.architecture} · {snapshot.hardware.cpu_cores} CPU cores · {memoryLabel(snapshot.hardware.ram_bytes)} RAM</small>
                <small>{snapshot.hardware.gpu_name || "No GPU detected"} · {memoryLabel(snapshot.hardware.vram_bytes)} VRAM · CUDA {snapshot.hardware.cuda_available ? "available" : "not detected"}</small>
              </div>
              <code title={snapshot.fingerprint}>SHA-256 {snapshot.fingerprint.slice(0, 12)}</code>
            </article>
          ))}</div>}
        <p className="quiet-note">{history.scope_note}</p>
      </>}
    </section>
  );
}
