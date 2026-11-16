import { useEffect, useState } from "react";
import { Check, Download, RefreshCcw, Search, ShieldCheck, Trash2 } from "lucide-react";
import { axiomApi, type RuntimeTraceReport } from "../api/client";

function displayDate(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

export default function RuntimeTracePanel() {
  const [report, setReport] = useState<RuntimeTraceReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [showClearConfirmation, setShowClearConfirmation] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [appliedSearch, setAppliedSearch] = useState("");
  const [loadingMore, setLoadingMore] = useState(false);

  const refresh = async (search = searchInput.trim()) => {
    setLoading(true);
    setError("");
    try {
      setReport(await axiomApi.runtimeTraces(50, 0, search));
      setAppliedSearch(search);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Runtime request traces are unavailable.");
    } finally {
      setLoading(false);
    }
  };

  const loadMore = async () => {
    if (!report?.has_more || loading || loadingMore) return;
    setLoadingMore(true);
    setError("");
    try {
      const next = await axiomApi.runtimeTraces(50, report.traces.length, appliedSearch);
      setReport((current) => current ? {
        ...next,
        traces: [...current.traces, ...next.traces],
      } : next);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Older runtime traces could not be loaded.");
    } finally {
      setLoadingMore(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const exportTraces = () => {
    if (!report) return;
    const payload = JSON.stringify({
      exported_at: new Date().toISOString(),
      schema_version: report.schema_version,
      privacy: report.privacy,
      traces: report.traces,
      scope_note: "This file contains request metadata only. AXIOM does not include prompt bodies or generated response text in traces.",
    }, null, 2);
    const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "axiom-runtime-traces.json";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    // Let the browser start the download before releasing its object URL.
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Exported the currently displayed metadata-only trace history.");
  };

  const clearTraces = async () => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await axiomApi.clearRuntimeTraces();
      await refresh();
      setShowClearConfirmation(false);
      setNotice(`Cleared ${result.cleared} in-memory trace records. Nothing was deleted from disk because traces are not persisted there.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Runtime trace history could not be cleared.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="surface" aria-labelledby="runtime-traces-title" style={{ marginTop: 18, padding: 18, border: "1px solid var(--line-strong)", borderRadius: 2, background: "var(--surface)" }}>
      <div className="surface-head">
        <div>
          <p className="surface-eyebrow">BOUNDED · IN-MEMORY · METADATA ONLY</p>
          <h3 id="runtime-traces-title"><ShieldCheck size={17} style={{ verticalAlign: "middle", marginRight: 8 }} />Runtime request trace history</h3>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading || busy}><RefreshCcw size={14} />{loading ? "Refreshing…" : "Refresh traces"}</button>
          <button type="button" className="button secondary-button" onClick={exportTraces} disabled={!report || report.traces.length === 0 || busy}><Download size={14} />Export JSON</button>
          <button type="button" className="button danger-button" onClick={() => setShowClearConfirmation(true)} disabled={!report || report.traces.length === 0 || busy}><Trash2 size={14} />Clear history</button>
        </div>
      </div>
      <p className="quiet-note">Successful and failed local inference/benchmark requests are correlated by request ID. This buffer retains at most {report?.max_retained ?? 200} recent events in process memory. Prompt text, generated responses, credentials, and environment variables are never stored in these records.</p>
      {error && <div className="inline-alert warning" role="alert">{error}</div>}
      {notice && <div className="inline-alert success" role="status"><Check size={15} /> {notice}</div>}
      <div className="event-filter"><Search size={15} /><input aria-label="Search runtime trace metadata" placeholder="Search request ID, operation, model, or status" value={searchInput} onChange={(event) => setSearchInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void refresh(searchInput.trim()); }} /><button type="button" className="button secondary-button" onClick={() => void refresh(searchInput.trim())} disabled={loading}>Search</button></div>
      {loading && !report ? <div className="loading-state" role="status">Loading runtime request traces…</div>
        : report?.traces.length ? <>
          <div className="info-grid">
            <div className="panel-surface" style={{ padding: 14 }}><p className="surface-eyebrow">RETAINED TRACES</p><strong style={{ fontSize: 22 }}>{report.total_retained}/{report.max_retained}</strong><p className="quiet-note">Newest entries first; buffer is memory-only</p></div>
            <div className="panel-surface" style={{ padding: 14 }}><p className="surface-eyebrow">SUCCESSFUL</p><strong style={{ fontSize: 22 }}>{report.traces.filter((trace) => trace.status === "success").length}</strong><p className="quiet-note">Among the displayed entries</p></div>
            <div className="panel-surface" style={{ padding: 14 }}><p className="surface-eyebrow">FAILED</p><strong style={{ fontSize: 22 }}>{report.traces.filter((trace) => trace.status === "error").length}</strong><p className="quiet-note">Error codes only, no raw messages</p></div>
          </div>
          <div style={{ overflowX: "auto", marginTop: 14 }}>
            <table className="data-table" aria-label="Runtime request trace history">
              <thead><tr><th>Request ID</th><th>Operation</th><th>Model</th><th>Status</th><th>Duration</th><th>Output tokens</th><th>Completed at</th></tr></thead>
              <tbody>{report.traces.map((trace) => <tr key={trace.request_id}>
                <td><code title={trace.request_id}>{trace.request_id.slice(0, 12)}…</code></td>
                <td>{trace.operation}</td><td>{trace.model}</td>
                <td><span className={`status-pill ${trace.status === "success" ? "success" : "danger"}`}>{trace.status.toUpperCase()}</span>{trace.error_code && <small style={{ display: "block" }}>{trace.error_code}</small>}</td>
                <td>{trace.duration_ms.toFixed(2)} ms</td>
                <td>{trace.completion_tokens ?? "—"}{trace.completion_tokens_per_second !== null ? <small style={{ display: "block" }}>{trace.completion_tokens_per_second} tok/s</small> : null}</td>
                <td>{displayDate(trace.completed_at)}</td>
              </tr>)}</tbody>
            </table>
          </div>
          <div className="pagination-actions">
            <small>Showing {report.traces.length} of {report.total} matching traces · {report.total_retained}/{report.max_retained} retained</small>
            {report.capacity_reached && <small>Oldest records may have rotated out of the bounded memory buffer.</small>}
            {report.has_more && <button type="button" className="button secondary-button" onClick={() => void loadMore()} disabled={loadingMore}>{loadingMore ? "Loading…" : "Load older traces"}</button>}
          </div>
        </> : <div className="empty-state"><ShieldCheck size={22} /><h4>No runtime traces retained</h4><p>Run local generation or a benchmark from this Runtime page, then refresh to inspect its request ID and timing metadata.</p></div>}
      {showClearConfirmation && <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setShowClearConfirmation(false); }}>
        <section className="dialog" role="dialog" aria-modal="true" aria-labelledby="clear-runtime-traces-title">
          <div className="dialog-head"><div><p className="eyebrow">CONFIRM CLEAR</p><h2 id="clear-runtime-traces-title">Clear runtime trace history?</h2></div></div>
          <p className="quiet-note">This clears the bounded in-memory trace buffer in the current API process. It cannot clear traces from other processes, and no trace file is written to disk.</p>
          <div className="integration-form-actions"><button type="button" className="button secondary-button" disabled={busy} onClick={() => setShowClearConfirmation(false)}>Cancel</button><button type="button" className="button danger-button" disabled={busy} onClick={() => void clearTraces()}>{busy ? "Clearing…" : "Clear in-memory history"}</button></div>
        </section>
      </div>}
    </section>
  );
}
