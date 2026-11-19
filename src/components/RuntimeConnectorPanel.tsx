import { useEffect, useState } from "react";
import { CheckCircle2, CircleHelp, RefreshCcw, XCircle } from "lucide-react";
import { axiomApi, type RuntimeConnectorReport } from "../api/client";

export default function RuntimeConnectorPanel() {
  const [report, setReport] = useState<RuntimeConnectorReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      setReport(await axiomApi.runtimeConnectors());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Runtime connector inventory is unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  return (
    <section className="surface runtime-connectors">
      <div className="surface-head">
        <div>
          <p className="surface-eyebrow">PASSIVE LOCAL DISCOVERY</p>
          <h2>Runtime connectors</h2>
        </div>
        <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading}>
          <RefreshCcw size={14} /> {loading ? "Checking" : "Refresh"}
        </button>
      </div>
      <p className="runtime-connectors-disclaimer">
        This checks for known executables and installed Python packages only. It does not start processes,
        contact endpoints, load models, or prove compatibility/readiness.
      </p>
      {error && <div className="inline-alert warning" role="alert">{error}</div>}
      {loading && <p className="loading-state" role="status">Inspecting local executable/package presence…</p>}
      {!loading && report && (
        <>
          <div className="runtime-connector-list">
            {report.connectors.map((connector) => (
              <div className="runtime-connector-row" key={connector.id}>
                <span className={`runtime-connector-mark ${connector.present ? "present" : "missing"}`} aria-hidden="true">
                  {connector.present ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                </span>
                <div className="runtime-connector-copy">
                  <b>{connector.name}</b>
                  <small>{connector.evidence} · {connector.kind}</small>
                </div>
                <span className={`status-pill ${connector.present ? "success" : "neutral"}`}>
                  <i />{connector.present ? "DISCOVERED" : "NOT FOUND"}
                </span>
              </div>
            ))}
          </div>
          <p className="quiet-note"><CircleHelp size={13} /> Discovery report generated {new Date(report.checked_at).toLocaleString()}. No connection was attempted.</p>
        </>
      )}
    </section>
  );
}
