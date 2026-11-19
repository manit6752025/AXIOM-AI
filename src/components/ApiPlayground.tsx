import { useState } from "react";
import {
  axiomApi,
  playgroundEndpoints,
  type PlaygroundEndpoint,
} from "../api/client";

type DeepResult = {
  path: PlaygroundEndpoint;
  label: string;
  ok: boolean;
  elapsedMs: number;
  summary: string;
};

function summarize(value: unknown): string {
  if (Array.isArray(value)) return `Array response · ${value.length} item(s)`;

  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const details = [
      "status",
      "version",
      "api_contract_version",
      "application_version",
      "total",
      "count",
    ]
      .filter((key) => record[key] !== undefined)
      .map((key) => `${key}: ${String(record[key])}`);

    if (details.length) return details.join(" · ");
    const keys = Object.keys(record).slice(0, 6);
    return keys.length ? `Object response · ${keys.join(", ")}` : "Empty object response";
  }

  if (typeof value === "string") return `Text response · ${value.length} characters`;
  return `${typeof value} response`;
}

export default function ApiPlayground() {
  const [selected, setSelected] = useState<PlaygroundEndpoint>(
    playgroundEndpoints[0].path,
  );
  const [response, setResponse] = useState<unknown>(null);
  const [hasResponse, setHasResponse] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [deepLoading, setDeepLoading] = useState(false);
  const [deepResults, setDeepResults] = useState<DeepResult[]>([]);
  const [deepComplete, setDeepComplete] = useState(false);

  const endpoint =
    playgroundEndpoints.find((item) => item.path === selected) ??
    playgroundEndpoints[0];

  async function runRequest() {
    setLoading(true);
    setError("");
    setHasResponse(false);

    try {
      const result = await axiomApi.playgroundRead(selected);
      setResponse(result);
      setHasResponse(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The API request failed.");
    } finally {
      setLoading(false);
    }
  }

  async function runDeepAnalysis() {
    setDeepLoading(true);
    setDeepComplete(false);
    setDeepResults([]);

    try {
      const results = await Promise.all(
        playgroundEndpoints.map(async (item): Promise<DeepResult> => {
          const started = performance.now();

          try {
            const value = await axiomApi.playgroundRead(item.path);
            return {
              path: item.path,
              label: item.label,
              ok: true,
              elapsedMs: Math.round(performance.now() - started),
              summary: summarize(value),
            };
          } catch (cause) {
            return {
              path: item.path,
              label: item.label,
              ok: false,
              elapsedMs: Math.round(performance.now() - started),
              summary:
                cause instanceof Error ? cause.message : "Request failed.",
            };
          }
        }),
      );

      setDeepResults(results);
      setDeepComplete(true);
    } finally {
      setDeepLoading(false);
    }
  }

  const passed = deepResults.filter((item) => item.ok).length;

  return (
    <section className="feature-page api-playground">
      <header className="page-header">
        <div>
          <p className="eyebrow">API INSPECTION / READ-ONLY</p>
          <h1>API Playground</h1>
          <p className="page-description">
            Inspect allowlisted local API endpoints, view JSON responses, and
            run a bounded diagnostic sweep.
          </p>
        </div>
      </header>

      <section className="surface">
        <div className="surface-head">
          <div>
            <p className="surface-eyebrow">ENDPOINT CATALOGUE</p>
            <h2>Choose an endpoint</h2>
          </div>
        </div>

        <p className="quiet-note">
          Requests use the predefined API allowlist, support GET only, and use the
          shared API client timeout. Arbitrary URLs and mutating methods are not
          supported.
        </p>

        <div className="field">
          <label htmlFor="api-endpoint">Endpoint</label>
          <div className="input-shell">
            <select
              id="api-endpoint"
              aria-label="Endpoint"
              value={selected}
              onChange={(event) => {
                setSelected(event.target.value as PlaygroundEndpoint);
                setHasResponse(false);
                setError("");
              }}
            >
              {playgroundEndpoints.map((item) => (
                <option key={item.path} value={item.path}>
                  {item.path} — {item.label}
                </option>
              ))}
            </select>
          </div>
          <small>{endpoint.description}</small>
        </div>

        <div className="page-actions">
          <button
            type="button"
            className="button primary-button"
            onClick={() => void runRequest()}
            disabled={loading || deepLoading}
          >
            {loading ? "Requesting…" : "Run read-only request"}
          </button>
          <button
            type="button"
            className="button secondary-button"
            onClick={() => void runDeepAnalysis()}
            disabled={loading || deepLoading}
          >
            {deepLoading ? "Analysing endpoints…" : "Run Deep Analysis"}
          </button>
        </div>

        {error && <div className="inline-alert danger" role="alert">{error}</div>}

        {loading && <p role="status">Waiting for the local API…</p>}

        {hasResponse && (
          <div>
            <p className="surface-eyebrow">RESPONSE / JSON</p>
            <pre className="api-playground-json">
              {JSON.stringify(response, null, 2)}
            </pre>
          </div>
        )}
      </section>

      <section className="surface">
        <div className="surface-head">
          <div>
            <p className="surface-eyebrow">BOUNDED ENDPOINT SWEEP</p>
            <h2>Deep Analysis</h2>
          </div>
          {deepComplete && (
            <span className={`status-pill ${passed === deepResults.length ? "success" : "warning"}`}>
              {passed}/{deepResults.length} passed
            </span>
          )}
        </div>

        <p className="quiet-note">
          Tests each allowlisted GET endpoint once, in parallel. Reports
          reachability, elapsed time, and a concise response summary. It does
          not run model inference or modify workspace data.
        </p>

        {deepLoading && <p role="status">Checking allowlisted endpoints…</p>}

        {deepComplete && (
          <div className="event-list" aria-live="polite">
            {deepResults.map((item) => (
              <div className="event-row" key={item.path}>
                <div>
                  <b>{item.label}</b>
                  <small>{item.path} · {item.elapsedMs} ms</small>
                  <small>{item.summary}</small>
                </div>
                <span className={`status-pill ${item.ok ? "success" : "warning"}`}>
                  {item.ok ? "PASS" : "FAIL"}
                </span>
              </div>
            ))}
          </div>
        )}

        {!deepComplete && !deepLoading && (
          <p className="quiet-note">Run Deep Analysis to inspect every allowed endpoint.</p>
        )}
      </section>
    </section>
  );
}
