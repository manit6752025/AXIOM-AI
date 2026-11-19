import { useState } from "react";

const endpoints = [
  { value: "/api/health", description: "Local service health and readiness." },
  { value: "/api/workspace/models", description: "Browse registered model metadata." },
  { value: "/api/workspace/datasets", description: "Browse dataset metadata." },
  { value: "/api/workspace/runtime", description: "Inspect local runtime status." },
  { value: "/api/workspace/events", description: "Inspect recent workspace events." },
  { value: "/api/workspace/diagnostics", description: "Inspect workspace diagnostics." },
  { value: "/api/workspace/workflows", description: "Inspect saved workflow designs." },
  { value: "/api/workspace/evaluations", description: "Inspect evaluation results." },
] as const;

export default function ApiPlayground() {
  const [selected, setSelected] = useState<string>(endpoints[0].value);
  const endpoint = endpoints.find((item) => item.value === selected) ?? endpoints[0];

  return (
    <section className="feature-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">API REFERENCE / READ-ONLY</p>
          <h1>API Playground</h1>
          <p className="page-description">
            Browse supported local API routes and their intended use.
            This reference does not send requests or change workspace data.
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

        <div className="field">
          <label htmlFor="api-endpoint">Endpoint</label>
          <div className="input-shell">
            <select
              id="api-endpoint"
              aria-label="Endpoint"
              value={selected}
              onChange={(event) => setSelected(event.target.value)}
            >
              {endpoints.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.value}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="event-list">
          <div className="event-row">
            <div>
              <b>Method</b>
              <small>GET · read-only reference</small>
            </div>
            <span className="status-pill success"><i />READ-ONLY</span>
          </div>
          <div className="event-row">
            <div>
              <b>Path</b>
              <small>{endpoint.value}</small>
            </div>
          </div>
          <p className="quiet-note">{endpoint.description}</p>
        </div>

        <div className="callout info">
          <div className="callout-icon">i</div>
          <div>
            <b>No request executed</b>
            <p>
              Selecting an endpoint only changes this reference view.
              It does not execute requests or mutate data.
            </p>
          </div>
        </div>
      </section>
    </section>
  );
}
