export type BenchmarkResourceSample = {
  id: string;
  sampled_at: string;
  model: string;
  device: string | null;
  cpu_percent: number | null;
  process_rss_delta_bytes: number | null;
  system_ram_used_delta_bytes: number | null;
  device_allocated_delta_bytes: number | null;
};

type MetricKey = "cpu_percent" | "process_rss_delta_bytes" | "system_ram_used_delta_bytes" | "device_allocated_delta_bytes";

const charts: Array<{ key: MetricKey; title: string; format: (value: number) => string }> = [
  { key: "cpu_percent", title: "System CPU after benchmark", format: (value) => `${value.toFixed(1)}%` },
  { key: "process_rss_delta_bytes", title: "Process RSS change", format: (value) => formatBytes(value) },
  { key: "system_ram_used_delta_bytes", title: "System RAM used change", format: (value) => formatBytes(value) },
  { key: "device_allocated_delta_bytes", title: "Device allocator change", format: (value) => formatBytes(value) },
];

function formatBytes(value: number): string {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  const abs = Math.abs(value);
  const unit = abs >= 1024 ** 3 ? "GiB" : abs >= 1024 ** 2 ? "MiB" : abs >= 1024 ? "KiB" : "B";
  const divisor = unit === "GiB" ? 1024 ** 3 : unit === "MiB" ? 1024 ** 2 : unit === "KiB" ? 1024 : 1;
  return sign + (abs / divisor).toFixed(unit === "B" ? 0 : 2) + " " + unit;
}

function TrendChart({ title, values, formatter }: { title: string; values: Array<{ id: string; value: number | null }>; formatter: (value: number) => string }) {
  const observed = values.map((item) => item.value).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const min = observed.length ? Math.min(0, ...observed) : 0;
  const max = observed.length ? Math.max(1, ...observed) : 1;
  const span = Math.max(1, max - min);
  const x = (index: number) => values.length < 2 ? 320 : 24 + index * 592 / (values.length - 1);
  const y = (value: number) => 82 - (value - min) / span * 66;
  const segments = values.slice(0, -1).flatMap((item, index) => {
    const next = values[index + 1];
    if (item.value === null || next.value === null || !Number.isFinite(item.value) || !Number.isFinite(next.value)) return [];
    return [{ id: item.id + next.id, x1: x(index), y1: y(item.value), x2: x(index + 1), y2: y(next.value) }];
  });
  return (
    <article className="benchmark-resource-trend">
      <div className="surface-head"><h4>{title}</h4><span className="status-pill">{observed.length}/{values.length} reported</span></div>
      {observed.length > 0 ? <svg viewBox="0 0 640 100" role="img" aria-label={title + " trend"}>
        <line x1="24" y1="16" x2="616" y2="16" stroke="var(--line)" strokeDasharray="3 4" />
        <line x1="24" y1="49" x2="616" y2="49" stroke="var(--line)" strokeDasharray="3 4" />
        <line x1="24" y1="82" x2="616" y2="82" stroke="var(--line)" strokeDasharray="3 4" />
        {segments.map((segment) => <line key={segment.id} x1={segment.x1} y1={segment.y1} x2={segment.x2} y2={segment.y2} stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" />)}
        {values.map((item, index) => item.value !== null && Number.isFinite(item.value)
          ? <circle key={item.id} cx={x(index)} cy={y(item.value)} r="4" fill="var(--accent)"><title>{formatter(item.value)}</title></circle>
          : null)}
      </svg> : <p className="quiet-note">Not reported by the telemetry provider for these benchmark runs.</p>}
      <div className="benchmark-resource-missing">
        {values.map((item, index) => <span key={item.id}>Run {index + 1}: {item.value === null || !Number.isFinite(item.value) ? "Not reported" : formatter(item.value)}</span>)}
      </div>
    </article>
  );
}

export default function BenchmarkResourceTrends({ samples }: { samples: BenchmarkResourceSample[] }) {
  if (samples.length === 0) return null;
  return (
    <section className="panel-surface benchmark-resource-history" aria-label="Benchmark resource trends" style={{ padding: 14, marginTop: 14 }}>
      <div className="surface-head">
        <div><p className="surface-eyebrow">BENCHMARK-LINKED · POINT-IN-TIME</p><h3>Resource trends</h3></div>
        <span className="status-pill">{samples.length}/20 runs</span>
      </div>
      <p className="quiet-note">Each successful native benchmark is paired with resource snapshots before and after the run. Resource deltas are host-wide observations, not isolated attribution to the model; other local activity may contribute. Missing metrics are shown as Not reported and are never estimated. The last 20 runs stay in browser memory only.</p>
      <div className="benchmark-resource-trend-grid">
        {charts.map((chart) => <TrendChart key={chart.key} title={chart.title} values={samples.map((sample) => ({ id: sample.id, value: sample[chart.key] }))} formatter={chart.format} />)}
      </div>
      <div className="benchmark-resource-run-index">
        {samples.map((sample, index) => <div key={sample.id}>
          <b>Run {index + 1} · {sample.model}</b>
          <span>{sample.device ?? "Device not reported"} · {new Date(sample.sampled_at).toLocaleString()}</span>
          <small>{[
            sample.cpu_percent === null ? "CPU not reported" : `CPU ${sample.cpu_percent.toFixed(1)}%`,
            sample.process_rss_delta_bytes === null ? "RSS change not reported" : `RSS ${formatBytes(sample.process_rss_delta_bytes)}`,
            sample.system_ram_used_delta_bytes === null ? "RAM change not reported" : `RAM ${formatBytes(sample.system_ram_used_delta_bytes)}`,
            sample.device_allocated_delta_bytes === null ? "Device allocation not reported" : `Device ${formatBytes(sample.device_allocated_delta_bytes)}`,
          ].join(" · ")}</small>
        </div>)}
      </div>
    </section>
  );
}
