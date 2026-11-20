import { useEffect, useState, type FormEvent } from "react";
import { Activity, Cpu, Play, RefreshCcw, Square } from "lucide-react";
import { axiomApi, type FineTuningJob, type FineTuningRuntime } from "../api/client";

const activeStates = new Set(["queued", "running", "cancelling", "finalizing"]);

export default function LocalFineTuningPanel() {
  const [runtime, setRuntime] = useState<FineTuningRuntime | null>(null);
  const [job, setJob] = useState<FineTuningJob | null>(null);
  const [jobHistory, setJobHistory] = useState<FineTuningJob[]>([]);
  const [jobHistoryTotal, setJobHistoryTotal] = useState(0);
  const [jobHistoryHasMore, setJobHistoryHasMore] = useState(false);
  const [jobHistoryLoading, setJobHistoryLoading] = useState(false);
  const [modelPath, setModelPath] = useState("");
  const [datasetPath, setDatasetPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [textField, setTextField] = useState("text");
  const [maxSteps, setMaxSteps] = useState("20");
  const [sequenceLength, setSequenceLength] = useState("512");
  const [learningRate, setLearningRate] = useState("0.0001");
  const [loraRank, setLoraRank] = useState("8");
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState("");

  const loadJobHistory = async (offset = 0, append = false) => {
    setJobHistoryLoading(true);
    try {
      const page = await axiomApi.fineTuningJobsPage({ limit: 10, offset });
      setJobHistory((current) => append ? [...current, ...page.jobs] : page.jobs);
      setJobHistoryTotal(page.total);
      setJobHistoryHasMore(page.has_more);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Fine-tuning job history is unavailable.");
    } finally {
      setJobHistoryLoading(false);
    }
  };

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const [available, saved] = await Promise.all([
        axiomApi.fineTuningRuntime(),
        axiomApi.fineTuningJobsPage({ limit: 10 }),
      ]);
      setRuntime(available);
      setJobHistory(saved.jobs);
      setJobHistoryTotal(saved.total);
      setJobHistoryHasMore(saved.has_more);
      if (!job && saved.jobs.length) setJob(saved.jobs[0]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Fine-tuning status is unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  useEffect(() => {
    if (!job || !activeStates.has(job.status)) return;
    const timer = window.setInterval(() => {
      axiomApi.fineTuningJob(job.id).then((updated) => {
        setJob(updated);
        setJobHistory((current) => current.map((item) => item.id === updated.id ? updated : item));
      }).catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : "Could not refresh the active fine-tuning job.");
      });
    }, 1800);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  const run = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setStarting(true);
    setError("");
    try {
      const created = await axiomApi.startFineTuningJob({
        model_path: modelPath.trim(),
        dataset_path: datasetPath.trim(),
        output_dir: outputDir.trim(),
        text_field: textField.trim() || "text",
        max_steps: Number(maxSteps),
        sequence_length: Number(sequenceLength),
        learning_rate: Number(learningRate),
        lora_rank: Number(loraRank),
      });
      setJob(created);
      setJobHistory((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      setJobHistoryTotal((current) => current + 1);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Fine-tuning job could not be started.");
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    if (!job) return;
    setCancelling(true);
    setError("");
    try {
      const updated = await axiomApi.cancelFineTuningJob(job.id);
      setJob(updated);
      setJobHistory((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Cancellation could not be requested.");
    } finally {
      setCancelling(false);
    }
  };

  const busy = Boolean(job && activeStates.has(job.status));
  const canStart = Boolean(runtime?.can_start && !busy && !starting && modelPath.trim() && datasetPath.trim() && outputDir.trim());

  return <section className="surface local-finetuning-panel" aria-label="Local LoRA fine-tuning">
    <div className="surface-head">
      <div><p className="surface-eyebrow">ACTUAL TRAINING · LOCAL MODEL FILES ONLY</p><h2>Local LoRA fine-tuning</h2></div>
      <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading || starting}><RefreshCcw size={14} /> {loading ? "Checking" : "Refresh"}</button>
    </div>
    <p>Runs real causal-language-model fine-tuning with a PEFT LoRA adapter. Base weights must already be in a local Transformers model folder; AXIOM will not download weights or execute remote model code. Training can consume substantial memory and may take time.</p>
    <div className="status-strip" aria-label="Fine-tuning dependencies">
      <div className="status-strip-item">
        <span className={`status-pill ${runtime?.can_start ? "success" : "warning"}`}><i />{loading ? "CHECKING" : runtime?.can_start ? "RUNTIME READY" : "SETUP REQUIRED"}</span>
        <div><b>Transformers + PEFT</b><span>{runtime?.can_start ? "Optional packages detected; model compatibility is checked at launch." : "Install the optional training dependencies in your active Python environment."}</span></div>
      </div>
    </div>
    {runtime && !runtime.can_start && <div className="inline-alert warning" role="status">{runtime.setup_note || <>Missing packages: {Object.entries(runtime.packages).filter(([, found]) => !found).map(([name]) => name).join(", ")}. Run <code>python3 -m pip install -e '.[training]'</code> from the repository using your active environment.</>}</div>}
    <form onSubmit={(event) => void run(event)}>
      <div className="two-column-grid">
        <label className="field"><span className="field-label-row"><span>Local base model directory</span></span><input value={modelPath} onChange={(event) => setModelPath(event.target.value)} required maxLength={4096} placeholder="/home/user/models/my-transformers-model" /><small>Folder must contain config.json and local model/tokenizer files.</small></label>
        <label className="field"><span className="field-label-row"><span>JSONL training dataset</span></span><input value={datasetPath} onChange={(event) => setDatasetPath(event.target.value)} required maxLength={4096} placeholder="/home/user/data/train.jsonl" /><small>UTF-8 JSONL, up to 50 MiB / 10,000 rows; default text field is text.</small></label>
      </div>
      <label className="field"><span className="field-label-row"><span>Adapter output directory</span></span><input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} required maxLength={4096} placeholder="/home/user/runs/my-lora-adapter" /><small>Parent directory must exist; output must be new or empty. Existing model/data files are not overwritten.</small></label>
      <div className="two-column-grid">
        <label className="field"><span className="field-label-row"><span>Text field</span></span><input value={textField} onChange={(event) => setTextField(event.target.value)} required maxLength={64} /><small>Can also read prompt/completion or messages records.</small></label>
        <label className="field"><span className="field-label-row"><span>Maximum optimization steps</span></span><input type="number" min={1} max={1000} step={1} value={maxSteps} onChange={(event) => setMaxSteps(event.target.value)} required /></label>
        <label className="field"><span className="field-label-row"><span>Sequence length</span></span><input type="number" min={32} max={2048} step={1} value={sequenceLength} onChange={(event) => setSequenceLength(event.target.value)} required /></label>
        <label className="field"><span className="field-label-row"><span>LoRA rank</span></span><input type="number" min={1} max={64} step={1} value={loraRank} onChange={(event) => setLoraRank(event.target.value)} required /></label>
        <label className="field"><span className="field-label-row"><span>Learning rate</span></span><input type="number" min={0.000001} max={0.01} step={0.000001} value={learningRate} onChange={(event) => setLearningRate(event.target.value)} required /></label>
      </div>
      <div className="integration-form-actions">
        <button type="submit" className="button primary-button" disabled={!canStart}><Play size={15} /> {starting ? "Starting…" : busy ? "Training is active" : "Start local fine-tuning"}</button>
        {job?.cancellation.requestable && <button type="button" className="button secondary-button" onClick={() => void cancel()} disabled={cancelling}><Square size={13} /> {cancelling ? "Requesting stop…" : "Cancel training"}</button>}
        {job && <span className="quiet-note"><Activity size={14} /> {job.completed_steps}/{job.config.max_steps} steps</span>}
      </div>
    </form>
    <section className="model-studio-history" aria-label="Fine-tuning job history">
      <div className="surface-head"><div><p className="surface-eyebrow">WORKSPACE JOBS</p><h3>Training history</h3></div><span className="quiet-note">{jobHistory.length} of {jobHistoryTotal}</span></div>
      {jobHistory.map((item) => (
        <article className="model-studio-result" key={item.id}>
          <div className="model-studio-result-copy"><b>{item.id.slice(0, 12)} · {item.config.model_path.split("/").filter(Boolean).at(-1) || item.config.model_path}</b><small>{item.completed_steps}/{item.config.max_steps} steps · {new Date(item.created_at).toLocaleString()}</small></div>
          <div className="model-studio-result-actions"><span className={`status-pill ${item.status === "completed" ? "success" : item.status === "failed" ? "warning" : "neutral"}`}><i />{item.status.toUpperCase()}</span><button type="button" className="button secondary-button small-button" aria-pressed={job?.id === item.id} onClick={() => setJob(item)}>{job?.id === item.id ? "Selected" : "Inspect"}</button></div>
        </article>
      ))}
      {jobHistory.length === 0 && !jobHistoryLoading && <p className="quiet-note">No saved fine-tuning jobs yet.</p>}
      {jobHistoryHasMore && <button type="button" className="button secondary-button" onClick={() => void loadJobHistory(jobHistory.length, true)} disabled={jobHistoryLoading}>{jobHistoryLoading ? "Loading…" : "Load more jobs"}</button>}
    </section>
    {error && <div className="inline-alert warning" role="alert">{error}</div>}
    {job && <section className="local-finetuning-job" role="region" aria-label="Fine-tuning job status">
      <div className="surface-head"><div><p className="surface-eyebrow">JOB {job.id.slice(0, 12)}</p><h3>Fine-tuning job</h3></div><span className={`status-pill ${job.status === "completed" ? "success" : job.status === "failed" ? "warning" : "neutral"}`}><i />{job.status.toUpperCase()}</span></div>
      {job.status === "cancelling" && <p className="quiet-note">Stopping at the next safe training boundary.</p>}
      {job.status === "finalizing" && <p className="quiet-note">Training steps are complete; AXIOM is writing adapter files. Cancellation is no longer available.</p>}
      {job.status === "cancelled" && <p className="quiet-note">Training stopped before adapter finalization.</p>}
      <div className="plan-result-grid"><div><small>Steps</small><b>{job.completed_steps}/{job.config.max_steps}</b></div><div><small>Latest loss</small><b>{typeof job.loss === "number" ? job.loss.toFixed(5) : "—"}</b></div><div><small>Output</small><b>{job.config.output_dir.split("/").filter(Boolean).at(-1) || job.config.output_dir}</b></div></div>
      {job.error && <div className="inline-alert warning" role="alert">{job.error}</div>}
      {job.result && <p className="quiet-note"><Cpu size={14} /> Adapter weights written · {job.result.dataset_rows} records · target modules: {job.result.target_modules.join(", ")}</p>}
      <pre className="local-finetuning-output" data-testid="fine-tuning-job-log">{job.logs.join("\n")}</pre>
    </section>}
  </section>;
}
