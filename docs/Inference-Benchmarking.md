# Local inference benchmarking

AXIOM's existing **Runtime → Local model inference** panel can run a measured benchmark against a model already available in the local Ollama service. The benchmark invokes the same loopback-only adapter as normal generation; it does not pull a model, start Ollama, call a cloud service, or run a synthetic calculator.

## Run a benchmark

1. Start the local Ollama service and ensure at least one model is already installed.
2. Open **Runtime** in the Control Center and wait for **OLLAMA ONLINE**.
3. Select the model, enter a representative prompt, choose the output-token limit (maximum 512 for benchmarks), and set temperature.
4. In **Inference benchmark**, choose the number of warm-up generations (0–2), measured iterations, and concurrency. The total measured requests (iterations × concurrency) cannot exceed 10.
5. Click **Benchmark this prompt**.

Warm-up requests are excluded from the measured timing. Measured requests are issued in waves at the chosen concurrency. The result shows wall-clock duration, request rate, completion-token rate where runtime token counts are available, and per-request latency samples including min, median, p95, and max.

## API contract

The endpoint is POST /api/inference/benchmark. It accepts:

| Field | Bounds | Purpose |
| --- | --- | --- |
| model | 1–120 characters | Exact local Ollama model name |
| prompt | 1–12,000 characters | Workload input sent to the local model |
| num_predict | 1–512 | Maximum generated tokens per request |
| temperature | 0–2 | Ollama sampling temperature |
| warmup_runs | 0–2 | Warm-up generations excluded from results |
| iterations | 1–10 | Number of measured waves |
| concurrency | 1–4 | Requests in each measured wave |
| total measured requests | iterations × concurrency ≤ 10 | Combined work limit |

The core also validates these bounds, so a non-API caller cannot bypass the workload limit.

## Measurement and privacy boundaries

- Per-request latency is measured by AXIOM with a monotonic clock around each actual local generation call; total wall time covers the measured waves, not warm-up.
- Completion-token throughput uses Ollama's reported eval_count values divided by total measured wall time. Coverage is explicitly reported; if token counts are unavailable, completion-token throughput is null rather than guessed.
- Concurrent requests are real requests to the local Ollama endpoint. They can change queueing, memory pressure, scheduling, and throughput; compare concurrency settings as different workloads.
- The response includes aggregate counts and latency samples, not generated output text. The API does not persist a benchmark result or prompt history.
- No quality score is computed. Throughput is not model quality and does not automatically establish model-fit compatibility.
- Compare results only when model, prompt, output-token limit, temperature, runtime configuration, hardware, and concurrency are kept consistent. Repeat runs to characterize variability.

The Runtime screen and API call use the existing loopback-only Ollama adapter at 127.0.0.1:11434. This is a local runtime measurement, not a cross-device leaderboard or a guaranteed performance estimate.
