# Local inference

AXIOM's local inference panel uses Ollama's API at `http://127.0.0.1:11434`. AXIOM can now start an Ollama service, stop only a process it started and can positively identify, and explicitly pull a model with bounded concurrency and progress reporting. Generation stays local; downloading a model is a separate, user-triggered network action.

## Requirements

1. Install Ollama for your operating system using the official instructions.
2. Start its local service if it is not already running.
3. Pull a model you have sufficient memory to run. A small starting example is:

   ```bash
   ollama pull qwen2.5:0.5b
   ```

Model download is an explicit user command, not an AXIOM side effect. The exact model available depends on your hardware and the models you've downloaded.

## Runtime request traces

The **Runtime** page includes a request history for local generation and benchmark calls. The API returns an `X-Request-ID` header and the response body carries the same identifier, allowing a call to be correlated with its trace. The in-process buffer keeps at most 200 records, newest first. It stores model name, operation, status, durations, token counts when available, bounded benchmark settings, and safe error codes. It does not accept or store prompt bodies, generated response text, credentials, or raw exception messages. Traces are memory-only and are lost when the API process restarts.

Use **Refresh traces** to reload the history, **Export JSON** to download the displayed metadata, and **Clear history** to clear the current process's buffer after explicit confirmation. Clear affects only this process; it does not remove files because trace history is not written to disk. See the runtime diagnostics API routes `GET /api/diagnostics/runtime-traces` and `DELETE /api/diagnostics/runtime-traces`.

## Benchmarking

The Runtime page can run a bounded benchmark against the same local Ollama service. It supports up to two excluded warm-up generations, up to ten measured requests across configurable concurrency, output limits up to 512 tokens per benchmark request, per-request latency statistics, wall time, and completion-token throughput when Ollama supplies token counts. The benchmark neither saves prompts/responses nor returns generated text in its result payload. See the [Inference Benchmarking guide](Inference-Benchmarking.md) for endpoint fields, limits, and measurement caveats.

## CLI

Run commands from the AXIOM environment:

```bash
axiom --no-banner infer status
axiom --no-banner infer models
axiom --no-banner infer run qwen2.5:0.5b --prompt "Explain local inference in one sentence." --num-predict 128
axiom --no-banner infer run qwen2.5:0.5b --prompt-file ./prompt.txt --json
```

Use `axiom --no-banner infer --help` and `axiom --no-banner infer run --help` for all options. The prompt limit is 12,000 characters, generation is bounded to 1–2,048 tokens, and the request timeout is finite.

## GUI

Start AXIOM's local API and open **Runtime** in Control Center. The **Local model inference** panel checks the local Ollama API, suggests model names reported by Ollama, and sends prompts through AXIOM's inference API. Generation can take time on CPU; the request timeout is bounded to five minutes. Output is displayed in the GUI and is not saved to workspace logs by this feature.

## Scope and limitations

- This is real text generation when the local Ollama API and chosen model are available; it is not a simulated output or a planning report.
- Runtime discovery remains a separate passive inventory and is not treated as proof of an active connection.
- AXIOM connects only to the constant IPv4 loopback endpoint `127.0.0.1:11434` and explicitly disables proxy use for these requests.
- AXIOM-managed service start forces `OLLAMA_HOST=127.0.0.1:11434`. A pre-existing service is not adopted or stopped; AXIOM refuses to stop processes it did not start and verify.
- Model pulls use an argument list rather than a shell, allow at most two concurrent jobs, cap each job at 30 minutes, sanitize progress output, and retain only a bounded in-memory job history. Downloading a model requires an explicit Pull model action or `axiom infer pull`.
- Process and model-pull management APIs are limited to local clients by default. Only set `AXIOM_ALLOW_REMOTE_RUNTIME_MANAGEMENT=1` on a trusted network when remote management is intentionally required.
- Ollama and AXIOM Native Engine have different model-format support. Ollama can serve Ollama-supported formats (including GGUF); the native AXIOM engine continues to support only its documented safetensors subset.
- If the service is offline or the model name is unknown, AXIOM reports the failure instead of substituting a fake response.


## Managed Ollama service and model pulls

The **Runtime → Local model inference** panel includes **Start Ollama**, **Stop AXIOM service**, and **Pull model** controls. Starting the service sets `OLLAMA_HOST=127.0.0.1:11434`; the service ownership record is stored locally with restrictive file permissions and contains no prompts, credentials, or model content. Stop is available only while AXIOM can verify the PID, executable, start identity, and `serve` command it launched. An independently running OS-managed Ollama service must be stopped with its own service manager.

Use the CLI equivalents:

```bash
axiom --no-banner infer service status --json
axiom --no-banner infer service start
axiom --no-banner infer service stop
axiom --no-banner infer pull qwen2.5:0.5b
axiom --no-banner infer ps --json
axiom --no-banner infer inspect qwen2.5:0.5b --json
axiom --no-banner infer unload qwen2.5:0.5b
```

A model pull has a job ID, status, sanitized progress lines, and a best-effort percentage when Ollama reports one. Job state is in-memory only, bounded to the latest 20 jobs, permits two concurrent pulls, and expires from the process history when newer jobs need room. A pull is terminated after 30 minutes or when explicitly cancelled from the GUI. The model download itself contacts the Ollama library; inference prompts are still sent only to the local loopback API.

## Loaded-model memory inventory

Use `axiom infer ps --json` or the **Show loaded models** control in the Runtime panel to inspect the point-in-time `/api/ps` inventory. AXIOM reports only models Ollama says are loaded, the size totals reported by Ollama, its `size_vram` value, and a derived non-VRAM remainder when both sizes are available. These values are runtime-reported estimates, not direct OS/GPU allocator measurements. The query does not request prompts, generated text, model templates, or license text.


## Streaming generation and cancellation

The Runtime inference form now uses `POST /api/inference/generate/stream` and newline-delimited JSON events: token deltas, a completion record with timings/token counts, or a safe error record. Text appears incrementally in the existing response area. **Stop generation** aborts the browser request; the daemon watches for disconnects and closes the upstream loopback response so Ollama can stop work promptly. Request traces retain only bounded metadata—not prompts or generated text.

This stream uses the same `127.0.0.1:11434` Ollama endpoint as non-streaming API calls. It does not convert Ollama model formats into AXIOM Native Engine formats.


## Advanced sampling options

The Runtime panel's **Advanced sampling options** section exposes Ollama's top-p, top-k, min-p, repetition penalty, optional seed, optional context window, and up to 16 newline-separated stop sequences. Values pass through request bounds and core validation before reaching the fixed loopback request. Min-p is disabled at `0`; positive values retain tokens whose probability is at least the configured fraction of the most likely token. Seed and context size are omitted when blank so Ollama can use the model/runtime defaults. The CLI supports the same settings:

```bash
axiom --no-banner infer run qwen2.5:0.5b --prompt "Explain AXIOM" --top-p 0.83 --top-k 36 --min-p 0.15 --repeat-penalty 1.2 --seed 123 --num-ctx 8192 --stop DONE --stop END
```

Top-p must be greater than 0 and at most 1; top-k is 0–100,000; min-p is 0–1 (with 0 disabling the cutoff); repetition penalty is 0.1–5; context size is 512–131,072 tokens; seed is a non-negative signed 64-bit integer; and each of at most 16 stop strings is limited to 256 characters. These are runtime options, not measured hardware-fit claims.


## AXIOM Engine Fabric

Engine Fabric now compares AXIOM Native Engine with optional local runtime APIs and can send bounded chat requests to Ollama, llama.cpp server, LM Studio, vLLM, and llama-cpp-python. It is a router and evidence-labelled capability catalogue, not a claim that AXIOM has implemented or embedded those runtimes internally. The AXIOM Native Engine retains its own implementation and strict trusted safetensors model paths. See [Engine Fabric](Engine-Fabric.md) for backend strengths, trade-offs, profiles, loopback boundaries, and CLI/API usage.
