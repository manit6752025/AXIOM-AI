# AXIOM Engine Fabric

The Engine Fabric is AXIOM's local execution control plane. It does not replace AXIOM Native Engine with an external runtime: it keeps the native transformer implementation first-class, discovers optional local backends, explains their trade-offs, recommends a profile, and dispatches to a selected compatible service when that is the best fit.

## What AXIOM brings together

| Engine | Main strengths | Trade-offs |
| --- | --- | --- |
| AXIOM Native Engine | AXIOM-owned forward pass and token loop; trusted model roots; floating-point safetensors plus validated single-file/split-shard dense GGUF for Llama, Mistral, Qwen2 and Qwen3; explicit device, precision and memory modes; cancellation and runtime telemetry | GGUF tensors are dequantized for floating-point execution, not run through low-bit kernels; EXL2/GPTQ/AWQ, unsupported tensor types/architectures and broader model families remain unsupported |
| Ollama | Simple local model lifecycle, straightforward model catalogue and chat API, broad user-friendly model distribution | Adds a separate service and offers fewer low-level scheduler/memory controls than specialist servers |
| llama.cpp server | Quantized GGUF, CPU/GPU hybrid offload, many hardware backends, OpenAI-compatible API | Binary flags/build options and model chat templates matter |
| LM Studio local server | Desktop model management and a convenient local API | Depends on its installed desktop/server runtime |
| vLLM | High-throughput GPU serving, continuous batching and KV-cache scheduling | More involved setup and most useful on appropriately provisioned accelerator systems |
| llama-cpp-python server | GGUF plus direct Python integration | Accelerator-specific wheels and local server configuration can be fiddly |

These are decision aids, not universal speed rankings. AXIOM does not claim an engine is faster, safer, or better on a machine without measurements from that machine.

## Control Center

Open Runtime in AXIOM Control Center. The Native Engine remains its own full panel. Engine Fabric shows runtime reachability, model IDs where the service reports them, profile-based recommendations, strengths and trade-offs, and a local chat form for compatible external services.

Profiles are:

- Balanced: practical general-purpose selection.
- Low memory: prioritize quantized or memory-efficient execution.
- High throughput: prioritize engines built for concurrent serving.
- Easy setup: prioritize approachable local model lifecycle.
- Portable / quantized: prioritize GGUF and broad CPU/accelerator support.
- Minimal background services: prefer AXIOM-owned execution when a supported checkpoint is already available.

The selected backend must respond on its fixed loopback endpoint before it can be used. A command found on PATH is only installation evidence, not proof that its service is running.

## API and CLI

~~~bash
axiom --no-banner engine backends --json
axiom --no-banner engine backends --profile low-memory
axiom --no-banner engine route-chat --backend llama-cpp --model local-model --prompt "Summarize this paragraph."
~~~

- GET /api/engine/backends?profile=balanced returns the engine matrix and a recommendation.
- POST /api/engine/backends/chat dispatches a bounded chat request to Ollama, llama.cpp, LM Studio, vLLM, or llama-cpp-python.
- AXIOM Native Engine continues to use GET /api/engine/status, GET /api/engine/models, POST /api/engine/generate, and POST /api/engine/generate/stream.

Advanced sampling accepts top-p, top-k, repetition penalty, seed, and stop sequences. Min-p is also exposed as `--min-p` in route-chat, auto-chat and route-benchmark. It is supported by AXIOM Native Engine and Ollama; nonzero values are rejected for other adapters instead of being silently ignored. Auto-routing filters by sampler compatibility before its bounded candidate limit so a supported match remains eligible.

## Security boundaries

External backends are selected by a stable ID, not by a user-supplied URL. Each adapter uses its documented local default endpoint on IPv4 loopback, bypasses environment proxies, limits request and response sizes, validates message roles and prompt/token bounds, and sends no provider credentials. AXIOM does not launch processes, pull models, contact cloud endpoints, or persist prompt/response content as part of engine discovery. Authentication-required endpoints are reported as such rather than receiving guessed credentials. If your local service uses another port or requires a key, configure it outside this feature and do not expose an unauthenticated local model API to an untrusted network.

Benchmark claims must be backed by measured runs on the target hardware. Hardware fit estimates, service presence, or model catalogue metadata are not substitutes for measured quality, memory and throughput.


## Evidence-based local benchmarking

The Engine Fabric includes an optional measured shootout for the selected external local runtime. It permits one warm-up, one to five measured requests, and up to 256 output tokens per request. The API discards each generation response and returns only per-request wall latency, median/P95 latency, end-to-end completion throughput when token counts are reported, run settings, and privacy flags. Results are kept in the current Control Center screen only; refreshing or closing the page clears the comparison. Different models and quantization levels are not apples-to-apples, so this is a tool for repeatable local observation, not a global leaderboard.
