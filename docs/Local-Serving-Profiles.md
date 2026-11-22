# Local native serving profiles

Serving profiles save a bounded configuration for AXIOM's existing native local model engine. They do not introduce a second runtime, store credentials, download checkpoints, or start a network listener.

## CLI

~~~bash
axiom engine profile create workstation \
  --model-path ~/models/qwen2.5-7b \
  --device auto --memory-mode layer_streaming --precision bf16 \
  --context-length 8192 --max-new-tokens 512 \\
  --max-queued-requests 4 --json
axiom engine profile list --json
axiom engine profile probe <profile-id> --json
axiom engine profile run <profile-id> --prompt "Explain the local inference boundary." --json
axiom engine profile delete <profile-id> --yes
~~~

Profiles can also be created, probed, run, and removed from Runtime → Serving profiles.

## Validated resource limits

Each profile binds to a model and optional LoRA adapter already discovered beneath AXIOM's trusted model roots. The API and CLI resolve the supplied paths through the native engine inventory before saving them. Persisted paths are checked again during readiness and generation; symbolic-link/traversal paths are not accepted by the native path resolver.

Profiles bound the configured context window (1–131,072 tokens, further capped by the checkpoint), maximum output (1–2,048 tokens), prompt text (1–12,000 characters), and request time (5–900 seconds). Device, memory mode, and precision are restricted to the options supported by AXIOM Native Engine. Before generation, the native tokenizer inspects the exact prompt and AXIOM rejects requests whose prompt tokens plus requested output exceed the profile context budget or require silent prompt truncation.

The native runtime remains single-worker (one active generation at a time). All native inference entry points use one process-wide FIFO admission manager: ordinary `axiom engine run` / API generation, serving-profile generation, native benchmark runs, and model unload share the same active slot and global ceiling of 32 waiting requests. The API server and a separately launched CLI process each own their in-process manager; separate processes do not share a queue.

Each serving profile can additionally opt into a FIFO queue of 0–32 waiting requests for that profile; the default is zero and rejects immediately. The per-profile limit is enforced together with the global ceiling, which counts ordinary and profile requests across the API process. All requests share the same FIFO order for the single native worker. Requests that exceed either bound receive HTTP 429 with `Retry-After: 1`. Queue wait time counts toward the profile request deadline; expired profile requests receive HTTP 504 and are removed. The ordinary native engine admission wait is bounded to 30 seconds. A client disconnect while queued removes its ticket; disconnect during generation signals cooperative engine cancellation. The API maps queue saturation, wait timeout, and cancellation to structured HTTP 429, 504, and 499 errors respectively.

Serving-profile generation metadata includes `queue_position`, `queued`, `waited_ms`, `concurrency_limit`, and `queue_limit`. The Runtime panel shows the returned queue position and wait time and provides **Cancel generation**. Direct CLI generation, benchmarks, and profile runs share the CLI process's admission manager; profile CLI output reports its wait position and duration.

## Readiness and lifecycle

`axiom engine profile probe` and `POST /api/engine/serving-profiles/{profile_id}/readiness` inspect checkpoint metadata, engine dependencies, selected-device availability, configured context support, and optional adapter compatibility. They do not load weights. The probe is a readiness assessment, not a promise that physical memory, model operators, or drivers will support a successful inference.

The first profile generation request loads the selected model into AXIOM's existing native runtime. It may reuse the currently loaded model when the engine's normal cache rules allow. Profile deletion removes only `.axiom/serving-profiles.json` metadata; model and adapter files are never deleted.

The workspace-local store is capped at 25 profiles and 128 KiB. The metadata file is written atomically with restrictive permissions where supported. Credentials, custom endpoints, prompts and generated responses are not persisted by the profile store.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/engine/serving-profiles` | List saved profiles |
| `POST` | `/api/engine/serving-profiles` | Create or update a profile by name |
| `GET` | `/api/engine/serving-profiles/{id}` | Read one profile |
| `POST` | `/api/engine/serving-profiles/{id}/readiness` | Run a metadata-only readiness probe |
| `POST` | `/api/engine/serving-profiles/{id}/generate` | Generate under the saved request limits |
| `DELETE` | `/api/engine/serving-profiles/{id}` | Delete profile metadata only |

Profile generation returns 422 for invalid prompt/context budgets, 409 when the profile is not ready, 429 when either admission queue limit is full, 499 when a queued/running client disconnects, and 504 when the configured request deadline expires. The route stays local to the AXIOM API; it does not bind a public listener or expose an OpenAI-compatible serving endpoint.