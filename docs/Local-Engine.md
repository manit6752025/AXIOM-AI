# AXIOM Native Engine: independent local inference execution

AXIOM Native Engine executes Llama, Mistral, Qwen2, and Qwen3-family causal language models directly inside AXIOM's Python process. PyTorch supplies tensor operations and device kernels; AXIOM implements the model's forward pass, RMSNorm, rotary position embeddings (RoPE), grouped-query attention, causal/sliding-window masks, per-layer key/value cache, SwiGLU MLP, LoRA deltas, sampling and token-by-token decoding. AXIOM uses PyTorch's scaled-dot-product attention kernel when available to benefit from fused/optimized kernels; the exact cache-aware causal and sliding-window mask is built by AXIOM, with a manual attention implementation as the fallback.

**It is not an Ollama wrapper.** The inference request does not start, call, or require an Ollama service. It does not call Transformers' AutoModel loader or model.generate. AXIOM runs supported floating-point safetensors and supported dense GGUF checkpoints through its own native execution code. GGUF input tensors are dequantized to floating-point before PyTorch execution, so GGUF source quantization does not imply low-bit runtime compute or memory savings. Tokenization is local using `tokenizer.json` or supported GGUF-embedded tokenizer metadata. Inference itself does not access the network; Model Studio provides explicit public-Hub download/install actions.

## Install the native runtime

```bash
python -m pip install -e '.[native-inference]'
```

This extra installs PyTorch, tokenizers, safetensors, gguf-py, and Jinja2 for sandboxed checkpoint chat templates and GGUF parsing/dequantization. The separate training extra retains Transformers/PEFT tooling for the existing training workflow; the native inference loop does not delegate generation to Transformers.

## Supported checkpoint format

For a safetensors checkpoint, a supported base-model directory must contain:
- `config.json` declaring model_type `llama`, `mistral`, `qwen2`, or `qwen3`, with RoPE settings implemented by the native decoder.
- `tokenizer.json`, used locally for encoding and decoding.
- `model.safetensors`, or `model.safetensors.index.json` plus all indexed safetensors shards.
- Standard floating-point tensors in float16, bfloat16, or float32 with expected names and shapes.

A supported GGUF directory must contain one GGUF file or a complete, metadata-validated split-shard checkpoint for the same dense Llama/Mistral/Qwen2/Qwen3 text architectures. Tokenization may use a local `tokenizer.json`, embedded Hugging Face tokenizer JSON, or one of AXIOM's supported GGUF tokenizer metadata formats. AXIOM validates architecture, tensor names/shapes, shard completeness, tokenizer vocabulary and supported tensor types before inference. GGUF Q4/Q5/Q8 source tensors are dequantized through gguf-py into the selected floating-point compute path; this is not a native low-bit kernel and does not preserve quantized memory usage.

Model Studio can pull compatible public Llama/Mistral/Qwen2/Qwen3 safetensors checkpoints directly into AXIOM's trusted native-model directory. It validates the architecture and Hub file inventory before downloading weights, fetches only allowlisted config/tokenizer/safetensors assets (never model-provided Python), stages downloads before publishing the directory, reports byte progress, checks free disk, enforces per-file/total limits, and refuses to overwrite an existing install. This safetensors pull path does not accept gated/private repositories. For GGUF, use the explicit GGUF catalogue/install flow and choose the AXIOM Native Engine target; for example, `axiom model studio install-gguf OWNER/REPOSITORY FILE.gguf --name my-model --native`. The native engine still rejects EXL2, GPTQ/AWQ low-bit weights, pickle-based `pytorch_model.bin`, custom remote model code, unsupported architectures/tensors, and RoPE scaling modes it does not implement. Qwen2/Qwen3 support is limited to their standard text decoders and implemented RoPE/query-key normalization settings; multimodal variants are not implied. It does not silently route generation through another engine. PyTorch provides tensor kernels, while AXIOM owns the supported model forward pass and decode loop.

## Trusted model roots

The engine resolves model and adapter paths against trusted local roots and checks the post-resolution canonical path so symlink escapes are rejected. Defaults include existing models, runs, and outputs directories below the working directory; ~/models; ~/Documents/models; ~/Documents/AXIOM-AI/models and runs; the Hugging Face cache; and AXIOM's local model/fine-tuning directories.

To add a trusted location, set AXIOM_MODEL_ROOTS before starting the API. Use the platform path separator between roots (colon on Linux/macOS, semicolon on Windows). For example:

```bash
export AXIOM_MODEL_ROOTS="$HOME/models:$HOME/Documents/AXIOM-AI/runs:/mnt/data/models"
```

Only add directories you trust. Model/config/tokenizer/index/adapter assets may be symlinks only when their resolved targets remain inside one of the configured trusted roots; external-target symlinks are rejected.

## CLI

Inventory the same trusted local model roots used by the Control Center:

```bash
axiom engine discover
axiom engine discover --json
axiom engine status
axiom engine status --json
axiom engine estimate --model-path "$HOME/models/my-llama-model" --context-length 4096 --batch-size 1 --ram-budget-gib 16 --vram-budget-gib 8
axiom engine estimate --model-path "$HOME/models/my-llama-model" --context-length 4096 --ram-budget-gib 16 --json
axiom engine pull Qwen/Qwen2.5-0.5B-Instruct
axiom engine run --model-path "$HOME/.local/share/axiom/models/qwen-qwen2.5-0.5b-instruct" --prompt "What is 2+2?" --max-new-tokens 64 --temperature 0 --device auto --memory-mode auto
axiom engine run --model-path "$HOME/models/my-llama-model" --prompt "Hello" --device cuda --memory-mode auto --precision fp16 --top-p 0.9 --top-k 40 --min-p 0.05 --seed 17
# Precision modes: auto (preserve checkpoint dtype), fp32, fp16, bf16
axiom engine run --model-path /absolute/path/to/model --prompt "Hello" --max-new-tokens 64 --temperature 0 --device cpu --stream
# Memory modes: auto, full, layer_streaming, disk_streaming
# Other explicit choices, when detected by the installed PyTorch build: cuda, rocm, mps, xpu
```

The `axiom engine estimate` command compares estimated weight, KV-cache, and workspace memory against one or both supplied budgets. Its formulas include explicit overhead, a safety margin, and an uncertainty range; use the JSON output to automate planning. INT4/INT8 rows are sizing references only and are marked as unsupported by the native loader. The recommendation selects among the currently native-supported floating-point candidates, but a budget fit is not a guarantee that a model will load successfully. The estimator reads model configuration metadata; it neither loads tensors nor predicts quality or speed.


Use `axiom engine pull OWNER/REPOSITORY` to start a public Hugging Face pull specifically for AXIOM Native Engine. This command accepts compatible Llama/Mistral/Qwen2/Qwen3 floating-point safetensors checkpoints, validates the architecture before downloading weights, writes into `~/.local/share/axiom/models`, and prints the path and matching run command. To install GGUF for native use instead, run `axiom model studio install-gguf OWNER/REPOSITORY FILE.gguf --name my-model --native`; the default destination is Ollama, while `--download-only` stores the file without runtime registration. A plain `axiom model studio pull` pulls a tag into Ollama, not AXIOM Native Engine. Discovery still inspects metadata and file presence only; it does not run inference. A listed model is a candidate for an actual local load, not a guarantee that its checkpoint tensors are compatible. The optional `--stream` mode displays cumulative decoded output as tokens arrive; `--json` remains a single final machine-readable response and cannot be combined with `--stream`.

## Control Center

Open **Runtime → AXIOM Local Engine**. The device-capability report distinguishes CPU, NVIDIA CUDA, AMD ROCm, Apple Metal/MPS, and Intel XPU availability, and reports free/total accelerator memory when the installed PyTorch driver exposes it. Choose **Auto-select available backend** or explicitly select a detected backend; an unavailable explicit choice is rejected rather than silently ignored. Choose **Compute precision** (`auto` preserves checkpoint dtype; `fp32`, `fp16`, `bf16` cast weights and supported linear operations to the selected floating-point dtype) and **Auto-fit** to compare checkpoint size, conservative KV/workspace estimates for the supported context, and observed free device/system memory; or explicitly choose **Full weights**, **Layer streaming + CPU KV cache**, or **Disk-resident streaming + CPU KV cache**. Auto-fit is advisory and returns its estimates/reason. In-RAM layer streaming keeps all transformer weights in system RAM and stages one layer at a time, while disk-resident streaming reads safetensors layers or dequantizes supported GGUF layer tensors on demand. Disk-resident streaming lowers full-model CPU and accelerator weight residency but causes extra disk I/O/dequantization work on every forward pass. Both modes trade throughput for weight residency; embeddings/output projection, LoRA tensors, activations, the active layer and KV cache still require memory. GGUF ingestion does not provide low-bit execution. EXL2/GPTQ/AWQ kernels and unsupported GGUF architectures/tensor types remain unsupported. Auto prefers an available CUDA/ROCm accelerator, then MPS, XPU and CPU. If a recognized out-of-memory error occurs while loading model weights on an accelerator in auto mode, AXIOM retries on CPU and records the fallback reason; an explicit device choice never silently switches. Device availability does not guarantee a model will fit memory or every operator is supported by a particular driver. Select a discovered model/adapter from the native inventory or enter its local path, then provide a prompt, output-token limit, temperature, top-p, top-k, min-p, repetition penalty and optional sampling seed. Min-p is disabled at `0`; otherwise it keeps tokens whose probability is at least the configured fraction of the highest-probability token, applied after top-k and top-p filtering. When `tokenizer_config.json` contains a chat template, AXIOM renders it in a sandbox before tokenization; Qwen2/Qwen3 text models without template metadata use the ChatML fallback. Base/completion checkpoints without an exported chat template receive the raw prompt. The result reports which format was used. Optional stop strings are entered one per line (maximum 16, 256 characters each). AXIOM withholds incomplete possible matches between tokens, stops on the first complete match, and excludes that string and later generated text from the final answer. For chat requests with prior messages, AXIOM drops the oldest non-system turns first when the prompt exceeds its token budget, preserving system messages and the newest user request where possible. The result reports original message count, dropped turns, tokens used, and whether hard token truncation was still needed; if the system message might be clipped, the result flags that explicitly. Combined chat input is bounded to 24,000 characters, and each user/prompt field is bounded separately. The completion includes a finish reason: `eos_token`, `max_new_tokens`, `stop_sequence`, or `cancelled`. After generation, AXIOM reports wall time plus measured prompt-prefill latency, summed cached-decode latency, time to first token (the initial prefill forward pass), and completion tokens per second. Streaming results report layer-weight bytes staged/released, CPU returns for in-RAM layer streaming, disk layer-tensor reads and read bytes for disk-resident mode, and cumulative KV-cache bytes copied to CPU. These transfer counters are not a peak-memory guarantee. The timing measurements exclude model loading and prompt tokenization; throughput is computed from completion token count divided by prefill plus decode compute time. Discovery lists metadata only; weights are validated and loaded only when generation starts. **Stop generation** aborts the client stream and signals cooperative cancellation through the API into AXIOM's native prompt-prefill/token loop; cancellation is checked between transformer layers and decoding steps. A disconnect during one individual tensor kernel is acted on when that kernel yields control. **Unload model** releases AXIOM's model/tokenizer references and best-effort empties the active CUDA, Metal/MPS, or XPU allocator cache when the runtime exposes that API. Framework and OS allocators may keep some reserved memory.

A standard bias-free CAUSAL_LM LoRA adapter may be selected. AXIOM applies its A/B projection deltas in its own linear-projection path. DoRA, QALoRA, per-module rank/alpha patterns, saved replacement modules, bias-enabled adapters, unsupported target modules, and pickle adapter weights are rejected.

Before generation, use **Check adapter compatibility** to run a metadata/tensor-shape preflight against the chosen base model. The report separates CPU, NVIDIA CUDA, AMD ROCm, Apple Metal/MPS, and Intel XPU availability and lists structural blockers by target module, rank, or shape. It does not allocate the full base model or guarantee device memory/driver/kernel success; actual generation remains the final execution test.

## Execution path

1. Validate paths against trusted roots and read bounded JSON configuration.
2. Check architecture/dimensions, safetensors index or GGUF shard metadata, tensor names/shapes, supported tensor types, and tokenizer vocabulary.
3. Select the requested or best available PyTorch backend (CUDA, ROCm, Metal/MPS, Intel XPU, or CPU), loading safetensors or dequantizing supported GGUF tensors onto the native execution path.
4. Tokenize locally and run AXIOM's embedding, RMSNorm, RoPE, grouped-query attention, KV cache, SwiGLU MLP and output-logit operations.
5. Select tokens through greedy decoding (temperature 0) or temperature/top-k/top-p/min-p sampling. Min-p uses the equivalent logit test `score >= max_score + log(min_p)`, avoiding a second softmax; passing filters retain the valid candidate distribution before token sampling. Each new token is passed through the cached model until the output limit or EOS token.

A checkpoint outside this contract is rejected with a compatibility error; the engine does not pretend it ran a model successfully.

## API and CLI

- GET /api/engine/status reports native dependency readiness, supported formats, observed execution-device availability, a recommended device, loaded backend and model state.
- GET /api/engine/telemetry and `axiom engine telemetry --json` return point-in-time CPU, process RSS, system RAM, device allocator and trusted-model-storage snapshots where supported. They omit raw paths, prompts, generated text, environment values and credentials; poll the endpoint for a time series.
- POST /api/engine/generate and POST /api/engine/generate/stream run AXIOM-native inference with a local model path, optional LoRA adapter, prompt, generation parameters, optional `device` choice (`auto`, `cpu`, `cuda`, `rocm`, `mps`, `xpu`), `memory_mode` (`auto`, `full`, `layer_streaming`, `disk_streaming`), and `precision` (`auto`, `fp32`, `fp16`, `bf16`). Stream events contain text updates and live inference/memory telemetry.
- POST /api/engine/unload releases the model cached by the current API process.
- `axiom engine status`, `axiom engine run`, and `axiom engine pull` use the same native engine.
- The OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions` endpoints expose only discovered trusted local Llama/Mistral/Qwen2/Qwen3 models. Set `stream: true` on `POST /v1/chat/completions` to receive SSE chunks. Chat messages are limited to system/user/assistant text; tools, multimodal content, multi-choice sampling, and continuous batching are not supported by this compatibility surface.

Weights may be placed locally or acquired through Model Studio's explicit public-Hub native pull/GGUF installation; always review model licenses. AXIOM does not claim supplied third-party base weights were pretrained by AXIOM. Native speed and memory depend on model size, precision, device, and—in GGUF mode—dequantization overhead. Native low-bit kernels and further architecture families need separate implementations and acceptance tests.
