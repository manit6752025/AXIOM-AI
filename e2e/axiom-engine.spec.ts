import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime page exposes direct AXIOM Engine inference with local model and adapter paths, then unloads it", async ({ page }) => {
  await ensureAuthenticated(page);
  let engineLoaded = false;
  let unloadCalls = 0;

  await page.route("**/api/workspace/runtime", (route) => route.fulfill({
    json: {
      status: "ready",
      hardware: {
        os_name: "Linux", architecture: "x86_64", cpu_cores: 8,
        ram_gb: 16, gpu_name: null, vram_gb: null, cuda_available: false,
      },
    },
  }));
  await page.route("**/api/inference/status", (route) => route.fulfill({ json: {
    backend: "ollama", endpoint: "http://127.0.0.1:11434", runtime_installed: false,
    status: "offline", models: [], error: "Ollama is offline for this test.", checked_at: "2026-10-21T00:00:00Z",
  } }));
  await page.route("**/api/diagnostics/runtime-connectors", (route) => route.fulfill({ json: {
    schema_version: 1, discovery_mode: "passive_local_discovery", checked_at: "2026-10-21T00:00:00Z",
    scope_note: "Passive discovery only.",
    connectors: [],
  } }));
  await page.route("**/api/engine/status", (route) => route.fulfill({ json: {
    engine: "axiom-native-pytorch-llama", ready: true, packages: { torch: true, tokenizers: true, safetensors: true, gguf: true, numpy: true },
    versions: { torch: "2.9.0", tokenizers: "0.22.0", safetensors: "0.6.0", gguf: "0.19.0", numpy: "2.4.6" },
    minimum_versions: { torch: "2.2.0", tokenizers: "0.19.0", safetensors: "0.4.0" },
    problems: [], execution_backend: "pytorch-native-axiom", transformers_generation_used: false, ollama_used: false,
    supported_model_types: ["llama", "mistral", "qwen2", "qwen3"], supported_weights: ["safetensors", "gguf-dequantized-float"],
    loaded: engineLoaded, architecture: engineLoaded ? "LlamaForCausalLM" : null, model_loading: "local_safetensors_and_gguf_if_available", remote_code: false,
    setup_note: "AXIOM Native Engine is ready for supported local Llama/Mistral safetensors and single-file GGUF checkpoints (GGUF dequantizes to floating-point; no low-bit kernels).",
    scope_note: "AXIOM-owned Llama/Mistral forward pass and autoregressive KV-cache decoder.", checked_at: "2026-10-22T00:00:00Z",
  } }));
  await page.route("**/api/engine/adapters/compatibility", (route) => route.fulfill({ json: {
    schema_version: 1,
    base_model: { name: "llama-test", model_type: "llama", architecture: "LlamaForCausalLM" },
    adapter: { name: "my-lora-adapter", rank: 8, alpha: 16, target_modules: ["q_proj", "v_proj"] },
    structurally_compatible: true, compatible_on_any_device: true,
    adapter_tensor_count: 4, checked_layers: 12,
    devices: [
      { name: "CPU", available: true, compatible: true, status: "compatible", reason: "Structural preflight passed." },
      { name: "CUDA", available: false, compatible: false, status: "unavailable", reason: "No CUDA device detected." },
      { name: "ROCm", available: false, compatible: false, status: "unavailable", reason: "No ROCm device detected." },
      { name: "Metal/MPS", available: false, compatible: false, status: "unavailable", reason: "MPS is unavailable." },
    ],
    issues: [], scope_note: "Metadata/tensor-shape preflight only; actual allocation and execution remain untested.",
  } }));
  await page.route("**/api/engine/models", (route) => route.fulfill({ json: {
    schema_version: 1,
    models: [{
      path: "/models/llama-test", name: "llama-test", model_type: "llama",
      architecture: "LlamaForCausalLM", weight_format: "safetensors", ready_for_native_attempt: true,
    }],
    adapters: [{ path: "/runs/my-lora-adapter", name: "my-lora-adapter", weight_format: "safetensors LoRA" }],
    truncated: false, scope_note: "Metadata-only discovery under trusted local roots; weights are validated on run.",
  } }));
  await page.route("**/api/engine/quantization/estimate", (route) => {
    const input = route.request().postDataJSON() as Record<string, unknown>;
    expect(input.model_path).toBe("/models/llama-test");
    expect(input.context_length).toBe(16);
    expect(input.batch_size).toBe(1);
    expect(input.ram_budget_gib).toBe(8);
    expect(input.vram_budget_gib).toBe(4);
    return route.fulfill({ json: {
      schema_version: 1,
      model: {
        name: "llama-test", model_type: "llama", architecture: "LlamaForCausalLM",
        estimated_parameter_count: 1200, parameter_count_method: "analytic config estimate",
      },
      request: { context_length: 16, batch_size: 1, ram_budget_gib: 8, vram_budget_gib: 4 },
      assumptions: { safety_margin_percent: 20, uncertainty_range_percent: { low: -15, high: 25 } },
      candidates: [
        { format: "int4", nominal_weight_bits: 4, estimated_weight_gib: 0.0000005, estimated_kv_cache_gib: 0.0000002, estimated_workspace_gib: 0.000001, estimated_total_gib: 0.000003, estimated_range_gib: [0.00000255, 0.00000375], fits_ram_budget: true, fits_vram_budget: true, native_checkpoint_supported: false, native_support_note: "Sizing only." },
        { format: "int8", nominal_weight_bits: 8, estimated_weight_gib: 0.000001, estimated_kv_cache_gib: 0.0000002, estimated_workspace_gib: 0.000001, estimated_total_gib: 0.0000035, estimated_range_gib: [0.000002975, 0.000004375], fits_ram_budget: true, fits_vram_budget: true, native_checkpoint_supported: false, native_support_note: "Sizing only." },
        { format: "bf16", nominal_weight_bits: 16, estimated_weight_gib: 0.0000013, estimated_kv_cache_gib: 0.0000002, estimated_workspace_gib: 0.000001, estimated_total_gib: 0.000004, estimated_range_gib: [0.0000034, 0.000005], fits_ram_budget: true, fits_vram_budget: true, native_checkpoint_supported: true, native_support_note: null },
        { format: "fp16", nominal_weight_bits: 16, estimated_weight_gib: 0.0000013, estimated_kv_cache_gib: 0.0000002, estimated_workspace_gib: 0.000001, estimated_total_gib: 0.000004, estimated_range_gib: [0.0000034, 0.000005], fits_ram_budget: true, fits_vram_budget: true, native_checkpoint_supported: true, native_support_note: null },
        { format: "fp32", nominal_weight_bits: 32, estimated_weight_gib: 0.0000025, estimated_kv_cache_gib: 0.0000004, estimated_workspace_gib: 0.000001, estimated_total_gib: 0.000006, estimated_range_gib: [0.0000051, 0.0000075], fits_ram_budget: true, fits_vram_budget: true, native_checkpoint_supported: true, native_support_note: null },
      ],
      recommendation: {
        format: "bf16", status: "fits_budget",
        note: "Lowest estimated-memory native format that fits every supplied budget. This is a planning suggestion, not a conversion and not a guarantee that the checkpoint's stored dtype or device kernels are compatible.",
      },
      scope_note: "Planning estimates only. They do not inspect tensor contents, measure physical memory, or predict model quality/speed.",
    } });
  });
  await page.route("**/api/engine/generate/stream", (route) => {
    engineLoaded = true;
    const requestBody = route.request().postDataJSON() as Record<string, unknown>;
    expect(requestBody.top_p).toBe(0.8);
    expect(requestBody.top_k).toBe(4);
    expect(requestBody.min_p).toBe(0.05);
    expect(requestBody.repetition_penalty).toBe(1.1);
    expect(requestBody.seed).toBe(42);
    expect(requestBody.stop_sequences).toEqual(["<END>", "###"]);
    expect(requestBody.memory_mode).toBe("auto");
    const completed = {
      engine: "axiom-native-engine", execution_backend: "pytorch-native-axiom", transformers_generation_used: false, ollama_used: false,
      model: "llama-test", architecture: "LlamaForCausalLM", model_type: "llama", device: "cpu", checkpoint_format: "safetensors", decoder: "axiom-autoregressive-kv-cache", attention_backend: "pytorch-scaled-dot-product-attention",
      adapter: "my-lora-adapter", response: "AXIOM local engine test response.",
      sampling: { temperature: 0.7, top_p: 0.8, top_k: 4, min_p: 0.05, repetition_penalty: 1.1, seed: 42 },
      stop_sequences: ["<END>", "###"], stop_reason: "stop_sequence",
      prefill_ms: 40.2, decode_ms: 30.3, time_to_first_token_ms: 40.2, generation_tokens_per_second: 85.106,
      prompt_tokens: 5, completion_tokens: 6, elapsed_ms: 120.5,
      generated_at: "2026-10-21T00:00:00Z", local_only: true,
    };
    return route.fulfill({
      status: 200,
      contentType: "application/x-ndjson",
      body: [
        JSON.stringify({ type: "token", response: "AXIOM local", completion_tokens: 2 }),
        JSON.stringify({ type: "complete", result: completed }),
      ].join("\n") + "\n",
    });
  });
  await page.route("**/api/engine/generate/batch", (route) => {
    const requestBody = route.request().postDataJSON() as Record<string, unknown>;
    expect(requestBody.prompts).toEqual(["First batch prompt", "Second batch prompt"]);
    expect(requestBody.top_p).toBe(0.8);
    expect(requestBody.top_k).toBe(4);
    expect(requestBody.min_p).toBe(0.05);
    expect(requestBody.repetition_penalty).toBe(1.1);
    expect(requestBody.seed).toBe(42);
    expect(requestBody.adapter_path).toBe("/runs/my-lora-adapter");
    return route.fulfill({ json: {
      engine: "axiom-native-engine",
      execution_backend: "pytorch-native-axiom",
      transformers_generation_used: false,
      ollama_used: false,
      model: "llama-test",
      architecture: "LlamaForCausalLM",
      model_type: "llama",
      batch_size: 2,
      prompt_tokens: 8,
      completion_tokens: 6,
      elapsed_ms: 45.5,
      memory_mode: "auto",
      checkpoint_format: "safetensors",
      local_only: true,
      results: [
        { index: 1, prompt_format: "plain", context_management: { prompt_tokens_before_truncation: 4, prompt_tokens_used: 4, token_truncation_applied: false }, prompt_tokens: 4, completion_tokens: 3, stop_reason: "max_new_tokens", response: "First batch result." },
        { index: 2, prompt_format: "plain", context_management: { prompt_tokens_before_truncation: 4, prompt_tokens_used: 4, token_truncation_applied: false }, prompt_tokens: 4, completion_tokens: 3, stop_reason: "eos_token", response: "Second batch result." },
      ],
    } });
  });
  await page.route("**/api/engine/unload", (route) => {
    unloadCalls += 1;
    engineLoaded = false;
    return route.fulfill({ json: {
      schema_version: 1, unloaded: true, loaded: false,
      scope_note: "The loaded model and tokenizer references were released by AXIOM. Framework or OS-level allocators may retain some reserved memory.",
    } });
  });

  await page.goto("/?page=runtime");

  const engine = page.locator(".axiom-engine-panel");
  await expect(engine.getByRole("heading", { name: "AXIOM Local Engine" })).toBeVisible();
  await expect(engine).toContainText("own PyTorch Llama/Mistral/Qwen2/Qwen3 forward pass");
  await expect(engine).toContainText("safetensors");
  await expect(engine).toContainText("GGUF");
  await expect(engine).toContainText("dequantizing GGUF tensors to floating-point");
  await expect(engine.getByText("ENGINE READY")).toBeVisible();

  await engine.getByLabel("Select discovered native model").selectOption("/models/llama-test");
  await expect(engine.getByLabel("Local Llama / Mistral model directory")).toHaveValue("/models/llama-test");
  await engine.getByLabel("Planner context length").fill("16");
  await engine.getByLabel("Planner batch size").fill("1");
  await engine.getByLabel("RAM budget GiB", { exact: true }).fill("8");
  await engine.getByLabel("VRAM budget GiB", { exact: true }).fill("4");
  await engine.getByRole("button", { name: "Estimate memory fit" }).click();
  const memoryEstimate = engine.getByRole("region", { name: "Quantization estimate results" });
  await expect(memoryEstimate).toContainText("Budget-based planning recommendation: BF16.");
  await expect(memoryEstimate).toContainText("3.15 KiB estimated");
  await expect(memoryEstimate.getByRole("article", { name: "int4 memory candidate" })).toContainText("Sizing only");
  await engine.getByLabel("Select discovered LoRA adapter").selectOption("/runs/my-lora-adapter");
  await expect(engine.getByLabel("Optional LoRA adapter directory")).toHaveValue("/runs/my-lora-adapter");
  await engine.getByRole("button", { name: "Check adapter compatibility" }).click();
  await expect(engine.getByRole("region", { name: "LoRA device compatibility report" })).toContainText("Adapter structure supported");
  await expect(engine.getByRole("region", { name: "LoRA device compatibility report" })).toContainText("CPU");
  await expect(engine.getByRole("region", { name: "LoRA device compatibility report" })).toContainText("ROCm");
  await engine.getByLabel("AXIOM Native Engine prompt").fill("Say hello.");
  await engine.getByLabel("Top-p").fill("0.8");
  await engine.getByLabel("Top-k").fill("4");
  await engine.getByLabel("Min-p").fill("0.05");
  await engine.getByLabel("Repetition penalty").fill("1.1");
  await engine.getByLabel("Sampling seed").fill("42");
  await engine.getByLabel("Stop sequences").fill("<END>\n###");
  await engine.getByRole("button", { name: "Run with AXIOM Engine" }).click();

  await expect(engine.getByTestId("axiom-engine-output")).toContainText("AXIOM local engine test response.");
  await expect(engine).toContainText("Finish: stop sequence · Stop sequences: 2 · Attention: pytorch-scaled-dot-product-attention · Sampling: temperature 0.7 · top-p 0.8 · top-k 4 · min-p 0.05 · repetition penalty 1.1 · seed 42");
  await expect(engine).toContainText("LlamaForCausalLM");
  await expect(engine.getByRole("button", { name: "Unload model" })).toBeVisible();

  await engine.getByLabel("AXIOM Native batch prompts").fill("First batch prompt\nSecond batch prompt");
  await engine.getByRole("button", { name: "Run native batch" }).click();
  const batchResults = engine.getByRole("region", { name: "Native batch results" });
  await expect(batchResults).toContainText("2 prompt results");
  await expect(batchResults).toContainText("First batch result.");
  await expect(batchResults).toContainText("Second batch result.");

  await engine.getByRole("button", { name: "Unload model" }).click();
  await expect(engine.getByRole("status")).toContainText("released its loaded model and tokenizer references");
  await expect(engine.getByRole("button", { name: "Unload model" })).toHaveCount(0);
  expect(unloadCalls).toBe(1);

  // Starting a second request and stopping it must keep the UI responsive and
  // preserve the partial-output/cancellation status rather than reporting a
  // successful completion.
  await page.route("**/api/engine/generate/stream", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 3000));
    try {
      await route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        body: JSON.stringify({
          type: "complete",
          result: {
            engine: "axiom-native-engine", execution_backend: "pytorch-native-axiom",
            transformers_generation_used: false, ollama_used: false,
            model: "llama-test", architecture: "LlamaForCausalLM", model_type: "llama",
            adapter: null, device: "cpu", checkpoint_format: "safetensors",
            decoder: "axiom-autoregressive-kv-cache", attention_backend: "pytorch-scaled-dot-product-attention", response: "Should not arrive",
            sampling: { temperature: 0.7, top_p: 0.9, top_k: 0, min_p: 0, repetition_penalty: 1, seed: null },
            stop_sequences: [], stop_reason: "cancelled",
            prefill_ms: 10, decode_ms: 1, time_to_first_token_ms: 10, generation_tokens_per_second: 90.909,
            prompt_tokens: 3, completion_tokens: 1, elapsed_ms: 100,
            generated_at: "2026-10-22T00:00:00Z", local_only: true,
          },
        }) + "\n",
      });
    } catch {
      // The browser has already aborted the request, which is the case under test.
    }
  });
  await engine.getByRole("button", { name: "Run with AXIOM Engine" }).click();
  await expect(engine.getByRole("button", { name: "Stop generation" })).toBeVisible();
  await engine.getByRole("button", { name: "Stop generation" }).click();
  await expect(engine.getByRole("alert")).toContainText("Generation cancelled after 0 token(s)");
  await expect(engine.getByTestId("axiom-engine-output")).toHaveCount(0);
});
