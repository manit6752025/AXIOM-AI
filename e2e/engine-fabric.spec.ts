import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

const backend = (id: string, name: string, status: "ready" | "setup_required", supportsChat: boolean, models: string[] = []) => ({
  id, name, kind: id === "axiom-native" ? "native" : "openai-compatible",
  endpoint: id === "axiom-native" ? null : "http://127.0.0.1:8080",
  supports_chat: supportsChat, formats: ["local model files"],
  strengths: ["Local execution", "No cloud routing", "Bounded API"],
  tradeoffs: ["Model/backend compatibility must be checked", "Measured speed varies by hardware"],
  best_for: ["balanced"], status, installed: status === "ready", reachable: status === "ready",
  models, model_count: models.length, error: status === "ready" ? null : "Install or configure a compatible local model.",
  scope: id === "axiom-native" ? "in_process_native_engine" : "fixed_loopback_only",
  generation: {
    transport: id === "axiom-native" ? "in_process" : id === "ollama" ? "ollama_options" : "openai_compatible",
    parameter_mapping: {
      max_new_tokens: id === "axiom-native" ? "max_new_tokens" : id === "ollama" ? "options.num_predict" : "max_tokens",
      temperature: id === "ollama" ? "options.temperature" : "temperature",
      top_p: id === "ollama" ? "options.top_p" : "top_p",
      top_k: id === "ollama" ? "options.top_k" : "top_k (extension)",
      min_p: id === "axiom-native" ? "min_p" : id === "ollama" ? "options.min_p" : "unsupported",
      repetition_penalty: id === "axiom-native" ? "repetition_penalty" : id === "ollama" ? "options.repeat_penalty" : id === "vllm" ? "repetition_penalty" : "repeat_penalty",
      seed: id === "ollama" ? "options.seed" : "seed",
      stop_sequences: id === "ollama" ? "options.stop" : "stop",
    },
    native_sampling_controls: id === "axiom-native" || id === "ollama"
      ? ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed", "stop_sequences"]
      : ["temperature", "top_p", "seed", "stop_sequences"],
    extension_sampling_controls: id === "axiom-native" || id === "ollama" ? [] : ["top_k", "repetition_penalty"],
    note: "Test fixture; parameter acceptance depends on the selected runtime.",
  },
});

test("Engine Fabric compares local engines and routes chat to the chosen service", async ({ page }) => {
  await ensureAuthenticated(page);
  let chatRequestBody: Record<string, unknown> | undefined;
  let benchmarkRequestBody: Record<string, unknown> | undefined;
  await page.route("**/api/engine/backends**", async (route) => {
    if (route.request().method() === "POST" && route.request().url().includes("/benchmark")) {
      benchmarkRequestBody = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({ json: {
        schema_version: 1, backend_id: "ollama", backend_name: "Ollama", model: "tiny-local",
        configuration: { prompt_characters: 18, max_new_tokens: 16, temperature: 0.2, top_p: 0.9, top_k: 0, min_p: 0.2, repetition_penalty: 1, seed: null, stop_sequence_count: 0, warmup_runs: 0, iterations: 2, measured_requests: 2 },
        measurements: {
          wall_time_ms: 42, requests_per_second: 47.6, completion_tokens_known: 2, total_completion_tokens: 14,
          completion_tokens_per_second: 350, latency_ms: { min: 20, median: 21, p95: 22, max: 22 },
          requests: [{ latency_ms: 20, completion_tokens: 7 }, { latency_ms: 22, completion_tokens: 7 }],
        },
        privacy: { prompt_returned: false, responses_returned: false, results_persisted: false },
        scope_note: "Measured local calls only.",
      } });
      return;
    }
    if (route.request().method() === "POST") {
      chatRequestBody = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({ json: {
        schema_version: 1, backend_id: "ollama", backend_name: "Ollama", model: "tiny-local",
        response: "A reply from the selected local backend.", done: true, prompt_eval_count: 5,
        eval_count: 7, total_duration_ms: 20, generated_at: "2026-10-24T00:00:00Z",
        execution: "local_loopback", privacy: { prompt_returned: false, response_persisted: false, cloud_provider_used: false },
      } });
      return;
    }
    await route.fulfill({ json: {
      schema_version: 1, checked_at: "2026-10-24T00:00:00Z", host: { os: "Linux", architecture: "x86_64" },
      profile: "balanced",
      profiles: [
        { id: "balanced", label: "Balanced" }, { id: "low-memory", label: "Low memory" },
        { id: "throughput", label: "High throughput" }, { id: "easy-setup", label: "Easy setup" },
        { id: "portable", label: "Portable / quantized" }, { id: "minimal-daemons", label: "Minimal background services" },
      ],
      backends: [
        backend("axiom-native", "AXIOM Native Engine", "setup_required", false),
        backend("ollama", "Ollama", "ready", true, ["qwen2.5:0.5b", "tiny-local"]),
        backend("llama-cpp", "llama.cpp server", "ready", true, ["tiny-local"]),
        backend("lm-studio", "LM Studio local server", "setup_required", true),
        backend("vllm", "vLLM local server", "setup_required", true),
        backend("llama-cpp-python", "llama-cpp-python server", "setup_required", true),
      ],
      recommendation: { profile: "balanced", backend_id: "llama-cpp", name: "llama.cpp server", status: "ready", reason: "A ready quantized local runtime is available." },
      security: { network_scope: "IPv4 loopback only", environment_proxies_used: false, remote_endpoints_configurable: false, credentials_sent: false, model_downloads_started: false, processes_launched: false },
      scope_note: "Only local APIs were inspected.",
    } });
  });
  await page.goto("/?page=runtime");
  const engineFabricPanel = page.locator(".engine-fabric-panel");
  await expect(engineFabricPanel.getByRole("heading", { name: "Choose the right local engine" })).toBeVisible();
  await expect(engineFabricPanel.getByText("AXIOM Native Engine", { exact: true }).first()).toBeVisible();
  await expect(engineFabricPanel.getByText("Recommended for this profile").first()).toBeVisible();
  await engineFabricPanel.getByLabel("Execution strategy", { exact: true }).selectOption("selected");
  await engineFabricPanel.getByLabel("Local engine", { exact: true }).selectOption("ollama");
  await engineFabricPanel.getByLabel("Model ID").fill("tiny-local");
  await engineFabricPanel.getByLabel("Min-p").fill("0.2");
  await engineFabricPanel.getByPlaceholder("Ask the selected local model to do something…").fill("Say hello locally.");
  await engineFabricPanel.getByRole("button", { name: "Run on selected engine" }).click();
  await expect(engineFabricPanel.getByRole("region", { name: "Engine Fabric response" })).toContainText("A reply from the selected local backend.");
  expect(chatRequestBody?.min_p).toBe(0.2);
  await engineFabricPanel.getByRole("button", { name: "Benchmark selected engine" }).click();
  await expect(engineFabricPanel.getByRole("table", { name: "Measured local engine comparison" })).toContainText("21 ms");
  await expect(engineFabricPanel.getByRole("table", { name: "Measured local engine comparison" })).toContainText("350 tokens/s");
  expect(benchmarkRequestBody?.min_p).toBe(0.2);
});
