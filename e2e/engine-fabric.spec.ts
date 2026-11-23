import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

const backend = (id: string, name: string, status: "ready" | "setup_required", supportsChat: boolean, models: string[] = []) => ({
  id, name, kind: id === "axiom-native" ? "native" : "openai-compatible",
  benchmark_requirements: {
    schema_version: 1,
    adapter_id: `axiom-engine-fabric-benchmark-v1:${id}`,
    backend_id: id,
    runtime: {
      kind: id === "axiom-native" ? "in_process" : "loopback_service",
      component: id,
      readiness_evidence: "Live adapter/model inventory check.",
      endpoint: id === "axiom-native" ? null : "http://127.0.0.1:11434",
      inventory_route: id === "ollama" ? "/api/tags" : "/v1/models",
    },
    model: { selection: "exact_model_id", inventory_source: "local adapter inventory", supported_formats: ["local model files"], must_be_discoverable: true, notes: "Exact model ID." },
    request: {
      route: "/api/chat", transport: "ollama_options", model_id_match: "exact",
      warmups: { minimum: 0, maximum: 1 },
      measured_requests: { minimum: 1, maximum: 5, concurrency: 1 },
      output_tokens: { minimum: 1, maximum: 256 },
      sampling: { native: ["temperature", "top_p"], extensions: [] },
    },
    privacy: { prompt_returned: false, responses_returned: false, results_persisted: false, cloud_provider_used: false },
    scope_note: "Adapter requirements only.",
  },
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
  let workspaceProfile: Record<string, unknown> | null = null;
  await page.route("**/api/engine/backends**", async (route) => {
    const requestUrl = new URL(route.request().url());
    const requestPath = requestUrl.pathname;
    const method = route.request().method();
    if (requestPath === "/api/engine/backends/profiles" && method === "GET") {
      await route.fulfill({ json: {
        schema_version: 1,
        profiles: workspaceProfile ? [workspaceProfile] : [],
        total: workspaceProfile ? 1 : 0,
        max_profiles: 25,
        credentials_persisted: false,
        custom_endpoints_persisted: false,
      } });
      return;
    }
    if (requestPath === "/api/engine/backends/profiles" && method === "POST") {
      const input = route.request().postDataJSON() as { name: string; backend_id: string; model: string };
      workspaceProfile = {
        id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        name: input.name,
        backend_id: input.backend_id,
        model: input.model,
        created_at: "2026-11-20T00:00:00Z",
        updated_at: "2026-11-20T00:00:00Z",
      };
      await route.fulfill({ status: 201, json: {
        profile: workspaceProfile,
        credentials_persisted: false,
        custom_endpoints_persisted: false,
      } });
      return;
    }
    if (requestPath.endsWith("/profiles/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/connect") && method === "POST") {
      await route.fulfill({ json: {
        profile_id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        backend_id: "ollama",
        model: "tiny-local",
        status: "verified",
        checked_at: "2026-11-20T00:00:00Z",
        connection_scope: "one_shot_local_health_and_model_inventory_check",
        persistent_connection: false,
        credentials_persisted: false,
        custom_endpoints_persisted: false,
        processes_started: false,
      } });
      return;
    }
    if (requestPath.endsWith("/profiles/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") && method === "DELETE") {
      const deleted = workspaceProfile;
      workspaceProfile = null;
      await route.fulfill({ json: { deleted, credentials_persisted: false } });
      return;
    }
    if (route.request().method() === "GET" && route.request().url().includes("/benchmark-requirements")) {
      await route.fulfill({ json: {
        schema_version: 1,
        backend_id: "ollama",
        adapter_id: "axiom-engine-fabric-benchmark-v1:ollama",
        requirements: backend("ollama", "Ollama", "ready", true, ["tiny-local"]).benchmark_requirements,
        preflight: { status: "ready", runtime_status: "ready", runtime_ready: true, model_id: "tiny-local", model_available: true, reason: null },
        can_benchmark: true,
      } });
      return;
    }
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
  await engineFabricPanel.getByRole("button", { name: "Check benchmark requirements" }).click();
  await expect(engineFabricPanel.getByText("Preflight: READY", { exact: false })).toBeVisible();
  await engineFabricPanel.getByRole("button", { name: "Benchmark selected engine" }).click();
  await expect(engineFabricPanel.getByRole("table", { name: "Measured local engine comparison" })).toContainText("21 ms");
  await expect(engineFabricPanel.getByRole("table", { name: "Measured local engine comparison" })).toContainText("350 tokens/s");
  expect(benchmarkRequestBody?.min_p).toBe(0.2);

  await engineFabricPanel.getByLabel("Engine Fabric profile name", { exact: true }).fill("Office local runtime");
  await engineFabricPanel.getByRole("button", { name: "Save selected backend + model" }).click();
  expect(workspaceProfile?.["backend_id"]).toBe("ollama");
  expect(workspaceProfile?.["model"]).toBe("tiny-local");
  const savedProfileSelect = engineFabricPanel.getByLabel("Saved Engine Fabric profile", { exact: true });
  await expect(savedProfileSelect).toContainText("Office local runtime · ollama · tiny-local");
  await savedProfileSelect.selectOption({ label: "Office local runtime · ollama · tiny-local" });
  await engineFabricPanel.getByRole("button", { name: "Apply", exact: true }).click();
  await engineFabricPanel.getByRole("button", { name: "Connect", exact: true }).click();
  await expect(engineFabricPanel.getByText("VERIFIED THIS SESSION", { exact: true })).toBeVisible();
  await engineFabricPanel.getByRole("button", { name: "Disconnect", exact: true }).click();
  await expect(engineFabricPanel.getByText("NOT CONNECTED", { exact: true })).toBeVisible();
  await expect(engineFabricPanel.getByLabel("Prompt")).toHaveValue("");
  await expect(engineFabricPanel.getByLabel("Model ID")).toHaveValue("");
  await expect(page.locator("body")).toContainText("external runtime remains running");
  await expect(page.locator("body")).toContainText("saved profile is retained for a later reconnect");
  await engineFabricPanel.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(savedProfileSelect).toHaveValue("");
  expect(workspaceProfile).toBeNull();
});
