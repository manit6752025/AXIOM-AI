import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime can start Ollama, pull a model with progress, and stop only the managed service", async ({ page }) => {
  // Model cards provide an "Unload from memory" action for each loaded model.
  // Endpoint acceptance: "/api/inference/models/running", "/api/inference/models/inspect", and "/api/inference/models/unload".
  await ensureAuthenticated(page);
  let online = false;
  let managed = false;
  let pullStatusChecks = 0;

  const report = () => ({
    backend: "ollama",
    endpoint: "http://127.0.0.1:11434",
    runtime_installed: true,
    status: online ? "online" : "offline",
    models: online && pullStatusChecks > 0 ? ["qwen2.5:0.5b"] : [],
    error: online ? null : "Ollama is offline in this test.",
    checked_at: "2026-10-24T00:00:00Z",
    service_managed_by_axiom: managed,
    service_can_start: !online && !managed,
    service_can_stop: managed,
    service_control_scope: "loopback_only",
    service_message: managed ? "AXIOM started this Ollama service." : "Ollama service status.",
  });

  await page.route("**/api/workspace/runtime", (route) => route.fulfill({ json: {
    status: "offline",
    hardware: { os_name: "Linux", architecture: "x86_64", cpu_cores: 8, ram_gb: 16, gpu_name: null, vram_gb: null, cuda_available: false },
  } }));
  await page.route("**/api/inference/status", (route) => route.fulfill({ json: report() }));
  await page.route("**/api/inference/service/start", async (route) => {
    online = true; managed = true;
    await route.fulfill({ json: { started: true, already_running: false, service: report() } });
  });
  await page.route("**/api/inference/service/stop", async (route) => {
    online = false; managed = false;
    await route.fulfill({ json: { stopped: true, service: report() } });
  });
  await page.route("**/api/inference/models/pull", async (route) => {
    await route.fulfill({ json: {
      id: "a".repeat(32), model: "qwen2.5:0.5b", status: "queued",
      created_at: "2026-10-24T00:00:00Z", updated_at: "2026-10-24T00:00:00Z",
      progress: "Queued", percent: null, logs: [], error: null, cancel_requested: false,
    } });
  });
  await page.route("**/api/inference/models/running", async (route) => {
    await route.fulfill({ json: {
      schema_version: 1, backend: "ollama", endpoint: "http://127.0.0.1:11434", count: 1,
      models: [{ name: "qwen2.5:0.5b", size_bytes: 800000000, size_vram_bytes: 300000000,
        size_ram_bytes: 500000000, memory_placement: "mixed", expires_at: "2026-10-24T10:00:00Z",
        digest_prefix: "abcdef123456", format: "gguf", family: "qwen2", parameter_size: "0.5B", quantization_level: "Q4_K_M" }],
      scope_note: "Point-in-time memory view. Runtime-reported estimates.",
    } });
  });
  await page.route("**/api/inference/models/inspect", async (route) => {
    await route.fulfill({ json: {
      schema_version: 1, model: "qwen2.5:0.5b", installed: true, size_bytes: 401000000,
      modified_at: "2026-10-24T00:00:00Z", format: "gguf", family: "qwen2", families: ["qwen2"],
      parameter_size: "0.5B", parameter_count: 494032768, quantization_level: "Q4_K_M", context_length: 32768,
      capabilities: ["completion", "tools"], template_configured: true, license_present: true,
      system_prompt_configured: false, scope_note: "Metadata remains local-only.",
    } });
  });
  await page.route("**/api/inference/models/unload", async (route) => {
    await route.fulfill({ json: {
      schema_version: 1, model: "qwen2.5:0.5b", unloaded: true, model_files_deleted: false,
      scope_note: "Ollama was asked to release this model from memory. Downloaded model files remain installed.",
    } });
  });
  await page.route("**/api/inference/models/pull/*", async (route) => {
    pullStatusChecks += 1;
    await route.fulfill({ json: {
      id: "a".repeat(32), model: "qwen2.5:0.5b", status: "completed",
      created_at: "2026-10-24T00:00:00Z", updated_at: "2026-10-24T00:00:01Z",
      progress: "Model downloaded and registered locally", percent: 100,
      logs: ["pulling manifest", "Model download complete"], error: null, cancel_requested: false,
    } });
  });

  await page.goto("/?page=runtime");
  const inferencePanel = page.locator(".ollama-inference-panel");
  await expect(inferencePanel.getByRole("heading", { name: "Local model inference" })).toBeVisible();
  await expect(inferencePanel.getByText("OLLAMA OFFLINE", { exact: true })).toBeVisible();
  await inferencePanel.getByRole("button", { name: "Start Ollama" }).click();
  await expect(inferencePanel.getByText("OLLAMA ONLINE", { exact: true })).toBeVisible();

  await inferencePanel.getByLabel("Ollama model tag").fill("qwen2.5:0.5b");
  await inferencePanel.getByRole("button", { name: "Pull model" }).click();
  await expect(inferencePanel.getByText("COMPLETED", { exact: true })).toBeVisible({ timeout: 5000 });
  await expect(inferencePanel.getByText("Model downloaded and registered locally")).toBeVisible();

  await inferencePanel.getByRole("button", { name: "Show loaded models" }).click();
  await expect(inferencePanel.getByRole("heading", { name: "Loaded models · 1" })).toBeVisible();
  await expect(inferencePanel.getByText("CPU + GPU", { exact: true })).toBeVisible();
  await expect(inferencePanel.getByText("GPU memory: 0.28 GiB", { exact: true })).toBeVisible();

  await inferencePanel.getByRole("button", { name: "Inspect model" }).click();
  await expect(inferencePanel.getByText("Q4_K_M", { exact: true })).toBeVisible();
  await expect(inferencePanel.getByText("32,768", { exact: true })).toBeVisible();
  await inferencePanel.getByRole("button", { name: "Unload qwen2.5:0.5b from memory" }).click();
  await expect(inferencePanel.getByText("Downloaded model files remain installed.", { exact: false })).toBeVisible();

  await inferencePanel.getByRole("button", { name: "Stop AXIOM service" }).click();
  await expect(inferencePanel.getByText("OLLAMA OFFLINE", { exact: true })).toBeVisible();
});
