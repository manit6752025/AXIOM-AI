import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test.describe("Measured local inference benchmark", () => {
  test("uses selected prompt settings, reports measured metrics, and returns no generated text", async ({ page }) => {
    let submitted: Record<string, unknown> | null = null;
    const benchmarkResult = {
      schema_version: 1,
      backend: "ollama",
      endpoint: "http://127.0.0.1:11434",
      model: "test-model",
      configuration: {
        prompt_characters: 25,
        num_predict: 64,
        temperature: 0.2,
        warmup_runs: 1,
        iterations: 2,
        concurrency: 1,
        measured_requests: 2,
      },
      measurements: {
        wall_time_ms: 420,
        requests_per_second: 4.762,
        completion_tokens_known: 2,
        prompt_tokens_known: 2,
        total_completion_tokens: 18,
        completion_tokens_per_second: 42.857,
        latency_ms: { min: 160, median: 200, p95: 240, max: 240 },
        requests: [
          { latency_ms: 160, prompt_tokens: 9, completion_tokens: 8, done: true },
          { latency_ms: 240, prompt_tokens: 9, completion_tokens: 10, done: true },
        ],
      },
      privacy: { prompt_returned: false, responses_returned: false, results_persisted: false },
      scope_note: "Measured loopback runtime timings; compare only identical configurations.",
    };

    await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.pathname === "/api/workspace/runtime") {
        await route.fulfill({ json: { status: "ready", hardware: {
          os_name: "Linux", architecture: "x86_64", cpu_cores: 8, ram_gb: 16,
          gpu_name: null, vram_gb: null, cuda_available: false,
        } } });
      } else if (url.pathname === "/api/inference/status") {
        await route.fulfill({ json: {
          backend: "ollama", endpoint: "http://127.0.0.1:11434", runtime_installed: true,
          status: "online", models: ["test-model"], error: null, checked_at: "2026-10-22T00:00:00Z",
        } });
      } else if (url.pathname === "/api/inference/benchmark" && request.method() === "POST") {
        submitted = request.postDataJSON() as Record<string, unknown>;
        await route.fulfill({ json: benchmarkResult });
      } else if (url.pathname === "/api/engine/status") {
        await route.fulfill({ json: {
          engine: "axiom-local-engine", ready: false,
          packages: {}, versions: {}, loaded: false, architecture: null,
          model_loading: "local_files_only", remote_code: false,
          setup_note: "Optional local dependencies are not installed.",
          scope_note: "Test stub; no engine was loaded.", checked_at: "2026-10-22T00:00:00Z",
        } });
      } else if (url.pathname === "/api/diagnostics/runtime-connectors") {
        await route.fulfill({ json: {
          schema_version: 1, discovery_mode: "passive_local_discovery",
          checked_at: "2026-10-22T00:00:00Z", connectors: [], scope_note: "Test stub.",
        } });
      } else if (url.pathname === "/api/diagnostics/runtime-traces" && request.method() === "GET") {
        await route.fulfill({ json: {
          schema_version: 1, traces: [], total_retained: 0, limit: 100, max_retained: 200,
          privacy: { prompt_bodies_stored: false, response_bodies_stored: false, disk_persistence: false },
        } });
      } else if (url.pathname === "/api/engine/models") {
        await route.fulfill({ json: {
          schema_version: 1, models: [], adapters: [], truncated: false,
          scope_note: "No compatible local models configured for this test.",
        } });
      } else if (url.pathname === "/api/models/studio/status") {
        await route.fulfill({ json: {
          schema_version: 1, ollama_installed: true,
          ollama_endpoint: "http://127.0.0.1:11434", download_limit_bytes: 1000000,
          gguf_directory: "models", catalogue: "Hugging Face", network_search_is_explicit: true,
        } });
      } else if (url.pathname === "/api/models/studio/jobs") {
        await route.fulfill({ json: { jobs: [] } });
      } else {
        await route.fulfill({ status: 404, json: { detail: "No test fixture for this endpoint." } });
      }
    });

    await ensureAuthenticated(page);
    await page.goto("/?page=runtime");
    const inferencePanel = page.locator(".ollama-inference-panel");
    await expect(page.getByRole("heading", { level: 1, name: "Runtime", exact: true })).toBeVisible();
    await expect(inferencePanel.getByRole("heading", { name: "Inference benchmark" })).toBeVisible();
    await expect(inferencePanel.getByText("OLLAMA ONLINE", { exact: true })).toBeVisible();

    await inferencePanel.getByPlaceholder("Ask your local model to do something…").fill("Describe local inference.");
    await inferencePanel.getByRole("spinbutton").first().fill("64");
    await inferencePanel.getByRole("spinbutton").nth(1).fill("0.2");
    await inferencePanel.getByRole("combobox", { name: "Benchmark warm-up generations" }).selectOption("1");
    await inferencePanel.getByRole("combobox", { name: "Benchmark iterations" }).selectOption("2");
    await inferencePanel.getByRole("combobox", { name: "Benchmark concurrency" }).selectOption("1");
    await inferencePanel.getByRole("button", { name: "Benchmark this prompt" }).click();

    await expect(inferencePanel.getByRole("region", { name: "Inference benchmark result" })).toBeVisible();
    await expect(inferencePanel.getByText("42.857 tokens/s", { exact: true })).toBeVisible();
    await expect(inferencePanel.getByText("200 ms", { exact: true })).toBeVisible();
    await expect(inferencePanel.getByText("P95 240 ms", { exact: true })).toBeVisible();
    const requestTable = page.getByRole("table", { name: "Per-request benchmark timings" });
    await expect(requestTable.getByRole("row")).toHaveCount(3);
    await expect(requestTable).toContainText("8");
    await expect(requestTable).toContainText("10");

    expect(submitted).toMatchObject({
      model: "test-model",
      prompt: "Describe local inference.",
      num_predict: 64,
      warmup_runs: 1,
      iterations: 2,
      concurrency: 1,
    });
    const visibleResult = await inferencePanel.getByRole("region", { name: "Inference benchmark result" }).innerText();
    expect(visibleResult).not.toContain("response text");
    expect(visibleResult).not.toContain("Describe local inference.");
  });

  test("blocks a requested workload above the ten measured-request limit", async ({ page }) => {
    let benchmarkRequests = 0;
    await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.pathname === "/api/workspace/runtime") {
        await route.fulfill({ json: { status: "ready", hardware: { os_name: "Linux", architecture: "x86_64", cpu_cores: 8, ram_gb: 16, gpu_name: null, vram_gb: null, cuda_available: false } } });
      } else if (url.pathname === "/api/inference/status") {
        await route.fulfill({ json: { backend: "ollama", endpoint: "http://127.0.0.1:11434", runtime_installed: true, status: "online", models: ["test-model"], error: null, checked_at: "2026-10-22T00:00:00Z" } });
      } else if (url.pathname === "/api/engine/status") {
        await route.fulfill({ json: { engine: "axiom-local-engine", ready: false, packages: {}, versions: {}, loaded: false, architecture: null, model_loading: "local_files_only", remote_code: false, setup_note: "Optional dependencies missing.", scope_note: "Test.", checked_at: "2026-10-22T00:00:00Z" } });
      } else if (url.pathname === "/api/diagnostics/runtime-connectors") {
        await route.fulfill({ json: { schema_version: 1, discovery_mode: "passive_local_discovery", checked_at: "2026-10-22T00:00:00Z", connectors: [], scope_note: "Test." } });
      } else if (url.pathname === "/api/diagnostics/runtime-traces" && request.method() === "GET") {
        await route.fulfill({ json: { schema_version: 1, traces: [], total_retained: 0, limit: 100, max_retained: 200, privacy: { prompt_bodies_stored: false, response_bodies_stored: false, disk_persistence: false } } });
      } else if (url.pathname === "/api/engine/models") {
        await route.fulfill({ json: { schema_version: 1, models: [], adapters: [], truncated: false, scope_note: "No compatible local models configured for this test." } });
      } else if (url.pathname === "/api/models/studio/status") {
        await route.fulfill({ json: { schema_version: 1, ollama_installed: true, ollama_endpoint: "http://127.0.0.1:11434", download_limit_bytes: 1000000, gguf_directory: "models", catalogue: "Hugging Face", network_search_is_explicit: true } });
      } else if (url.pathname === "/api/models/studio/jobs") {
        await route.fulfill({ json: { jobs: [] } });
      } else if (url.pathname === "/api/inference/benchmark") {
        benchmarkRequests += 1;
        await route.fulfill({ status: 500, json: { detail: "Should not be called" } });
      } else {
        await route.fulfill({ status: 404, json: { detail: "No test fixture." } });
      }
    });

    await ensureAuthenticated(page);
    await page.goto("/?page=runtime");
    const inferencePanel = page.locator(".ollama-inference-panel");
    await inferencePanel.getByPlaceholder("Ask your local model to do something…").fill("Test bounded work.");
    await inferencePanel.getByRole("combobox", { name: "Benchmark iterations" }).selectOption("10");
    await inferencePanel.getByRole("combobox", { name: "Benchmark concurrency" }).selectOption("2");
    await inferencePanel.getByRole("button", { name: "Benchmark this prompt" }).click();
    await expect(page.getByText(/iterations × concurrency must not exceed 10/i)).toBeVisible();
    expect(benchmarkRequests).toBe(0);
  });
});
