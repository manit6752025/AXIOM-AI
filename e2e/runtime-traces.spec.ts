import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime request traces show safe metadata, export JSON, and clear the in-memory buffer", async ({ page }) => {
  let traces: Array<Record<string, unknown>> = [
    {
      request_id: "ab12cd34ef56ab12cd34ef56ab12cd34",
      operation: "inference.generate",
      model: "local-test-model",
      status: "success",
      started_at: "2026-10-22T08:00:00+00:00",
      completed_at: "2026-10-22T08:00:01+00:00",
      duration_ms: 1001.2,
      prompt_characters: 38,
      requested_output_tokens: 64,
      temperature: 0.2,
      warmup_runs: null,
      iterations: null,
      concurrency: null,
      measured_requests: null,
      prompt_tokens: 10,
      completion_tokens: 20,
      completion_tokens_known: null,
      completion_tokens_per_second: null,
      error_code: null,
    },
    {
      request_id: "de98fa76bc54de98fa76bc54de98fa76",
      operation: "inference.benchmark",
      model: "local-test-model",
      status: "error",
      started_at: "2026-10-22T08:01:00+00:00",
      completed_at: "2026-10-22T08:01:01+00:00",
      duration_ms: 55.6,
      prompt_characters: 38,
      requested_output_tokens: 64,
      temperature: 0.2,
      warmup_runs: 1,
      iterations: 2,
      concurrency: 1,
      measured_requests: 2,
      prompt_tokens: null,
      completion_tokens: null,
      completion_tokens_known: null,
      completion_tokens_per_second: null,
      error_code: "runtime_unavailable",
    },
  ];
  let clearCount = 0;

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
        status: "online", models: ["local-test-model"], error: null, checked_at: "2026-10-22T00:00:00Z",
      } });
    } else if (url.pathname === "/api/diagnostics/runtime-traces" && request.method() === "GET") {
      await route.fulfill({ json: {
        schema_version: 1,
        traces,
        total_retained: traces.length,
        limit: 100,
        max_retained: 200,
        privacy: { prompt_bodies_stored: false, response_bodies_stored: false, disk_persistence: false },
      } });
    } else if (url.pathname === "/api/diagnostics/runtime-traces" && request.method() === "DELETE") {
      clearCount += 1;
      traces = [];
      await route.fulfill({ json: { cleared: 2, remaining: 0, disk_persistence: false } });
    } else if (url.pathname === "/api/engine/status") {
      await route.fulfill({ json: {
        engine: "axiom-local-engine", ready: false, packages: {}, versions: {}, loaded: false,
        architecture: null, model_loading: "local_files_only", remote_code: false,
        setup_note: "Optional local dependencies are not installed.", scope_note: "Test stub.",
        checked_at: "2026-10-22T00:00:00Z",
      } });
    } else if (url.pathname === "/api/diagnostics/runtime-connectors") {
      await route.fulfill({ json: {
        schema_version: 1, discovery_mode: "passive_local_discovery", checked_at: "2026-10-22T00:00:00Z",
        connectors: [], scope_note: "Test stub.",
      } });
    } else if (url.pathname === "/api/models/studio/status") {
      await route.fulfill({ json: {
        schema_version: 1, ollama_installed: true, ollama_endpoint: "http://127.0.0.1:11434",
        download_limit_bytes: 1000000, gguf_directory: "models", catalogue: "Hugging Face",
        network_search_is_explicit: true,
      } });
    } else if (url.pathname === "/api/models/studio/jobs") {
      await route.fulfill({ json: { jobs: [] } });
    } else {
      await route.fulfill({ status: 404, json: { detail: "No fixture for this runtime-trace test." } });
    }
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");
  const tracesPanel = page.getByRole("region", { name: "Runtime request trace history", exact: true });
  await expect(page.getByRole("heading", { name: "Runtime request trace history" })).toBeVisible();
  await expect(page.getByRole("table", { name: "Runtime request trace history" }).getByRole("row")).toHaveCount(3);
  await expect(page.getByText("inference.generate", { exact: true })).toBeVisible();
  await expect(page.getByText("runtime_unavailable", { exact: true })).toBeVisible();
  await expect(tracesPanel).not.toContainText("prompt_body");
  await expect(tracesPanel).not.toContainText("response_body");

  const downloadPromise = page.waitForEvent("download");
  await tracesPanel.getByRole("button", { name: "Export JSON" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("axiom-runtime-traces.json");
  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();
  const exportedJson = await readFile(downloadPath as string, "utf8");
  expect(exportedJson).toContain('"prompt_bodies_stored": false');
  expect(exportedJson).toContain("ab12cd34ef56ab12cd34ef56ab12cd34");
  expect(exportedJson).not.toContain("prompt_text");
  expect(exportedJson).not.toContain("response_text");

  await tracesPanel.getByRole("button", { name: "Clear history" }).click();
  await expect(page.getByRole("dialog", { name: "Clear runtime trace history?" })).toBeVisible();
  await page.getByRole("button", { name: "Clear in-memory history" }).click();
  await expect(page.getByRole("heading", { name: "No runtime traces retained" })).toBeVisible();
  expect(clearCount).toBe(1);
  await expect(tracesPanel.getByRole("button", { name: "Export JSON" })).toBeDisabled();
});
