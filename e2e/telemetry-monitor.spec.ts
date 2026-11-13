import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime resource monitoring is opt-in, bounded, exportable, and clearable", async ({ page }) => {
  let telemetryCalls = 0;

  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());

    if (url.pathname === "/api/workspace/runtime") {
      await route.fulfill({ json: { status: "ready", hardware: {
        os_name: "Linux", architecture: "x86_64", cpu_cores: 8, ram_gb: 16,
        gpu_name: null, vram_gb: null, cuda_available: false,
      } } });
    } else if (url.pathname === "/api/engine/telemetry" && request.method() === "GET") {
      telemetryCalls += 1;
      await route.fulfill({ json: {
        schema_version: 1,
        sampled_at: `2026-11-13T00:00:${String(telemetryCalls).padStart(2, "0")}+00:00`,
        sample_type: "point_in_time",
        cpu: { logical_cores: 8, system_percent: 12.5 + telemetryCalls, process_percent: 2.5 },
        memory: { total_bytes: 16 * 1024 ** 3, available_bytes: 8 * 1024 ** 3, used_bytes: 8 * 1024 ** 3, percent: 50 },
        process: { pid: 7, rss_bytes: 128 * 1024 ** 2, peak_rss_bytes: 256 * 1024 ** 2, threads: 4 },
        devices: [],
        loaded_runtime: {
          loaded: false, model: null, architecture: null, device: null, memory_mode: null,
          precision: null, last_generation: null, device_allocator: null,
        },
        disks: [],
        privacy_scope: {
          prompt_content_included: false, generated_content_included: false, model_paths_included: false,
          environment_values_included: false, credentials_included: false, network_access_performed: false,
        },
        scope_note: "Test telemetry contains resource values only.",
      } });
    } else if (url.pathname === "/api/diagnostics/runtime-traces" && request.method() === "GET") {
      await route.fulfill({ json: {
        schema_version: 1, traces: [], total_retained: 0, limit: 100, max_retained: 200,
        privacy: { prompt_bodies_stored: false, response_bodies_stored: false, disk_persistence: false },
      } });
    } else if (url.pathname === "/api/diagnostics/runtime-connectors") {
      await route.fulfill({ json: {
        schema_version: 1, discovery_mode: "passive_local_discovery", checked_at: "2026-11-13T00:00:00Z",
        connectors: [], scope_note: "Test stub.",
      } });
    } else if (url.pathname === "/api/engine/status") {
      await route.fulfill({ json: {
        engine: "axiom-local-engine", ready: false, packages: {}, versions: {}, loaded: false,
        architecture: null, model_loading: "local_files_only", remote_code: false,
        setup_note: "Optional local dependencies are not installed.", scope_note: "Test stub.",
        checked_at: "2026-11-13T00:00:00Z",
      } });
    } else if (url.pathname === "/api/inference/status") {
      await route.fulfill({ json: {
        backend: "ollama", endpoint: "http://127.0.0.1:11434", runtime_installed: false,
        status: "offline", models: [], error: null, checked_at: "2026-11-13T00:00:00Z",
      } });
    } else if (url.pathname === "/api/models/studio/status") {
      await route.fulfill({ json: {
        schema_version: 1, ollama_installed: false, ollama_endpoint: "http://127.0.0.1:11434",
        download_limit_bytes: 1000000, gguf_directory: "models", catalogue: "Hugging Face",
        network_search_is_explicit: true,
      } });
    } else if (url.pathname === "/api/models/studio/jobs") {
      await route.fulfill({ json: { jobs: [] } });
    } else {
      await route.fulfill({ status: 404, json: { detail: "No fixture for this resource-monitor test." } });
    }
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");

  const monitor = page.getByRole("region", { name: "Resource monitoring", exact: true });
  await expect(monitor.getByRole("heading", { name: "Resource monitoring" })).toBeVisible();
  await expect(monitor.getByText("No resource samples collected")).toBeVisible();
  expect(telemetryCalls).toBe(0);

  await monitor.getByLabel("Sample interval").selectOption("1");
  await monitor.getByRole("button", { name: "Start monitoring" }).click();
  await expect.poll(() => telemetryCalls, { timeout: 10_000 }).toBeGreaterThanOrEqual(2);
  await expect(monitor.getByText(/\d+\/120/)).toBeVisible();
  await expect(monitor.getByRole("table", { name: "Resource telemetry samples" })).toBeVisible();

  await monitor.getByRole("button", { name: "Stop monitoring" }).click();
  const downloadPromise = page.waitForEvent("download");
  await monitor.getByRole("button", { name: "Export JSON" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("axiom-resource-telemetry.json");
  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();
  const exported = await (await import("node:fs/promises")).readFile(downloadPath as string, "utf8");
  const payload = JSON.parse(exported);
  expect(payload.sample_count).toBeGreaterThanOrEqual(2);
  expect(payload.privacy.model_paths_included).toBe(false);
  expect(payload.privacy.prompt_content_included).toBe(false);

  await monitor.getByRole("button", { name: "Clear history" }).click();
  await expect(monitor.getByText("No resource samples collected")).toBeVisible();
  await expect(monitor.getByRole("button", { name: "Export JSON" })).toBeDisabled();
  expect(telemetryCalls).toBeGreaterThanOrEqual(2);
});
