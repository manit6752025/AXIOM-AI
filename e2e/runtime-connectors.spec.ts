import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime page displays passive connector discovery without claiming connection", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.route("**/api/workspace/runtime", (route) => route.fulfill({
    json: {
      status: "offline",
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
  await page.route("**/api/diagnostics/runtime-connectors", (route) => route.fulfill({
    json: {
      schema_version: 1,
      discovery_mode: "passive_local_discovery",
      checked_at: "2026-10-20T00:00:00Z",
      scope_note: "Presence is not a connection.",
      connectors: [
        { id: "ollama", name: "Ollama", kind: "executable", present: true, status: "discovered_not_connected", potential_capabilities: ["local_inference"], evidence: "found on PATH", connection_checked: false, note: "No process contacted." },
        { id: "vllm-cli", name: "vLLM CLI", kind: "executable", present: false, status: "not_detected", potential_capabilities: ["serving"], evidence: "not present", connection_checked: false, note: "No process contacted." },
      ],
    },
  }));

  await page.goto("/?page=runtime");
  const connectorsPanel = page.locator(".runtime-connectors");
  await expect(connectorsPanel.getByRole("heading", { name: "Runtime connectors" })).toBeVisible();
  await expect(connectorsPanel.getByText("Ollama", { exact: true })).toBeVisible();
  await expect(connectorsPanel.getByText("DISCOVERED", { exact: true })).toBeVisible();
  await expect(connectorsPanel.getByText("vLLM CLI", { exact: true })).toBeVisible();
  await expect(page.locator("body")).toContainText("does not start processes");
  await expect(page.locator("body")).not.toContainText("CONNECTED");
});
