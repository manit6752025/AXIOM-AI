import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Dashboard hardware history is opt-in, captures on demand, and clears retained records", async ({ page }) => {
  let enabled = false;
  let snapshots: Array<Record<string, unknown>> = [];
  let getCalls = 0;
  let captureCalls = 0;
  let maxSnapshots = 20;
  let maxAgeDays = 90;

  const report = () => ({
    schema_version: 1,
    settings: { enabled, max_snapshots: maxSnapshots, max_age_days: maxAgeDays },
    total: snapshots.length,
    snapshots,
    privacy: {
      host_identity_included: false,
      network_identifiers_included: false,
      filesystem_paths_included: false,
      credentials_included: false,
      model_or_dataset_content_included: false,
      automatic_sampling: false,
    },
    scope_note: "Explicit local hardware snapshots only. No automatic sampling.",
  });

  await ensureAuthenticated(page);
  await page.route("**/api/workspace/hardware-snapshots**", async (route) => {
    const method = route.request().method();
    const url = new URL(route.request().url());
    if (method === "GET" && url.pathname === "/api/workspace/hardware-snapshots") {
      getCalls += 1;
      await route.fulfill({ json: report() });
      return;
    }
    if (method === "PUT" && url.pathname.endsWith("/settings")) {
      const body = route.request().postDataJSON() as { enabled: boolean; max_snapshots: number; max_age_days: number };
      enabled = body.enabled;
      maxSnapshots = body.max_snapshots;
      maxAgeDays = body.max_age_days;
      await route.fulfill({ json: report() });
      return;
    }
    if (method === "POST" && url.pathname === "/api/workspace/hardware-snapshots") {
      captureCalls += 1;
      if (!enabled) {
        await route.fulfill({ status: 409, json: { detail: "Hardware snapshot history is disabled." } });
        return;
      }
      const hardware = {
        os_name: "Linux", architecture: "x86_64", cpu_cores: 8,
        ram_bytes: 32 * 1024 ** 3, gpu_name: "Test GPU", vram_bytes: 12 * 1024 ** 3,
        cuda_available: true,
      };
      snapshots = [{
        id: "a".repeat(32), captured_at: "2026-11-21T09:30:00+00:00",
        fingerprint: "b".repeat(64), source: "explicit_capture",
        software_version: "test", hardware,
      }];
      await route.fulfill({ status: 201, json: { snapshot: snapshots[0], history: report() } });
      return;
    }
    if (method === "DELETE") {
      snapshots = [];
      await route.fulfill({ json: report() });
      return;
    }
    await route.fulfill({ status: 405, json: { detail: "Method not allowed." } });
  });

  await page.goto("/?page=dashboard");
  const panel = page.getByRole("region", { name: "Hardware snapshot history" });
  await expect(panel.getByRole("heading", { name: "Hardware snapshot history" })).toBeVisible();
  await expect(panel.getByText("CAPTURE OFF")).toBeVisible();
  await expect(panel.getByRole("button", { name: "Capture snapshot" })).toBeDisabled();
  expect(captureCalls).toBe(0);
  expect(getCalls).toBeGreaterThan(0);

  await panel.getByLabel("Maximum retained snapshots").fill("5");
  await panel.getByLabel("Maximum snapshot age (days)").fill("30");
  await panel.getByRole("button", { name: "Enable history" }).click();
  await expect(panel.getByText("CAPTURE ENABLED")).toBeVisible();
  await expect(panel.getByText("30 day retention")).toBeVisible();

  await panel.getByRole("button", { name: "Capture snapshot" }).click();
  await expect(panel.getByText("Hardware snapshot captured and retained locally.")).toBeVisible();
  await expect(panel.getByText(/Test GPU/)).toBeVisible();
  expect(captureCalls).toBe(1);

  page.on("dialog", (dialog) => void dialog.accept());
  await panel.getByRole("button", { name: "Clear history" }).click();
  await expect(panel.getByText(/No hardware snapshots have been captured/)).toBeVisible();
  await expect(panel.getByText("Hardware snapshot history cleared.")).toBeVisible();
});
