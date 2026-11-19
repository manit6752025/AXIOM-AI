import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("API Playground exposes bounded read-only endpoints and Deep Analysis", async ({ page }) => {
  test.setTimeout(30_000);

  const requests: Array<{
    url: string;
    method: string;
    postData: string | null;
  }> = [];

  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    requests.push({
      url: url.pathname + url.search,
      method: request.method(),
      postData: request.postData(),
    });

    const path = url.pathname;

    if (path.endsWith("/health")) {
      await route.fulfill({ json: { status: "ok", version: "test" } });
    } else if (path.endsWith("/meta")) {
      await route.fulfill({
        json: { api_contract_version: 1, application_version: "test" },
      });
    } else if (path.endsWith("/runtime")) {
      await route.fulfill({ json: { status: "offline", hardware: { os_name: "Linux", architecture: "x86_64", cpu_cores: 4, ram_gb: 8, gpu_name: null, vram_gb: null, cuda_available: false } } });
    } else if (path.endsWith("/models")) {
      await route.fulfill({
        json: { models: [], total: 0, limit: 5, offset: 0, has_more: false },
      });
    } else if (path.endsWith("/datasets")) {
      await route.fulfill({ json: { datasets: [] } });
    } else if (path.endsWith("/events")) {
      await route.fulfill({ json: { events: [] } });
    } else if (path.endsWith("/diagnostics/workspace")) {
      await route.fulfill({
        json: {
          checks: [],
          summary: { passed: 0, warnings: 0, failed: 0 },
          status: "pass",
        },
      });
    } else if (path.endsWith("/diagnostics/roadmap")) {
      await route.fulfill({
        json: { integrity: { ok: true }, total: 120, counts: {}, items: [] },
      });
    } else if (path.endsWith("/diagnostics/runtime-connectors")) {
      await route.fulfill({
        json: {
          schema_version: 1,
          discovery_mode: "passive_local_discovery",
          checked_at: new Date().toISOString(),
          connectors: [],
          scope_note: "Test fixture; no runtime was launched.",
        },
      });
    } else {
      await route.fulfill({ status: 404, json: { detail: "Not found" } });
    }
  });

  await ensureAuthenticated(page);



  // ensureAuthenticated() already opened the authenticated Control Center.
  // Do not reload the page here; navigate using the existing session.
  const navigation = page.getByLabel("Primary navigation");
  await expect(navigation).toBeVisible({ timeout: 10_000 });
  await navigation.getByText("API Playground", { exact: true }).click();

  await expect(
    page.getByRole("heading", { name: /API Playground/i }),
  ).toBeVisible({ timeout: 10_000 });
  await expect(page.getByLabel("Endpoint")).toBeVisible({ timeout: 10_000 });

  const select = page.getByLabel("Endpoint");
  const options = await select
    .locator("option")
    .evaluateAll((nodes) =>
      nodes.map((node) => (node as HTMLOptionElement).value),
    );

  expect(options.length).toBeGreaterThanOrEqual(8);
  expect(options.every((option) => option.startsWith("/api/"))).toBe(true);
  expect(
    options.some((option) => option.includes("/api/jobs") || option.includes("/delete")),
  ).toBe(false);

  await select.selectOption("/api/meta");
  await page.getByRole("button", { name: "Run read-only request" }).click();
  await expect(page.locator(".api-playground-json")).toContainText(
    '"api_contract_version": 1',
  );

  const beforeDeepAnalysis = requests.length;
  await page.getByRole("button", { name: /Run Deep Analysis/i }).click();

  await expect(
    page.getByText(`${options.length}/${options.length} passed`, { exact: true }),
  ).toBeVisible({ timeout: 10_000 });

  const sweepRequests = requests.slice(beforeDeepAnalysis);
  const sweepPaths = new Set(
    sweepRequests.map((request) =>
      new URL(request.url, "http://axiom.test").pathname,
    ),
  );

  for (const option of options) {
    const expectedPath = new URL(option, "http://axiom.test").pathname;
    expect(sweepPaths.has(expectedPath), `Deep Analysis did not request ${expectedPath}`).toBe(true);
  }

  expect(
    requests.every((request) => request.method === "GET" && request.postData === null),
  ).toBe(true);

  // Confirm the authenticated session survives a reload on the same origin.
  console.log("[AXIOM E2E] storage BEFORE reload", await page.evaluate(() => ({
    flags: Object.fromEntries(
      Object.keys(localStorage)
        .filter((key) => /remember|session|setup|booted/i.test(key) && !/admin/i.test(key))
        .map((key) => [key, localStorage.getItem(key)]),
    ),
    testAdminPresent: Boolean(localStorage.getItem("axiom-admin-v2")),
    tabSessionFlags: Object.fromEntries(
      Object.keys(sessionStorage)
        .filter((key) => /remember|session/i.test(key))
        .map((key) => [key, sessionStorage.getItem(key)]),
    ),
  })));
  await page.reload();

  await page.waitForTimeout(300);
  console.log("[AXIOM E2E] state AFTER reload", await page.evaluate(() => ({
    url: location.href,
    bodyText: document.body?.innerText?.slice(0, 500) ?? "",
    flags: Object.fromEntries(
      Object.keys(localStorage)
        .filter((key) => /remember|session|setup|booted/i.test(key) && !/admin/i.test(key))
        .map((key) => [key, localStorage.getItem(key)]),
    ),
    testAdminPresent: Boolean(localStorage.getItem("axiom-admin-v2")),
    tabSessionFlags: Object.fromEntries(
      Object.keys(sessionStorage)
        .filter((key) => /remember|session/i.test(key))
        .map((key) => [key, sessionStorage.getItem(key)]),
    ),
  })));
  await expect(page.getByLabel("Primary navigation")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByLabel("Endpoint")).toBeVisible();

  // Verify hardware diagnostics through the real runtime snapshot.
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Diagnostics", exact: true })).toBeVisible();
  const hardwareRow = page.locator(".diagnostic-row").filter({ hasText: "Hardware detection" });
  await expect(hardwareRow).toContainText("PASS", { timeout: 10_000 });
});
