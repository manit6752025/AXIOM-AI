import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("global backend status retries offline connections and shows request loading", async ({ page }) => {
  let allowHealth = false;
  let healthCalls = 0;
  await page.route("**/api/health", async (route) => {
    healthCalls += 1;
    if (!allowHealth) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Test backend unavailable" }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok", version: "test" }),
    });
  });

  await ensureAuthenticated(page);
  const status = page.getByRole("group", { name: "Backend connection" });
  await expect(status.getByText("OFFLINE", { exact: true })).toBeVisible();
  allowHealth = true;
  await status.getByRole("button", { name: "Retry backend connection" }).click();
  await expect(status.getByText("ONLINE", { exact: true })).toBeVisible();
  expect(healthCalls).toBeGreaterThanOrEqual(2);

  let releaseRequest: (() => void) | undefined;
  const requestGate = new Promise<void>((resolve) => { releaseRequest = resolve; });
  await page.route("**/api/workspace/models?*", async (route) => {
    await requestGate;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ models: [], total: 0, limit: 100, offset: 0, has_more: false }),
    });
  });

  await page.goto("/?page=models");
  await expect(status.getByRole("status").filter({ hasText: /LOADING/ })).toBeVisible();
  releaseRequest?.();
  await expect(status.getByRole("status").filter({ hasText: /LOADING/ })).toHaveCount(0);
});

test("backend connection indicator and retry action persist outside the dashboard", async ({ page }) => {
  await ensureAuthenticated(page);
  const status = page.getByRole("group", { name: "Backend connection" });
  await expect(status).toBeVisible();
  await page.goto("/?page=settings");
  await expect(status).toBeVisible();
  await expect(status.getByRole("button", { name: "Retry backend connection" })).toBeVisible();
});
