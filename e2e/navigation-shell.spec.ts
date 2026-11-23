import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Control Center remains compact and usable from 320px phones through tablets", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.goto("/?page=settings");

  const viewports = [
    { width: 320, height: 700 },
    { width: 360, height: 780 },
    { width: 390, height: 844 },
    { width: 480, height: 850 },
    { width: 768, height: 1024 },
    { width: 1024, height: 768 },
  ];

  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    await expect(page.locator("header.topbar")).toBeVisible();
    await expect(page.locator("#main-content")).toBeVisible();

    const dimensions = await page.evaluate(() => ({
      viewport: window.innerWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
    }));
    expect(dimensions.document, `document overflow at ${viewport.width}px`).toBeLessThanOrEqual(viewport.width + 1);
    expect(dimensions.body, `body overflow at ${viewport.width}px`).toBeLessThanOrEqual(viewport.width + 1);

    if (viewport.width <= 780) {
      const columns = await page.locator(".settings-grid").evaluate((element) =>
        getComputedStyle(element).gridTemplateColumns.split(" ").filter(Boolean).length,
      );
      expect(columns, `settings columns at ${viewport.width}px`).toBe(1);
    }
  }

  await page.setViewportSize({ width: 320, height: 700 });
  const navigation = page.getByLabel("Primary navigation");
  await expect(navigation).not.toHaveClass(/is-open/);
  await page.getByRole("button", { name: "Open navigation" }).click();
  await expect(navigation).toHaveClass(/is-open/);
  await navigation.getByRole("button", { name: "Models" }).click();
  await expect(page).toHaveURL(/page=models/);
  await expect(navigation).not.toHaveClass(/is-open/);
  await expect(page.locator(".topbar-context")).toContainText("Models");
});


test("an open Control Center tab reloads the rebuilt UI without clearing browser state", async ({ page }) => {
  let fingerprint = "build-before";
  await page.route("**/api/ui/build", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    headers: { "Cache-Control": "no-store" },
    body: JSON.stringify({ fingerprint }),
  }));
  await ensureAuthenticated(page);
  await page.evaluate(() => {
    localStorage.setItem("axiom-refresh-test-local", "keep-local");
    sessionStorage.setItem("axiom-refresh-test-session", "keep-session");
  });

  const originalOrigin = await page.evaluate(() => location.origin);
  const reload = page.waitForEvent("framenavigated", {
    predicate: (frame) => frame === page.mainFrame(),
    timeout: 10_000,
  });
  fingerprint = "build-after";
  await reload;
  await page.waitForLoadState("domcontentloaded");

  await expect.poll(() => page.evaluate(() => ({
    origin: location.origin,
    local: localStorage.getItem("axiom-refresh-test-local"),
    session: sessionStorage.getItem("axiom-refresh-test-session"),
  }))).toEqual({
    origin: originalOrigin,
    local: "keep-local",
    session: "keep-session",
  });
  await expect(page.getByLabel("Primary navigation")).toBeVisible();
});
