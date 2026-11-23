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
