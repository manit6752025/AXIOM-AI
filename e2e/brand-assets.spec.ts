import { expect, test } from "@playwright/test";

async function expectBrandImageLoaded(page: import("@playwright/test").Page) {
  const mark = page.locator("img.axiom-mark").first();
  await expect(mark).toBeVisible();
  await expect.poll(() => mark.evaluate((image: HTMLImageElement) => ({
    complete: image.complete,
    width: image.naturalWidth,
    height: image.naturalHeight,
    source: image.currentSrc,
  }))).toMatchObject({ complete: true, width: 128, height: 128 });
  await expect(mark).toHaveAttribute("src", "./axiom-mark.svg");
}

test("the shared AXIOM mark loads on welcome, setup, and sign-in screens under /ui/", async ({ page }) => {
  const markResponses: Array<{ url: string; status: number }> = [];
  page.on("response", (response) => {
    if (response.url().endsWith("/axiom-mark.svg")) {
      markResponses.push({ url: response.url(), status: response.status() });
    }
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Build AI/i })).toBeVisible();
  await expectBrandImageLoaded(page);

  await page.getByRole("button", { name: /Begin setup/i }).click();
  await expect(page.getByRole("heading", { name: "Create your administrator." })).toBeVisible();
  await expectBrandImageLoaded(page);

  await page.getByLabel("Username").fill("AXIOM E2E Admin");
  await page.getByLabel("Password", { exact: true }).fill("AxiomE2E2026");
  await page.getByLabel("Confirm password").fill("AxiomE2E2026");
  await page.getByRole("button", { name: /Create administrator/i }).click();

  await expect(page.getByRole("heading", { name: "Welcome back." })).toBeVisible({ timeout: 10_000 });
  await expectBrandImageLoaded(page);

  expect(markResponses.length).toBeGreaterThanOrEqual(3);
  expect(markResponses.every((item) => item.status === 200)).toBe(true);
  expect(markResponses.every((item) => new URL(item.url).pathname === "/axiom-mark.svg" || new URL(item.url).pathname === "/ui/axiom-mark.svg")).toBe(true);
});
