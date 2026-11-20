import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Diagnostics downloads the privacy-safe support bundle from the API", async ({ page }) => {
  await ensureAuthenticated(page);
  let requested = false;
  await page.route("**/api/diagnostics/support-bundle", async (route) => {
    requested = true;
    await route.fulfill({
      status: 200,
      contentType: "application/zip",
      headers: { "Cache-Control": "no-store" },
      body: Buffer.from("PK\\x03\\x04AXIOM support bundle fixture"),
    });
  });
  await page.goto("/?page=diagnostics");
  const downloadButton = page.getByRole("button", { name: "Download support bundle" });
  await expect(downloadButton).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await downloadButton.click();
  const download = await downloadPromise;
  expect(requested).toBe(true);
  expect(download.suggestedFilename()).toBe("axiom-support-bundle.zip");
  await expect(page.getByText("Privacy-safe support bundle downloaded. Review it before sharing.")).toBeVisible();
});
