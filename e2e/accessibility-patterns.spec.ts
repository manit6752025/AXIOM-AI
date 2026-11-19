import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("shared confirmation dialog exposes labels and traps keyboard focus", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.locator(".profile-trigger").click();
  await page.getByRole("menuitem", { name: "Change password" }).click();

  const dialog = page.getByRole("dialog", { name: "Change password" });
  await expect(dialog).toBeVisible();
  await expect(page.getByLabel("Current password")).toBeVisible();
  await expect(page.getByLabel("New password", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Confirm new password")).toBeVisible();
  await expect(page.getByRole("button", { name: "Close change password dialog" })).toBeFocused();

  await page.keyboard.press("Shift+Tab");
  await expect(page.locator("#confirm-password")).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Close change password dialog" })).toBeFocused();

  await page.locator("#confirm-password").fill("partial");
  await expect(page.locator("#confirm-password")).toHaveAttribute("aria-describedby", "confirm-password-hint");
  await expect(page.locator("#confirm-password")).toHaveAttribute("aria-invalid", "false");
  await expect(page.locator("#confirm-password-hint")).toHaveText("Passwords do not match");

  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
});

test("empty workspace views announce their no-data state", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.route("**/api/workspace/models?*", (route) => route.fulfill({
    json: { models: [], total: 0, limit: 100, offset: 0, has_more: false },
  }));
  await page.goto("/?page=models");
  await expect(page.getByRole("status").filter({ has: page.getByRole("heading", { name: "Registry is empty" }) })).toBeVisible();
});
