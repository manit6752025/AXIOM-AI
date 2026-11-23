import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("global command palette searches pages and executes theme and security actions", async ({ page }) => {
  await ensureAuthenticated(page);

  await page.keyboard.press("Control+k");
  const search = page.getByRole("textbox", { name: "Search pages and actions" });
  await expect(search).toBeFocused();
  await expect(page.getByRole("listbox", { name: "Pages and actions" })).toBeVisible();

  await search.fill("switch to light theme");
  const themeAction = page.getByRole("option", { name: /Switch to light theme/ });
  await expect(themeAction).toBeVisible();
  await page.keyboard.press("Enter");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(search).toHaveAttribute("aria-expanded", "false");

  await page.keyboard.press("Control+k");
  await search.fill("change administrator password");
  await expect(page.getByRole("option", { name: /Change administrator password/ })).toBeVisible();
  await page.getByRole("option", { name: /Change administrator password/ }).click();
  await expect(page.getByRole("dialog", { name: "Change password" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "Change password" })).not.toBeVisible();
});

test("global command palette navigates to pages and reports no matches", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.keyboard.press("Control+k");

  const search = page.getByRole("textbox", { name: "Search pages and actions" });
  await search.fill("datasets");
  await expect(page.getByRole("option", { name: /^Datasets/ })).toBeVisible();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/page=datasets/);
  await expect(page.locator(".topbar-context")).toContainText("Datasets");

  await page.keyboard.press("Control+k");
  await search.fill("nothing matches this command");
  await expect(page.getByText(/No matching pages or actions/)).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(search).toHaveAttribute("aria-expanded", "false");
});
