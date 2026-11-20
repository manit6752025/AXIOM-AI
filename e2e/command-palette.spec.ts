import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("global command palette searches pages and executes theme and security actions", async ({ page }) => {
  await ensureAuthenticated(page);

  await page.keyboard.press("Control+k");
  const search = page.getByRole("combobox", { name: "Search pages and actions" });
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

  const search = page.getByRole("combobox", { name: "Search pages and actions" });
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


test("command palette announces result count and active keyboard selection", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.keyboard.press("Control+k");

  const search = page.getByRole("combobox", { name: "Search pages and actions" });
  const announcement = page.locator("#command-palette-announcement");
  await expect(search).toBeFocused();
  await expect(search).toHaveAttribute("aria-expanded", "true");
  await expect(search).toHaveAttribute("aria-autocomplete", "list");
  await expect(announcement).toContainText("results");

  const firstId = await search.getAttribute("aria-activedescendant");
  expect(firstId).toBeTruthy();
  const firstOption = page.locator(`[id="${firstId}"]`);
  await expect(firstOption).toHaveAttribute("aria-selected", "true");
  await expect(firstOption).toHaveAttribute("aria-posinset", "1");

  await search.press("ArrowDown");
  const secondId = await search.getAttribute("aria-activedescendant");
  expect(secondId).toBeTruthy();
  expect(secondId).not.toBe(firstId);
  await expect(page.locator(`[id="${secondId}"]`)).toHaveAttribute("aria-selected", "true");
  await expect(announcement).toContainText("selected");

  await search.press("End");
  const lastId = await search.getAttribute("aria-activedescendant");
  await expect(page.locator(`[id="${lastId}"]`)).toHaveAttribute("aria-posinset", await page.locator("#workspace-search-results [role=\"option\"]").count() + "");

  await search.press("Home");
  await expect(page.locator(`[id="${await search.getAttribute("aria-activedescendant")}"]`)).toHaveAttribute("aria-posinset", "1");

  await search.fill("no-result-axiom-command-xyz");
  await expect(search).not.toHaveAttribute("aria-activedescendant");
  await expect(announcement).toContainText("No pages or actions match");
  await search.press("Escape");
  await expect(search).toHaveAttribute("aria-expanded", "false");
});
