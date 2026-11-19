import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("desktop sidebar, topbar and theme preferences remain stable", async ({ page }) => {
  await ensureAuthenticated(page);

  const navigation = page.getByLabel("Primary navigation");
  const shell = page.locator(".app-shell");
  await expect(navigation).toBeVisible();
  await expect(page.locator("header.topbar")).toBeVisible();
  await expect(shell).not.toHaveClass(/sidebar-collapsed/);

  await navigation.getByRole("button", { name: "Collapse sidebar" }).click();
  await expect(shell).toHaveClass(/sidebar-collapsed/);
  await navigation.getByRole("button", { name: "Expand sidebar" }).click();
  await expect(shell).not.toHaveClass(/sidebar-collapsed/);

  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect.poll(() => page.evaluate(() => localStorage.getItem("axiom-theme"))).toBe("light");

  await page.reload();
  await expect(page.getByLabel("Primary navigation")).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.locator("header.topbar")).toBeVisible();
});

test("mobile sidebar navigation closes after selection and browser history restores page state", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.setViewportSize({ width: 390, height: 844 });

  const navigation = page.getByLabel("Primary navigation");
  const sidebar = navigation;
  await page.getByRole("button", { name: "Open navigation" }).click();
  await expect(sidebar).toHaveClass(/is-open/);

  await sidebar.getByRole("button", { name: "Settings" }).click();
  await expect(page).toHaveURL(/page=settings/);
  await expect(page.locator(".topbar-context")).toContainText("Settings");
  await expect(sidebar).not.toHaveClass(/is-open/);
  await expect(sidebar.getByRole("button", { name: "Settings" })).toHaveAttribute("aria-current", "page");

  await page.goBack();
  await expect(page.locator(".topbar-context")).toContainText("Control center");
  await expect(sidebar.getByRole("button", { name: "Control center" })).toHaveAttribute("aria-current", "page");
});
