import { expect, test } from "@playwright/test";
import { redactErrorText } from "../src/utils/errorReporting";
import { ensureAuthenticated } from "./helpers/auth";

test("redacts credentials and input values before an error can be retained", () => {
  const report = redactErrorText('Request failed: Authorization: Bearer top-secret api_key=abc123 password: "correct-horse" prompt=private-input');
  expect(report).toContain("[REDACTED]");
  expect(report).not.toContain("top-secret");
  expect(report).not.toContain("abc123");
  expect(report).not.toContain("correct-horse");
  expect(report).not.toContain("private-input");
});

test("settings explains local-only redacted frontend reports", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.goto("/?page=settings");
  await expect(page.getByRole("heading", { name: "Frontend error reports" })).toBeVisible();
  await expect(page.getByText("Reports are never sent automatically.", { exact: false })).toBeVisible();
  await expect(page.getByText("No frontend errors retained", { exact: true })).toBeVisible();
});
