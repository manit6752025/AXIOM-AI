import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Settings explains local data retention and links to runtime trace controls", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.goto("/?page=settings");

  await expect(page.getByRole("heading", { level: 1, name: "Settings", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Data and retention", exact: true })).toBeVisible();
  await expect(page.getByText(/metadata-only records in memory/i)).toBeVisible();

  const guide = page.getByRole("link", { name: "Read the data retention and deletion guide" });
  await expect(guide).toHaveAttribute(
    "href",
    "https://github.com/manit6752025/AXIOM-AI/blob/latest/development/docs/DATA-RETENTION.md",
  );

  await page.getByRole("button", { name: "Review traces" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Runtime", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "Runtime request trace history" })).toBeVisible();
});

test("Settings persists the selected accent palette and maximum session timeout", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.goto("/?page=settings");

  await page.getByLabel("Session timeout (minutes)").fill("300");
  await page.getByLabel("Accent color").selectOption("violet");
  await page.getByRole("button", { name: "Save preferences" }).click();

  await expect(page.getByText("Preferences saved for this browser profile.")).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-accent", "violet");
  await expect(page.getByText("300 MINUTES")).toBeVisible();

  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-accent", "violet");
  await expect(page.getByLabel("Session timeout (minutes)")).toHaveValue("300");
  await expect(page.getByLabel("Accent color")).toHaveValue("violet");
});


test("Settings exports and imports a workspace metadata archive", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.goto("/?page=settings");

  await expect(page.getByRole("heading", { name: "Workspace metadata transfer", exact: true })).toBeVisible();
  await expect(page.getByText(/MCP permission grants.*model weights/i)).toBeVisible();

  await page.route("**/api/workspace/metadata/export", async (route) => {
    await route.fulfill({
      status: 200,
      headers: { "Content-Type": "application/vnd.axiom.workspace-metadata+zip" },
      body: Buffer.from("PK\\u0003\\u0004AXIOM-E2E"),
    });
  });
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Export workspace metadata" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("axiom-workspace-metadata.zip");

  let uploadedBody: Buffer | null = null;
  await page.route("**/api/workspace/metadata/import?replace=true", async (route) => {
    uploadedBody = route.request().postDataBuffer() ?? null;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema_version: 1,
        mode: "replace",
        imported: { models: 1 },
        excluded: [],
      }),
    });
  });
  await page.getByLabel("Replace included metadata collections").check();
  await page.getByLabel("Import workspace metadata archive").setInputFiles({
    name: "workspace.zip",
    mimeType: "application/vnd.axiom.workspace-metadata+zip",
    buffer: Buffer.from("PK\\u0003\\u0004AXIOM-IMPORT"),
  });
  const confirmation = page.getByRole("dialog", { name: "Replace included collections?" });
  await expect(confirmation).toBeVisible();
  await expect(confirmation.getByRole("button", { name: "Cancel" })).toBeFocused();
  await confirmation.getByRole("button", { name: "Replace included collections" }).click();
  await expect(page.locator(".inline-alert.success[role=\"status\"]")).toContainText("Metadata replace completed");
  expect(uploadedBody?.toString()).toBe("PK\\u0003\\u0004AXIOM-IMPORT");
});
