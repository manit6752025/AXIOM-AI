import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Diagnostics exports only redacted frontend error metadata and never uploads it", async ({ page }) => {
  const rawSecret = "Bearer SUPER_SECRET_TOKEN";
  const rawEmail = "private.person@example.invalid";
  const rawPrompt = "PRIVATE_PROMPT_VALUE_DO_NOT_EXPORT";
  const rawPath = "/home/private-user/customer-dataset.jsonl";
  let reportUploadRequests = 0;

  await page.route("**/api/workspace/models**", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        headers: { "X-Request-ID": "private-request-id-123" },
        body: JSON.stringify({
          detail: `${rawSecret} ${rawEmail} ${rawPrompt} ${rawPath}`,
          message: "customer message must not be captured",
        }),
      });
      return;
    }
    await route.continue();
  });
  await page.route("**/api/frontend-errors**", async (route) => {
    reportUploadRequests += 1;
    await route.fulfill({ status: 500 });
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=diagnostics");

  const reportTable = page.getByRole("table", { name: "Redacted frontend error events" });
  await expect(reportTable).toBeVisible();
  await expect(reportTable).toContainText("api_http_error");
  await expect(reportTable).toContainText("500");
  await expect(reportTable).toContainText("diagnostics");

  await page.evaluate(() => {
    const unsafeError = new TypeError("RAW_STACK_AND_SECRET_VALUE");
    window.dispatchEvent(new ErrorEvent("error", {
      message: "RAW_WINDOW_ERROR_SECRET",
      error: unsafeError,
    }));
  });
  await expect(reportTable).toContainText("window.error");

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Download redacted report" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("axiom-frontend-error-report.json");

  const contents = await readFile((await download.path()) as string, "utf8");
  const exported = JSON.parse(contents) as {
    schema_version: number;
    events: Array<Record<string, unknown>>;
    privacy: Record<string, unknown>;
  };
  expect(exported.schema_version).toBe(1);
  expect(exported.events.length).toBeGreaterThanOrEqual(2);
  expect(exported.events.some((event) =>
    event.source === "api.request" &&
    event.category === "api_http_error" &&
    event.http_status === 500 &&
    event.page === "diagnostics"
  )).toBe(true);
  expect(exported.events.some((event) =>
    event.source === "window.error" &&
    event.category === "runtime_error" &&
    event.error_name === "TypeError"
  )).toBe(true);
  expect(exported.privacy).toMatchObject({
    storage: "in_memory_only",
    automatic_upload: false,
    raw_messages_stored: false,
    stack_traces_stored: false,
    request_urls_stored: false,
    request_ids_stored: false,
    request_bodies_stored: false,
    prompts_or_dataset_values_stored: false,
    credentials_stored: false,
    max_entries: 50,
  });

  for (const secret of [
    rawSecret, rawEmail, rawPrompt, rawPath, "private-request-id-123",
    "customer message must not be captured", "RAW_STACK_AND_SECRET_VALUE",
    "RAW_WINDOW_ERROR_SECRET",
  ]) {
    expect(contents).not.toContain(secret);
  }
  for (const event of exported.events) {
    expect(Object.keys(event).sort()).toEqual([
      "category", "error_name", "http_status", "id", "occurred_at", "page", "source",
    ]);
  }
  expect(reportUploadRequests).toBe(0);

  await page.getByRole("button", { name: "Clear in-memory report" }).click();
  await expect(page.getByText("No frontend errors recorded", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Download redacted report" })).toBeDisabled();
  expect(reportUploadRequests).toBe(0);
});
