import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Datasets inventory offers a bounded, non-mutating cleaning preview", async ({ page }) => {
  const datasetWriteRequests: string[] = [];
  let previewPayload: Record<string, unknown> | null = null;
  let previewCalls = 0;

  await page.route("**/api/workspace/datasets**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "GET" && url.pathname === "/api/workspace/datasets") {
      await route.fulfill({
        json: {
          datasets: [{ path: "data/train.jsonl", size_bytes: 128 }],
          total: 1,
          limit: 500,
          offset: 0,
          has_more: false,
          truncated: false,
          scan_limit: 5000,
          filters: { path: null, search: url.searchParams.get("search") ?? "" },
        },
      });
      return;
    }
    await route.continue();
  });

  await page.route("**/api/workspace/dataset-cleaning-preview", async (route) => {
    const request = route.request();
    if (request.method() !== "POST") {
      await route.continue();
      return;
    }
    previewCalls += 1;
    previewPayload = request.postDataJSON() as Record<string, unknown>;
    await route.fulfill({
      json: {
        preview: {
          path: "data/train.jsonl",
          file_name: "train.jsonl",
          preview_only: true,
          output_written: false,
          file_size_bytes: 128,
          bytes_scanned: 128,
          lines_scanned: 4,
          kept_rows: 2,
          removed_rows: 2,
          removed_empty_rows: 1,
          removed_invalid_rows: 1,
          removed_non_object_rows: 0,
          removed_schema_rows: 0,
          removed_duplicate_rows: 0,
          overlong_rows: 0,
          rewritten_rows: 1,
          estimated_output_bytes: 100,
          truncated: false,
          truncation_reasons: [],
          changes: [
            { operation: "remove_empty", rows: 1 },
            { operation: "remove_invalid", rows: 1 },
          ],
          warnings: [],
          limits: { max_bytes: 64 * 1024 * 1024, max_rows: 100000, max_line_bytes: 1024 * 1024 },
        },
      },
    });
  });

  await ensureAuthenticated(page);
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith("/api/workspace/dataset") && request.method() !== "GET") {
      datasetWriteRequests.push(`${request.method()} ${url.pathname}`);
    }
  });
  await page.goto("/?page=datasets");
  await expect(page.getByRole("heading", { level: 1, name: "Datasets", exact: true })).toBeVisible();
  await expect(page.locator(".module-row").filter({ has: page.getByRole("button", { name: "Preview clean", exact: true }) }).getByText("data/train.jsonl", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Preview clean", exact: true }).click();
  await expect(page.getByText("Preview only", { exact: true })).toBeVisible();
  await expect(page.getByText("No output file was created or changed.", { exact: false })).toBeVisible();
  await expect(page.getByText("2 rows removed by current rules", { exact: true })).toBeVisible();

  expect(previewCalls).toBe(1);
  expect(previewPayload).toMatchObject({
    path: "data/train.jsonl",
    max_bytes: expect.any(Number),
    max_rows: expect.any(Number),
    max_line_bytes: expect.any(Number),
  });
  expect(Number(previewPayload?.max_bytes)).toBeGreaterThan(0);
  expect(Number(previewPayload?.max_rows)).toBeGreaterThan(0);
  expect(Number(previewPayload?.max_line_bytes)).toBeGreaterThan(0);
  expect(datasetWriteRequests).toEqual(["POST /api/workspace/dataset-cleaning-preview"]);
});
