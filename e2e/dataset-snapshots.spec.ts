import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Datasets creates immutable content-addressed snapshots and verifies parent lineage", async ({ page }) => {
  const firstId = "a".repeat(32);
  const secondId = "b".repeat(32);
  const firstHash = "1".repeat(64);
  const secondHash = "2".repeat(64);
  const firstManifestHash = "3".repeat(64);
  const firstRecord = {
    sequence: 1,
    manifest: {
      schema_version: 1,
      snapshot_id: firstId,
      created_at: "2026-11-21T00:00:00+00:00",
      source_path: "train.jsonl",
      file_name: "train.jsonl",
      format: "jsonl",
      file_size_bytes: 128,
      sha256: firstHash,
      object_name: firstHash + ".jsonl",
      parent_snapshot_id: null,
      parent_manifest_sha256: null,
    },
    manifest_sha256: firstManifestHash,
    previous_record_sha256: "0".repeat(64),
    record_sha256: "4".repeat(64),
  };
  const secondRecord = {
    sequence: 2,
    manifest: {
      ...firstRecord.manifest,
      snapshot_id: secondId,
      created_at: "2026-11-21T00:01:00+00:00",
      sha256: secondHash,
      object_name: secondHash + ".jsonl",
      parent_snapshot_id: firstId,
      parent_manifest_sha256: firstManifestHash,
    },
    manifest_sha256: "5".repeat(64),
    previous_record_sha256: firstRecord.record_sha256,
    record_sha256: "6".repeat(64),
  };
  const records = [firstRecord, secondRecord];
  const creationBodies: Array<Record<string, unknown>> = [];
  let verificationCalls = 0;

  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/workspace/datasets" && request.method() === "GET") {
      await route.fulfill({ json: {
        datasets: [{ path: "train.jsonl", size_bytes: 128 }],
        total: 1, limit: 500, offset: 0, has_more: false, truncated: false,
        scan_limit: 5000, filters: { path: null, search: "" },
      } });
    } else if (url.pathname === "/api/workspace/dataset-snapshots" && request.method() === "GET") {
      await route.fulfill({ json: {
        schema_version: 1,
        snapshots: [...records].reverse(),
        total: records.length,
        limit: 50,
        offset: 0,
        has_more: false,
        immutable: true,
      } });
    } else if (url.pathname === "/api/workspace/dataset-snapshots" && request.method() === "POST") {
      const body = request.postDataJSON() as Record<string, unknown>;
      creationBodies.push(body);
      const record = creationBodies.length === 1 ? firstRecord : secondRecord;
      await route.fulfill({ status: 201, json: { snapshot: record, total: creationBodies.length } });
    } else if (url.pathname === "/api/workspace/dataset-snapshots/" + firstId + "/verify") {
      verificationCalls += 1;
      await route.fulfill({ json: {
        schema_version: 1,
        snapshot_id: firstId,
        valid: true,
        expected_sha256: firstHash,
        file_size_bytes: 128,
        manifest_sha256: firstManifestHash,
        lineage_valid: true,
        immutable: true,
      } });
    } else if (url.pathname === "/api/workspace/dataset-manifest") {
      await route.fulfill({ json: { manifest: {
        schema_version: 1, generated_at: "2026-11-21T00:00:00Z",
        path: "train.jsonl", lineage: { relation: "source", parent_manifest_sha256: null },
        file_name: "train.jsonl", format: "jsonl", file_size_bytes: 128,
        bytes_hashed: 128, full_content_hashed: true, sha256: firstHash,
        partial_sha256: null, status: "complete", modified_at: null,
        limits: { max_bytes: 128 },
      } } });
    } else if (url.pathname === "/api/workspace/dataset-cleaning-preview") {
      await route.fulfill({ json: { preview: {
        path: "train.jsonl", file_name: "train.jsonl", preview_only: true,
        output_written: false, file_size_bytes: 128, bytes_scanned: 128,
        lines_scanned: 2, kept_rows: 2, removed_rows: 0, removed_empty_rows: 0,
        removed_invalid_rows: 0, removed_non_object_rows: 0, removed_schema_rows: 0,
        removed_duplicate_rows: 0, overlong_rows: 0, rewritten_rows: 0,
        estimated_output_bytes: 128, truncated: false, truncation_reasons: [],
        changes: [], warnings: [], limits: { max_bytes: 128, max_rows: 100, max_line_bytes: 1024 },
      } } });
    } else {
      await route.fulfill({ status: 404, json: { detail: "No fixture for this dataset snapshot test." } });
    }
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=datasets");
  await expect(page.getByRole("heading", { level: 1, name: "Datasets", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Immutable snapshot history" })).toBeVisible();

  await page.getByRole("button", { name: "Snapshot", exact: true }).click();
  await expect(page.getByText(firstHash.slice(0, 16), { exact: false })).toBeVisible();
  expect(creationBodies[0]).toMatchObject({ path: "train.jsonl", max_bytes: expect.any(Number) });
  expect(creationBodies[0].parent_snapshot_id).toBeUndefined();

  await page.getByLabel("Parent dataset snapshot").selectOption(firstId);
  await page.getByRole("button", { name: "Snapshot", exact: true }).click();
  await expect(page.getByText(secondHash.slice(0, 16), { exact: false })).toBeVisible();
  expect(creationBodies[1]).toMatchObject({
    path: "train.jsonl",
    max_bytes: expect.any(Number),
    parent_snapshot_id: firstId,
  });

  const history = page.locator(".surface").filter({ has: page.getByRole("heading", { name: "Immutable snapshot history" }) });
  await history.getByRole("button", { name: "Verify" }).first().click();
  await expect(page.getByText("Snapshot content hash and lineage verified.")).toBeVisible();
  expect(verificationCalls).toBe(1);
});
