import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

const model = {
  id: "a".repeat(32),
  name: "Metadata-only model",
  source: "models/metadata-only",
  format: "gguf",
  parameters: "7B",
  quantization: "Q4_K_M",
  architecture: "QwenForCausalLM",
  license: "Apache-2.0",
  tags: ["local"],
  favorite: false,
  content_sha256: null,
};

test("model registry export downloads metadata rather than model weights", async ({ page }) => {
  let exportCalls = 0;
  await page.route("**/api/workspace/models**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/workspace/models/export") {
      exportCalls += 1;
      await route.fulfill({ json: { schema_version: 1, models: [model] } });
      return;
    }
    if (url.pathname === "/api/workspace/models" && route.request().method() === "GET") {
      await route.fulfill({ json: {
        models: [model], total: 1, limit: 25, offset: 0, has_more: false,
        filters: { search: "", favorite_only: false, format: null, tag: null, sort_by: "name", sort_order: "asc" },
      } });
      return;
    }
    await route.continue();
  });
  await page.route("**/api/model-studio/status", (route) => route.fulfill({ json: {
    schema_version: 1, ollama_installed: false, ollama_endpoint: "http://127.0.0.1:11434",
    download_limit_bytes: 1000000, gguf_directory: "models", catalogue: "Hugging Face",
    network_search_is_explicit: true,
  } }));
  await page.route("**/api/model-studio/jobs**", (route) => route.fulfill({ json: { jobs: [] } }));

  await ensureAuthenticated(page);
  await page.goto("/?page=models");
  await expect(page.getByRole("heading", { level: 1, name: "Models", exact: true })).toBeVisible();
  await expect(page.getByText("Registry metadata only", { exact: true })).toBeVisible();

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Export", exact: true }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("axiom-model-registry.json");
  expect(exportCalls).toBe(1);
});

test("dataset manifest export is a JSON provenance file, not a dataset copy", async ({ page }) => {
  const manifest = {
    schema_version: 1,
    generated_at: "2026-11-20T00:00:00Z",
    path: "train.jsonl",
    lineage: { relation: "source", parent_manifest_sha256: null },
    file_name: "train.jsonl",
    format: "jsonl",
    file_size_bytes: 128,
    bytes_hashed: 128,
    full_content_hashed: true,
    sha256: "a".repeat(64),
    partial_sha256: null,
    status: "complete",
    modified_at: null,
    limits: { max_bytes: 64 * 1024 * 1024 },
  };
  await page.route("**/api/workspace/datasets*", (route) => route.fulfill({ json: {
    datasets: [{ path: "train.jsonl", size_bytes: 128 }],
    total: 1, limit: 100, offset: 0, has_more: false, truncated: false, scan_limit: 5000,
    filters: { path: null, search: "" },
  } }));
  await page.route("**/api/workspace/dataset-manifest**", (route) => route.fulfill({ json: { manifest } }));

  await ensureAuthenticated(page);
  await page.goto("/?page=datasets");
  await expect(page.getByText("train.jsonl", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Manifest", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Manifest: train.jsonl" })).toBeVisible();
  await expect(page.getByText("Manifest contains file metadata and checksums only, never dataset values.", { exact: true })).toBeVisible();

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Export JSON", exact: true }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("train.jsonl.manifest.json");
});

test("support bundle download is labelled for review before sharing", async ({ page }) => {
  let requests = 0;
  await page.route("**/api/diagnostics/support-bundle", async (route) => {
    requests += 1;
    await route.fulfill({
      status: 200,
      headers: { "Content-Type": "application/zip" },
      body: Buffer.from("PK\\u0003\\u0004AXIOM-SUPPORT"),
    });
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=diagnostics");
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Download support bundle" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("axiom-support-bundle.zip");
  expect(requests).toBe(1);
  await expect(page.getByText("Privacy-safe support bundle downloaded. Review it before sharing.", { exact: true })).toBeVisible();
});
