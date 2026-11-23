import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Models can inspect bounded local provenance evidence without loading weights", async ({ page }) => {
  let provenancePayload: Record<string, unknown> | null = null;
  const report = {
    schema_version: 1,
    model_directory: "models/demo",
    registry_metadata: { status: "not_linked", evidence_origin: null },
    source_evidence: {
      status: "not_independently_verified",
      declared_source: null,
      source_url: null,
      source_revision: null,
      source_attribution: null,
      network_used: false,
      note: "A local path does not prove publisher identity.",
    },
    format_evidence: {
      detected_formats: ["safetensors"],
      weight_file_count: 1,
      weight_size_bytes: 13,
      format_file_counts: { safetensors: 1 },
      pickle_capable_weight_files_present: false,
      files: [{ path: "model.safetensors", format: "safetensors", size_bytes: 13 }],
    },
    hash_evidence: {
      algorithm: "SHA-256",
      status: "complete",
      checkpoint_sha256: "0123456789abcdef".repeat(4),
      hashed_file_count: 1,
      files: [{
        path: "model.safetensors",
        format: "safetensors",
        size_bytes: 13,
        sha256: "abcdef0123456789".repeat(4),
      }],
      maximum_total_weight_bytes: 16 * 1024 ** 3,
      maximum_weight_files: 512,
    },
    license_evidence: {
      status: "local_files_found",
      human_review_required: true,
      declared_license: null,
      registry_review_state: null,
      registry_review_notes: null,
      files: [{ path: "LICENSE", size_bytes: 20, sha256: "a".repeat(64), hash_status: "complete" }],
      note: "Local license files require human review.",
    },
    security_evidence: {
      symlink_files_skipped: 1,
      symlink_directories_skipped: 0,
      remote_code_executed: false,
      network_used: false,
      model_weights_loaded: false,
    },
    scope_note: "Read-only provenance; hashes are not publisher authentication or a malware verdict.",
  };

  await page.route("**/api/workspace/models", (route) => route.fulfill({
    json: { models: [], total: 0, limit: 25, offset: 0, has_more: false },
  }));
  await page.route("**/api/workspace/models/discover", (route) => route.fulfill({
    json: {
      candidates: [{
        path: "demo",
        format: "Safetensors",
        has_config: true,
        weight_file_count: 1,
        weight_size_bytes: 13,
      }],
      truncated: false,
    },
  }));
  await page.route("**/api/workspace/models/provenance", async (route) => {
    provenancePayload = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ json: { provenance: report } });
  });
  await page.route("**/api/model-studio/status", (route) => route.fulfill({ json: {
    schema_version: 1,
    ollama_installed: false,
    ollama_endpoint: "http://127.0.0.1:11434",
    download_limit_bytes: 1000000,
    gguf_directory: "models",
    catalogue: "Hugging Face",
    network_search_is_explicit: true,
  } }));
  await page.route("**/api/model-studio/jobs**", (route) => route.fulfill({ json: { jobs: [] } }));

  await ensureAuthenticated(page);
  await page.goto("/?page=models");
  await page.getByRole("button", { name: "Discover local" }).click();
  await expect(page.getByRole("heading", { name: "Local model folders" })).toBeVisible();
  await page.getByRole("button", { name: "Inspect provenance" }).click();

  await expect(page.getByRole("heading", { name: "Model provenance and security evidence" })).toBeVisible();
  await expect(page.getByText("Checkpoint SHA-256", { exact: true })).toBeVisible();
  await expect(page.getByText("0123456789abcdef…", { exact: true })).toBeVisible();
  await expect(page.getByText("LICENSE", { exact: true })).toBeVisible();
  await expect(page.getByText("not independently verified", { exact: false })).toBeVisible();
  expect(provenancePayload).toEqual({ path: "demo" });
});
