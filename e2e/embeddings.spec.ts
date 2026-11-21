import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime embeds bounded local text and displays vector metadata only", async ({ page }) => {
  let requestPayload: Record<string, unknown> | null = null;
  await page.route("**/api/engine/embeddings", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    requestPayload = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ json: {
      schema_version: 1,
      engine: "axiom-local-embeddings",
      model: { name: "tiny-encoder", dimension: 32 },
      device: "cpu",
      request: { input_count: 1, chunk_count: 1, batch_size: 8, max_length: 256,
        chunk_size_chars: 2048, chunk_overlap_chars: 128, normalized: true, input_characters: 18 },
      chunks: [{ input_index: 1, chunk_index: 1, start_char: 0, end_char: 18, characters: 18, embedding_index: 1 }],
      embeddings: [Array.from({ length: 32 }, (_, index) => index / 32)],
      privacy: { source_text_returned: false, source_text_persisted: false, model_files_uploaded: false, network_used: false, results_persisted: false },
      scope_note: "Local only.",
      generated_at: "2026-11-21T00:00:00Z",
    } });
  });
  await page.route("**/api/engine/embeddings/unload", async (route) => {
    await route.fulfill({ json: { schema_version: 1, unloaded: true, scope_note: "Unloaded." } });
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");
  await expect(page.getByRole("heading", { name: "Local text embeddings", exact: true })).toBeVisible();
  await page.getByLabel("Local embedding model path").fill("models/tiny-encoder");
  await page.getByLabel("Text to embed").fill("AXIOM stays local.");
  await page.getByLabel("Execution device").selectOption("cpu");
  await page.getByRole("button", { name: "Generate local embeddings" }).click();

  await expect(page.getByText("LOCAL EMBEDDINGS READY", { exact: true })).toBeVisible();
  await expect(page.getByText("32 dimensions · 1 chunk(s) · cpu · L2 normalized", { exact: true })).toBeVisible();
  expect(requestPayload).toMatchObject({
    model_path: "models/tiny-encoder",
    texts: ["AXIOM stays local."],
    device: "cpu",
  });
  expect(JSON.stringify(requestPayload)).not.toContain("AXIOM stays local.");
});
