import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime locally reranks bounded candidates and displays score metadata only", async ({ page }) => {
  let payload: Record<string, unknown> | null = null;
  await page.route("**/api/engine/rerank", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    payload = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ json: {
      schema_version: 1,
      engine: "axiom-local-cross-encoder",
      model: { name: "tiny-reranker", score_mode: "sigmoid" },
      device: "cpu",
      request: { document_count: 2, batch_size: 8, max_length: 256, query_characters: 5, total_document_characters: 20 },
      results: [
        { rank: 1, document_index: 2, score: 0.91, document_characters: 10 },
        { rank: 2, document_index: 1, score: 0.35, document_characters: 10 },
      ],
      privacy: { query_returned: false, documents_returned: false, text_persisted: false, model_files_uploaded: false, network_used: false },
      scope_note: "Local ranking signal.",
    } });
  });
  await page.route("**/api/engine/rerank/unload", async (route) => {
    await route.fulfill({ json: { schema_version: 1, unloaded: true, scope_note: "Unloaded." } });
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");
  await expect(page.getByRole("heading", { name: "Local cross-encoder reranking", exact: true })).toBeVisible();
  await page.getByLabel("Local reranker model path").fill("models/tiny-reranker");
  await page.getByLabel("Query").fill("AXIOM");
  await page.getByLabel("Candidate documents (one per line)").fill("First secret document\nSecond secret document");
  await page.getByRole("button", { name: "Rerank candidates locally" }).click();

  await expect(page.getByText("RERANKING COMPLETE", { exact: true })).toBeVisible();
  await expect(page.getByText("Candidate 2", { exact: true })).toBeVisible();
  const resultPanel = await page.locator(".reranking-result").innerText();
  expect(resultPanel).not.toContain("First secret document");
  expect(resultPanel).not.toContain("Second secret document");
  expect(payload).toMatchObject({
    model_path: "models/tiny-reranker",
    query: "AXIOM",
    documents: ["First secret document", "Second secret document"],
  });
});
