import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime creates a workspace knowledge base and ingests a local document", async ({ page }) => {
  let createPayload: Record<string, unknown> | null = null;
  let ingestPayload: Record<string, unknown> | null = null;
  const base = {
    id: "b".repeat(32), name: "Engineering", description: "Team notes",
    created_at: "2026-11-21T00:00:00Z", updated_at: "2026-11-21T00:00:00Z",
    document_count: 0, total_bytes: 0,
  };
  const document = {
    id: "d".repeat(32), source_path: "documents/guide.md", file_name: "guide.md", format: "md",
    content_sha256: "a".repeat(64), size_bytes: 22, character_count: 22,
    ingested_at: "2026-11-21T00:00:00Z",
  };
  let hasDocument = false;
  await page.route("**/api/workspace/knowledge-bases", async (route) => {
    if (route.request().method() === "POST") {
      createPayload = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({ status: 201, json: { schema_version: 1, knowledge_base: base } });
      return;
    }
    await route.fulfill({
      json: {
        schema_version: 1,
        knowledge_bases: [{ ...base, document_count: hasDocument ? 1 : 0 }],
        total: 1,
        limits: { max_knowledge_bases: 64, max_documents_per_base: 1000 },
      },
    });
  });
  await page.route("**/api/workspace/knowledge-bases/**/documents**", async (route) => {
    const method = route.request().method();
    if (method === "POST") {
      ingestPayload = route.request().postDataJSON() as Record<string, unknown>;
      hasDocument = true;
      await route.fulfill({
        status: 201,
        json: {
          schema_version: 1,
          document,
          source_text_returned: false,
          source_text_persisted: true,
          network_used: false,
        },
      });
      return;
    }
    if (method === "DELETE") {
      hasDocument = false;
      await route.fulfill({
        json: { schema_version: 1, removed: document, source_file_modified: false },
      });
      return;
    }
    await route.fulfill({
      json: {
        schema_version: 1,
        knowledge_base: { ...base, document_count: hasDocument ? 1 : 0 },
        documents: hasDocument ? [document] : [],
        total: hasDocument ? 1 : 0,
      },
    });
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");
  await expect(page.getByRole("heading", { name: "Workspace knowledge bases", exact: true })).toBeVisible();
  await page.getByLabel("Knowledge base name").fill("Engineering");
  await page.getByLabel("Description (optional)").fill("Team notes");
  await page.getByRole("button", { name: "Create knowledge base" }).click();
  await expect(page.getByText("Created Engineering.", { exact: true })).toBeVisible();
  await page.getByLabel("Workspace-relative document path").fill("documents/guide.md");
  await page.getByRole("button", { name: "Ingest document" }).click();
  await expect(page.getByText("guide.md", { exact: true })).toBeVisible();
  expect(createPayload).toMatchObject({ name: "Engineering", description: "Team notes" });
  expect(ingestPayload).toEqual({ path: "documents/guide.md" });

  let searchPayload: Record<string, unknown> | null = null;
  await page.route("**/api/workspace/knowledge-bases/*/search", async (route) => {
    searchPayload = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ json: {
      schema_version: 1,
      engine: "axiom-hybrid-knowledge-search",
      knowledge_base: base,
      query: { character_count: 5, candidate_documents: 1, chunks_scanned: 1, result_limit: 8,
        chunk_size_chars: 1200, chunk_overlap_chars: 160, vector_enabled: true, vector_weight: 0.65,
        formats: [], source_prefixes: ["documents"], document_ids: [] },
      results: [{
        score: 0.91, lexical_score: 0.82, vector_score: 0.97, snippet: "AXIOM local retrieval excerpt.",
        citation: { document_id: document.id, file_name: document.file_name, source_path: document.source_path,
          format: "md", content_sha256: document.content_sha256, chunk_index: 1, start_char: 0,
          end_char: 22, character_count: 22 },
      }],
      total_results: 1,
      privacy: { source_documents_modified: false, results_persisted: false, network_used: false,
        full_source_documents_returned: false, snippets_returned: true },
      scope_note: "Local search with source offsets.",
    } });
  });
  await page.getByLabel("Search query").fill("AXIOM");
  await page.getByLabel("Source path prefix (optional)").fill("documents");
  await page.getByLabel("Local embedding model path (optional hybrid search)").fill("models/local-encoder");
  await page.getByLabel("Vector score weight").fill("0.65");
  await page.getByRole("button", { name: "Search knowledge base" }).click();
  await expect(page.getByText("HYBRID SEARCH", { exact: true })).toBeVisible();
  await expect(page.getByText("AXIOM local retrieval excerpt.", { exact: true })).toBeVisible();
  const searchResults = page.locator(".knowledge-search-results");
  await expect(searchResults.getByText("documents/guide.md", { exact: false })).toBeVisible();
  await expect(searchResults.getByText("SHA-256 " + document.content_sha256, { exact: false })).toBeVisible();
  expect(searchPayload).toMatchObject({
    query: "AXIOM",
    source_prefixes: ["documents"],
    embedding_model_path: "models/local-encoder",
    vector_weight: 0.65,
    max_documents: 64,
  });

  await page.getByRole("button", { name: "Remove copy" }).click();
  await expect(page.getByText("Removed the ingested copy of guide.md; the source file was left unchanged.", { exact: true })).toBeVisible();
  await expect(page.getByText("No documents yet", { exact: true })).toBeVisible();
});
