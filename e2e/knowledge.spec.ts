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
  await page.route("**/api/workspace/knowledge-bases/*/documents", async (route) => {
    if (route.request().method() === "POST") {
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
});
