import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

type PromptVariable = {
  name: string;
  type: "string" | "integer" | "number" | "boolean" | "choice";
  required?: boolean;
  choices?: string[];
  min_length?: number | null;
  max_length?: number | null;
};
type PromptTemplate = {
  id: string;
  name: string;
  description: string;
  body: string;
  variables: PromptVariable[];
  version: number;
  created_at: string;
  updated_at: string;
  content_sha256: string;
};

test("Prompt Studio creates, edits, renders, versions and deletes workspace templates", async ({ page }) => {
  await ensureAuthenticated(page);
  const templates = new Map<string, PromptTemplate>();
  const timestamps = { created_at: "2026-11-21T00:00:00Z", updated_at: "2026-11-21T00:00:00Z" };
  const hash = "a".repeat(64);
  const id = "a".repeat(32);

  await page.route("**/api/prompt-templates**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const segments = url.pathname.split("/").filter(Boolean);
    const templateId = segments[2] ?? "";

    if (request.method() === "GET" && segments.length === 2) {
      const search = (url.searchParams.get("search") ?? "").toLowerCase();
      const all = [...templates.values()].filter((item) =>
        !search || `${item.name} ${item.description} ${item.body}`.toLowerCase().includes(search));
      await route.fulfill({
        json: {
          schema_version: 1,
          templates: all.map((item) => ({
            id: item.id, name: item.name, description: item.description,
            version: item.version, created_at: item.created_at, updated_at: item.updated_at,
            content_sha256: item.content_sha256, variable_count: item.variables.length,
          })),
          total: all.length, limit: Number(url.searchParams.get("limit") ?? 50),
          offset: Number(url.searchParams.get("offset") ?? 0), has_more: false,
        },
      });
      return;
    }

    const current = templates.get(templateId);
    if (request.method() === "GET" && segments.length === 3) {
      if (!current) { await route.fulfill({ status: 404, json: { detail: "Not found" } }); return; }
      await route.fulfill({ json: { template: current } });
      return;
    }
    if (request.method() === "GET" && segments[3] === "versions" && segments.length === 4) {
      if (!current) { await route.fulfill({ status: 404, json: { detail: "Not found" } }); return; }
      const versions = [{
        version: current.version, updated_at: current.updated_at, name: current.name,
        content_sha256: current.content_sha256, current: true,
      }];
      await route.fulfill({ json: { template_id: templateId, versions, total: versions.length } });
      return;
    }
    if (request.method() === "GET" && segments[3] === "versions" && segments.length === 5) {
      if (!current) { await route.fulfill({ status: 404, json: { detail: "Not found" } }); return; }
      await route.fulfill({ json: { revision: current } });
      return;
    }
    if (request.method() === "POST" && segments.length === 2) {
      const input = request.postDataJSON() as Omit<PromptTemplate, "id" | "version" | "content_sha256">;
      const created = {
        ...input, id, version: 1, content_sha256: hash, ...timestamps,
      };
      templates.set(id, created);
      await route.fulfill({ status: 201, json: { template: created } });
      return;
    }
    if (request.method() === "PUT" && segments.length === 3) {
      if (!current) { await route.fulfill({ status: 404, json: { detail: "Not found" } }); return; }
      const input = request.postDataJSON() as {
        expected_version: number; name: string; description: string; body: string; variables: PromptVariable[];
      };
      if (input.expected_version !== current.version) {
        await route.fulfill({ status: 409, json: { detail: "This template changed." } });
        return;
      }
      const updated = {
        ...current, ...input, version: current.version + 1, content_sha256: "b".repeat(64),
        updated_at: "2026-11-21T00:01:00Z",
      };
      templates.set(templateId, updated);
      await route.fulfill({ json: { template: updated } });
      return;
    }
    if (request.method() === "POST" && segments[3] === "render") {
      if (!current) { await route.fulfill({ status: 404, json: { detail: "Not found" } }); return; }
      const input = request.postDataJSON() as { values: Record<string, string | number | boolean> };
      const rendered = current.body.replace(/\\{\\{\\s*([A-Za-z][A-Za-z0-9_]*)\\s*\\}\\}/g, (_match, key: string) =>
        String(input.values[key] ?? ""));
      await route.fulfill({
        json: {
          template_id: current.id, template_name: current.name, version: current.version,
          rendered, variables_used: current.variables.map((item) => item.name),
          execution_performed: false, model_invoked: false,
        },
      });
      return;
    }
    if (request.method() === "DELETE" && segments.length === 3) {
      if (!current) { await route.fulfill({ status: 404, json: { detail: "Not found" } }); return; }
      if (Number(url.searchParams.get("expected_version")) !== current.version) {
        await route.fulfill({ status: 409, json: { detail: "Version mismatch" } });
        return;
      }
      templates.delete(templateId);
      await route.fulfill({ json: { deleted_id: templateId, deleted: true } });
      return;
    }
    await route.fulfill({ status: 405, json: { detail: "Method not allowed" } });
  });

  await page.goto("/?page=prompt-studio");
  await expect(page.getByRole("heading", { level: 1, name: "Prompt Studio", exact: true })).toBeVisible();

  await page.getByLabel("Create template").fill("Welcome prompt");
  await page.getByRole("button", { name: "Create", exact: true }).click();
  const templateButton = page.locator("button.prompt-studio-template").filter({ hasText: "Welcome prompt" });
  await expect(templateButton).toBeVisible();
  await templateButton.click();

  await page.getByLabel(/Template body/).fill("Summarize {{topic}} in a {{tone}} tone.");
  await page.getByLabel(/Typed variable definitions/).fill(JSON.stringify([
    { name: "topic", type: "string", required: true, min_length: 1, max_length: 120 },
    { name: "tone", type: "choice", required: true, choices: ["formal", "friendly"] },
  ], null, 2));
  await page.getByRole("button", { name: "Save revision" }).click();
  await expect(page.getByText("Saved Welcome prompt as version 2.")).toBeVisible();

  await page.getByLabel(/Variable values/).fill(JSON.stringify({ topic: "local models", tone: "friendly" }));
  await page.getByRole("button", { name: "Render preview" }).click();
  await expect(page.getByLabel("Rendered prompt")).toContainText("Summarize local models in a friendly tone.");
  await expect(page.getByText(/No model or tool was invoked/)).toBeVisible();
  await expect(page.getByText("Version 2", { exact: true })).toBeVisible();

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Delete template" }).click();
  await expect(templateButton).toHaveCount(0);
  await expect(page.getByText(/No templates yet/)).toBeVisible();
});
