import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Agent Profiles creates, versions and deletes non-default profiles", async ({ page }) => {
  await ensureAuthenticated(page);
  const profiles = new Map<string, {
    id: string; name: string; version: number; task_capabilities: string[];
    created_at: string; updated_at: string; built_in: boolean;
  }>();
  profiles.set("default", { id: "default", name: "default", version: 1, task_capabilities: [], created_at: "2026-10-20T00:00:00Z", updated_at: "2026-10-20T00:00:00Z", built_in: true });
  const nextId = "b".repeat(32);
  await page.route("**/api/agent-profiles**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const id = url.pathname.split("/").filter(Boolean).at(-1) ?? "";
    if (request.method() === "GET") {
      await route.fulfill({ json: { schema_version: 1, profiles: [...profiles.values()], capability_catalog: ["model.inspect", "dataset.profile", "training.plan", "evaluation.review", "runtime.inspect", "mcp.inspect"], authorization_note: "Task profiles do not grant MCP/tool permissions." } });
      return;
    }
    if (request.method() === "POST") {
      const payload = request.postDataJSON() as { name: string; task_capabilities: string[] };
      const profile = { id: nextId, name: payload.name, version: 1, task_capabilities: payload.task_capabilities, created_at: "2026-10-20T00:00:00Z", updated_at: "2026-10-20T00:00:00Z", built_in: false };
      profiles.set(nextId, profile);
      await route.fulfill({ status: 201, json: { profile } });
      return;
    }
    if (request.method() === "PUT") {
      const current = profiles.get(id);
      const payload = request.postDataJSON() as { task_capabilities: string[] };
      if (!current) { await route.fulfill({ status: 404, json: { detail: "Not found" } }); return; }
      const updated = { ...current, version: current.version + 1, task_capabilities: payload.task_capabilities, updated_at: "2026-10-20T00:01:00Z" };
      profiles.set(id, updated);
      await route.fulfill({ json: { profile: updated } });
      return;
    }
    if (request.method() === "DELETE") {
      profiles.delete(id);
      await route.fulfill({ json: { deleted_id: id, remaining: profiles.size } });
      return;
    }
    await route.fulfill({ status: 405 });
  });

  await page.goto("/?page=agent-profiles");
  await expect(page.getByRole("heading", { level: 1, name: "Agent Profiles", exact: true })).toBeVisible();
  await page.getByLabel("Create profile").fill("Dataset reviewer");
  await page.getByRole("button", { name: "Create default-deny" }).click();
  await expect(page.locator("button.agent-profile-select").filter({ hasText: "Dataset reviewer" })).toBeVisible();
  await page.getByLabel("dataset.profile").check();
  await page.getByRole("button", { name: "Save profile" }).click();
  await expect(page.getByText(/Saved Dataset reviewer at profile version 2/)).toBeVisible();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Delete Dataset reviewer" }).click();
  await expect(page.getByRole("button", { name: /Dataset reviewer/ })).toHaveCount(0);
  await expect(page.locator("body")).toContainText("do not grant MCP/tool permissions");
});
