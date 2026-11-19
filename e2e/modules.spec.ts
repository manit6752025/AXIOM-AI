import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test.describe("AXIOM module explorer", () => {
  test.beforeEach(async ({ page }) => {
    await ensureAuthenticated(page);
  });

  test("lists existing GUI surfaces and honestly labels CLI-only entry points", async ({ page }) => {
    await page.goto("/?page=modules");

    await expect(page.getByRole("heading", { name: "Platform module directory" })).toBeVisible();
    await expect(page.getByRole("group", { name: "Agent Optimizer" })).toBeVisible();
    await expect(page.getByRole("group", { name: "Dataset engineering and quality" })).toBeVisible();
    await expect(page.getByRole("group", { name: "Developer tool catalog" })).toContainText("CLI surface");
    await expect(page.getByRole("group", { name: "Training planner and experiments" })).toContainText("Planning + local LoRA");
    await expect(page.getByRole("link", { name: "Open API" })).toHaveAttribute("href", "/api/docs");
  });

  test("searches the directory and opens the existing page instead of simulating a command", async ({ page }) => {
    await page.goto("/?page=modules");
    await page.getByRole("textbox", { name: "Search AXIOM modules" }).fill("dataset quality");

    const datasetModule = page.getByRole("group", { name: "Dataset engineering and quality" });
    await expect(datasetModule).toBeVisible();
    await expect(page.getByRole("group", { name: "SuperCompress context utilities" })).toHaveCount(0);

    await datasetModule.getByRole("button", { name: "Open" }).click();
    await expect(page).toHaveURL(/page=datasets/);
    await expect(page.locator(".page-header h1")).toContainText("Datasets");
  });

  test("opens the new wired GUI pages from the module directory", async ({ page }) => {
    await page.route("**/api/agent-profiles**", (route) => route.fulfill({ json: {
      schema_version: 1,
      profiles: [{ id: "default", name: "default", version: 1, task_capabilities: [], created_at: "2026-10-20T00:00:00Z", updated_at: "2026-10-20T00:00:00Z", built_in: true }],
      capability_catalog: ["model.inspect", "dataset.profile", "training.plan", "evaluation.review", "runtime.inspect", "mcp.inspect"],
      authorization_note: "Task capabilities are descriptive metadata and do not grant MCP/tool permissions.",
    } }));

    await page.goto("/?page=modules");
    const apiEntry = page.getByRole("group", { name: "API Playground" });
    await expect(apiEntry).toContainText("GUI · GET only");
    await apiEntry.getByRole("button", { name: "Open" }).click();
    await expect(page).toHaveURL(/page=api-playground/);
    await expect(page.locator(".page-header h1")).toHaveText("API Playground");

    await page.goto("/?page=modules");
    const profilesEntry = page.getByRole("group", { name: "Agent task profiles" });
    await expect(profilesEntry).toContainText("GUI + CLI metadata");
    await profilesEntry.getByRole("button", { name: "Open" }).click();
    await expect(page).toHaveURL(/page=agent-profiles/);
    await expect(page.locator(".page-header h1")).toHaveText("Agent Profiles");
  });

  test("does not imply that roadmap planning work is already delivered", async ({ page }) => {
    await page.goto("/?page=modules");
    await expect(page.getByText(/does not mark planned roadmap items as implemented/i)).toBeVisible();
    await expect(page.getByRole("group", { name: "Workflow builder" })).toContainText("Design only");
  });
  
  test("opens the native engine and Engine Fabric from Runtime", async ({ page }) => {
    await page.goto("/?page=runtime");
    await expect(page.getByRole("heading", { name: "AXIOM Local Engine" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Choose the right local engine" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Run a local chat" })).toBeVisible();
  });

});
