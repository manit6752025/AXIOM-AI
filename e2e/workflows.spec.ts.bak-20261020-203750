import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

const workflow = {
  id: "workflow-1",
  name: "Dataset to plan",
  created_at: "2026-10-19T00:00:00Z",
  updated_at: "2026-10-19T00:00:00Z",
  schema_version: 1,
  revision: 1,
  versions: [],
  nodes: [
    { id: "dataset", type: "dataset.profile" as const },
    { id: "plan", type: "training.plan" as const },
  ],
  edges: [{ source: "dataset", target: "plan" }],
  execution_status: "not_executable" as const,
};

test("workflow autosave and dry-run stay planning-only", async ({ page }) => {
  await ensureAuthenticated(page);
  let currentWorkflow = {
    ...workflow,
    versions: [] as Array<{
      revision: number;
      updated_at: string;
      name: string;
      nodes: typeof workflow.nodes;
      edges: typeof workflow.edges;
    }>,
  };
  let createdJob = false;

  await page.route("**/api/workflows**", async (route) => {
    const request = route.request();
    if (request.method() === "GET") {
      await route.fulfill({ json: { workflows: [currentWorkflow] } });
      return;
    }
    if (request.method() === "PUT") {
      const payload = request.postDataJSON() as { name: string; nodes: typeof workflow.nodes; edges: typeof workflow.edges };
      currentWorkflow = {
        ...currentWorkflow,
        ...payload,
        revision: currentWorkflow.revision + 1,
        updated_at: "2026-10-19T00:01:00Z",
        versions: [...currentWorkflow.versions, {
          revision: currentWorkflow.revision,
          updated_at: currentWorkflow.updated_at,
          name: currentWorkflow.name,
          nodes: currentWorkflow.nodes,
          edges: currentWorkflow.edges,
        }],
      };
      await route.fulfill({ json: { workflow: currentWorkflow } });
      return;
    }
    await route.continue();
  });
  await page.route("**/api/workflows/templates", (route) =>
    route.fulfill({ json: { templates: [] } }),
  );
  await page.route("**/api/workflows/workflow-1/dry-run", (route) =>
    route.fulfill({
      json: {
        dry_run: {
          workflow_id: "workflow-1",
          workflow_revision: currentWorkflow.revision,
          schema_version: 1,
          status: "planning_only",
          execution_status: "not_executable",
          executed: false,
          job_created: false,
          node_count: 2,
          steps: [
            { node_id: "dataset", type: "dataset.profile", state: "not_run" },
            { node_id: "plan", type: "training.plan", state: "not_run" },
          ],
          unresolved_inputs: [
            { node_id: "dataset", type: "dataset.profile", reason: "No path is configured." },
          ],
          resource_summary: {
            available_hardware: {
              os_name: "test",
              architecture: "x86_64",
              cpu_cores: 8,
              ram_gb: 32,
              gpu_name: "Test GPU",
              vram_gb: 16,
              cuda_available: true,
            },
            estimated_training_plan: {
              method: "qlora",
              precision: "4-bit",
              lora_rank: 16,
              batch_size: 1,
              gradient_accumulation: 8,
              sequence_length: 2048,
              learning_rate: 0.0002,
              estimated_vram_gb: 8.2,
              fits_hardware: true,
              reason: "Estimated plan fits current hardware.",
            },
            estimated_vram_gb: 8.2,
            fits_hardware: true,
            parameters_billions: 7,
            method: "qlora",
          },
          reproducibility_manifest: {
            schema_version: 1,
            generated_at: "2026-10-19T00:01:01Z",
            workflow_id: "workflow-1",
            workflow_revision: currentWorkflow.revision,
            graph_sha256: "a".repeat(64),
            seed: 42,
            seed_applied: false,
            execution_status: "not_executable",
            software: {
              axiom: "0.3.0",
              python: "3.12.0",
              platform: "Test",
              architecture: "x86_64",
              dependencies: { torch: null },
            },
            hardware: {
              os_name: "test",
              architecture: "x86_64",
              cpu_cores: 8,
              ram_gb: 32,
              gpu_name: "Test GPU",
              vram_gb: 16,
              cuda_available: true,
            },
            planning_settings: {
              parameter_billions: 7,
              requested_method: "auto",
              selected_method: "qlora",
            },
          },
          warnings: ["No job was created or executed."],
        },
      },
    }),
  );
  await page.route("**/api/jobs/**", async (route) => {
    createdJob = true;
    await route.fulfill({ status: 404 });
  });

  await page.goto("/?page=workflows");
  await page.getByRole("button", { name: /Dataset to plan/ }).click();
  const updateResponse = page.waitForResponse((response) =>
    response.url().endsWith("/api/workflows/workflow-1") && response.request().method() === "PUT",
  );
  await page.getByLabel("Workflow name").fill("Autosaved plan");
  await updateResponse;
  await expect(page.getByRole("status").filter({ hasText: "Autosaved · revision 2 · design only" })).toBeVisible();

  await page.getByRole("button", { name: "Dry-run saved graph" }).click();
  await expect(page.getByText("Workflow dry-run summary")).toBeVisible();
  await expect(page.getByText("NOT EXECUTED")).toBeVisible();
  await expect(page.locator("body")).toContainText("8.2 GB");
  await expect(page.getByText(/Seed 42 \(not applied\)/)).toBeVisible();
  expect(createdJob).toBe(false);
});
