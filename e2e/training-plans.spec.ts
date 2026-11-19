import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

const basePlan = {
  method: "qlora",
  precision: "4-bit",
  lora_rank: 16,
  batch_size: 1,
  gradient_accumulation: 16,
  sequence_length: 4096,
  learning_rate: 0.0002,
  estimated_vram_gb: 7.84,
  fits_hardware: true,
  reason: "Configuration is estimated to fit available GPU memory.",
};

const hardware = {
  os_name: "TestOS",
  architecture: "x86_64",
  cpu_cores: 8,
  ram_gb: 32,
  gpu_name: "Test GPU",
  vram_gb: 16,
  cuda_available: true,
};

const left = {
  schema_version: 1,
  id: "plan-left",
  name: "Seven billion QLoRA",
  created_at: "2026-11-18T10:00:00Z",
  model: "models/qwen",
  dataset: "data/train.jsonl",
  notes: null,
  tags: ["baseline"],
  parameter_billions: 7,
  requested_method: "qlora",
  planner_algorithm_version: 1,
  axiom_version: "0.3.0-beta.8",
  status: "planned",
  hardware,
  plan: basePlan,
};

const right = {
  ...left,
  id: "plan-right",
  name: "Thirteen billion QLoRA",
  parameter_billions: 13,
  hardware: { ...hardware, vram_gb: 24 },
  plan: { ...basePlan, sequence_length: 2048, learning_rate: 0.0001 },
};

const snapshot = {
  format: "application/vnd.axiom.training-plan+json;version=1",
  schema_version: 1,
  planner: { algorithm_version: 1, axiom_version: "0.3.0-beta.8", snapshot_kind: "saved_estimate" },
  inputs: { parameter_billions: 7, requested_method: "qlora", model: "models/qwen", dataset: "data/train.jsonl" },
  hardware,
  plan: basePlan,
  execution: { started: false, scope_note: "This is the exact saved planning snapshot; no training job was started." },
  fingerprint: { algorithm: "sha256", value: "a".repeat(64) },
};

test("Training prints a reproducible saved plan and diffs two snapshots", async ({ page }) => {
  await ensureAuthenticated(page);

  await page.route("**/api/workspace/runtime", (route) => route.fulfill({
    json: { status: "offline", hardware },
  }));
  await page.route("**/api/workspace/experiments**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith("/plan-left/reproducible")) {
      await route.fulfill({ json: { snapshot } });
      return;
    }
    if (url.pathname.endsWith("/plan-left/diff/plan-right")) {
      await route.fulfill({ json: { diff: {
        schema_version: 1,
        left: { id: "plan-left", name: left.name, fingerprint: "a".repeat(64) },
        right: { id: "plan-right", name: right.name, fingerprint: "b".repeat(64) },
        identical: false,
        changes: [
          { path: "inputs.parameter_billions", category: "inputs", before_present: true, before: 7, after_present: true, after: 13 },
          { path: "hardware.vram_gb", category: "hardware", before_present: true, before: 16, after_present: true, after: 24 },
          { path: "plan.sequence_length", category: "estimate", before_present: true, before: 4096, after_present: true, after: 2048 },
        ],
        scope_note: "Saved snapshots only; no planner calculation or training execution.",
      } } });
      return;
    }
    await route.fulfill({ json: {
      experiments: [right, left],
      total: 2,
      limit: 100,
      offset: 0,
      has_more: false,
    } });
  });
  await page.route("**/api/fine-tuning/runtime", (route) => route.fulfill({
    json: {
      backend: "transformers-peft-local-lora",
      can_start: false,
      packages: { torch: false, transformers: false, peft: false, accelerate: false },
      model_loading: "local_files_only",
      scope_note: "Optional local runtime is unavailable in this browser test.",
    },
  }));
  await page.route("**/api/fine-tuning/jobs**", (route) => route.fulfill({ json: { jobs: [] } }));

  await page.goto("/?page=training");
  await expect(page.getByRole("heading", { name: "Saved plan tools" })).toBeVisible();

  await page.getByLabel("Saved plan to print").selectOption("plan-left");
  await page.getByRole("button", { name: "Print canonical JSON" }).click();
  await expect(page.getByText("Saved plan fingerprint")).toBeVisible();
  await expect(page.locator(".plan-reproducible-json")).toContainText("application/vnd.axiom.training-plan+json");
  await expect(page.locator(".plan-reproducible-json")).toContainText("models/qwen");

  await page.getByLabel("Baseline saved plan").selectOption("plan-left");
  await page.getByLabel("Comparison saved plan").selectOption("plan-right");
  await page.getByRole("button", { name: "Diff saved plans" }).click();
  await expect(page.getByText("3 recorded field(s) changed")).toBeVisible();
  await expect(page.getByText("inputs.parameter_billions", { exact: true })).toBeVisible();
  await expect(page.getByText("hardware.vram_gb", { exact: true })).toBeVisible();
  const diffTable = page.getByRole("table", { name: "Saved plan field differences" });
  await expect(diffTable).toBeVisible();
  await expect(diffTable.getByRole("columnheader")).toHaveText(["Field", "Category", "Baseline", "Comparison"]);
  await expect(diffTable.getByRole("row")).toHaveCount(4);
});
