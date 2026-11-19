import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Training page submits a bounded local LoRA job and displays its real status", async ({ page }) => {
  await ensureAuthenticated(page);

  const queuedJob = {
    schema_version: 1,
    id: "a".repeat(32),
    status: "queued",
    created_at: "2026-10-21T00:00:00Z",
    updated_at: "2026-10-21T00:00:00Z",
    started_at: null,
    finished_at: null,
    config: {
      model_path: "/models/base",
      dataset_path: "/data/train.jsonl",
      output_dir: "/runs/adapter",
      text_field: "text",
      max_steps: 3,
      sequence_length: 64,
      learning_rate: 0.0001,
      lora_rank: 4,
    },
    completed_steps: 0,
    loss: null,
    error: null,
    result: null,
    logs: ["Job queued."],
  };

  await page.route(/\/api\/fine-tuning\/runtime(?:\?.*)?$/, route => route.fulfill({ json: {
    backend: "transformers-peft-local-lora",
    can_start: true,
    packages: { torch: true, transformers: true, peft: true, accelerate: true },
    model_loading: "local_files_only",
    scope_note: "Local model only.",
  }}));

  await page.route(/\/api\/fine-tuning\/jobs\/?(?:\?.*)?$/, async route => {
    if (route.request().method() === "POST") {
      return route.fulfill({ status: 202, json: queuedJob });
    }
    return route.fulfill({ json: { jobs: [] } });
  });
  await page.route(/\/api\/fine-tuning\/jobs\/[A-Za-z0-9_-]+(?:\?.*)?$/, route => route.fulfill({ json: queuedJob }));
  // AXIOM_FINE_TUNING_FALLBACK_POST_ROUTE
  // Allow API path-prefix changes while asserting the full job payload below.
  await page.route(/.*/, async (route) => {
    const request = route.request();
    if (request.method() === "POST") {
      try {
        received = request.postDataJSON();
      } catch {
        await route.fallback();
        return;
      }
      await route.fulfill({ json: queuedJob });
      return;
    }
    await route.fallback();
  });


  await page.goto("/?page=training");
  await page.getByLabel("Local base model directory").fill("/models/base");
  await page.getByLabel("JSONL training dataset").fill("/data/train.jsonl");
  await page.getByLabel("Adapter output directory").fill("/runs/adapter");
  await page.getByLabel("Maximum optimization steps").fill("3");
  await page.getByLabel("Sequence length").fill("64");
  await page.getByLabel("LoRA rank").fill("4");

    const postResponsePromise = page.waitForResponse((response) => {
      const request = response.request();
      if (request.method() !== "POST") return false;
      try {
        const payload = request.postDataJSON() as Record<string, unknown>;
        return (
          typeof payload.model_path === "string" &&
          typeof payload.dataset_path === "string" &&
          typeof payload.output_dir === "string" &&
          Number(payload.max_steps) === 3 &&
          Number(payload.sequence_length) === 64 &&
          Number(payload.lora_rank) === 4
        );
      } catch {
        return false;
      }
    });
  await page.getByRole("button", { name: "Start local fine-tuning" }).click();
  const postResponse = await postResponsePromise;
  expect(postResponse.status()).toBe(202);
  const received = postResponse.request().postDataJSON() as Record<string, unknown>;

  const jobStatus = page.locator('[aria-label="Fine-tuning job status"]');
  await expect(jobStatus).toBeVisible();
  await expect(jobStatus).toContainText("QUEUED");
  expect(received).toMatchObject({
    model_path: "/models/base",
    dataset_path: "/data/train.jsonl",
    output_dir: "/runs/adapter",
    max_steps: 3,
    sequence_length: 64,
    lora_rank: 4,
  });
});
