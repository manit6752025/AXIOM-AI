import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Training page submits a bounded local LoRA job and displays its status", async ({
  page,
}) => {
  const queuedJob = {
    schema_version: 1,
    id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
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

  let submittedConfig: Record<string, unknown> | undefined;
  let postRequestCount = 0;

  await ensureAuthenticated(page);

  await page.route("**/api/fine-tuning/**", async (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    const method = request.method();

    if (pathname.endsWith("/runtime") && method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          backend: "transformers-peft-local-lora",
          can_start: true,
          packages: {
            torch: true,
            transformers: true,
            peft: true,
            accelerate: true,
          },
          model_loading: "local_files_only",
          scope_note: "Local model only.",
        }),
      });
      return;
    }

    if (pathname.endsWith("/jobs") && method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ jobs: [] }),
      });
      return;
    }

    if (pathname.endsWith("/jobs") && method === "POST") {
      postRequestCount += 1;

      try {
        submittedConfig = request.postDataJSON() as Record<string, unknown>;
      } catch {
        await route.fulfill({
          status: 400,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Invalid test request JSON" }),
        });
        return;
      }

      await route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify(queuedJob),
      });
      return;
    }

    if (/\/jobs\/[^/]+$/.test(pathname) && method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(queuedJob),
      });
      return;
    }

    await route.continue();
  });

  await page.goto("/?page=training");

  await expect(
    page.getByRole("heading", { name: "Local LoRA fine-tuning" }),
  ).toBeVisible();

  await expect(page.getByText("RUNTIME READY")).toBeVisible({
    timeout: 10_000,
  });

  await page.getByLabel("Local base model directory").fill("/models/base");
  await page.getByLabel("JSONL training dataset").fill("/data/train.jsonl");
  await page.getByLabel("Adapter output directory").fill("/runs/adapter");
  await page.getByLabel("Maximum optimization steps").fill("3");
  await page.getByLabel("Sequence length").fill("64");
  await page.getByLabel("LoRA rank").fill("4");

  const submitButton = page.getByRole("button", {
    name: "Start local fine-tuning",
  });
  await expect(submitButton).toBeEnabled({ timeout: 10_000 });

  const postResponsePromise = page.waitForResponse(
    (response) => {
      const request = response.request();
      const { pathname } = new URL(response.url());

      return (
        request.method() === "POST" &&
        pathname.endsWith("/api/fine-tuning/jobs")
      );
    },
    { timeout: 10_000 },
  );

  await submitButton.click();

  const postResponse = await postResponsePromise;
  expect(postResponse.status()).toBe(202);
  expect(postRequestCount).toBe(1);

  expect(submittedConfig).toMatchObject({
    model_path: "/models/base",
    dataset_path: "/data/train.jsonl",
    output_dir: "/runs/adapter",
    text_field: "text",
    max_steps: 3,
    sequence_length: 64,
    learning_rate: 0.0001,
    lora_rank: 4,
  });

  const jobStatus = page.locator('[aria-label="Fine-tuning job status"]');
  await expect(jobStatus).toBeVisible({ timeout: 10_000 });
  await expect(jobStatus).toContainText(/queued/i);
});
