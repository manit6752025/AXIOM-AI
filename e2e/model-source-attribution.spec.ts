import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Models registration records source attribution and human-managed license review", async ({ page }) => {
  const submitted: Record<string, unknown>[] = [];
  const savedModels: Record<string, unknown>[] = [];

  await page.route("**/api/workspace/models**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());

    if (url.pathname === "/api/workspace/models" && request.method() === "GET") {
      await route.fulfill({
        json: {
          models: savedModels,
          total: savedModels.length,
          limit: 25,
          offset: 0,
          has_more: false,
        },
      });
      return;
    }

    if (url.pathname === "/api/workspace/models" && request.method() === "POST") {
      const payload = request.postDataJSON() as Record<string, unknown>;
      submitted.push(payload);
      const model = { ...payload, id: "e".repeat(32) };
      savedModels.splice(0, savedModels.length, model);
      await route.fulfill({ status: 201, json: { model } });
      return;
    }

    await route.continue();
  });

  await page.route("**/api/model-studio/status", (route) => route.fulfill({ json: {
    schema_version: 1,
    ollama_installed: false,
    ollama_endpoint: "http://127.0.0.1:11434",
    download_limit_bytes: 1000000,
    gguf_directory: "models",
    catalogue: "Hugging Face",
    network_search_is_explicit: true,
  } }));
  await page.route("**/api/model-studio/jobs**", (route) => route.fulfill({ json: { jobs: [] } }));

  await ensureAuthenticated(page);
  await page.goto("/?page=models");
  await expect(page.getByRole("heading", { level: 1, name: "Models", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Register model" }).click();

  await page.getByPlaceholder("Qwen 3 8B").fill("Attributed Qwen");
  await page.getByPlaceholder("./models/qwen3-8b").fill("models/attributed-qwen");
  await page.getByPlaceholder("safetensors, gguf, ...").fill("safetensors");
  await page.getByPlaceholder("Apache-2.0").fill("Apache-2.0");
  await page.getByLabel("License review state").selectOption("restricted");
  await page.getByLabel("Source attribution").fill("Example Research Group");
  await page.getByLabel("Source URL").fill("https://huggingface.co/example/attributed-qwen");
  await page.getByLabel("Source revision").fill("0123456789abcdef");
  await page.getByLabel("License review notes").fill("Non-commercial use only; checked the model card.");

  await page.getByRole("button", { name: "Save to registry" }).click();
  await expect.poll(() => submitted.length).toBe(1);
  expect(submitted[0]).toMatchObject({
    name: "Attributed Qwen",
    source: "models/attributed-qwen",
    license: "Apache-2.0",
    license_review: "restricted",
    source_attribution: "Example Research Group",
    source_url: "https://huggingface.co/example/attributed-qwen",
    source_revision: "0123456789abcdef",
    license_review_notes: "Non-commercial use only; checked the model card.",
  });

  await expect(page.getByText("Attributed Qwen", { exact: true })).toBeVisible();
  await expect(page.locator(".model-registry-row")).toContainText("Example Research Group");
  await expect(page.locator(".model-registry-row")).toContainText("licence review restricted");
  await expect(page.getByRole("link", { name: "Source page" })).toHaveAttribute(
    "href",
    "https://huggingface.co/example/attributed-qwen",
  );
});
