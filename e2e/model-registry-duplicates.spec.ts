import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Models form preserves entered metadata and surfaces duplicate source conflicts", async ({ page }) => {
  let postCount = 0;
  await page.route("**/api/workspace/models**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/workspace/models" && request.method() === "GET") {
      await route.fulfill({
        json: { models: [], total: 0, limit: 25, offset: 0, has_more: false },
      });
      return;
    }
    if (url.pathname === "/api/workspace/models" && request.method() === "POST") {
      postCount += 1;
      await route.fulfill({
        status: 409,
        json: { detail: "Model source already registered: org/demo-model" },
      });
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
  await page.getByRole("button", { name: "Register model" }).click();
  await page.getByPlaceholder("Qwen 3 8B").fill("Duplicate source test");
  await page.getByPlaceholder("./models/qwen3-8b").fill("org/demo-model");
  await page.getByPlaceholder("safetensors, gguf, ...").fill("gguf");
  await page.getByRole("button", { name: "Save to registry" }).click();

  await expect.poll(() => postCount).toBe(1);
  const notice = page.locator(".notice-toast");
  await expect(notice).toContainText("Model source already registered: org/demo-model");
  await expect(page.getByRole("heading", { name: "Register model metadata" })).toBeVisible();
  await expect(page.getByPlaceholder("Qwen 3 8B")).toHaveValue("Duplicate source test");
  await expect(page.getByPlaceholder("./models/qwen3-8b")).toHaveValue("org/demo-model");
});
