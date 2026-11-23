import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("model registry removal requires an explicit accessible confirmation", async ({ page }) => {
  const id = "a".repeat(32);
  const model = {
    id, name: "Confirmation model", source: "models/confirmation-model", format: "gguf",
    parameters: "7B", quantization: "Q4_K_M", architecture: "QwenForCausalLM",
    license: "Apache-2.0", tags: ["local"], favorite: false, content_sha256: null,
  };
  let records = [model];
  let deleteCalls = 0;

  await page.route("**/api/workspace/models**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/workspace/models" && request.method() === "GET") {
      await route.fulfill({ json: {
        models: records, total: records.length, limit: 25, offset: 0, has_more: false,
        filters: { search: "", favorite_only: false, format: null, tag: null, sort_by: "name", sort_order: "asc" },
      } });
      return;
    }
    if (url.pathname === `/api/workspace/models/${id}` && request.method() === "DELETE") {
      deleteCalls += 1;
      records = [];
      await route.fulfill({ json: { removed: model, weights_deleted: false } });
      return;
    }
    await route.continue();
  });
  await page.route("**/api/model-studio/status", (route) => route.fulfill({ json: {
    schema_version: 1, ollama_installed: false, ollama_endpoint: "http://127.0.0.1:11434",
    download_limit_bytes: 1000000, gguf_directory: "models", catalogue: "Hugging Face",
    network_search_is_explicit: true,
  } }));
  await page.route("**/api/model-studio/jobs**", (route) => route.fulfill({ json: { jobs: [] } }));

  await ensureAuthenticated(page);
  await page.goto("/?page=models");
  await expect(page.getByText("Confirmation model", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Remove Confirmation model from registry" }).click();

  let dialog = page.getByRole("dialog", { name: "Remove Confirmation model?" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Cancel" })).toBeFocused();
  await expect(dialog).toContainText("Local model weights will not be deleted.");
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).not.toBeVisible();
  expect(deleteCalls).toBe(0);
  await expect(page.getByText("Confirmation model", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Remove Confirmation model from registry" }).click();
  dialog = page.getByRole("dialog", { name: "Remove Confirmation model?" });
  await dialog.getByRole("button", { name: "Remove registry entry" }).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByText("Confirmation model", { exact: true })).toHaveCount(0);
  expect(deleteCalls).toBe(1);
  await expect(page.getByText(/Local weights were not deleted/)).toBeVisible();
});
