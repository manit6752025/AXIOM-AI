import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("local model inspection summary is shown and can prefill registry metadata", async ({ page }) => {
  const inspectRequests: Array<Record<string, unknown>> = [];

  await page.route("**/api/workspace/models**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());

    if (url.pathname.endsWith("/discover") && request.method() === "GET") {
      await route.fulfill({
        json: {
          candidates: [{
            path: "qwen-inspect",
            lineage: { relation: "source", parent_manifest_sha256: null },
            format: "Safetensors",
            has_config: true,
            weight_file_count: 2,
            weight_size_bytes: 7340032,
          }],
          truncated: false,
        },
      });
      return;
    }

    if (url.pathname.endsWith("/inspect") && request.method() === "POST") {
      inspectRequests.push(request.postDataJSON() as Record<string, unknown>);
      await route.fulfill({
        json: {
          inspection: {
            path: "models/qwen-inspect",
            architecture: "Qwen3ForCausalLM",
            model_type: "qwen3",
            parameter_count: 8000000000,
            model_format: "Safetensors",
            weight_size_bytes: 7340032,
            has_config: true,
            has_tokenizer: true,
            has_safetensors: true,
            has_gguf: false,
            has_pytorch_weights: false,
            capabilities: ["Text generation", "Chat"],
            estimated_vram_gb: 9.4,
            status: "ready",
          },
        },
      });
      return;
    }

    if (url.pathname === "/api/workspace/models" && request.method() === "GET") {
      await route.fulfill({
        json: {
          models: [],
          total: 0,
          limit: Number(url.searchParams.get("limit") ?? "25"),
          offset: Number(url.searchParams.get("offset") ?? "0"),
          has_more: false,
        },
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
  await expect(page.getByRole("heading", { level: 1, name: "Models", exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Discover local" }).click();
  await expect(page.getByText("models/qwen-inspect", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Inspect", exact: true }).click();
  const summary = page.locator("section").filter({ has: page.getByRole("heading", { name: "Model inspection summary" }) });
  await expect(summary.getByText("Qwen3ForCausalLM", { exact: true })).toBeVisible();
  await expect(summary.getByText("Safetensors", { exact: true })).toBeVisible();
  await expect(summary.getByText("Text generation · Chat", { exact: true })).toBeVisible();
  await expect(summary).toContainText("Tokenizer found");
  await expect(summary).toContainText("8,000,000,000 parameters");
  await expect(summary).toContainText("ready");

  await page.getByRole("button", { name: "Use metadata", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Register model metadata" })).toBeVisible();
  await expect(page.getByPlaceholder("Qwen 3 8B")).toHaveValue("qwen-inspect");
  await expect(page.getByPlaceholder("./models/qwen3-8b")).toHaveValue("models/qwen-inspect");
  await expect(page.getByPlaceholder("Qwen3ForCausalLM")).toHaveValue("Qwen3ForCausalLM");
  await expect(page.getByLabel("Weight size in bytes")).toHaveValue("7340032");
  expect(inspectRequests).toEqual([{ path: "qwen-inspect" }, { path: "qwen-inspect" }]);
});
