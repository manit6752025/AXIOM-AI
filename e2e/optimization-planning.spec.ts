import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

const profile = {
  number: 2,
  key: "reasoning",
  name: "Reasoning Agent",
  description: "Long-context analytical tasks",
  context_tokens: 32768,
  temperature: 0.2,
  preferred_quantization: "Q4_K_M",
};

const plan = {
  profile,
  model: {
    source: "Qwen/Qwen3-8B",
    local_path: "models/qwen3-8b",
    parameter_billions: 8,
    estimated_fp16_gb: 16,
    parameter_source: "model metadata",
    files: 2,
    weights: ["model.safetensors"],
  },
  system: {
    os: "linux",
    arch: "x86_64",
    cpu_count: 8,
    ram_gb: 32,
    gpu_available: true,
    gpu_name: "Test GPU",
    vram_gb: 12,
    disk_free_gb: 100,
  },
  optimization: {
    target_tokens_per_second: 15,
    recommended_quantization: "Q4_K_M",
    recommended_format: "GGUF",
    runtime_bundle: true,
    benchmark_required: true,
  },
};

async function prepareOptimizer(page) {
  await page.route("**/api/optimizer/profiles", (route) =>
    route.fulfill({ json: { profiles: [profile] } }),
  );
  let submitted: Record<string, unknown> | null = null;
  await page.route("**/api/optimizer/plan", async (route) => {
    submitted = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ json: plan });
  });
  await ensureAuthenticated(page);
  return {
    getSubmitted: () => submitted,
  };
}

test("Agent Optimizer submits the chosen profile, model and throughput target", async ({ page }) => {
  const { getSubmitted } = await prepareOptimizer(page);
  await page.goto("/?page=optimizer");

  await expect(page.getByRole("heading", { name: "Make a large model fit your device" })).toBeVisible();
  await page.getByLabel("Agent profile").selectOption("2");
  await page.getByLabel("Hugging Face model / local path").fill("Qwen/Qwen3-8B");
  await page.getByLabel("Target tokens/sec").fill("15");
  await page.getByRole("button", { name: "OPTIMIZE MODEL" }).click();

  await expect(page.getByText("Q4_K_M", { exact: true })).toBeVisible();
  await expect(page.getByText("Qwen/Qwen3-8B", { exact: true })).toBeVisible();
  await expect(page.locator("b").filter({ hasText: /^15 tok\\/s$/ })).toBeVisible();
  expect(getSubmitted()).toEqual({
    model: "Qwen/Qwen3-8B",
    profile: 2,
    target_tokens_per_second: 15,
  });
});

test("Quantization Lab renders size and fit comparisons from the bounded plan", async ({ page }) => {
  const { getSubmitted } = await prepareOptimizer(page);
  await page.goto("/?page=quantizer");

  await expect(page.getByRole("heading", { name: "Make a large model fit your device" })).toBeVisible();
  await page.getByLabel("Agent profile").selectOption("2");
  await page.getByLabel("Hugging Face model / local model").fill("Qwen/Qwen3-8B");
  await page.getByLabel("Performance target").selectOption("15");
  await page.getByRole("button", { name: "ANALYZE & OPTIMIZE" }).click();

  await expect(page.getByText("Optimization result", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Estimated weight sizes" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "INT4" })).toBeVisible();
  await expect(page.getByText("Q4_K_M", { exact: true })).toBeVisible();
  await expect(page.locator("b").filter({ hasText: /^15 tok\\/s$/ })).toBeVisible();
  expect(getSubmitted()).toEqual({
    model: "Qwen/Qwen3-8B",
    profile: 2,
    target_tokens_per_second: 15,
  });
});
