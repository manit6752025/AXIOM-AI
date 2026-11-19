import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("model registry filters and saved presets use the real typed API query contract", async ({ page }) => {
  const requests: URL[] = [];
  const models = [
    { id: "a".repeat(32), name: "Qwen Seven", source: "models/qwen-seven", format: "gguf", parameters: "7B", quantization: "Q4_K_M", architecture: "QwenForCausalLM", license: "Apache-2.0", tags: ["local", "chat"], favorite: true, content_sha256: null },
    { id: "b".repeat(32), name: "Qwen Mini", source: "models/qwen-mini", format: "gguf", parameters: "1.5B", quantization: "Q4_K_M", architecture: "QwenForCausalLM", license: "Apache-2.0", tags: ["local", "chat"], favorite: false, content_sha256: null },
    { id: "c".repeat(32), name: "Llama Base", source: "models/llama", format: "safetensors", parameters: "8B", quantization: "", architecture: "LlamaForCausalLM", license: "Meta", tags: ["training"], favorite: false, content_sha256: null },
  ];
  const fitReport = {
    schema_version: 1,
    model: { id: "a".repeat(32), name: "Qwen Seven", format: "gguf", parameters: "7B", parameter_count_billions: 7, parameter_basis: "unit_suffix", quantization: "Q4_K_M", architecture: "QwenForCausalLM" },
    hardware: { os_name: "Linux", architecture: "x86_64", cpu_cores: 8, ram_gb: 16, gpu_name: "Test GPU", vram_gb: 4, vram_probe: "NVIDIA nvidia-smi only", unified_memory_detected: false },
    quantization_bits_per_parameter: { low: 4, high: 5 },
    estimated_weight_gb: { low: 3.5, high: 4.38 },
    estimated_runtime_memory_gb: { low: 3.85, high: 6.57 },
    system_ram_fit: { status: "likely", physical_gb: 16, budget_gb: 11.2, reserve_percent: 30, required_gb: { low: 3.85, high: 6.57 } },
    gpu_memory_fit: { status: "unlikely", physical_gb: 4, budget_gb: 3.4, reserve_percent: 15, required_gb: { low: 3.85, high: 6.57 } },
    format_compatibility: { status: "candidate_not_verified", likely_runtimes: ["Ollama", "llama.cpp"], guidance: "GGUF is a runtime candidate, not a verified load." },
    warnings: ["This is a metadata-based screening estimate, not a benchmark.", "GPU memory measurement is limited."],
    scope_note: "Estimate only. Runtime compatibility and actual performance require loading and measuring this model with a specific local backend.",
  };

  await page.route("**/api/workspace/models**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname.endsWith("/compatibility") && request.method() === "GET") {
      await route.fulfill({ json: { report: fitReport } });
      return;
    }
    if (url.pathname !== "/api/workspace/models" || request.method() !== "GET") {
      await route.continue();
      return;
    }
    requests.push(url);
    const search = (url.searchParams.get("search") ?? "").toLowerCase();
    const format = (url.searchParams.get("format") ?? "").toLowerCase();
    const tag = (url.searchParams.get("tag") ?? "").toLowerCase();
    const favorites = url.searchParams.get("favorite_only") === "true";
    const sortBy = url.searchParams.get("sort_by") ?? "name";
    const sortOrder = url.searchParams.get("sort_order") ?? "asc";
    const filtered = models.filter((model) =>
      (!search || [model.name, model.source, model.format, model.parameters ?? "", model.architecture ?? "", model.license ?? "", ...model.tags].some((field) => field.toLowerCase().includes(search)))
      && (!format || format === "all" || model.format.toLowerCase() === format)
      && (!tag || model.tags.some((item) => item.toLowerCase() === tag))
      && (!favorites || model.favorite)
    );
    filtered.sort((left, right) => {
      const a = sortBy === "parameters" ? Number.parseFloat(left.parameters ?? "0") : sortBy === "favorite" ? Number(left.favorite) : String(left[sortBy as keyof typeof left] ?? "").toString().toLowerCase();
      const b = sortBy === "parameters" ? Number.parseFloat(right.parameters ?? "0") : sortBy === "favorite" ? Number(right.favorite) : String(right[sortBy as keyof typeof right] ?? "").toString().toLowerCase();
      const result = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
      return sortOrder === "desc" ? -result : result;
    });
    const offset = Number(url.searchParams.get("offset") ?? "0");
    const limit = Number(url.searchParams.get("limit") ?? "25");
    const items = filtered.slice(offset, offset + limit);
    await route.fulfill({ json: {
      models: items, total: filtered.length, limit, offset, has_more: offset + limit < filtered.length,
      filters: { search, favorite_only: favorites, format: format || null, tag: tag || null, sort_by: sortBy, sort_order: sortOrder },
    } });
  });

  await page.route("**/api/model-studio/status", (route) => route.fulfill({ json: {
    schema_version: 1, ollama_installed: false, ollama_endpoint: "http://127.0.0.1:11434",
    download_limit_bytes: 1000000, gguf_directory: "models", catalogue: "Hugging Face",
    network_search_is_explicit: true,
  } }));
  await page.route("**/api/model-studio/jobs**", (route) => route.fulfill({ json: { jobs: [] } }));

  await ensureAuthenticated(page);
  await page.goto("/?page=models");
  await expect(page.getByRole("heading", { level: 1, name: "Models", exact: true })).toBeVisible();
  await expect(page.getByText("Qwen Seven", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Check hardware fit for Qwen Seven" }).click();
  await expect(page.getByRole("heading", { name: "Hardware fit · Qwen Seven" })).toBeVisible();
  await expect(page.getByText("3.50–4.38 GB", { exact: true })).toBeVisible();
  await expect(page.getByText("UNLIKELY", { exact: true })).toBeVisible();
  await expect(page.getByText("MEMORY ESTIMATE · NOT A BENCHMARK", { exact: true })).toBeVisible();

  await page.getByRole("textbox", { name: "Search model registry" }).fill("qwen");
  await page.getByRole("textbox", { name: "Exact model format" }).fill("gguf");
  await page.getByRole("textbox", { name: "Exact model tag" }).fill("chat");
  await page.getByRole("checkbox", { name: "Favorites only" }).check();
  await page.getByLabel("Sort model registry by").selectOption("parameters");
  await page.getByLabel("Model sort order").selectOption("desc");
  await page.getByRole("button", { name: "Apply filters" }).click();

  await expect(page.getByText("Qwen Seven", { exact: true })).toBeVisible();
  await expect(page.getByText("Qwen Mini", { exact: true })).toHaveCount(0);
  const filteredRequest = requests.at(-1);
  expect(filteredRequest?.searchParams.get("search")).toBe("qwen");
  expect(filteredRequest?.searchParams.get("format")).toBe("gguf");
  expect(filteredRequest?.searchParams.get("tag")).toBe("chat");
  expect(filteredRequest?.searchParams.get("favorite_only")).toBe("true");
  expect(filteredRequest?.searchParams.get("sort_by")).toBe("parameters");
  expect(filteredRequest?.searchParams.get("sort_order")).toBe("desc");

  await page.getByRole("textbox", { name: "Saved filter name" }).fill("Favorite Qwen chat models");
  await page.getByRole("button", { name: "Save current filters" }).click();
  await expect(page.getByText("Favorite Qwen chat models", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(page.getByText("Qwen Mini", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Apply", exact: true }).last().click();
  await expect(page.getByText("Qwen Seven", { exact: true })).toBeVisible();
  await expect(page.getByText("Qwen Mini", { exact: true })).toHaveCount(0);
});
