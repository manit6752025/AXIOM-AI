import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Model Studio searches GGUF files and submits a bounded local install job", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.route("**/api/models/studio/status", route => route.fulfill({ json: {
    schema_version: 1, ollama_installed: true, ollama_endpoint: "http://127.0.0.1:11434",
    download_limit_bytes: 21474836480, gguf_directory: "/local/models/gguf",
    catalogue: "Hugging Face public model API", network_search_is_explicit: true,
  }}));
  await page.route("**/api/models/studio/jobs**", async route => {
    if (route.request().method() === "POST") return route.fulfill({ status: 202, json: {
      schema_version: 1, id: "b".repeat(32), status: "queued", operation: "install_gguf",
      created_at: "2026-10-21T00:00:00Z", updated_at: "2026-10-21T00:00:00Z",
      config: { operation: "install_gguf", repo_id: "example/tiny-gguf", filename: "tiny-Q4_K_M.gguf", model_name: "axiom-tiny", register_ollama: true },
      progress_bytes: 0, total_bytes: null, logs: ["Job queued."], error: null, result: null,
    }});
    return route.fulfill({ json: { jobs: [] } });
  });
  await page.route(/\/api\/models\/studio\/search(?:\?.*)?$/, route => route.fulfill({ json: {
    provider: "huggingface", query: "tiny GGUF", count: 1, note: "metadata",
    results: [{ id: "example/tiny-gguf", author: "example", downloads: 20, likes: 2,
      pipeline_tag: "text-generation", last_modified: "2026-10-20", tags: ["gguf"], url: "https://huggingface.co/example/tiny-gguf" }],
  }}));
  await page.route(/\/api\/models\/studio\/files(?:\?.*)?$/, route => route.fulfill({ json: {
    repo_id: "example/tiny-gguf", count: 1, url: "https://huggingface.co/example/tiny-gguf", note: "file",
    files: [{ filename: "tiny-Q4_K_M.gguf", size_bytes: 1234, quantization: "Q4_K_M" }],
  }}));

  await page.goto("/?page=models");
  await expect(page.getByRole("heading", { name: "Model Studio" })).toBeVisible();
  await page.getByLabel("Model search").fill("tiny GGUF");
  await page.getByRole("button", { name: "Search GGUF catalogue" }).click();
  await expect(page.getByText("example/tiny-gguf", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Choose GGUF file" }).click();
  await expect(page.getByLabel("GGUF file", { exact: true })).toContainText("tiny-Q4_K_M.gguf");
  await page.getByLabel("Local model name").fill("axiom-tiny");
  await page.getByRole("button", { name: "Download GGUF and register" }).click();
  await expect(page.getByRole("region", { name: "Model Studio job status" })).toContainText("QUEUED");
});

test("Model Studio requires explicit consent before sending prompts to a remote provider", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.route("**/api/models/studio/status", route => route.fulfill({ json: { schema_version: 1, ollama_installed: false, ollama_endpoint: "http://127.0.0.1:11434", download_limit_bytes: 21474836480, gguf_directory: "/local/models/gguf", catalogue: "Hugging Face public model API", network_search_is_explicit: true } }));
  await page.route("**/api/models/studio/jobs**", route => route.fulfill({ json: { jobs: [] } }));
  let chatWasSent = false;
  await page.route("**/api/models/studio/remote/models", route => route.fulfill({ json: { provider: "openai-compatible", base_url: "https://provider.example/api/v1", models: ["example/chat"], count: 1, credential_storage: "not_saved", note: "not stored" } }));
  await page.route("**/api/models/studio/remote/chat", async route => { chatWasSent = true; return route.fulfill({ json: { provider: "openai-compatible", base_url: "https://provider.example/api/v1", model: "example/chat", response: "remote reply", prompt_tokens: 3, completion_tokens: 2, credential_storage: "not_saved", generated_at: "2026-10-21T00:00:00Z" } }); });
  await page.goto("/?page=models");
  await page.getByLabel("Remote provider base URL").fill("https://provider.example/api/v1");
  await page.getByLabel("Remote provider API key").fill("test-key");
  await page.getByRole("button", { name: "Test connection and list models" }).click();
  await expect(page.getByLabel("Remote model ID")).toHaveValue("example/chat");
  const sendPrompt = page.getByRole("button", { name: "Send prompt to online model" });
  await expect(sendPrompt).toBeDisabled();
  expect(chatWasSent).toBe(false);
  await page.getByRole("checkbox", { name: /I understand this prompt will leave my computer/ }).check();
  await expect(sendPrompt).toBeEnabled();
  await sendPrompt.click();
  await expect(page.getByText("remote reply", { exact: true })).toBeVisible();
  expect(chatWasSent).toBe(true);
});


test("Model Studio installs GGUF directly into AXIOM Native Engine trusted storage", async ({ page }) => {
  await ensureAuthenticated(page);
  await page.route("**/api/models/studio/status", route => route.fulfill({ json: {
    schema_version: 1, ollama_installed: false, ollama_endpoint: "http://127.0.0.1:11434",
    download_limit_bytes: 21474836480, gguf_directory: "/local/models/gguf",
    native_model_root: "/local/axiom/models", native_model_download_limit_bytes: 171798691840,
    native_model_file_limit_bytes: 21474836480, native_pull_available: true,
    catalogue: "Hugging Face public model API", network_search_is_explicit: true,
  }}));
  let submitted: Record<string, unknown> | null = null;
  await page.route("**/api/models/studio/jobs**", async route => {
    if (route.request().method() === "POST") {
      submitted = route.request().postDataJSON() as Record<string, unknown>;
      return route.fulfill({ status: 202, json: {
        schema_version: 1, id: "c".repeat(32), status: "queued", operation: "install_gguf",
        created_at: "2026-10-25T00:00:00Z", updated_at: "2026-10-25T00:00:00Z",
        config: submitted, progress_bytes: 0, total_bytes: null, logs: ["Job queued."], error: null, result: null,
      }});
    }
    return route.fulfill({ json: { jobs: [] }});
  });
  await page.route("**/api/models/studio/search**", route => route.fulfill({ json: {
    provider: "huggingface", query: "tiny GGUF", count: 1, note: "metadata",
    results: [{ id: "example/tiny-gguf", author: "example", downloads: 20, likes: 2,
      pipeline_tag: "text-generation", last_modified: "2026-10-20", tags: ["gguf"], url: "https://huggingface.co/example/tiny-gguf" }],
  }}));
  await page.route("**/api/models/studio/files**", route => route.fulfill({ json: {
    repo_id: "example/tiny-gguf", count: 1, url: "https://huggingface.co/example/tiny-gguf", note: "file",
    files: [{ filename: "tiny-Q4_K_M.gguf", size_bytes: 1234, quantization: "Q4_K_M" }],
  }}));

  await page.goto("/?page=models");
  await page.getByLabel("Model search").fill("tiny GGUF");
  await page.getByRole("button", { name: "Search GGUF catalogue" }).click();
  await page.getByRole("button", { name: "Choose GGUF file" }).click();
  await page.getByLabel("Local model name").fill("axiom-tiny");
  await page.getByLabel("Install directly into AXIOM Native Engine").check();
  await expect(page.getByRole("button", { name: "Install for AXIOM Native Engine" })).toBeEnabled();
  await page.getByRole("button", { name: "Install for AXIOM Native Engine" }).click();
  await expect.poll(() => submitted).toMatchObject({
    operation: "install_gguf",
    repo_id: "example/tiny-gguf",
    filename: "tiny-Q4_K_M.gguf",
    model_name: "axiom-tiny",
    register_ollama: false,
    install_native: true,
  });
  await expect(page.getByRole("region", { name: "Model Studio job status" })).toContainText("QUEUED");
});
