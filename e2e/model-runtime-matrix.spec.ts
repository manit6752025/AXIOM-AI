import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Models page displays explicit local runtime version and format evidence", async ({ page }) => {
  await page.route("**/api/workspace/models/runtime-matrix**", async (route) => {
    await route.fulfill({
      json: {
        matrix: {
          schema_version: 1,
          probed_at: "2026-11-20T00:00:00Z",
          cache_ttl_seconds: 30,
          network_access: false,
          model_load_tested: false,
          runtimes: [
            {
              id: "ollama",
              name: "Ollama",
              availability: "installed",
              version: "ollama version is 0.5.0",
              version_evidence: "ollama --version",
              evidence_method: "ollama --version",
              candidate_formats: ["GGUF"],
            },
            {
              id: "llama.cpp",
              name: "llama.cpp",
              availability: "not_detected",
              version: null,
              version_evidence: null,
              evidence_method: "llama.cpp CLI --version",
              candidate_formats: ["GGUF"],
            },
            {
              id: "transformers",
              name: "Hugging Face Transformers",
              availability: "installed",
              version: "4.55.0",
              version_evidence: "importlib.metadata.version('transformers')",
              evidence_method: "importlib.metadata.version('transformers')",
              candidate_formats: ["Safetensors", "PyTorch"],
            },
          ],
          formats: [
            {
              format: "GGUF",
              candidates: [
                {
                  runtime_id: "ollama",
                  runtime_name: "Ollama",
                  availability: "installed",
                  version: "ollama version is 0.5.0",
                  version_evidence: "ollama --version",
                  status: "candidate_not_verified",
                },
                {
                  runtime_id: "llama.cpp",
                  runtime_name: "llama.cpp",
                  availability: "not_detected",
                  version: null,
                  version_evidence: null,
                  status: "candidate_not_verified",
                },
              ],
            },
            {
              format: "Safetensors",
              candidates: [
                {
                  runtime_id: "transformers",
                  runtime_name: "Hugging Face Transformers",
                  availability: "installed",
                  version: "4.55.0",
                  version_evidence: "importlib.metadata.version('transformers')",
                  status: "candidate_not_verified",
                },
              ],
            },
            {
              format: "PyTorch",
              candidates: [
                {
                  runtime_id: "transformers",
                  runtime_name: "Hugging Face Transformers",
                  availability: "installed",
                  version: "4.55.0",
                  version_evidence: "importlib.metadata.version('transformers')",
                  status: "candidate_not_verified",
                },
              ],
            },
          ],
          limitations: ["A missing runtime or unavailable version probe is not interpreted as proof that a format is unsupported."],
          scope_note: "Local runtime and version evidence only. Format matches are candidates, not proof that a particular model can be loaded.",
        },
      },
    });
  });
  await page.route("**/api/workspace/models**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/workspace/models/runtime-matrix") {
      await route.fallback();
      return;
    }
    if (url.pathname === "/api/workspace/models" && route.request().method() === "GET") {
      await route.fulfill({ json: { models: [], total: 0, limit: 25, offset: 0, has_more: false } });
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
  await page.getByRole("button", { name: "Compatibility matrix" }).click();

  const matrix = page.locator("section").filter({ has: page.getByRole("heading", { name: "Format/runtime compatibility" }) });
  await expect(matrix).toContainText("GGUF");
  await expect(matrix).toContainText("Ollama: installed · ollama version is 0.5.0");
  await expect(matrix).toContainText("llama.cpp: not detected");
  await expect(matrix).toContainText("Safetensors");
  await expect(matrix).toContainText("Hugging Face Transformers: installed · 4.55.0");
  await expect(matrix).toContainText("Local runtime and version evidence only");
  await expect(matrix).toContainText("No network requests or model loads are performed");
});
