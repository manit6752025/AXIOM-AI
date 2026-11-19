import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("unified LLM client runs local chats and requires consent for cloud chats", async ({ page }) => {
  const chatRequests: Array<Record<string, unknown>> = [];
  let modelCatalogueCalls = 0;

  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;

    if (path === "/api/workspace/runtime") {
      await route.fulfill({ json: { status: "ready", hardware: {
        os_name: "Linux", architecture: "x86_64", cpu_cores: 8, ram_gb: 16,
        gpu_name: null, vram_gb: null, cuda_available: false,
      } } });
    } else if (path === "/api/inference/status") {
      await route.fulfill({ json: {
        backend: "ollama", endpoint: "http://127.0.0.1:11434", runtime_installed: false,
        status: "offline", models: [], error: "Ollama is offline for this test.", checked_at: "2026-10-24T00:00:00Z",
      } });
    } else if (path === "/api/engine/status") {
      await route.fulfill({ json: {
        engine: "axiom-native-pytorch-llama", ready: false, packages: {}, versions: {},
        loaded: false, architecture: null, model_loading: "local_safetensors_only", remote_code: false,
        setup_note: "Optional dependencies are not installed in this test.", scope_note: "Test fixture.",
        checked_at: "2026-10-24T00:00:00Z",
      } });
    } else if (path === "/api/engine/models") {
      await route.fulfill({ json: { schema_version: 1, models: [], adapters: [], truncated: false, scope_note: "Test fixture." } });
    } else if (path === "/api/diagnostics/runtime-traces" && request.method() === "GET") {
      await route.fulfill({ json: {
        schema_version: 1, traces: [], total_retained: 0, limit: 50, max_retained: 200,
        privacy: { prompt_bodies_stored: false, response_bodies_stored: false, disk_persistence: false },
      } });
    } else if (path === "/api/diagnostics/runtime-connectors") {
      await route.fulfill({ json: {
        schema_version: 1, discovery_mode: "passive_local_discovery", checked_at: "2026-10-24T00:00:00Z",
        connectors: [], scope_note: "Test fixture.",
      } });
    } else if (path === "/api/llm/providers") {
      await route.fulfill({ json: {
        schema_version: 1, credentials_persisted: false,
        providers: [
          { id: "axiom-local", label: "AXIOM Native Engine", kind: "local", requires_api_key: false, supports_model_listing: true, privacy: "local-only", note: "Local test model." },
          { id: "ollama-local", label: "Ollama (local)", kind: "local", requires_api_key: false, supports_model_listing: true, privacy: "local-only", note: "Local Ollama test model." },
          { id: "deepseek", label: "DeepSeek", kind: "cloud", requires_api_key: true, supports_model_listing: true, privacy: "remote-consent-required", note: "Cloud test provider." },
        ],
      } });
    } else if (path === "/api/llm/models" && request.method() === "POST") {
      modelCatalogueCalls += 1;
      const body = request.postDataJSON() as { provider: string; api_key?: string };
      if (body.provider === "deepseek") expect(body.api_key).toBe("never-return-this");
      else expect(body.api_key).toBeUndefined();
      await route.fulfill({ json: {
        provider: body.provider,
        models: body.provider === "axiom-local" ? ["tiny-llama"] : body.provider === "ollama-local" ? ["qwen2.5:0.5b"] : ["deepseek-chat"],
        count: 1, credential_storage: body.provider === "deepseek" ? "not_saved" : "not_applicable",
        execution: body.provider === "deepseek" ? "remote" : "local",
        note: "Test fixture.",
      } });
    } else if (path === "/api/llm/chat" && request.method() === "POST") {
      const body = request.postDataJSON() as Record<string, unknown>;
      chatRequests.push(body);
      await route.fulfill({ json: {
        provider: body.provider,
        model: body.model,
        response: body.provider === "axiom-local" ? "local-only answer" : body.provider === "ollama-local" ? "ollama local answer" : "cloud answer",
        prompt_tokens: 4, completion_tokens: 3,
        credential_storage: body.provider === "deepseek" ? "not_saved" : "not_applicable",
        execution: body.provider === "deepseek" ? "remote" : "local",
        device: body.provider === "axiom-local" ? "cpu" : body.provider === "ollama-local" ? "ollama" : undefined,
        generated_at: "2026-10-24T00:00:00Z",
      } });
    } else {
      await route.fulfill({ status: 404, json: { detail: "No fixture registered for " + path } });
    }
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");
  const panel = page.locator(".llm-client-panel");
  await expect(panel.getByRole("heading", { name: "Unified LLM client" })).toBeVisible();

  await panel.getByLabel("Model ID").fill("tiny-llama");
  await panel.getByLabel("Prompt").fill("Hello local model");
  await panel.getByRole("button", { name: "Generate locally" }).click();
  await expect.poll(() => chatRequests.length).toBe(1);
  await expect(panel.locator(".llm-message.assistant-message").last()).toContainText("local-only answer");
  expect(chatRequests).toHaveLength(1);
  expect(chatRequests[0]["provider"]).toBe("axiom-local");
  expect(chatRequests[0]["consent"]).toBe(true);

  await panel.getByLabel("Prompt").fill("Continue our previous answer");
  await panel.getByRole("button", { name: "Generate locally" }).click();
  await expect(panel.locator(".llm-message.assistant-message")).toHaveCount(2);
  expect(chatRequests).toHaveLength(2);
  expect(chatRequests[1]["provider"]).toBe("axiom-local");
  const continuedMessages = chatRequests[1]["messages"] as Array<{ role: string; content: string }>;
  expect(continuedMessages.some((message) => message.role === "user" && message.content === "Hello local model")).toBe(true);
  expect(continuedMessages.some((message) => message.role === "assistant" && message.content === "local-only answer")).toBe(true);
  expect(continuedMessages.some((message) => message.role === "user" && message.content === "Continue our previous answer")).toBe(true);

  await panel.getByLabel("LLM provider", { exact: true }).selectOption("ollama-local");
  await panel.getByRole("button", { name: "Load models" }).click();
  await expect(panel.getByLabel("Model ID")).toHaveValue("qwen2.5:0.5b");
  await panel.getByLabel("Prompt").fill("Hello Ollama local model");
  await panel.getByRole("button", { name: "Generate locally" }).click();
  await expect(panel.locator(".llm-message.assistant-message").last()).toContainText("ollama local answer");
  expect(chatRequests).toHaveLength(3);
  expect(chatRequests[2]["provider"]).toBe("ollama-local");
  expect(chatRequests[2]["consent"]).toBe(true);
  expect(chatRequests[2]["api_key"]).toBeUndefined();
  expect((chatRequests[2]["messages"] as Array<{ role: string; content: string }>).some((message) => message.content === "Hello Ollama local model")).toBe(true);

  await panel.getByLabel("LLM provider", { exact: true }).selectOption("deepseek");
  await panel.getByLabel("Provider API key").fill("never-return-this");
  await panel.getByRole("button", { name: "Load models" }).click();
  await expect(panel.getByLabel("Model ID")).toHaveValue("deepseek-chat");
  await expect.poll(() => modelCatalogueCalls).toBeGreaterThan(1);

  await panel.getByLabel("Prompt").fill("Hello cloud model");
  const consent = panel.getByRole("checkbox", { name: /this conversation will be sent to DeepSeek/i });
  await expect(panel.getByRole("button", { name: "Send and generate" })).toBeDisabled();
  await consent.check();
  await panel.getByRole("button", { name: "Send and generate" }).click();
  await expect(panel.getByText("cloud answer", { exact: true })).toBeVisible();

  expect(chatRequests).toHaveLength(4);
  expect(chatRequests[3]["provider"]).toBe("deepseek");
  expect(chatRequests[3]["consent"]).toBe(true);
  expect(chatRequests[3]["api_key"]).toBe("never-return-this");
  const messages = chatRequests[3]["messages"] as Array<{ role: string; content: string }>;
  expect(messages.some((message) => message.role === "user" && message.content === "Hello cloud model")).toBe(true);
  await expect(panel).not.toContainText("never-return-this");
});
