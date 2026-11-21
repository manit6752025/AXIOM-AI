import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime creates, probes, uses, and deletes a bounded native serving profile", async ({ page }) => {
  await ensureAuthenticated(page);
  const model = {
    path: "/workspace/models/tiny-qwen",
    name: "tiny-qwen",
    model_type: "qwen2",
    architecture: "Qwen2ForCausalLM",
    weight_format: "safetensors",
    max_position_embeddings: 8192,
    effective_max_position_embeddings: 8192,
    rope_scaling_type: null,
    ready_for_native_attempt: true,
  };
  let profiles: Array<Record<string, unknown>> = [];
  let createPayload: Record<string, unknown> | null = null;
  let generatePayload: Record<string, unknown> | null = null;

  await page.route("**/api/engine/models", (route) =>
    route.fulfill({ json: {
      schema_version: 1, models: [model], adapters: [], truncated: false,
      scope_note: "Local metadata discovery only.",
    } }),
  );
  await page.route("**/api/engine/serving-profiles", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: {
        schema_version: 1, profiles, total: profiles.length, max_profiles: 25,
        credentials_persisted: false, network_used: false, weights_loaded: false,
      } });
      return;
    }
    createPayload = route.request().postDataJSON() as Record<string, unknown>;
    const profile = {
      id: "a".repeat(32),
      ...createPayload,
      created_at: "2026-11-21T00:00:00Z",
      updated_at: "2026-11-21T00:00:00Z",
    };
    profiles = [profile];
    await route.fulfill({ status: 201, json: {
      profile,
      model_context: {
        model_type: "qwen2", architecture: "Qwen2ForCausalLM",
        checkpoint_format: "safetensors", max_position_embeddings: 8192,
        effective_max_position_embeddings: 8192,
      },
      scope: { weights_loaded: false, service_started: false, network_used: false },
    } });
  });
  await page.route("**/api/engine/serving-profiles/*/readiness", async (route) => {
    await route.fulfill({ json: {
      schema_version: 1, profile_id: "a".repeat(32), profile_name: "Runtime smoke",
      ready: true, checks: { model: { available: true }, device: { available: true } },
      problems: [], weights_loaded: false, service_started: false, network_used: false,
      probed_at: "2026-11-21T00:00:00Z",
      scope_note: "Metadata-only readiness probe.",
    } });
  });
  await page.route("**/api/engine/serving-profiles/*/generate", async (route) => {
    generatePayload = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ json: {
      schema_version: 1,
      profile: { id: "a".repeat(32), name: "Runtime smoke" },
      request_budget: {
        prompt_characters: 12, prompt_tokens: 8, max_new_tokens: 64,
        requested_context_tokens: 72, context_limit_tokens: 4096,
      },
      result: { response: "Serving profile response stays local." },
      scope: { local_only: true, network_used: false, profile_persisted: false },
    } });
  });
  await page.route("**/api/engine/serving-profiles/*", async (route) => {
    if (route.request().method() === "DELETE") {
      await route.fulfill({ json: { deleted: profiles[0], model_files_deleted: false } });
      profiles = [];
      return;
    }
    await route.continue();
  });

  await page.goto("/?page=runtime");
  await expect(page.getByRole("heading", { name: "Serving profiles" })).toBeVisible();
  await page.getByLabel("Serving profile name").fill("Runtime smoke");
  await page.getByLabel("Serving model checkpoint").selectOption(model.path);
  await page.getByLabel("Serving device").selectOption("cpu");
  await page.getByLabel("Serving context length").fill("4096");
  await page.getByLabel("Serving maximum output tokens").fill("64");
  await page.getByRole("button", { name: "Save serving profile" }).click();

  await expect.poll(() => createPayload).not.toBeNull();
  expect(createPayload).toMatchObject({
    name: "Runtime smoke",
    model_path: model.path,
    device_preference: "cpu",
    context_length: 4096,
    max_new_tokens: 64,
    max_concurrent_requests: 1,
    max_queued_requests: 0,
  });
  await expect(page.getByText("Runtime smoke", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Probe" }).click();
  await expect(page.getByText("READY", { exact: true })).toBeVisible();

  await page.getByLabel("Profile for generation").selectOption("a".repeat(32));
  await page.getByLabel("Serving profile prompt").fill("Hello local model");
  await page.getByRole("button", { name: "Generate through profile" }).click();
  await expect.poll(() => generatePayload).not.toBeNull();
  expect(generatePayload).toMatchObject({ prompt: "Hello local model" });
  await expect(page.getByLabel("Serving profile generated response")).toContainText("Serving profile response stays local.");

  await page.getByRole("button", { name: "Delete serving profile Runtime smoke" }).click();
  await expect(page.getByText("0 profile(s)", { exact: true })).toBeVisible();
});
