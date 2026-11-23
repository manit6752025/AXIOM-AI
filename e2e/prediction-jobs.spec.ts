import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime starts, tracks, and cancels a bounded native prediction job", async ({ page }) => {
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
  const jobId = "d".repeat(32);
  const now = "2026-11-23T00:00:00Z";
  let job: Record<string, unknown> | null = null;
  let startPayload: Record<string, unknown> | null = null;
  let cancelRequests = 0;

  const jobScope = {
    prompts_returned: false,
    predictions_returned: false,
    logs_returned: false,
    absolute_paths_returned: false,
    output_file_published_only_after_completion: true,
    job_state_persisted: true,
    server_restart_clears_job_state: false,
    generation_resumed_after_restart: false,
  };
  const pageScope = {
    prompts_returned: false,
    predictions_returned: false,
    logs_returned: false,
    absolute_paths_returned: false,
    job_state_persisted: true,
  };

  await page.route("**/api/engine/status", (route) => route.fulfill({ json: {
    engine: "axiom-native-engine",
    ready: true,
    packages: { torch: true, safetensors: true },
    versions: { torch: "2.6.0", safetensors: "0.5.0" },
    devices: [{
      id: "cpu", name: "CPU", available: true, reason: "Available",
      memory_free_gib: 4, memory_total_gib: 8,
    }],
    recommended_device: "cpu",
    loaded: false,
    architecture: null,
    model_loading: "local_safetensors_and_gguf_if_available",
    remote_code: false,
    supported_model_types: ["llama", "mistral", "qwen2", "qwen3"],
    supported_weights: ["safetensors"],
    setup_note: "Local native runtime is ready.",
    scope_note: "Status only; no model weights were loaded.",
    checked_at: now,
  } }));

  await page.route("**/api/engine/models", (route) => route.fulfill({ json: {
    schema_version: 1,
    models: [model],
    adapters: [],
    truncated: false,
    scope_note: "Local model metadata discovery only.",
  } }));

  await page.route("**/api/engine/telemetry", (route) => route.fulfill({ json: {
    schema_version: 1,
    sampled_at: now,
    sample_type: "point_in_time",
    cpu: { logical_cores: 4, system_percent: 5, process_percent: 1 },
    memory: { total_bytes: 8 * 1024 ** 3, available_bytes: 4 * 1024 ** 3, used_bytes: 4 * 1024 ** 3, percent: 50 },
    process: { pid: 1, rss_bytes: 128 * 1024 ** 2, peak_rss_bytes: 128 * 1024 ** 2, threads: 2 },
    devices: [{ id: "cpu", name: "CPU", available: true, memory_free_gib: 4, memory_total_gib: 8 }],
    loaded_runtime: {
      loaded: false, model: null, architecture: null, device: null, memory_mode: null,
      precision: null, last_generation: null, device_allocator: null,
    },
    disks: [],
    privacy_scope: {
      prompt_content_included: false, generated_content_included: false, model_paths_included: false,
      environment_values_included: false, credentials_included: false, network_access_performed: false,
    },
    scope_note: "A point-in-time resource snapshot.",
  } }));

  await page.route("**/api/engine/benchmark-profiles**", (route) => route.fulfill({ json: {
    schema_version: 1, total: 0, profiles: [],
    privacy: { prompt_content_included: false, model_paths_included: false, credentials_included: false },
  } }));

  await page.route("**/api/engine/serving-profiles**", (route) => route.fulfill({ json: {
    schema_version: 1, profiles: [], total: 0, max_profiles: 25,
    credentials_persisted: false, network_used: false, weights_loaded: false,
  } }));

  await page.route("**/api/engine/telemetry-history**", (route) => route.fulfill({ json: {
    schema_version: 1,
    settings: { enabled: false, max_samples: 120, max_age_days: 30 },
    total: 0,
    samples: [],
    privacy: {
      process_ids_included: false, model_names_or_paths_included: false,
      prompt_or_generated_content_included: false, environment_values_included: false,
      credentials_included: false, network_identifiers_included: false, automatic_sampling: false,
    },
  } }));

  await page.route("**/api/prediction-jobs**", async (route) => {
    const request = route.request();
    if (request.method() === "GET") {
      await route.fulfill({ json: { schema_version: 1, jobs: job ? [job] : [], limit: 20, scope: pageScope } });
      return;
    }
    if (request.method() === "POST") {
      startPayload = request.postDataJSON() as Record<string, unknown>;
      job = {
        id: jobId,
        status: "running",
        created_at: now,
        started_at: now,
        finished_at: null,
        dataset_path: "data/evaluation.jsonl",
        output_path: "data/predictions.jsonl",
        artifact_ready: false,
        rows_total: 4,
        rows_completed: 1,
        prompts_truncated: 0,
        evaluation_run_ids: [],
        error_code: null,
        scope: jobScope,
      };
      await route.fulfill({ status: 202, json: {
        schema_version: 1,
        job,
        limits: {
          rows: 100, input_bytes: 48 * 1024 ** 2, input_line_bytes: 768 * 1024,
          prompt_characters: 8000, generated_tokens_per_row: 64, batch_rows: 4,
          prediction_characters_per_row: 16000, output_bytes: 64 * 1024 ** 2,
          output_line_bytes: 256 * 1024,
        },
        scope: {
          network_accessed: false, model_downloaded: false, prompts_returned: false,
          predictions_returned: false, logs_returned: false, job_state_persisted: true,
          output_published_only_after_completion: true,
        },
      } });
      return;
    }
    await route.continue();
  });

  await page.route("**/api/prediction-jobs/*/cancel", async (route) => {
    cancelRequests += 1;
    if (job) job = { ...job, status: "cancelling" };
    await route.fulfill({ json: {
      schema_version: 1,
      job,
      cancellation_accepted: true,
      note: "Cancellation is cooperative and is processed at the next safe generation boundary.",
    } });
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");

  const engine = page.getByRole("region", { name: "AXIOM local engine" });
  await expect(engine.getByRole("heading", { name: "AXIOM Local Engine" })).toBeVisible();
  await expect(engine.getByRole("region", { name: "Prediction jobs" })).toBeVisible();

  await engine.getByLabel("Select discovered native model").selectOption(model.path);
  await engine.getByRole("button", { name: "Start prediction job" }).click();

  await expect.poll(() => startPayload).not.toBeNull();
  expect(startPayload).toMatchObject({
    model_path: model.path,
    dataset_path: "data/evaluation.jsonl",
    output_path: "data/predictions.jsonl",
    prompt_field: "prompt",
    reference_field: "reference",
    prediction_field: "prediction",
    max_new_tokens: 64,
    device: "auto",
    precision: "auto",
  });

  const history = engine.getByRole("region", { name: "Saved prediction job history" });
  await expect(history.getByRole("article", { name: "Prediction job " + jobId })).toBeVisible();
  await expect(history.getByText("data/predictions.jsonl · running", { exact: true })).toBeVisible();
  await expect(history.getByLabel("Progress for prediction job " + jobId)).toHaveAttribute("value", "1");
  await expect(history.getByRole("button", { name: "Cancel prediction job " + jobId })).toBeVisible();

  await history.getByRole("button", { name: "Cancel prediction job " + jobId }).click();
  await expect.poll(() => cancelRequests).toBe(1);
  await expect(history.getByText("data/predictions.jsonl · cancelling", { exact: true })).toBeVisible();
  await expect(engine.getByRole("status").filter({ hasText: "Cancellation requested." })).toBeVisible();
  await expect(history.getByText("Output not published until completion", { exact: false })).toBeVisible();
});
