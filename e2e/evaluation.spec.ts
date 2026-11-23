import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

const evaluationReport = {
  metric_schema_version: 1,
  evaluation_type: "offline_text_metrics",
  path: "evaluation.jsonl",
  file_name: "evaluation.jsonl",
  dataset_manifest: {
    sha256: "a".repeat(64),
    partial_sha256: null,
    bytes_hashed: 128,
    file_size_bytes: 128,
    full_content_hashed: true,
  },
  fields: { prediction: "prediction", reference: "reference", split: null, selected_split: null },
  counts: {
    lines_scanned: 2,
    valid_samples: 2,
    invalid_json_rows: 0,
    non_object_rows: 0,
    overlong_rows: 0,
    missing_value_rows: 0,
    non_string_value_rows: 0,
    filtered_split_rows: 0,
  },
  metrics: {
    exact_match: 0.5,
    normalized_exact_match: 0.5,
    token_precision_micro: 0.75,
    token_recall_micro: 0.75,
    token_f1_micro: 0.75,
  },
    confidence_intervals: {
      confidence_level: 0.95,
      method: "Wilson score interval",
      exact_match: { low: 0.1, high: 0.9 },
      normalized_exact_match: { low: 0.1, high: 0.9 },
    },
  truncated: false,
  truncation_reasons: [],
  limits: { max_bytes: 67108864, max_rows: 100000, max_line_bytes: 1048576 },
  privacy: { raw_values_returned: false, raw_values_persisted: false },
};

test("evaluation form submits selected fields and displays persisted aggregate results", async ({ page }) => {
  await ensureAuthenticated(page);
  let runs: Array<Record<string, unknown>> = [];
  let submitted: Record<string, unknown> | null = null;

  await page.route("**/api/workspace/datasets*", (route) =>
    route.fulfill({ json: { datasets: [{ path: "evaluation.jsonl", size_bytes: 128 }] } }),
  );
  const savedExperiments = [
    { id: "11111111111111111111111111111111", name: "Baseline snapshot", parameter_billions: 7,
      requested_method: "qlora", plan: { method: "qlora", precision: "bf16" }, hardware: { gpu_name: "Test GPU" },
      created_at: "2026-10-19T00:00:00Z", tags: [] },
    { id: "22222222222222222222222222222222", name: "Candidate snapshot", parameter_billions: 7,
      requested_method: "lora", plan: { method: "lora", precision: "bf16" }, hardware: { gpu_name: "Test GPU" },
      created_at: "2026-10-19T00:00:00Z", tags: [] },
  ];
  await page.route("**/api/workspace/experiments*", (route) =>
    route.fulfill({ json: { experiments: savedExperiments, total: savedExperiments.length, limit: 20, offset: 0, has_more: false } }),
  );
  await page.route("**/api/evaluation/runs*", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: { runs, total: runs.length, limit: 20, offset: 0, has_more: false } });
      return;
    }
    submitted = route.request().postDataJSON() as Record<string, unknown>;
    const run = {
      schema_version: 1,
      id: "evaluation-1",
      name: "evaluation offline evaluation",
      created_at: "2026-10-19T00:00:00Z",
      model_label: null,
      baseline: false,
      comparison: null,
      status: "complete",
      execution_status: "precomputed_outputs_only",
      output_manifest: { format: "application/vnd.axiom.evaluation-report+json;version=1", sha256: "b".repeat(64), bytes: 500 },
      report: evaluationReport,
    };
    runs = [run];
    await route.fulfill({ status: 201, json: { run } });
  });

  await page.goto("/?page=evaluation");
  await expect(page.getByLabel("JSONL dataset")).toHaveValue("evaluation.jsonl");
  await page.getByLabel("Prediction field").fill("prediction");
  await page.getByLabel("Reference field").fill("reference");
  await page.getByLabel("Link to saved experiment").selectOption("11111111111111111111111111111111");
  await page.getByRole("button", { name: "Run offline evaluation" }).click();

  await expect.poll(() => submitted).not.toBeNull();
  expect(submitted).toMatchObject({
    path: "evaluation.jsonl",
    prediction_field: "prediction",
    reference_field: "reference",
    baseline: false,
    experiment_id: "11111111111111111111111111111111",
  });
  await expect(page.getByText(/2 samples · exact 0.500/)).toBeVisible();
  await expect(page.getByText("Precomputed outputs only", { exact: true })).toBeVisible();

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Export evaluation offline evaluation report as JSON" }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("axiom-evaluation-evaluation-1.json");
  const exported = JSON.parse(await readFile((await download.path()) as string, "utf8"));
  expect(exported).toMatchObject({
    id: "evaluation-1",
    report: { privacy: { raw_values_returned: false, raw_values_persisted: false } },
  });
  expect(JSON.stringify(exported)).not.toContain("PRIVATE_PREDICTION_TEXT");
});


test("evaluation compares repeated experiment results and reports sample-size limits", async ({ page }) => {
  await ensureAuthenticated(page);
  const experiments = [
    { id: "11111111111111111111111111111111", name: "Baseline snapshot", parameter_billions: 7,
      requested_method: "qlora", plan: { method: "qlora", precision: "bf16" }, hardware: { gpu_name: "Test GPU" },
      created_at: "2026-10-19T00:00:00Z", tags: [] },
    { id: "22222222222222222222222222222222", name: "Candidate snapshot", parameter_billions: 7,
      requested_method: "lora", plan: { method: "lora", precision: "bf16" }, hardware: { gpu_name: "Test GPU" },
      created_at: "2026-10-19T00:00:00Z", tags: [] },
  ];
  let comparisonPayload: Record<string, unknown> | null = null;
  await page.route("**/api/workspace/datasets*", (route) =>
    route.fulfill({ json: { datasets: [], total: 0, limit: 500, offset: 0, has_more: false } }),
  );
  await page.route("**/api/workspace/experiments*", (route) =>
    route.fulfill({ json: { experiments, total: 2, limit: 20, offset: 0, has_more: false } }),
  );
  await page.route("**/api/evaluation/runs*", (route) =>
    route.fulfill({ json: { runs: [], total: 0, limit: 20, offset: 0, has_more: false } }),
  );
  await page.route("**/api/evaluation/experiment-comparisons", async (route) => {
    comparisonPayload = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ json: { comparison: {
      schema_version: 1,
      comparison_type: "axiom-repeated-offline-evaluation",
      metric: "exact_match",
      corpus: { comparison_corpus_sha256: "a".repeat(64), reference_field: "reference",
        split_field: null, selected_split: null, sample_count: 3 },
      baseline: { experiment_id: experiments[0].id, experiment_name: experiments[0].name,
        plan_fingerprint: "b".repeat(64), replicate_count: 2, excluded_incomplete_count: 0,
        mean: 0.9, sample_standard_deviation: 0.05, minimum: 0.85, maximum: 0.95, run_ids: ["b1", "b2"] },
      candidate: { experiment_id: experiments[1].id, experiment_name: experiments[1].name,
        plan_fingerprint: "c".repeat(64), replicate_count: 2, excluded_incomplete_count: 0,
        mean: 0.8, sample_standard_deviation: 0.04, minimum: 0.76, maximum: 0.84, run_ids: ["c1", "c2"] },
      difference: { candidate_minus_baseline_mean: -0.1, direction: "regressed" },
      statistical_test: { method: "Welch two-sided unequal-variance t-test", status: "computed",
        minimum_replicates_per_group: 2, t_statistic: -2.4, degrees_of_freedom: 1.9,
        p_value: 0.048, degenerate_zero_variance: false },
      generated_at: "2026-11-21T00:00:00Z",
      scope_note: "P-values assume independent repeated evaluation runs.",
    } } });
  });

  await page.goto("/?page=evaluation");
  await expect(page.getByLabel("Baseline experiment").locator("option", { hasText: "Baseline snapshot" })).toHaveCount(1);
  await page.getByLabel("Baseline experiment").selectOption(experiments[0].id);
  await page.getByLabel("Candidate experiment").selectOption(experiments[1].id);
  await page.getByLabel("Comparison metric").selectOption("exact_match");
  await page.getByRole("button", { name: "Compare saved experiments" }).click();

  await expect.poll(() => comparisonPayload).not.toBeNull();
  expect(comparisonPayload).toMatchObject({
    baseline_experiment_id: experiments[0].id,
    candidate_experiment_id: experiments[1].id,
    metric: "exact_match",
  });
  await expect(page.getByText("REGRESSED", { exact: true })).toBeVisible();
  await expect(page.getByText(/Welch two-sided p = 0.048000/)).toBeVisible();
  await expect(page.getByText(/2 complete replicates/).first()).toBeVisible();
});

test("evaluation distinguishes undefined zero-variance Welch results from insufficient replicates", async ({ page }) => {
  await ensureAuthenticated(page);
  const experiments = [
    { id: "11111111111111111111111111111111", name: "Stable baseline", parameter_billions: 7,
      requested_method: "qlora", plan: { method: "qlora", precision: "bf16" }, hardware: { gpu_name: "Test GPU" },
      created_at: "2026-10-19T00:00:00Z", tags: [] },
    { id: "22222222222222222222222222222222", name: "Stable candidate", parameter_billions: 7,
      requested_method: "lora", plan: { method: "lora", precision: "bf16" }, hardware: { gpu_name: "Test GPU" },
      created_at: "2026-10-19T00:00:00Z", tags: [] },
  ];
  const reason = "Both groups have zero within-group variance; a finite t statistic and inferential p-value are undefined.";
  await page.route("**/api/workspace/datasets*", (route) =>
    route.fulfill({ json: { datasets: [], total: 0, limit: 500, offset: 0, has_more: false } }),
  );
  await page.route("**/api/workspace/experiments*", (route) =>
    route.fulfill({ json: { experiments, total: 2, limit: 20, offset: 0, has_more: false } }),
  );
  await page.route("**/api/evaluation/runs*", (route) =>
    route.fulfill({ json: { runs: [], total: 0, limit: 20, offset: 0, has_more: false } }),
  );
  await page.route("**/api/evaluation/experiment-comparisons", (route) =>
    route.fulfill({ json: { comparison: {
      schema_version: 1,
      comparison_type: "axiom-repeated-offline-evaluation",
      metric: "exact_match",
      corpus: { comparison_corpus_sha256: "a".repeat(64), reference_field: "reference",
        split_field: null, selected_split: null, sample_count: 4 },
      baseline: { experiment_id: experiments[0].id, experiment_name: experiments[0].name,
        plan_fingerprint: "b".repeat(64), replicate_count: 3, excluded_incomplete_count: 0,
        mean: 0.5, sample_standard_deviation: 0, minimum: 0.5, maximum: 0.5, run_ids: ["b1", "b2", "b3"] },
      candidate: { experiment_id: experiments[1].id, experiment_name: experiments[1].name,
        plan_fingerprint: "c".repeat(64), replicate_count: 3, excluded_incomplete_count: 0,
        mean: 0.75, sample_standard_deviation: 0, minimum: 0.75, maximum: 0.75, run_ids: ["c1", "c2", "c3"] },
      difference: { candidate_minus_baseline_mean: 0.25, direction: "improved" },
      statistical_test: { method: "Welch two-sided unequal-variance t-test", status: "degenerate_zero_variance",
        minimum_replicates_per_group: 2, t_statistic: null, degrees_of_freedom: null,
        p_value: null, degenerate_zero_variance: true, note: reason },
      generated_at: "2026-11-21T00:00:00Z",
      scope_note: "Welch's test assumes independent repeated evaluation runs.",
    } } }),
  );

  await page.goto("/?page=evaluation");
  await page.getByLabel("Baseline experiment").selectOption(experiments[0].id);
  await page.getByLabel("Candidate experiment").selectOption(experiments[1].id);
  await page.getByLabel("Comparison metric").selectOption("exact_match");
  await page.getByRole("button", { name: "Compare saved experiments" }).click();

  await expect(page.getByText("Welch test undefined: " + reason, { exact: true })).toBeVisible();
  await expect(page.getByText("Welch test unavailable: at least two complete replicates per experiment are required.", { exact: true })).toHaveCount(0);
});
