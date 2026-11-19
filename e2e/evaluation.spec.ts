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
  await page.route("**/api/workspace/experiments*", (route) =>
    route.fulfill({ json: { experiments: [], total: 0, limit: 20, offset: 0, has_more: false } }),
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
  await page.getByRole("button", { name: "Run offline evaluation" }).click();

  await expect.poll(() => submitted).not.toBeNull();
  expect(submitted).toMatchObject({
    path: "evaluation.jsonl",
    prediction_field: "prediction",
    reference_field: "reference",
    baseline: false,
  });
  await expect(page.getByText(/2 samples · exact 0.500/)).toBeVisible();
  await expect(page.getByText("Precomputed outputs only", { exact: true })).toBeVisible();
});
