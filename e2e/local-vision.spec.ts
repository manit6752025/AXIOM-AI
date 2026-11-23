import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime only sends an explicit, bounded local vision request after user submission", async ({ page }) => {
  const requests: Array<Record<string, unknown>> = [];
  await page.route("**/api/engine/vision", async (route) => {
    const payload = route.request().postDataJSON() as Record<string, unknown>;
    requests.push(payload);
    await route.fulfill({ json: {
      schema_version: 1,
      model: "llava:latest",
      response: "The image contains a labelled network diagram.",
      image: { format: "png", width: 800, height: 600, size_bytes: 12_345 },
      generation: { max_new_tokens: 96, eval_count: 42, total_duration_ms: 1200 },
      privacy: {
        local_loopback_only: true,
        network_used: false,
        image_returned: false,
        image_persisted_by_axiom: false,
        prompt_persisted_by_axiom: false,
        model_downloaded: false,
        model_weights_loaded_by_axiom: false,
      },
      scope_note: "Sent only to the local Ollama loopback service.",
    } });
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");
  await expect(page.getByRole("heading", { name: "Local image understanding", exact: true })).toBeVisible();
  expect(requests).toHaveLength(0);

  await page.getByLabel("Installed local vision model").fill("llava:latest");
  await page.getByLabel("Workspace image path").fill("data/images/topology.png");
  await page.getByLabel("Image question").fill("Describe the devices and links.");
  await page.getByLabel("Maximum response tokens").fill("96");
  await page.getByRole("button", { name: "Analyze image locally" }).click();

  await expect(page.getByText("LOCAL VISION COMPLETE", { exact: true })).toBeVisible();
  await expect(page.getByText("The image contains a labelled network diagram.", { exact: true })).toBeVisible();
  expect(requests).toEqual([{
    model: "llava:latest",
    image_path: "data/images/topology.png",
    prompt: "Describe the devices and links.",
    max_new_tokens: 96,
  }]);
});
