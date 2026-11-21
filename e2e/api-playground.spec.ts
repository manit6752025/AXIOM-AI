import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("API Playground exposes bounded read-only endpoints and Deep Analysis", async ({ page }) => {
  test.setTimeout(30_000);

  const requests: Array<{
    url: string;
    method: string;
    postData: string | null;
  }> = [];

  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    requests.push({
      url: url.pathname + url.search,
      method: request.method(),
      postData: request.postData(),
    });

    const path = url.pathname;

    if (path === "/api/contract/examples") {
      await route.fulfill({ json: {
        schema_version: 1,
        api_contract_version: 1,
        openapi_version: "3.1.0",
        application_version: "test",
        source_sha256: "a".repeat(64),
        operation_count: 2,
        total_operations: 2,
        truncated: false,
        privacy: {
          credential_values_included: false,
          environment_values_included: false,
          schema_defaults_replayed: false,
          requests_executed: false,
        },
        operations: [
          {
            operation_id: "get_api_meta",
            identifier: "GetApiMeta",
            method: "GET",
            path: "/api/meta",
            summary: "Contract metadata",
            description: "",
            tags: ["metadata"],
            side_effecting: false,
            parameters: [],
            request: null,
            response: {
              status_code: "200",
              description: "Successful response",
              content_type: "application/json",
              schema: { type: "object" },
              example: { api_contract_version: 1, application_version: "test" },
              redacted_fields: [],
              template_fields: [],
              typescript_type: "{ api_contract_version: number; }",
            },
            typescript: "export type GetApiMetaRequest = void;\\nexport type GetApiMetaResponse = { api_contract_version: number; };",
            curl_template: "curl -X GET http://127.0.0.1:40507/api/meta",
            scope_note: "Generated schema example only.",
          },
          {
            operation_id: "post_llm_chat",
            identifier: "PostLlmChat",
            method: "POST",
            path: "/api/llm/chat",
            summary: "Cloud or local chat",
            description: "",
            tags: ["llm-client"],
            side_effecting: true,
            parameters: [],
            request: {
              required: true,
              content_type: "application/json",
              schema: {
                type: "object",
                properties: {
                  api_key: { type: "string", format: "password", writeOnly: true, "x-axiom-redacted": true },
                  provider: { type: "string" },
                },
              },
              example: { provider: "deepseek" },
              redacted_fields: ["$.body.api_key"],
              template_fields: [],
              typescript_type: "{ api_key?: string; provider: string; }",
            },
            response: {
              status_code: "200",
              description: "Successful response",
              content_type: "application/json",
              schema: { type: "object" },
              example: { execution: "remote" },
              redacted_fields: [],
              template_fields: [],
              typescript_type: "{ execution: string; }",
            },
            typescript: "export type PostLlmChatRequest = { api_key?: string; provider: string; };",
            curl_template: "curl -X POST http://127.0.0.1:40507/api/llm/chat",
            scope_note: "Generated schema example only.",
          },
        ],
        scope_note: "Generated examples are reference only and are not executed.",
      } });
    } else if (path.endsWith("/health")) {
      await route.fulfill({ json: { status: "ok", version: "test" } });
    } else if (path.endsWith("/meta")) {
      await route.fulfill({
        json: { api_contract_version: 1, application_version: "test" },
      });
    } else if (path.endsWith("/runtime")) {
      await route.fulfill({ json: { status: "offline", hardware: { os_name: "Linux", architecture: "x86_64", cpu_cores: 4, ram_gb: 8, gpu_name: null, vram_gb: null, cuda_available: false } } });
    } else if (path.endsWith("/models")) {
      await route.fulfill({
        json: { models: [], total: 0, limit: 5, offset: 0, has_more: false },
      });
    } else if (path.endsWith("/datasets")) {
      await route.fulfill({ json: { datasets: [] } });
    } else if (path === "/api/security/audit") {
      await route.fulfill({ json: {
        schema_version: 1,
        generated_at: "2026-11-20T06:00:00Z",
        status: "pass",
        workspace: { name: "test-workspace" },
        summary: {
          passed: 0,
          warnings: 0,
          failed: 0,
          informational: 0,
          api_scope_tokens_enabled: true,
        },
        checks: [],
      } });
    } else if (path === "/api/security/events") {
      await route.fulfill({ json: {
        schema_version: 1,
        events: [],
        total: 0,
        limit: 50,
        offset: 0,
        has_more: false,
        integrity: {
          schema_version: 1,
          valid: true,
          event_count: 0,
          max_events: 2000,
          head_hash: "0".repeat(64),
          capacity_reached: false,
        },
        privacy: {
          event_summaries_allowlisted: true,
          credential_values_stored: false,
          arbitrary_user_text_stored: false,
          reviewer_identity_authenticated: false,
          journal_is_append_only: true,
        },
      } });
    } else if (path.endsWith("/events")) {
      await route.fulfill({ json: { events: [] } });
    } else if (path.endsWith("/diagnostics/workspace")) {
      await route.fulfill({
        json: {
          checks: [],
          summary: { passed: 0, warnings: 0, failed: 0 },
          status: "pass",
        },
      });
    } else if (path.endsWith("/diagnostics/roadmap")) {
      await route.fulfill({
        json: { integrity: { ok: true }, total: 120, counts: {}, items: [] },
      });
    } else if (path.endsWith("/diagnostics/runtime-connectors")) {
      await route.fulfill({
        json: {
          schema_version: 1,
          discovery_mode: "passive_local_discovery",
          checked_at: new Date().toISOString(),
          connectors: [],
          scope_note: "Test fixture; no runtime was launched.",
        },
      });
    } else {
      await route.fulfill({ status: 404, json: { detail: "Not found" } });
    }
  });

  await ensureAuthenticated(page);



  // ensureAuthenticated() already opened the authenticated Control Center.
  // Do not reload the page here; navigate using the existing session.
  const navigation = page.getByLabel("Primary navigation");
  await expect(navigation).toBeVisible({ timeout: 10_000 });
  await navigation.getByText("API Playground", { exact: true }).click();

  await expect(
    page.getByRole("heading", { name: /API Playground/i }),
  ).toBeVisible({ timeout: 10_000 });
  await expect(page.getByLabel("Endpoint")).toBeVisible({ timeout: 10_000 });

  const generated = page.locator(".api-contract-examples");
  await expect(generated.getByRole("heading", { name: "Typed API examples" })).toBeVisible();
  const generatedSelect = generated.getByLabel("Generated API operation");
  await generatedSelect.selectOption("post_llm_chat");
  await expect(generated.locator(".api-playground-json").first()).toContainText("deepseek");
  await expect(generated.locator(".api-playground-json").nth(2)).toContainText("PostLlmChatRequest");
  await expect(generated).toContainText("$.body.api_key");
  await expect(generated).toContainText("never submitted");

  const select = page.getByLabel("Endpoint");
  const options = await select
    .locator("option")
    .evaluateAll((nodes) =>
      nodes.map((node) => (node as HTMLOptionElement).value),
    );

  expect(options.length).toBeGreaterThanOrEqual(8);
  expect(options.every((option) => option.startsWith("/api/"))).toBe(true);
  expect(
    options.some((option) => option.includes("/api/jobs") || option.includes("/delete")),
  ).toBe(false);

  await select.selectOption("/api/meta");
  await page.getByRole("button", { name: "Run read-only request" }).click();
  const readOnlyResponse = page
    .locator("section.surface")
    .filter({ has: page.getByRole("button", { name: "Run read-only request" }) })
    .locator(".api-playground-json");
  await expect(readOnlyResponse).toHaveCount(1);
  await expect(readOnlyResponse).toContainText('"api_contract_version": 1');

  const beforeDeepAnalysis = requests.length;
  await page.getByRole("button", { name: /Run Deep Analysis/i }).click();

  await expect(
    page.getByText(`${options.length}/${options.length} passed`, { exact: true }),
  ).toBeVisible({ timeout: 10_000 });

  const sweepRequests = requests.slice(beforeDeepAnalysis);
  const sweepPaths = new Set(
    sweepRequests.map((request) =>
      new URL(request.url, "http://axiom.test").pathname,
    ),
  );

  for (const option of options) {
    const expectedPath = new URL(option, "http://axiom.test").pathname;
    expect(sweepPaths.has(expectedPath), `Deep Analysis did not request ${expectedPath}`).toBe(true);
  }

  expect(
    requests.every((request) => request.method === "GET" && request.postData === null),
  ).toBe(true);

  // Confirm the authenticated session survives a reload on the same origin.
  console.log("[AXIOM E2E] storage BEFORE reload", await page.evaluate(() => ({
    flags: Object.fromEntries(
      Object.keys(localStorage)
        .filter((key) => /remember|session|setup|booted/i.test(key) && !/admin/i.test(key))
        .map((key) => [key, localStorage.getItem(key)]),
    ),
    testAdminPresent: Boolean(localStorage.getItem("axiom-admin-v2")),
    tabSessionFlags: Object.fromEntries(
      Object.keys(sessionStorage)
        .filter((key) => /remember|session/i.test(key))
        .map((key) => [key, sessionStorage.getItem(key)]),
    ),
  })));
  await page.reload();

  await page.waitForTimeout(300);
  console.log("[AXIOM E2E] state AFTER reload", await page.evaluate(() => ({
    url: location.href,
    bodyText: document.body?.innerText?.slice(0, 500) ?? "",
    flags: Object.fromEntries(
      Object.keys(localStorage)
        .filter((key) => /remember|session|setup|booted/i.test(key) && !/admin/i.test(key))
        .map((key) => [key, localStorage.getItem(key)]),
    ),
    testAdminPresent: Boolean(localStorage.getItem("axiom-admin-v2")),
    tabSessionFlags: Object.fromEntries(
      Object.keys(sessionStorage)
        .filter((key) => /remember|session/i.test(key))
        .map((key) => [key, sessionStorage.getItem(key)]),
    ),
  })));
  await expect(page.getByLabel("Primary navigation")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByLabel("Endpoint")).toBeVisible();

  // Verify hardware diagnostics through the real runtime snapshot.
  await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Diagnostics", exact: true })).toBeVisible();
  const hardwareRow = page.locator(".diagnostic-row").filter({ hasText: "Hardware detection" });
  await expect(hardwareRow).toContainText("PASS", { timeout: 10_000 });
});
