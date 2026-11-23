import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("MCP trust must be explicit and per-server ownership stays user-declared", async ({ page }) => {
  await ensureAuthenticated(page);
  let handshakeCalls = 0;
  let savedTrust: { trust_status: string; owner_type: string; owner_label: string | null } | null = null;

  await page.route("**/api/security/tool-permissions", (route) => route.fulfill({ json: {
    active_profile: "default",
    profiles: [{ name: "default", allowed_tools: [] }],
    tools: [{ name: "axiom_info", description: "Read-only platform information" }],
  } }));
  await page.route("**/api/mcp/servers", (route) => route.fulfill({ json: {
    schema_version: 1,
    total: 1,
    servers: [{
      name: "Unverified MCP",
      transport: "stdio",
      configured: true,
      executable: "python",
      argument_count: 2,
      trust_status: savedTrust?.trust_status || "unverified",
      owner_type: savedTrust?.owner_type || "unknown",
      owner_label: savedTrust?.owner_label || null,
      ownership_verified: false,
      connection_state: "not_inspected",
      last_inspected_at: null,
      last_connection_error: null,
    }],
    inspection_note: "Inventory is passive.",
  } }));
  await page.route("**/api/mcp/servers/Unverified%20MCP/trust", async (route) => {
    expect(route.request().method()).toBe("PATCH");
    savedTrust = route.request().postDataJSON();
    await route.fulfill({ json: {
      name: "Unverified MCP",
      ...savedTrust,
      ownership_verified: false,
    } });
  });
  await page.route("**/api/mcp/servers/Unverified%20MCP/handshake", async (route) => {
    handshakeCalls += 1;
    await route.fulfill({ json: {
      name: "Unverified MCP",
      transport: "stdio",
      status: "ok",
      protocol_version: "2025-11-25",
      server_name: "Test MCP",
      server_version: "1.0",
      capabilities: { tools: true, resources: false, prompts: false, logging: false, completions: false },
      duration_ms: 5,
      connection_state_persisted: true,
      connection_state: "connected",
      last_inspected_at: "2026-11-20T12:00:00Z",
      last_connection_error: null,
      tool_execution_performed: false,
      schemas_enumerated: false,
    } });
  });

  await page.goto("/?page=mcp");
  await expect(page.getByText("UNVERIFIED", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Inspect handshake" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Browse schemas" })).toBeDisabled();
  await expect(page.getByText(/Ownership is user-declared and not verified/)).toBeVisible();
  expect(handshakeCalls).toBe(0);

  await page.getByLabel("Trust status for Unverified MCP").selectOption("trusted");
  await page.getByLabel("Owner type for Unverified MCP").selectOption("workspace");
  await page.getByLabel("Owner label for Unverified MCP").fill("Research Team");
  await page.getByRole("button", { name: "Save trust" }).click();

  await expect(page.getByText("TRUSTED", { exact: true })).toBeVisible();
  expect(savedTrust).toEqual({
    trust_status: "trusted",
    owner_type: "workspace",
    owner_label: "Research Team",
  });
  await expect(page.getByRole("button", { name: "Inspect handshake" })).toBeEnabled();

  await page.getByRole("button", { name: "Inspect handshake" }).click();
  await expect(page.getByText("HANDSHAKE OK")).toBeVisible();
  await expect(page.getByText(/Last check: connected/)).toBeVisible();
  expect(handshakeCalls).toBe(1);

  await page.getByLabel("Trust status for Unverified MCP").selectOption("blocked");
  await page.getByRole("button", { name: "Save trust" }).click();
  await expect(page.getByText("BLOCKED", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Inspect handshake" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Browse schemas" })).toBeDisabled();
  expect(handshakeCalls).toBe(1);
});
