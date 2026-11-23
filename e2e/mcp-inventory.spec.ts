import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("MCP inventory stays passive until an explicit protocol handshake", async ({ page }) => {
  await ensureAuthenticated(page);
  let handshakeCalls = 0;

  await page.route("**/api/security/tool-permissions", (route) => route.fulfill({ json: {
    active_profile: "default",
    profiles: [{ name: "default", allowed_tools: [] }],
    tools: [{ name: "axiom_info", description: "Read-only platform information" }],
  } }));
  await page.route("**/api/mcp/servers", (route) => route.fulfill({ json: {
    schema_version: 1,
    total: 1,
    servers: [{
      name: "Local MCP",
      transport: "stdio",
      configured: true,
      executable: "python",
      argument_count: 2,
      connection_state: "not_inspected",
    }],
    inspection_note: "Inventory is passive.",
  } }));
  await page.route((url) => url.pathname.startsWith("/api/mcp/servers/") && url.pathname.endsWith("/handshake"), async (route) => {
    handshakeCalls += 1;
    expect(route.request().method()).toBe("POST");
    await route.fulfill({ json: {
      name: "Local MCP",
      transport: "stdio",
      status: "ok",
      protocol_version: "2025-11-25",
      server_name: "Local test server",
      server_version: "1.0",
      capabilities: { tools: true, resources: false, prompts: false, logging: false, completions: false },
      duration_ms: 15,
      tool_execution_performed: false,
      schemas_enumerated: false,
    } });
  });

  await page.goto("/?page=mcp");
  await expect(page.getByRole("heading", { name: "Registered MCP servers" })).toBeVisible();
  await expect(page.getByText("Local MCP", { exact: true })).toBeVisible();
  await expect(page.getByText(/Not inspected/)).toBeVisible();
  expect(handshakeCalls).toBe(0);

  await page.getByRole("button", { name: "Inspect handshake" }).click();
  await expect(page.getByText("HANDSHAKE OK")).toBeVisible();
  await expect(page.getByText(/protocol 2025-11-25/)).toBeVisible();
  await expect(page.getByText(/no tools executed · no schemas enumerated/)).toBeVisible();
  expect(handshakeCalls).toBe(1);
});
