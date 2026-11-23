import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("MCP Control Center registers versioned manifests and re-reviews stale scoped grants", async ({ page }) => {
  await ensureAuthenticated(page);
  const manifest = (version: string) => ({
    manifest_schema_version: 1,
    id: "com.example.research",
    name: "Example research",
    version,
    description: "A declarative research plugin.",
    tools: [
      { name: "search", description: "Search approved local data.", scopes: ["read"] },
      { name: "annotate", description: "Write a review annotation.", scopes: ["read", "write"] },
    ],
  });
  let currentManifest: ReturnType<typeof manifest> | null = null;
  let manifestSha = "a".repeat(64);
  let permissionState: "default_deny" | "granted" | "stale_manifest" = "default_deny";
  let grant: Array<{ name: string; scopes: string[] }> = [];
  const calls: Array<{ method: string; body: Record<string, unknown> | null }> = [];

  const inventory = () => ({
    store_schema_version: 1,
    manifests: currentManifest ? [{
      manifest: currentManifest,
      sha256: manifestSha,
      permission_status: permissionState,
      granted_tools: permissionState === "granted" ? grant : [],
    }] : [],
    maximum_manifests: 50,
    maximum_tools_per_manifest: 32,
    execution_note: "Manifests are declarative metadata and permission records only. AXIOM does not load plugin code or execute manifest-defined commands.",
  });

  await page.route("**/api/security/tool-permissions", (route) => route.fulfill({
    json: {
      active_profile: "default",
      profiles: [{ name: "default", allowed_tools: [] }],
      tools: [{ name: "axiom_info", description: "Read AXIOM runtime metadata." }],
    },
  }));
  await page.route("**/api/mcp/servers", (route) => route.fulfill({
    json: {
      schema_version: 1,
      total: 0,
      servers: [],
      inspection_note: "Registered server information is read without starting external processes.",
    },
  }));
  await page.route("**/api/security/plugins**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/security/plugins" && request.method() === "GET") {
      await route.fulfill({ json: inventory() });
      return;
    }
    if (url.pathname === "/api/security/plugins/manifests" && request.method() === "POST") {
      const payload = request.postDataJSON() as { manifest: ReturnType<typeof manifest> };
      currentManifest = payload.manifest;
      manifestSha = currentManifest.version === "1.0.0" ? "a".repeat(64) : "b".repeat(64);
      if (grant.length) permissionState = "stale_manifest";
      else permissionState = "default_deny";
      calls.push({ method: "POST", body: payload });
      await route.fulfill({ status: 201, json: {
        manifest: currentManifest,
        sha256: manifestSha,
        permission_status: "default_deny",
      } });
      return;
    }
    if (url.pathname.endsWith("/permissions") && request.method() === "PUT") {
      const payload = request.postDataJSON() as { manifest_sha256: string; tools: typeof grant };
      calls.push({ method: "PUT", body: payload });
      if (payload.manifest_sha256 !== manifestSha) {
        await route.fulfill({ status: 422, json: { detail: "Manifest changed." } });
        return;
      }
      grant = payload.tools;
      permissionState = "granted";
      await route.fulfill({ json: {
        plugin_id: "com.example.research",
        manifest_sha256: manifestSha,
        tools: grant,
      } });
      return;
    }
    if (url.pathname.endsWith("/permissions") && request.method() === "DELETE") {
      grant = [];
      permissionState = "default_deny";
      calls.push({ method: "DELETE", body: null });
      await route.fulfill({ json: { plugin_id: "com.example.research", revoked: true } });
      return;
    }
    await route.fulfill({ status: 404, json: { detail: "Unexpected plugin API route." } });
  });

  await page.goto("/?page=mcp");
  await expect(page.getByRole("heading", { name: "Declarative plugin manifests" })).toBeVisible();
  await expect(page.getByText("No plugin manifests registered")).toBeVisible();

  await page.getByLabel("Plugin manifest JSON").fill(JSON.stringify(manifest("1.0.0"), null, 2));
  await page.getByRole("button", { name: "Validate and register" }).click();
  await expect(page.getByText("com.example.research · 1.0.0")).toBeVisible();
  await expect(page.getByText("DEFAULT DENY")).toBeVisible();
  await expect(page.getByText(/does not load plugin code/i)).toBeVisible();

  const searchTool = page.locator(".permission-row").filter({ hasText: "Search approved local data." });
  await searchTool.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Save scoped grants" }).click();
  await expect(page.getByText("GRANTED", { exact: true })).toBeVisible();
  expect(calls.filter((call) => call.method === "PUT")).toHaveLength(1);
  expect(calls.find((call) => call.method === "PUT")?.body).toEqual({
    manifest_sha256: "a".repeat(64),
    tools: [{ name: "search", scopes: ["read"] }],
  });

  await page.getByLabel("Plugin manifest JSON").fill(JSON.stringify(manifest("1.1.0"), null, 2));
  await page.getByRole("button", { name: "Validate and register" }).click();
  await expect(page.getByText("STALE MANIFEST", { exact: true })).toBeVisible();
  await expect(page.getByText(/previous grant.*denied/i)).toBeVisible();

  await page.locator(".permission-row").filter({ hasText: "Search approved local data." }).getByRole("checkbox").check();
  await page.getByRole("button", { name: "Save scoped grants" }).click();
  await expect(page.getByText("GRANTED", { exact: true })).toBeVisible();
  expect(calls.filter((call) => call.method === "PUT")).toHaveLength(2);
  expect(calls.filter((call) => call.method === "PUT")[1].body).toEqual({
    manifest_sha256: "b".repeat(64),
    tools: [{ name: "search", scopes: ["read"] }],
  });

  await page.getByRole("button", { name: "Revoke all grants" }).click();
  await expect(page.getByText("DEFAULT DENY", { exact: true })).toBeVisible();
  expect(calls.filter((call) => call.method === "DELETE")).toHaveLength(1);
});
