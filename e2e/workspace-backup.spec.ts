import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Diagnostics downloads and restores a full workspace into a new isolated workspace", async ({ page }) => {
  const defaultWorkspace = {
    id: "local-1111111111111111",
    name: "Original workspace",
    created_at: "2026-11-22T00:00:00Z",
    updated_at: "2026-11-22T00:00:00Z",
    is_default: true,
    status: "ready" as const,
  };
  const restoredWorkspace = {
    id: "ws-" + "2".repeat(32),
    name: "Restored workspace",
    created_at: "2026-11-22T00:01:00Z",
    updated_at: "2026-11-22T00:01:00Z",
    is_default: false,
    status: "ready" as const,
  };
  let restored = false;
  let restoreRequestName: string | null = null;
  await page.addInitScript(() => {
    const originalClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () {
      if (this.download === "axiom-workspace-backup.zip") {
        (window as Window & { __axiomBackupHref?: string }).__axiomBackupHref = this.href;
      }
      originalClick.call(this);
    };
  });
  await page.route((url) => new URL(url).pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/workspaces" && request.method() === "GET") {
      await route.fulfill({ json: {
        schema_version: 1,
        active_workspace_id: defaultWorkspace.id,
        default_workspace_id: defaultWorkspace.id,
        workspaces: restored ? [defaultWorkspace, restoredWorkspace] : [defaultWorkspace],
        scope_note: "Workspace filesystem paths remain local and are not returned by this API.",
      } });
    } else if (url.pathname === "/api/workspaces/backup/restore" && request.method() === "POST") {
      restoreRequestName = url.searchParams.get("name");
      const body = request.postDataBuffer();
      expect(body?.length).toBeGreaterThan(0);
      restored = true;
      await route.fulfill({ status: 201, json: {
        schema_version: 1,
        workspace: restoredWorkspace,
        backup: {
          schema_version: 1,
          backup_id: "a".repeat(32),
          workspace_label: "Original workspace",
          file_count: 3,
          total_bytes: 120,
          archive_bytes: body?.length ?? 0,
          manifest_sha256: "b".repeat(64),
          skipped: { symlink_files: 0, symlink_directories: 0, special_files: 0, excluded_directories: 0 },
          integrity: "verified",
          scope_note: "Checksums verified.",
        },
        scope_note: "The backup was restored into a new managed workspace. The active workspace was not modified.",
      } });
    } else if (url.pathname === "/api/health") {
      await route.fulfill({ json: { status: "ok", version: "0.3.0-test" } });
    } else if (url.pathname === "/api/workspace/runtime") {
      await route.fulfill({ json: { status: "ready", hardware: {
        os_name: "Linux", architecture: "x86_64", cpu_cores: 8, ram_gb: 16,
        gpu_name: null, vram_gb: null,
      } } });
    } else if (url.pathname === "/api/workspace/models") {
      await route.fulfill({ json: { models: [], total: 0, offset: 0, limit: 100, has_more: false } });
    } else if (url.pathname === "/api/workspace/datasets") {
      await route.fulfill({ json: { datasets: [] } });
    } else if (url.pathname === "/api/diagnostics/workspace") {
      await route.fulfill({ json: {
        schema_version: 1, generated_at: "2026-11-22T00:00:00Z", status: "pass",
        workspace: { name: "Original workspace", project_config_present: true, is_project: true },
        summary: { passed: 1, warnings: 0, failed: 0, informational: 0, missing_dependencies: [], outdated_dependencies: [] },
        checks: [],
      } });
    } else if (url.pathname === "/api/security/audit") {
      await route.fulfill({ json: {
        schema_version: 1, generated_at: "2026-11-22T00:00:00Z", status: "pass",
        workspace: { name: "Original workspace" },
        summary: { passed: 1, warnings: 0, failed: 0, informational: 0, api_scope_tokens_enabled: true },
        checks: [],
      } });
    } else if (url.pathname === "/api/security/events") {
      await route.fulfill({ json: {
        schema_version: 1, events: [], total: 0, limit: 50, offset: 0, has_more: false,
        integrity: { schema_version: 1, valid: true, event_count: 0, max_events: 2000, head_hash: "0".repeat(64), capacity_reached: false },
        privacy: { event_summaries_allowlisted: true, credential_values_stored: false, arbitrary_user_text_stored: false, reviewer_identity_authenticated: false, journal_is_append_only: true },
      } });
    } else {
      await route.fulfill({ status: 404, json: { detail: "No fixture for workspace backup acceptance." } });
    }
  });

  // The backup is an anchor navigation/download rather than a fetch() call.
  // Install a dedicated route after the API fixture to verify the exact selected workspace.
  await page.route("**/api/workspaces/*/backup", async (route) => {
    const url = new URL(route.request().url());
    expect(url.pathname).toBe(`/api/workspaces/${defaultWorkspace.id}/backup`);
    await route.fulfill({
      status: 200,
      contentType: "application/zip",
      headers: {
        "Content-Disposition": 'attachment; filename="axiom-workspace-backup.zip"',
        "Cache-Control": "no-store",
      },
      body: Buffer.from("AXIOM test ZIP fixture"),
    });
  });

  await ensureAuthenticated(page);
  await page.evaluate((workspaceId) => localStorage.setItem("axiom-active-workspace", workspaceId), defaultWorkspace.id);
  await page.goto("/?page=diagnostics");
  await expect(page.getByRole("heading", { name: "Full workspace backup and restore" })).toBeVisible();

  page.once("dialog", (dialog) => dialog.accept());
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download full workspace backup" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("axiom-workspace-backup.zip");
  const backupHref = await page.evaluate(() =>
    (window as Window & { __axiomBackupHref?: string }).__axiomBackupHref ?? null
  );
  expect(backupHref).not.toBeNull();
  expect(new URL(backupHref!).pathname).toBe(`/api/workspaces/${defaultWorkspace.id}/backup`);

  await page.getByLabel("Backup archive (.zip)").setInputFiles({
    name: "project-backup.zip",
    mimeType: "application/zip",
    buffer: Buffer.from("AXIOM uploaded test ZIP"),
  });
  await page.getByLabel("Name for restored workspace").fill("Restored workspace");
  page.once("dialog", (dialog) => {
    expect(dialog.message()).toContain("current workspace will not be overwritten");
    dialog.accept();
  });
  await page.getByRole("button", { name: "Verify and restore as new workspace" }).click();

  expect(restoreRequestName).toBe("Restored workspace");
  await expect(page.locator(".notice-toast")).toContainText(
    'Restored 3 file(s) into new workspace "Restored workspace". Your current workspace remains active.',
  );
  expect(await page.evaluate(() => localStorage.getItem("axiom-active-workspace"))).toBe(defaultWorkspace.id);

  await page.getByRole("button", { name: /Switch workspace. Current workspace: Original workspace/ }).click();
  await expect(page.getByRole("menuitemradio", { name: /Restored workspace/ })).toBeVisible();
});
