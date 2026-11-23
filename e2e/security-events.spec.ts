import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Diagnostics lists, verifies, reviews, and exports the redacted security event journal", async ({ page }) => {
  const originalEvent = {
    seq: 1,
    event_id: "a".repeat(32),
    occurred_at: "2026-11-20T06:00:00+00:00",
    event_type: "mcp.profile.created",
    summary: "Created a default-deny MCP profile",
    prev_hash: "0".repeat(64),
    hash: "1".repeat(64),
  };
  let reviewed = false;
  let reviewerLabel: string | null = null;

  const integrity = () => ({
    schema_version: 1,
    valid: true,
    event_count: reviewed ? 2 : 1,
    max_events: 2000,
    head_hash: reviewed ? "2".repeat(64) : originalEvent.hash,
    capacity_reached: false,
  });
  const pageReport = () => ({
    schema_version: 1,
    events: [{
      ...originalEvent,
      reviewed,
      reviewed_at: reviewed ? "2026-11-20T06:01:00+00:00" : null,
      reviewer_label: reviewerLabel,
      reviewer_identity_verified: reviewed ? false : null,
    }],
    total: 1,
    limit: 50,
    offset: 0,
    has_more: false,
    integrity: integrity(),
    privacy: {
      event_summaries_allowlisted: true,
      credential_values_stored: false,
      arbitrary_user_text_stored: false,
      reviewer_identity_authenticated: false,
      journal_is_append_only: true,
    },
  });

  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/health") {
      await route.fulfill({ json: { status: "ok", version: "test" } });
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
        schema_version: 1, generated_at: "2026-11-20T06:00:00Z", status: "pass",
        workspace: { name: "test-workspace", project_config_present: true, is_project: true },
        summary: { passed: 1, warnings: 0, failed: 0, informational: 0, missing_dependencies: [], outdated_dependencies: [] },
        checks: [],
      } });
    } else if (url.pathname === "/api/security/audit") {
      await route.fulfill({ json: {
        schema_version: 1, generated_at: "2026-11-20T06:00:00Z", status: "pass",
        workspace: { name: "test-workspace" },
        summary: { passed: 1, warnings: 0, failed: 0, informational: 0, api_scope_tokens_enabled: true },
        checks: [],
      } });
    } else if (url.pathname === "/api/security/events" && request.method() === "GET") {
      await route.fulfill({ json: pageReport() });
    } else if (url.pathname === "/api/security/events/verify") {
      await route.fulfill({ json: integrity() });
    } else if (url.pathname === "/api/security/events/export") {
      await route.fulfill({ json: {
        schema_version: 1,
        exported_at: "2026-11-20T06:02:00+00:00",
        integrity: integrity(),
        events: reviewed
          ? [
              originalEvent,
              {
                seq: 2,
                event_id: "b".repeat(32),
                occurred_at: "2026-11-20T06:01:00+00:00",
                event_type: "security.event.reviewed",
                summary: "Recorded an administrator review acknowledgement",
                prev_hash: originalEvent.hash,
                hash: "2".repeat(64),
                target_event_id: originalEvent.event_id,
                reviewer_label: reviewerLabel,
              },
            ]
          : [originalEvent],
        privacy: {
          credential_values_stored: false,
          arbitrary_user_text_stored: false,
          reviewer_identity_authenticated: false,
        },
      } });
    } else if (url.pathname.endsWith("/review") && request.method() === "POST") {
      const payload = request.postDataJSON() as { reviewer_label: string };
      reviewerLabel = payload.reviewer_label;
      reviewed = true;
      await route.fulfill({ json: {
        review: {
          seq: 2, event_id: "b".repeat(32), occurred_at: "2026-11-20T06:01:00+00:00",
          event_type: "security.event.reviewed",
          summary: "Recorded an administrator review acknowledgement",
          prev_hash: originalEvent.hash, hash: "2".repeat(64),
          target_event_id: originalEvent.event_id, reviewer_label: reviewerLabel,
        },
        already_reviewed: false,
        reviewer_identity_verified: false,
      } });
    } else {
      await route.fulfill({ status: 404, json: { detail: "No fixture for this security-events test." } });
    }
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=diagnostics");
  const journal = page.getByRole("region", { name: /Security event journal/i });
  await expect(page.getByRole("heading", { name: "Security event journal" })).toBeVisible();
  await expect(journal).toContainText("INTEGRITY VALID");
  await expect(journal).toContainText("Created a default-deny MCP profile");
  await expect(journal).toContainText("self-attested");

  await page.getByLabel("Reviewer label").fill("Test Admin");
  await journal.getByRole("button", { name: "Mark reviewed" }).click();
  await expect(journal).toContainText("Reviewed by Test Admin · self-attested");

  await journal.getByRole("button", { name: "Verify chain" }).click();
  await expect(page.getByText(/Security event chain verified/)).toBeVisible();

  const downloadPromise = page.waitForEvent("download");
  await journal.getByRole("button", { name: "Export redacted JSON" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("axiom-security-events.json");
  const path = await download.path();
  expect(path).not.toBeNull();
  const exported = await readFile(path as string, "utf8");
  expect(exported).toContain("security.event.reviewed");
  expect(exported).toContain("Test Admin");
  expect(exported).toContain('"reviewer_identity_authenticated": false');
  expect(exported).not.toContain("Authorization");
  expect(exported).not.toContain("password");
});
