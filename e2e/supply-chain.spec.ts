import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Diagnostics exports SPDX and requires explicit consent for dependency advisory lookups", async ({ page }) => {
  const requests: Array<{ method: string; body: Record<string, unknown> | null }> = [];
  const sbom = {
    spdxVersion: "SPDX-2.3",
    dataLicense: "CC0-1.0",
    SPDXID: "SPDXRef-DOCUMENT",
    name: "AXIOM installed Python environment",
    documentNamespace: "https://spdx.org/spdxdocs/axiom-test",
    creationInfo: {
      creators: ["Tool: AXIOM-AI"],
      created: "2026-11-21T00:00:00Z",
      comment: "Metadata-only package inventory.",
    },
    documentDescribes: ["SPDXRef-Package-abc"],
    packages: [{
      name: "fastapi",
      SPDXID: "SPDXRef-Package-abc",
      versionInfo: "0.115.0",
      downloadLocation: "NOASSERTION",
      filesAnalyzed: false,
      licenseConcluded: "NOASSERTION",
      licenseDeclared: "NOASSERTION",
      copyrightText: "NOASSERTION",
      primaryPackagePurpose: "LIBRARY",
      externalRefs: [{
        referenceCategory: "PACKAGE-MANAGER",
        referenceType: "purl",
        referenceLocator: "pkg:pypi/fastapi@0.115.0",
      }],
    }],
    relationships: [{
      spdxElementId: "SPDXRef-DOCUMENT",
      relationshipType: "DESCRIBES",
      relatedSpdxElement: "SPDXRef-Package-abc",
    }],
    annotations: [{
      annotationDate: "2026-11-22T00:00:00Z",
      annotationType: "OTHER",
      annotator: "Tool: AXIOM-AI",
      comment: "AXIOM inventory: schema_version=1; package_count=1; relationship_count=1; content_sha256=" + "a".repeat(64) + "; network_access_performed=false; host_paths_collected=false.",
    }],
  };
  const vulnerabilityReport = {
    schema_version: 1,
    generated_at: "2026-11-21T00:00:00Z",
    scanner: { name: "pip-audit", version: "2.10.1" },
    status: "complete",
    scan_scope: "active Python environment; installed distributions",
    network_accessed: true,
    dependency_count: 12,
    vulnerable_dependency_count: 1,
    vulnerability_count: 1,
    findings: [{
      id: "PYSEC-2026-1",
      package: "example-package",
      installed_version: "1.0.0",
      aliases: ["CVE-2026-0001"],
      fix_versions: ["1.0.1"],
    }],
    exit_code: 1,
    privacy: {
      credentials_included: false,
      environment_values_included: false,
      package_file_contents_included: false,
      network_identifiers_included: false,
    },
    limitations: ["Only installed Python packages were scanned."],
  };
  const integrity = {
    schema_version: 1,
    valid: true,
    event_count: 0,
    max_events: 2000,
    head_hash: "0".repeat(64),
    capacity_reached: false,
  };

  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/health") {
      await route.fulfill({ json: { status: "ok", version: "0.3.0-test" } });
    } else if (url.pathname === "/api/workspace/runtime") {
      await route.fulfill({ json: { status: "ready", hardware: {
        os_name: "Linux",
        architecture: "x86_64",
        cpu_cores: 8,
        ram_gb: 16,
        gpu_name: null,
        vram_gb: null,
      } } });
    } else if (url.pathname === "/api/workspace/models") {
      await route.fulfill({ json: { models: [], total: 0, offset: 0, limit: 100, has_more: false } });
    } else if (url.pathname === "/api/workspace/datasets") {
      await route.fulfill({ json: { datasets: [] } });
    } else if (url.pathname === "/api/diagnostics/workspace") {
      await route.fulfill({ json: {
        schema_version: 1,
        generated_at: "2026-11-21T00:00:00Z",
        status: "pass",
        workspace: { name: "test-workspace", project_config_present: true, is_project: true },
        summary: { passed: 1, warnings: 0, failed: 0, informational: 0, missing_dependencies: [], outdated_dependencies: [] },
        checks: [],
      } });
    } else if (url.pathname === "/api/security/audit") {
      await route.fulfill({ json: {
        schema_version: 1,
        generated_at: "2026-11-21T00:00:00Z",
        status: "pass",
        workspace: { name: "test-workspace" },
        summary: { passed: 1, warnings: 0, failed: 0, informational: 0, api_scope_tokens_enabled: true },
        checks: [],
      } });
    } else if (url.pathname === "/api/security/events") {
      await route.fulfill({ json: {
        schema_version: 1,
        events: [],
        total: 0,
        limit: 50,
        offset: 0,
        has_more: false,
        integrity,
        privacy: {
          event_summaries_allowlisted: true,
          credential_values_stored: false,
          arbitrary_user_text_stored: false,
          reviewer_identity_authenticated: false,
          journal_is_append_only: true,
        },
      } });
    } else if (url.pathname === "/api/security/sbom" && request.method() === "GET") {
      await route.fulfill({ json: sbom });
    } else if (url.pathname === "/api/security/dependency-audit" && request.method() === "POST") {
      requests.push({
        method: request.method(),
        body: request.postDataJSON() as Record<string, unknown>,
      });
      await route.fulfill({ json: vulnerabilityReport });
    } else {
      await route.fulfill({ status: 404, json: { detail: "No fixture for this supply-chain test." } });
    }
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=diagnostics");
  await expect(page.getByRole("heading", { level: 1, name: "Diagnostics" })).toBeVisible();

  const sbomDownloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export SPDX SBOM" }).click();
  const sbomDownload = await sbomDownloadPromise;
  expect(sbomDownload.suggestedFilename()).toBe("axiom-installed-environment.spdx.json");
  const sbomPath = await sbomDownload.path();
  expect(sbomPath).not.toBeNull();
  const exportedSbom = JSON.parse(await readFile(sbomPath as string, "utf8")) as typeof sbom;
  expect(exportedSbom.spdxVersion).toBe("SPDX-2.3");
  expect(exportedSbom.packages[0].externalRefs[0].referenceLocator).toBe("pkg:pypi/fastapi@0.115.0");

  let consentMessage = "";
  page.once("dialog", async (dialog) => {
    consentMessage = dialog.message();
    await dialog.dismiss();
  });
  await page.getByRole("button", { name: "Scan dependencies (network)" }).first().click();
  expect(consentMessage).toContain("may contact external package vulnerability advisory services");
  expect(requests).toHaveLength(0);

  page.once("dialog", async (dialog) => {
    consentMessage = dialog.message();
    await dialog.accept();
  });
  const auditDownloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Scan dependencies (network)" }).click();
  const auditDownload = await auditDownloadPromise;
  expect(auditDownload.suggestedFilename()).toBe("axiom-dependency-vulnerability-report.json");
  const auditPath = await auditDownload.path();
  expect(auditPath).not.toBeNull();
  const exportedAudit = JSON.parse(await readFile(auditPath as string, "utf8")) as typeof vulnerabilityReport;
  expect(exportedAudit.status).toBe("complete");
  expect(exportedAudit.vulnerability_count).toBe(1);
  expect(exportedAudit.findings[0].aliases).toContain("CVE-2026-0001");
  expect(requests).toEqual([{
    method: "POST",
    body: { allow_network: true, timeout_seconds: 90 },
  }]);
});
