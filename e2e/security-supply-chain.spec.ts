import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Diagnostics exports a local SPDX SBOM and gates online vulnerability scans behind explicit confirmation", async ({ page }) => {
  const sbom = {
    spdxVersion: "SPDX-2.3",
    dataLicense: "CC0-1.0",
    SPDXID: "SPDXRef-DOCUMENT",
    name: "AXIOM installed Python environment",
    documentNamespace: "https://spdx.org/spdxdocs/axiom-test",
    creationInfo: {
      creators: ["Tool: AXIOM-AI"],
      created: "2026-11-21T00:00:00Z",
      comment: "metadata only",
    },
    documentDescribes: ["SPDXRef-Package-test"],
    packages: [{
      name: "fastapi",
      SPDXID: "SPDXRef-Package-test",
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
      relatedSpdxElement: "SPDXRef-Package-test",
    }],
    annotations: [],
  };
  const vulnerabilityReport = {
    schema_version: 1,
    generated_at: "2026-11-21T00:01:00Z",
    scanner: { name: "pip-audit", version: "2.9.0" },
    status: "complete",
    network_accessed: true,
    dependency_count: 1,
    vulnerable_dependency_count: 1,
    vulnerability_count: 1,
    findings: [{
      id: "PYSEC-2026-1",
      package: "fastapi",
      installed_version: "0.90.0",
      aliases: ["CVE-2026-0001"],
      fix_versions: ["0.115.0"],
    }],
    privacy: {
      credentials_included: false,
      environment_values_included: false,
      package_file_contents_included: false,
      network_identifiers_included: false,
    },
    limitations: ["Test advisory result."],
  };
  let auditCalls = 0;
  let auditRequest: Record<string, unknown> | null = null;

  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/health") {
      await route.fulfill({ json: { status: "ok", version: "test" } });
    } else if (url.pathname === "/api/workspace/runtime") {
      await route.fulfill({ json: { status: "ready", hardware: {
        cpu_name: "Test CPU",
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
        summary: {
          passed: 1, warnings: 0, failed: 0, informational: 0,
          missing_dependencies: [], outdated_dependencies: [],
        },
        checks: [],
      } });
    } else if (url.pathname === "/api/security/audit") {
      await route.fulfill({ json: {
        schema_version: 1,
        generated_at: "2026-11-21T00:00:00Z",
        status: "pass",
        workspace: { name: "test-workspace" },
        summary: {
          passed: 1, warnings: 0, failed: 0, informational: 0,
          api_scope_tokens_enabled: true,
        },
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
          tool_decisions_audited: true,
          argument_values_stored: false,
          credential_values_stored: false,
          arbitrary_user_text_stored: false,
          reviewer_identity_authenticated: false,
          journal_is_append_only: true,
        },
      } });
    } else if (url.pathname === "/api/security/sbom") {
      await route.fulfill({ json: sbom });
    } else if (url.pathname === "/api/security/dependency-audit" && request.method() === "POST") {
      auditCalls += 1;
      auditRequest = request.postDataJSON() as Record<string, unknown>;
      await route.fulfill({ json: vulnerabilityReport });
    } else {
      await route.fulfill({ status: 404, json: { detail: "No fixture for this supply-chain test." } });
    }
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=diagnostics");
  const supplyChain = page.locator(".surface").filter({
    has: page.getByRole("heading", { name: "Software supply chain" }),
  });
  await expect(page.getByRole("heading", { name: "Software supply chain" })).toBeVisible();
  expect(auditCalls).toBe(0);

  const sbomDownload = page.waitForEvent("download");
  await supplyChain.getByRole("button", { name: "Export SPDX 2.3 SBOM" }).click();
  const sbomFile = await sbomDownload;
  expect(sbomFile.suggestedFilename()).toBe("axiom-installed-environment.spdx.json");
  const sbomPath = await sbomFile.path();
  expect(sbomPath).not.toBeNull();
  const exportedSbom = JSON.parse(await readFile(sbomPath as string, "utf8"));
  expect(exportedSbom.spdxVersion).toBe("SPDX-2.3");
  expect(exportedSbom.packages[0].externalRefs[0].referenceLocator).toBe("pkg:pypi/fastapi@0.115.0");
  expect(exportedSbom.axiom.network_access_performed).toBe(false);

  const rejectedDialogPromise = page.waitForEvent("dialog");
  page.once("dialog", (dialog) => void dialog.dismiss());
  await supplyChain.getByRole("button", { name: "Scan dependencies (network)" }).click();
  const rejectedDialog = await rejectedDialogPromise;
  expect(rejectedDialog.type()).toBe("confirm");
  expect(rejectedDialog.message()).toContain("external package vulnerability advisory services");
  expect(auditCalls).toBe(0);

  const acceptedDialogPromise = page.waitForEvent("dialog");
  page.once("dialog", (dialog) => void dialog.accept());
  const auditDownload = page.waitForEvent("download");
  await supplyChain.getByRole("button", { name: "Scan dependencies (network)" }).click();
  const acceptedDialog = await acceptedDialogPromise;
  expect(acceptedDialog.type()).toBe("confirm");
  const auditFile = await auditDownload;
  expect(auditFile.suggestedFilename()).toBe("axiom-dependency-vulnerability-report.json");
  expect(auditRequest).toEqual({ allow_network: true, timeout_seconds: 90 });
  expect(auditCalls).toBe(1);
  const auditPath = await auditFile.path();
  expect(auditPath).not.toBeNull();
  const exportedAudit = JSON.parse(await readFile(auditPath as string, "utf8"));
  expect(exportedAudit.vulnerability_count).toBe(1);
  expect(exportedAudit.findings[0].id).toBe("PYSEC-2026-1");
  expect(JSON.stringify(exportedAudit)).not.toContain("Authorization");
});
