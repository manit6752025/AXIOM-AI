import assert from "node:assert/strict";
import test from "node:test";

import {
  AxiomAPIError,
  AxiomClient,
  AxiomConfigurationError,
  AxiomNetworkError,
} from "../dist/index.js";

function jsonResponse(value, status = 200, headers = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

test("sends bearer, workspace and request ID headers on every request", async () => {
  const requests = [];
  const fetchImpl = async (input, init = {}) => {
    requests.push({ url: new URL(input.toString()), init, headers: new Headers(init.headers) });
    return jsonResponse({ status: "ok", version: "test" }, 200, { "x-request-id": "server-id" });
  };
  const client = new AxiomClient({
    baseUrl: "http://127.0.0.1:40507",
    token: "read-token",
    workspaceId: "ws-123",
    headers: { "X-SDK-Test": "typescript" },
    fetchImpl,
  });

  const health = await client.health();
  assert.equal(health.status, "ok");
  assert.equal(requests.length, 1);
  assert.equal(requests[0].headers.get("authorization"), "Bearer read-token");
  assert.equal(requests[0].headers.get("x-axiom-workspace-id"), "ws-123");
  assert.equal(requests[0].headers.get("x-sdk-test"), "typescript");
  assert.match(requests[0].headers.get("x-request-id"), /^[a-f0-9]{32}$/);
  assert.equal(requests[0].init.redirect, "error");
  client.close();
});

test("serializes query parameters and JSON request bodies", async () => {
  const requests = [];
  const client = new AxiomClient({
    fetchImpl: async (input, init = {}) => {
      requests.push({ url: new URL(input.toString()), init, headers: new Headers(init.headers) });
      return jsonResponse({ workspace: { id: "ws-new", name: "Restored" } }, 201);
    },
  });

  await client.models({ limit: 15, offset: 2, search: "qwen 3" });
  const modelQuery = requests[0].url.searchParams;
  assert.equal(modelQuery.get("limit"), "15");
  assert.equal(modelQuery.get("offset"), "2");
  assert.equal(modelQuery.get("search"), "qwen 3");

  await client.createWorkspace("Research workspace");
  assert.equal(requests[1].headers.get("content-type"), "application/json");
  assert.deepEqual(JSON.parse(requests[1].init.body), { name: "Research workspace" });
  client.close();
});

test("converts AXIOM error envelopes into typed errors", async () => {
  const client = new AxiomClient({
    fetchImpl: async () => jsonResponse({
      request_id: "from-body",
      error: {
        code: "workspace_not_found",
        message: "Workspace not found.",
        details: { workspace_id: "missing" },
      },
    }, 404, { "x-request-id": "from-header" }),
  });

  await assert.rejects(
    () => client.health(),
    (error) => {
      assert.ok(error instanceof AxiomAPIError);
      assert.equal(error.statusCode, 404);
      assert.equal(error.code, "workspace_not_found");
      assert.equal(error.requestId, "from-header");
      assert.deepEqual(error.details, { workspace_id: "missing" });
      return true;
    },
  );
  client.close();
});

test("restores ZIP data and downloads workspace backups as bytes", async () => {
  const requests = [];
  const archive = new Blob(["PK\x03\x04archive"], { type: "application/zip" });
  const client = new AxiomClient({
    fetchImpl: async (input, init = {}) => {
      const url = new URL(input.toString());
      requests.push({ url, init, headers: new Headers(init.headers) });
      if (url.pathname.endsWith("/backup")) return new Response(new Uint8Array([80, 75, 3, 4]));
      return jsonResponse({
        schema_version: 1,
        workspace: { id: "ws-restored", name: url.searchParams.get("name") },
        backup: { integrity: "verified" },
        scope_note: "new workspace",
      }, 201);
    },
  });

  const downloaded = await client.downloadWorkspaceBackup("ws-source");
  assert.deepEqual([...downloaded], [80, 75, 3, 4]);
  const restored = await client.restoreWorkspaceBackup(archive, "Restored workspace");
  assert.equal(restored.workspace.id, "ws-restored");
  assert.equal(requests[1].url.searchParams.get("name"), "Restored workspace");
  assert.equal(requests[1].headers.get("content-type"), "application/zip");
  assert.ok(requests[1].init.body instanceof Blob);
  assert.equal(await requests[1].init.body.text(), await archive.text());
  assert.equal(requests[1].init.signal instanceof AbortSignal, true);
  client.close();
});

test("rejects insecure remote origins, invalid paths and protected custom headers", () => {
  assert.throws(() => new AxiomClient({ baseUrl: "http://api.example.com" }), AxiomConfigurationError);
  assert.throws(() => new AxiomClient({ baseUrl: "https://user:pass@example.com" }), AxiomConfigurationError);
  assert.throws(() => new AxiomClient({ headers: { Authorization: "override" } }), AxiomConfigurationError);
  const client = new AxiomClient();
  assert.throws(() => client.request("GET", "https://example.invalid/api/health"), AxiomConfigurationError);
  assert.throws(() => client.request("GET", "/api/../openapi.json"), AxiomConfigurationError);
  client.close();
});

test("does not retry failures and converts aborted requests into network errors", async () => {
  let calls = 0;
  const client = new AxiomClient({
    timeoutMs: 10,
    fetchImpl: async (_input, init = {}) => {
      calls += 1;
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        }, { once: true });
      });
    },
  });

  await assert.rejects(() => client.health(), AxiomNetworkError);
  assert.equal(calls, 1);
  client.close();
});

test("refuses calls after close", async () => {
  const client = new AxiomClient();
  client.close();
  await assert.rejects(() => client.health(), AxiomConfigurationError);
});
