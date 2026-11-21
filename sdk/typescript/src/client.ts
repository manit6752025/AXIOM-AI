import { AxiomAPIError, AxiomConfigurationError, AxiomNetworkError, type AxiomErrorEnvelope } from "./errors.js";
import type {
  AxiomAPIContractMetadata,
  AxiomClientOptions,
  AxiomEngineStatus,
  AxiomEngineTelemetry,
  AxiomHealth,
  AxiomKnowledgeBase,
  AxiomPluginManifestRegistry,
  AxiomRequestOptions,
  AxiomTelemetryHistory,
  AxiomTelemetryHistorySettings,
  AxiomWorkspaceBackupRestoreResult,
  AxiomWorkspaceCatalog,
  AxiomWorkspaceSummary,
} from "./types.js";

export const API_CONTRACT_VERSION = 1 as const;
export const DEFAULT_BASE_URL = "http://127.0.0.1:40507";

type HTTPMethod = "GET" | "HEAD" | "POST" | "PUT" | "PATCH" | "DELETE";

function requestId(): string {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID().replaceAll("-", "");
  }
  return Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
}

function validateHeaders(headers: Record<string, string> | undefined): Record<string, string> {
  const result: Record<string, string> = {};
  const protectedHeaders = new Set([
    "authorization",
    "x-request-id",
    "x-axiom-workspace-id",
    "host",
    "content-length",
  ]);
  for (const [name, value] of Object.entries(headers ?? {})) {
    if (!name.trim() || typeof value !== "string") {
      throw new AxiomConfigurationError("Additional headers need non-empty names and string values.");
    }
    if (protectedHeaders.has(name.toLowerCase())) {
      throw new AxiomConfigurationError(`The ${name} header is managed by AxiomClient.`);
    }
    if (/[\r\n\0]/.test(name + value)) {
      throw new AxiomConfigurationError("Additional headers must not contain control characters.");
    }
    result[name] = value;
  }
  return result;
}

export class AxiomClient {
  readonly baseUrl: URL;
  readonly timeoutMs: number;
  private readonly token: string | undefined;
  private readonly workspaceId: string | undefined;
  private readonly additionalHeaders: Record<string, string>;
  private readonly fetchImpl: typeof fetch;
  private closed = false;

  constructor(options: AxiomClientOptions = {}) {
    const baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    let parsed: URL;
    try {
      parsed = new URL(baseUrl);
    } catch {
      throw new AxiomConfigurationError("baseUrl must be a valid HTTP(S) origin.");
    }
    if (
      !["http:", "https:"].includes(parsed.protocol)
      || !parsed.hostname
      || parsed.username
      || parsed.password
      || parsed.pathname !== "/"
      || parsed.search
      || parsed.hash
    ) {
      throw new AxiomConfigurationError("baseUrl must be an HTTP(S) origin without credentials, path, query or fragment.");
    }
    const local = ["localhost", "127.0.0.1", "[::1]", "::1"].includes(parsed.hostname.toLowerCase());
    if (parsed.protocol !== "https:" && !local) {
      throw new AxiomConfigurationError("Non-loopback AXIOM services must use HTTPS.");
    }
    if (options.token !== undefined && (!options.token.trim() || /[\r\n]/.test(options.token))) {
      throw new AxiomConfigurationError("Bearer token must be a non-empty single-line string.");
    }
    if (options.workspaceId !== undefined && (!options.workspaceId.trim() || options.workspaceId.length > 128)) {
      throw new AxiomConfigurationError("Workspace ID must contain 1 to 128 characters.");
    }
    const timeout = options.timeoutMs ?? 30_000;
    if (!Number.isSafeInteger(timeout) || timeout < 1 || timeout > 900_000) {
      throw new AxiomConfigurationError("timeoutMs must be an integer between 1 and 900000.");
    }
    this.baseUrl = parsed;
    this.timeoutMs = timeout;
    this.token = options.token;
    this.workspaceId = options.workspaceId;
    this.additionalHeaders = validateHeaders(options.headers);
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  private url(path: string, params?: AxiomRequestOptions["params"]): URL {
    if (!path.startsWith("/api/") || path.startsWith("//") || path.includes("\\") || path.includes("#")) {
      throw new AxiomConfigurationError("SDK paths must be relative AXIOM /api/ routes.");
    }
    if (path.split("/").some((part) => part === "." || part === "..")) {
      throw new AxiomConfigurationError("SDK paths must not contain dot segments.");
    }
    const url = new URL(path, this.baseUrl);
    for (const [key, value] of Object.entries(params ?? {})) {
      if (value !== undefined && value !== null) url.searchParams.set(key, String(value));
    }
    return url;
  }

  private async send(
    method: HTTPMethod,
    path: string,
    options: AxiomRequestOptions & { body?: BodyInit | null; jsonBody?: unknown; contentType?: string } = {},
  ): Promise<Response> {
    if (this.closed) throw new AxiomConfigurationError("AxiomClient is closed.");
    if (options.body != null && options.jsonBody !== undefined) {
      throw new AxiomConfigurationError("Supply body or jsonBody, not both.");
    }
    const id = requestId();
    const headers = new Headers({
      Accept: "application/json",
      "User-Agent": "@axiom-ai/sdk/1.0.0",
      ...this.additionalHeaders,
    });
    for (const [name, value] of Object.entries(options.headers ?? {})) {
      const lower = name.toLowerCase();
      if (["authorization", "x-request-id", "x-axiom-workspace-id", "host", "content-length"].includes(lower)) {
        throw new AxiomConfigurationError(`The ${name} header is managed by AxiomClient.`);
      }
      headers.set(name, value);
    }
    headers.set("X-Request-ID", id);
    if (this.workspaceId) headers.set("X-AXIOM-Workspace-ID", this.workspaceId);
    if (this.token) headers.set("Authorization", `Bearer ${this.token}`);
    let body = options.body ?? null;
    if (options.jsonBody !== undefined) {
      try {
        body = JSON.stringify(options.jsonBody, (_key, value) => {
          if (typeof value === "number" && !Number.isFinite(value)) {
            throw new TypeError("Non-finite JSON numbers are not supported.");
          }
          return value;
        });
      } catch {
        throw new AxiomConfigurationError("Request body must contain finite JSON-compatible values.");
      }
      headers.set("Content-Type", "application/json");
    } else if (body !== null && options.contentType) {
      headers.set("Content-Type", options.contentType);
    }
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 900_000) {
      throw new AxiomConfigurationError("timeoutMs must be an integer between 1 and 900000.");
    }
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
    const externalSignal = options.signal;
    const abortExternal = () => controller.abort(externalSignal?.reason);
    if (externalSignal?.aborted) abortExternal();
    else externalSignal?.addEventListener("abort", abortExternal, { once: true });
    try {
      return await this.fetchImpl(this.url(path, options.params), {
        method,
        headers,
        body,
        redirect: "error",
        signal: controller.signal,
      });
    } catch (cause) {
      const message = timedOut
        ? "AXIOM request timed out."
        : externalSignal?.aborted
          ? "AXIOM request was cancelled."
          : "AXIOM service could not be reached.";
      throw new AxiomNetworkError(message, id, { cause });
    } finally {
      clearTimeout(timer);
      externalSignal?.removeEventListener("abort", abortExternal);
    }
  }

  private async checkedResponse(
    method: HTTPMethod,
    path: string,
    options: AxiomRequestOptions & { body?: BodyInit | null; jsonBody?: unknown; contentType?: string } = {},
  ): Promise<Response> {
    const response = await this.send(method, path, options);
    if (!response.ok) {
      const text = await response.text();
      let payload: AxiomErrorEnvelope = {};
      try {
        payload = text ? JSON.parse(text) as AxiomErrorEnvelope : {};
      } catch {
        payload = {};
      }
      const error = payload.error ?? {};
      const message = error.message ?? payload.detail ?? `AXIOM API request failed with HTTP ${response.status}.`;
      throw new AxiomAPIError(
        message,
        response.status,
        error.code ?? null,
        response.headers.get("X-Request-ID") ?? payload.request_id ?? null,
        error.details,
      );
    }
    return response;
  }

  async request<TResponse>(
    method: HTTPMethod,
    path: string,
    options: AxiomRequestOptions & { jsonBody?: unknown } = {},
  ): Promise<TResponse> {
    const response = await this.checkedResponse(method, path, options);
    if (response.status === 204) return undefined as TResponse;
    const text = await response.text();
    if (!text) return undefined as TResponse;
    try {
      return JSON.parse(text) as TResponse;
    } catch {
      throw new AxiomNetworkError(
        "AXIOM returned a successful response that was not valid JSON.",
        response.headers.get("X-Request-ID"),
      );
    }
  }

  private async requestBytes(method: HTTPMethod, path: string, options: AxiomRequestOptions = {}): Promise<Uint8Array> {
    const response = await this.checkedResponse(method, path, options);
    return new Uint8Array(await response.arrayBuffer());
  }

  health(): Promise<AxiomHealth> {
    return this.request("GET", "/api/health");
  }

  meta(): Promise<AxiomAPIContractMetadata> {
    return this.request("GET", "/api/meta");
  }

  openapi(): Promise<Record<string, unknown>> {
    return this.request("GET", "/api/openapi.json");
  }

  runtime(): Promise<Record<string, unknown>> {
    return this.request("GET", "/api/workspace/runtime");
  }

  models(options: { limit?: number; offset?: number; search?: string } = {}): Promise<Record<string, unknown>> {
    return this.request("GET", "/api/workspace/models", { params: options });
  }

  datasets(options: { limit?: number; offset?: number; search?: string } = {}): Promise<Record<string, unknown>> {
    return this.request("GET", "/api/workspace/datasets", { params: options });
  }

  workspaces(): Promise<AxiomWorkspaceCatalog> {
    return this.request("GET", "/api/workspaces");
  }

  createWorkspace(name: string): Promise<{ workspace: AxiomWorkspaceSummary }> {
    return this.request("POST", "/api/workspaces", { jsonBody: { name } });
  }

  renameWorkspace(workspaceId: string, name: string): Promise<{ workspace: AxiomWorkspaceSummary }> {
    return this.request("PATCH", `/api/workspaces/${encodeURIComponent(workspaceId)}`, {
      jsonBody: { name },
    });
  }

  downloadWorkspaceBackup(workspaceId: string, options: { timeoutMs?: number; signal?: AbortSignal } = {}): Promise<Uint8Array> {
    return this.requestBytes("GET", `/api/workspaces/${encodeURIComponent(workspaceId)}/backup`, {
      ...options,
      timeoutMs: options.timeoutMs ?? 900_000,
    });
  }

  restoreWorkspaceBackup(
    archive: Blob,
    name: string,
    options: { timeoutMs?: number; signal?: AbortSignal } = {},
  ): Promise<AxiomWorkspaceBackupRestoreResult> {
    if (!(archive instanceof Blob) || archive.size === 0) {
      throw new AxiomConfigurationError("Backup archive must be a non-empty Blob.");
    }
    return this.request("POST", "/api/workspaces/backup/restore", {
      params: { name },
      body: archive,
      contentType: "application/zip",
      ...options,
      timeoutMs: options.timeoutMs ?? 900_000,
    });
  }

  engineStatus(): Promise<AxiomEngineStatus> {
    return this.request("GET", "/api/engine/status");
  }

  engineTelemetry(): Promise<AxiomEngineTelemetry> {
    return this.request("GET", "/api/engine/telemetry");
  }

  telemetryHistory(limit = 500): Promise<AxiomTelemetryHistory> {
    return this.request("GET", "/api/engine/telemetry-history", { params: { limit } });
  }

  configureTelemetryHistory(settings: AxiomTelemetryHistorySettings): Promise<AxiomTelemetryHistory> {
    return this.request("PUT", "/api/engine/telemetry-history/settings", { jsonBody: settings });
  }

  recordTelemetrySample(): Promise<{ sample: AxiomTelemetryHistory["samples"][number]; history: AxiomTelemetryHistory }> {
    return this.request("POST", "/api/engine/telemetry-history");
  }

  clearTelemetryHistory(): Promise<AxiomTelemetryHistory> {
    return this.request("DELETE", "/api/engine/telemetry-history");
  }

  knowledgeBases(): Promise<{ knowledge_bases: AxiomKnowledgeBase[]; [key: string]: unknown }> {
    return this.request("GET", "/api/workspace/knowledge-bases");
  }

  createKnowledgeBase(name: string, description = ""): Promise<{ knowledge_base: AxiomKnowledgeBase; [key: string]: unknown }> {
    return this.request("POST", "/api/workspace/knowledge-bases", { jsonBody: { name, description } });
  }

  searchKnowledgeBase(
    baseId: string,
    query: string,
    options: Record<string, unknown> = {},
  ): Promise<Record<string, unknown>> {
    return this.request(
      "POST",
      `/api/workspace/knowledge-bases/${encodeURIComponent(baseId)}/search`,
      { jsonBody: { query, ...options } },
    );
  }

  pluginManifests(): Promise<AxiomPluginManifestRegistry> {
    return this.request("GET", "/api/security/plugins");
  }

  close(): void {
    this.closed = true;
  }
}
