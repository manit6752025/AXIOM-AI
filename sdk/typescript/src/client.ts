import { AxiomAPIError, AxiomConfigurationError, AxiomNetworkError, type AxiomErrorEnvelope } from "./errors.js";
import type {
  AxiomAPIContractMetadata,
  AxiomClientOptions,
  AxiomEngineStatus,
  AxiomEngineTelemetry,
  AxiomHealth,
  AxiomKnowledgeBase,
  AxiomPluginManifestRegistry,
  AxiomPredictionJobCancelResult,
  AxiomPredictionJobPage,
  AxiomPredictionJobStartRequest,
  AxiomPredictionJobStartResult,
  AxiomPredictionJobResumeResult,
  AxiomPredictionJobStatusResult,
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

interface PendingResponse {
  response: Response;
  requestId: string;
  cleanup: () => void;
  timedOut: () => boolean;
  externallyAborted: () => boolean;
}

function hasControlCharacters(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code < 0x20 || code === 0x7f) return true;
  }
  return false;
}

function parseErrorEnvelope(text: string): AxiomErrorEnvelope {
  if (!text) return {};
  try {
    const value: unknown = JSON.parse(text);
    return typeof value === "object" && value !== null && !Array.isArray(value)
      ? value as AxiomErrorEnvelope
      : {};
  } catch {
    return {};
  }
}

function requestId(): string {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID().replaceAll("-", "");
  }
  if (cryptoApi && typeof cryptoApi.getRandomValues === "function") {
    const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  }
  throw new AxiomConfigurationError("A cryptographically secure random source is required for request IDs.");
}

function validateHeaders(headers: Record<string, string> | undefined): Record<string, string> {
  const result: Record<string, string> = {};
  const seenNames = new Set<string>();
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
    const lower = name.toLowerCase();
    if (protectedHeaders.has(lower)) {
      throw new AxiomConfigurationError(`The ${name} header is managed by AxiomClient.`);
    }
    if (seenNames.has(lower)) {
      throw new AxiomConfigurationError(`The ${name} header was supplied more than once.`);
    }
    seenNames.add(lower);
    if (hasControlCharacters(name + value)) {
      throw new AxiomConfigurationError("Additional headers must not contain control characters.");
    }
    result[name] = value;
  }
  return result;
}

export class AxiomClient {
  readonly baseUrl: string;
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
    const localHostname = parsed.hostname.toLowerCase().replace(/\.$/, "");
    const local = ["localhost", "127.0.0.1", "[::1]", "::1"].includes(localHostname);
    if (parsed.protocol !== "https:" && !local) {
      throw new AxiomConfigurationError("Non-loopback AXIOM services must use HTTPS.");
    }
    if (
      options.token !== undefined
      && (!options.token.trim() || hasControlCharacters(options.token))
    ) {
      throw new AxiomConfigurationError("Bearer token must be a non-empty printable single-line string.");
    }
    if (
      options.workspaceId !== undefined
      && (
        !options.workspaceId.trim()
        || options.workspaceId.length > 128
        || hasControlCharacters(options.workspaceId)
      )
    ) {
      throw new AxiomConfigurationError("Workspace ID must contain 1 to 128 printable characters.");
    }
    const timeout = options.timeoutMs ?? 30_000;
    if (!Number.isSafeInteger(timeout) || timeout < 1 || timeout > 900_000) {
      throw new AxiomConfigurationError("timeoutMs must be an integer between 1 and 900000.");
    }
    this.baseUrl = parsed.origin;
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
  ): Promise<PendingResponse> {
    if (this.closed) throw new AxiomConfigurationError("AxiomClient is closed.");
    if (options.body != null && options.jsonBody !== undefined) {
      throw new AxiomConfigurationError("Supply body or jsonBody, not both.");
    }
    const id = requestId();
    const headers = new Headers({
      Accept: "application/json",
      ...this.additionalHeaders,
    });
    const requestHeaderNames = new Set<string>();
    for (const [name, value] of Object.entries(options.headers ?? {})) {
      const lower = name.toLowerCase();
      if (["authorization", "x-request-id", "x-axiom-workspace-id", "host", "content-length"].includes(lower)) {
        throw new AxiomConfigurationError(`The ${name} header is managed by AxiomClient.`);
      }
      if (!name.trim() || hasControlCharacters(name + value)) {
        throw new AxiomConfigurationError("Per-request headers must use printable names and values.");
      }
      if (requestHeaderNames.has(lower)) {
        throw new AxiomConfigurationError(`The ${name} header was supplied more than once.`);
      }
      requestHeaderNames.add(lower);
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
    const url = this.url(path, options.params);
    const controller = new AbortController();
    let didTimeOut = false;
    let cleaned = false;
    const timer = setTimeout(() => {
      didTimeOut = true;
      controller.abort();
    }, timeoutMs);
    const externalSignal = options.signal;
    const abortExternal = () => controller.abort(externalSignal?.reason);
    if (externalSignal?.aborted) abortExternal();
    else externalSignal?.addEventListener("abort", abortExternal, { once: true });
    const cleanup = () => {
      if (cleaned) return;
      cleaned = true;
      clearTimeout(timer);
      externalSignal?.removeEventListener("abort", abortExternal);
    };
    try {
      const response = await this.fetchImpl(url, {
        method,
        headers,
        body,
        redirect: "error",
        signal: controller.signal,
      });
      return {
        response,
        requestId: id,
        cleanup,
        timedOut: () => didTimeOut,
        externallyAborted: () => Boolean(externalSignal?.aborted),
      };
    } catch (cause) {
      cleanup();
      const message = didTimeOut
        ? "AXIOM request timed out."
        : externalSignal?.aborted
          ? "AXIOM request was cancelled."
          : "AXIOM service could not be reached.";
      throw new AxiomNetworkError(message, id, { cause });
    }
  }

  private responseReadError(pending: PendingResponse, cause: unknown): AxiomNetworkError {
    const message = pending.timedOut()
      ? "AXIOM request timed out while reading the response."
      : pending.externallyAborted()
        ? "AXIOM request was cancelled while reading the response."
        : "AXIOM response body could not be read.";
    return new AxiomNetworkError(message, pending.requestId, { cause });
  }

  private async checkedResponse(
    method: HTTPMethod,
    path: string,
    options: AxiomRequestOptions & { body?: BodyInit | null; jsonBody?: unknown; contentType?: string } = {},
  ): Promise<PendingResponse> {
    const pending = await this.send(method, path, options);
    const response = pending.response;
    if (!response.ok) {
      let text: string;
      try {
        text = await response.text();
      } catch (cause) {
        pending.cleanup();
        throw this.responseReadError(pending, cause);
      }
      pending.cleanup();
      const payload = parseErrorEnvelope(text);
      const error = payload.error ?? {};
      const message = error.message ?? payload.detail ?? `AXIOM API request failed with HTTP ${response.status}.`;
      throw new AxiomAPIError(
        message,
        response.status,
        error.code ?? null,
        response.headers.get("X-Request-ID") ?? payload.request_id ?? pending.requestId,
        error.details,
      );
    }
    return pending;
  }

  async request<TResponse>(
    method: HTTPMethod,
    path: string,
    options: AxiomRequestOptions & { jsonBody?: unknown; body?: BodyInit | null; contentType?: string } = {},
  ): Promise<TResponse> {
    const pending = await this.checkedResponse(method, path, options);
    const response = pending.response;
    if (response.status === 204) {
      pending.cleanup();
      return undefined as TResponse;
    }
    let text: string;
    try {
      text = await response.text();
    } catch (cause) {
      pending.cleanup();
      throw this.responseReadError(pending, cause);
    }
    pending.cleanup();
    if (!text) return undefined as TResponse;
    try {
      return JSON.parse(text) as TResponse;
    } catch {
      throw new AxiomNetworkError(
        "AXIOM returned a successful response that was not valid JSON.",
        response.headers.get("X-Request-ID") ?? pending.requestId,
      );
    }
  }

  private async requestBytes(method: HTTPMethod, path: string, options: AxiomRequestOptions = {}): Promise<Uint8Array> {
    const pending = await this.checkedResponse(method, path, options);
    try {
      return new Uint8Array(await pending.response.arrayBuffer());
    } catch (cause) {
      throw this.responseReadError(pending, cause);
    } finally {
      pending.cleanup();
    }
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
    if (!name.trim() || name.length > 64 || hasControlCharacters(name)) {
      throw new AxiomConfigurationError("Workspace name must contain 1 to 64 printable characters.");
    }
    return this.request("POST", "/api/workspaces", { jsonBody: { name } });
  }

  renameWorkspace(workspaceId: string, name: string): Promise<{ workspace: AxiomWorkspaceSummary }> {
    if (!workspaceId.trim() || workspaceId.length > 128 || hasControlCharacters(workspaceId)) {
      throw new AxiomConfigurationError("Workspace ID must contain 1 to 128 printable characters.");
    }
    if (!name.trim() || name.length > 64 || hasControlCharacters(name)) {
      throw new AxiomConfigurationError("Workspace name must contain 1 to 64 printable characters.");
    }
    return this.request("PATCH", `/api/workspaces/${encodeURIComponent(workspaceId)}`, {
      jsonBody: { name },
    });
  }

  downloadWorkspaceBackup(workspaceId: string, options: { timeoutMs?: number; signal?: AbortSignal } = {}): Promise<Uint8Array> {
    return this.requestBytes("GET", `/api/workspaces/${encodeURIComponent(workspaceId)}/backup`, {
      ...options,
      timeoutMs: options.timeoutMs ?? this.timeoutMs,
    });
  }

  async downloadWorkspaceBackupStream(
    workspaceId: string,
    options: { timeoutMs?: number; signal?: AbortSignal } = {},
  ): Promise<ReadableStream<Uint8Array>> {
    const pending = await this.checkedResponse(
      "GET",
      `/api/workspaces/${encodeURIComponent(workspaceId)}/backup`,
      { ...options, timeoutMs: options.timeoutMs ?? this.timeoutMs },
    );
    const body = pending.response.body;
    if (!body) {
      pending.cleanup();
      throw new AxiomNetworkError("AXIOM returned an empty backup stream.", pending.requestId);
    }
    const reader = body.getReader();
    return new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const chunk = await reader.read();
          if (chunk.done) {
            pending.cleanup();
            controller.close();
          } else {
            controller.enqueue(chunk.value);
          }
        } catch (cause) {
          pending.cleanup();
          controller.error(new AxiomNetworkError(
            pending.timedOut()
              ? "AXIOM backup download timed out."
              : "AXIOM backup stream failed.",
            pending.requestId,
            { cause },
          ));
        }
      },
      async cancel(reason) {
        pending.cleanup();
        await reader.cancel(reason);
      },
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
    if (!name.trim() || name.length > 64 || hasControlCharacters(name)) {
      throw new AxiomConfigurationError("Restored workspace name must contain 1 to 64 printable characters.");
    }
    return this.request("POST", "/api/workspaces/backup/restore", {
      params: { name },
      body: archive,
      contentType: "application/zip",
      ...options,
      timeoutMs: options.timeoutMs ?? this.timeoutMs,
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


  private validatePredictionJobId(jobId: string): void {
    if (!/^[0-9a-f]{32}$/.test(jobId)) {
      throw new AxiomConfigurationError("Prediction job ID must be 32 lowercase hexadecimal characters.");
    }
  }

  predictionJobs(limit = 20): Promise<AxiomPredictionJobPage> {
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 50) {
      throw new AxiomConfigurationError("Prediction job list limit must be between 1 and 50.");
    }
    return this.request("GET", "/api/prediction-jobs", { params: { limit } });
  }

  startPredictionJob(request: AxiomPredictionJobStartRequest): Promise<AxiomPredictionJobStartResult> {
    if (!request || typeof request !== "object") {
      throw new AxiomConfigurationError("Prediction job start request must be an object.");
    }
    const allowedFields = new Set([
      "model_path", "adapter_path", "dataset_path", "output_path",
      "prompt_field", "reference_field", "prediction_field",
      "max_new_tokens", "temperature", "seed", "device", "precision",
    ]);
    if (Object.keys(request).some((key) => !allowedFields.has(key))) {
      throw new AxiomConfigurationError("Prediction job start request contains unsupported fields.");
    }
    for (const [field, value, maximum] of [
      ["model_path", request.model_path, 4096],
      ["dataset_path", request.dataset_path, 512],
      ["output_path", request.output_path, 512],
    ] as const) {
      if (typeof value !== "string" || !value.trim() || value.length > maximum) {
        throw new AxiomConfigurationError(`${field} must contain 1 to ${maximum} characters.`);
      }
    }
    if (request.max_new_tokens !== undefined && (!Number.isSafeInteger(request.max_new_tokens) || request.max_new_tokens < 1 || request.max_new_tokens > 64)) {
      throw new AxiomConfigurationError("max_new_tokens must be an integer between 1 and 64.");
    }
    if (request.temperature !== undefined && (!Number.isFinite(request.temperature) || request.temperature < 0 || request.temperature > 2)) {
      throw new AxiomConfigurationError("temperature must be a finite number between 0 and 2.");
    }
    if (request.seed !== undefined && request.seed !== null && (!Number.isSafeInteger(request.seed) || request.seed < 0)) {
      throw new AxiomConfigurationError("seed must be a non-negative safe integer.");
    }
    if (request.adapter_path !== undefined && request.adapter_path !== null && (
      typeof request.adapter_path !== "string" || !request.adapter_path.trim() || request.adapter_path.length > 4096
    )) {
      throw new AxiomConfigurationError("adapter_path must contain 1 to 4096 characters when supplied.");
    }
    const fieldNames = [
      ["prompt_field", request.prompt_field ?? "prompt"],
      ["reference_field", request.reference_field ?? "reference"],
      ["prediction_field", request.prediction_field ?? "prediction"],
    ] as const;
    const fieldPattern = /^[A-Za-z_][A-Za-z0-9_.-]{0,159}$/;
    for (const [field, value] of fieldNames) {
      if (typeof value !== "string" || !fieldPattern.test(value)) {
        throw new AxiomConfigurationError(`${field} must be a valid JSON field name of at most 160 characters.`);
      }
    }
    if (new Set(fieldNames.map(([, value]) => value)).size !== fieldNames.length) {
      throw new AxiomConfigurationError("Prompt, reference, and prediction fields must be different.");
    }
    if (!request.dataset_path.startsWith("data/")) {
      throw new AxiomConfigurationError("Prediction input must be a workspace-relative path under data/.");
    }
    if (
      !request.output_path.startsWith("data/")
      || request.output_path.split("/").length !== 2
      || !request.output_path.toLowerCase().endsWith(".jsonl")
    ) {
      throw new AxiomConfigurationError("Prediction output must be a direct-child data/NAME.jsonl path.");
    }
    if (request.output_path === request.dataset_path) {
      throw new AxiomConfigurationError("Prediction output must differ from the source dataset.");
    }
    const devices = new Set(["auto", "cpu", "cuda", "rocm", "mps", "xpu"]);
    const precisions = new Set(["auto", "fp32", "fp16", "bf16"]);
    if (request.device !== undefined && (typeof request.device !== "string" || !devices.has(request.device))) {
      throw new AxiomConfigurationError("device is not a supported local prediction device.");
    }
    if (request.precision !== undefined && (typeof request.precision !== "string" || !precisions.has(request.precision))) {
      throw new AxiomConfigurationError("precision is not a supported local prediction precision.");
    }
    if (request.device === "mps" && request.precision === "bf16") {
      throw new AxiomConfigurationError("BF16 precision is not supported on Metal/MPS.");
    }
    return this.request("POST", "/api/prediction-jobs", { jsonBody: request });
  }

  predictionJob(jobId: string): Promise<AxiomPredictionJobStatusResult> {
    this.validatePredictionJobId(jobId);
    return this.request("GET", `/api/prediction-jobs/${encodeURIComponent(jobId)}`);
  }

  resumePredictionJob(jobId: string): Promise<AxiomPredictionJobResumeResult> {
    this.validatePredictionJobId(jobId);
    return this.request("POST", `/api/prediction-jobs/${encodeURIComponent(jobId)}/resume`);
  }

  cancelPredictionJob(jobId: string): Promise<AxiomPredictionJobCancelResult> {
    this.validatePredictionJobId(jobId);
    return this.request("POST", `/api/prediction-jobs/${encodeURIComponent(jobId)}/cancel`);
  }

  pluginManifests(): Promise<AxiomPluginManifestRegistry> {
    return this.request("GET", "/api/security/plugins");
  }

  close(): void {
    this.closed = true;
  }
}
