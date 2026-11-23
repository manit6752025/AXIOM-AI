export const FRONTEND_ERROR_EVENT = "axiom:frontend-error-recorded";
export const MAX_FRONTEND_ERROR_ENTRIES = 50;

export type FrontendErrorSource = "api.request" | "window.error" | "window.unhandledrejection";
export type FrontendErrorCategory =
  | "api_http_error"
  | "api_network_error"
  | "api_timeout"
  | "runtime_error"
  | "unhandled_rejection";
export type FrontendErrorName =
  | "ApiError"
  | "Error"
  | "TypeError"
  | "RangeError"
  | "SyntaxError"
  | "ReferenceError"
  | "URIError"
  | "AbortError"
  | "NetworkError"
  | "UnknownError";

export type FrontendErrorEntry = {
  id: number;
  occurred_at: string;
  source: FrontendErrorSource;
  category: FrontendErrorCategory;
  error_name: FrontendErrorName;
  page: string;
  http_status: number | null;
};

export type FrontendErrorReport = {
  schema_version: 1;
  generated_at: string;
  events: FrontendErrorEntry[];
  privacy: {
    storage: "in_memory_only";
    automatic_upload: false;
    raw_messages_stored: false;
    stack_traces_stored: false;
    request_urls_stored: false;
    request_ids_stored: false;
    request_bodies_stored: false;
    prompts_or_dataset_values_stored: false;
    credentials_stored: false;
    max_entries: 50;
  };
};

type FrontendErrorInput = {
  source: FrontendErrorSource;
  category: FrontendErrorCategory;
  errorName?: unknown;
  httpStatus?: number;
};

const PAGE_IDS = new Set([
  "dashboard", "models", "datasets", "training", "workflows", "evaluation",
  "runtime", "mcp", "modules", "roadmap", "diagnostics", "logs", "settings",
  "api-playground", "optimizer", "quantizer",
]);
const ERROR_NAMES = new Set<FrontendErrorName>([
  "ApiError", "Error", "TypeError", "RangeError", "SyntaxError",
  "ReferenceError", "URIError", "AbortError", "NetworkError", "UnknownError",
]);
const entries: FrontendErrorEntry[] = [];
let nextId = 1;

function safePageId(): string {
  try {
    const page = new URLSearchParams(window.location.search).get("page") ?? "dashboard";
    return PAGE_IDS.has(page) ? page : "unknown";
  } catch {
    return "unknown";
  }
}

function safeErrorName(value: unknown): FrontendErrorName {
  if (typeof value === "string" && ERROR_NAMES.has(value as FrontendErrorName)) {
    return value as FrontendErrorName;
  }
  return "UnknownError";
}

export function recordFrontendError(input: FrontendErrorInput): void {
  const httpStatus = Number.isInteger(input.httpStatus) && input.httpStatus! >= 400 && input.httpStatus! <= 599
    ? input.httpStatus!
    : null;
  entries.push({
    id: nextId++,
    occurred_at: new Date().toISOString(),
    source: input.source,
    category: input.category,
    error_name: safeErrorName(input.errorName),
    page: safePageId(),
    http_status: httpStatus,
  });
  if (entries.length > MAX_FRONTEND_ERROR_ENTRIES) {
    entries.splice(0, entries.length - MAX_FRONTEND_ERROR_ENTRIES);
  }
  window.dispatchEvent(new Event(FRONTEND_ERROR_EVENT));
}

export function getFrontendErrorReport(): FrontendErrorReport {
  return {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    events: entries.map((entry) => ({ ...entry })),
    privacy: {
      storage: "in_memory_only",
      automatic_upload: false,
      raw_messages_stored: false,
      stack_traces_stored: false,
      request_urls_stored: false,
      request_ids_stored: false,
      request_bodies_stored: false,
      prompts_or_dataset_values_stored: false,
      credentials_stored: false,
      max_entries: MAX_FRONTEND_ERROR_ENTRIES,
    },
  };
}

export function clearFrontendErrorReport(): void {
  entries.length = 0;
  window.dispatchEvent(new Event(FRONTEND_ERROR_EVENT));
}

export function installFrontendErrorHandlers(): () => void {
  const onWindowError = (event: ErrorEvent) => {
    recordFrontendError({
      source: "window.error",
      category: "runtime_error",
      errorName: event.error && typeof event.error === "object" ? (event.error as { name?: unknown }).name : "UnknownError",
    });
  };
  const onUnhandledRejection = (event: PromiseRejectionEvent) => {
    const reason = event.reason;
    recordFrontendError({
      source: "window.unhandledrejection",
      category: "unhandled_rejection",
      errorName: reason && typeof reason === "object" ? (reason as { name?: unknown }).name : "UnknownError",
    });
  };
  window.addEventListener("error", onWindowError);
  window.addEventListener("unhandledrejection", onUnhandledRejection);
  return () => {
    window.removeEventListener("error", onWindowError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
  };
}
