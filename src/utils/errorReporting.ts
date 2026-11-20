export type FrontendErrorRecord = {
  id: string;
  occurredAt: string;
  source: "api" | "window" | "promise";
  message: string;
};

const MAX_RECORDS = 25;
const records: FrontendErrorRecord[] = [];

/** Error text can originate outside AXIOM. Never retain values from it. */
export function redactErrorText(value: unknown): string {
  const text = typeof value === "string" ? value : value instanceof Error ? value.message : "Unexpected frontend error.";
  return text
    .replace(/\b(bearer|basic)\s+[a-z0-9._~+/-]+=*/gi, "$1 [REDACTED]")
    .replace(/([?&](?:api[_-]?key|access[_-]?token|token|secret|password|passwd|prompt|data)=)[^&#\s]*/gi, "$1[REDACTED]")
    .replace(/(["']?(?:api[_-]?key|access[_-]?token|token|secret|password|passwd|prompt|data)["']?\s*[:=]\s*)["']?[^,}\]\s"']+["']?/gi, "$1[REDACTED]")
    .replace(/(authorization\s*[:=]\s*)[^,;\n]*/gi, "$1[REDACTED]")
    .replace(/\s+/g, " ").trim().slice(0, 512) || "Unexpected frontend error.";
}

export function frontendErrorRecords(): readonly FrontendErrorRecord[] { return records; }

export function clearFrontendErrorRecords(): void {
  records.splice(0, records.length);
  window.dispatchEvent(new CustomEvent("axiom:frontend-error"));
}

export function reportFrontendError(source: FrontendErrorRecord["source"], error: unknown): FrontendErrorRecord {
  const record: FrontendErrorRecord = {
    id: globalThis.crypto?.randomUUID?.() ?? `frontend-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    occurredAt: new Date().toISOString(), source, message: redactErrorText(error),
  };
  records.unshift(record);
  records.splice(MAX_RECORDS);
  window.dispatchEvent(new CustomEvent("axiom:frontend-error", { detail: record }));
  return record;
}
