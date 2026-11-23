/** JSON values accepted and returned by the AXIOM API. */
export type JSONPrimitive = string | number | boolean | null;
export type JSONValue = JSONPrimitive | JSONValue[] | { [key: string]: JSONValue };

export interface AxiomPredictionJobSummary {
  id: string;
  status: string;
  created_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  dataset_path?: string;
  output_path?: string;
  artifact_ready?: boolean;
  rows_total?: number;
  rows_completed?: number;
  resumable?: boolean;
  resumption_count?: number;
  error_code?: string | null;
  [key: string]: unknown;
}

export interface AxiomPredictionJobStartRequest {
  model_path: string;
  dataset_path: string;
  output_path: string;
  adapter_path?: string | null;
  prompt_field?: string;
  reference_field?: string;
  prediction_field?: string;
  max_new_tokens?: number;
  temperature?: number;
  seed?: number | null;
  device?: "auto" | "cpu" | "cuda" | "rocm" | "mps" | "xpu";
  precision?: "auto" | "fp32" | "fp16" | "bf16";
}

export interface AxiomPredictionJobPage {
  schema_version: 1;
  jobs: AxiomPredictionJobSummary[];
  limit: number;
  scope?: Record<string, unknown>;
}

export interface AxiomPredictionJobStatusResult {
  schema_version: 1;
  job: AxiomPredictionJobSummary;
}

export interface AxiomPredictionJobStartResult {
  schema_version: 1;
  job: AxiomPredictionJobSummary;
  limits?: Record<string, number>;
  scope?: Record<string, unknown>;
}

export interface AxiomPredictionJobCancelResult {
  schema_version: 1;
  job: AxiomPredictionJobSummary;
  cancellation_accepted: boolean;
  note?: string;
}

export interface AxiomPredictionJobResumeScope {
  dataset_fingerprint_verified: true;
  model_and_adapter_fingerprints_verified: true;
  checkpoint_identity_and_digest_verified: true;
  generation_configuration_fingerprint_verified: true;
  partial_output_published: false;
  prompts_returned: false;
  predictions_returned: false;
  generation_resumed_after_restart: true;
}

export interface AxiomPredictionJobResumeResult {
  schema_version: 1;
  job: AxiomPredictionJobSummary;
  scope: AxiomPredictionJobResumeScope;
}

export interface AxiomClientOptions {
  /** API origin. Defaults to the local AXIOM daemon. */
  baseUrl?: string;
  /** Optional AXIOM read or write bearer token. */
  token?: string;
  /** Optional workspace routing selector. */
  workspaceId?: string;
  /** Per-request timeout in milliseconds. */
  timeoutMs?: number;
  /** Additional non-sensitive headers. SDK security headers are managed internally. */
  headers?: Record<string, string>;
  /** Optional fetch implementation for testing or supported runtimes. */
  fetchImpl?: typeof fetch;
}

export interface AxiomRequestOptions {
  params?: Record<string, string | number | boolean | null | undefined>;
  headers?: Record<string, string>;
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface AxiomHealth {
  status: string;
  version: string;
}

export interface AxiomAPIContractMetadata {
  api_contract_version: 1;
  application_version: string;
  openapi_url: string;
  compatibility_policy: string;
}

export interface AxiomWorkspaceSummary {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  is_default: boolean;
  status: string;
  managed?: boolean;
}

export interface AxiomWorkspaceCatalog {
  schema_version: 1;
  active_workspace_id: string;
  default_workspace_id: string;
  workspaces: AxiomWorkspaceSummary[];
  scope_note: string;
}

export interface AxiomWorkspaceBackupReport {
  schema_version: 1;
  backup_id: string;
  workspace_label: string;
  file_count: number;
  total_bytes: number;
  archive_bytes?: number;
  manifest_sha256?: string;
  integrity?: "verified";
  skipped: {
    symlink_files: number;
    symlink_directories: number;
    special_files: number;
    excluded_directories: number;
  };
  scope_note: string;
}

export interface AxiomWorkspaceBackupRestoreResult {
  schema_version: 1;
  workspace: AxiomWorkspaceSummary;
  backup: AxiomWorkspaceBackupReport;
  scope_note: string;
}

export interface AxiomTelemetryHistorySettings {
  enabled: boolean;
  max_samples: number;
  max_age_days: number;
}

export interface AxiomResourceTelemetryMetrics {
  sampled_at: string;
  sample_type: "point_in_time";
  cpu: {
    logical_cores: number | null;
    system_percent: number | null;
    process_percent: number | null;
  };
  memory: {
    total_bytes: number | null;
    available_bytes: number | null;
    used_bytes: number | null;
    percent: number | null;
  };
  process: {
    rss_bytes: number | null;
    peak_rss_bytes: number | null;
    threads: number | null;
  };
  devices: Array<{
    available: boolean;
    memory_free_gib: number | null;
    memory_total_gib: number | null;
  }>;
  loaded_runtime: {
    loaded: boolean;
    device: string | null;
    memory_mode: "full" | "layer_streaming" | "disk_streaming" | null;
    precision: string | null;
    device_allocator: Record<string, string | number | boolean | null> | null;
  };
}

export interface AxiomResourceTelemetryRecord {
  id: string;
  sampled_at: string;
  fingerprint: string;
  metrics: AxiomResourceTelemetryMetrics;
}

export interface AxiomTelemetryHistory {
  schema_version: 1;
  settings: AxiomTelemetryHistorySettings;
  total: number;
  samples: AxiomResourceTelemetryRecord[];
  privacy: {
    process_ids_included: false;
    model_names_or_paths_included: false;
    prompt_or_generated_content_included: false;
    environment_values_included: false;
    credentials_included: false;
    network_identifiers_included: false;
    automatic_sampling: false;
  };
}

export interface AxiomEngineStatus {
  [key: string]: unknown;
}

export interface AxiomEngineTelemetry {
  sampled_at: string;
  sample_type: "point_in_time";
  cpu: Record<string, number | null>;
  memory: Record<string, number | null>;
  process: Record<string, number | string | null>;
  devices: Array<Record<string, boolean | number | string | null>>;
  loaded_runtime: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AxiomKnowledgeBase {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  [key: string]: unknown;
}

export interface AxiomPluginManifestRegistry {
  store_schema_version: 1;
  manifests: Array<{
    manifest: {
      manifest_schema_version: 1;
      id: string;
      name: string;
      version: string;
      description: string;
      tools: Array<{
        name: string;
        description: string;
        scopes: Array<"read" | "write" | "execution" | "network" | "secrets">;
      }>;
    };
    sha256: string;
    permission_status: "default_deny" | "granted" | "stale_manifest";
    granted_tools: Array<{ name: string; scopes: string[] }>;
  }>;
  maximum_manifests: number;
  maximum_tools_per_manifest: number;
  execution_note: string;
}
