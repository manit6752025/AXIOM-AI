const API_BASE = import.meta.env.VITE_AXIOM_API_URL || "";


export const playgroundEndpoints = [
  { path: "/api/health", label: "Health", description: "Process liveness and API version." },
  { path: "/api/meta", label: "Contract metadata", description: "API contract version and compatibility policy." },
  { path: "/api/workspace/runtime", label: "Runtime snapshot", description: "Current hardware snapshot and runtime readiness." },
  { path: "/api/workspace/models?offset=0&limit=5", label: "Models (first 5)", description: "A small, bounded page of registered model metadata." },
  { path: "/api/workspace/datasets", label: "Dataset inventory", description: "Workspace JSONL inventory; no row values are requested." },
  { path: "/api/workspace/events", label: "Recent events", description: "Recent workspace actions for local troubleshooting." },
  { path: "/api/diagnostics/workspace", label: "Workspace diagnostics", description: "Redacted checks for workspace paths, config shape, and dependencies." },
  { path: "/api/diagnostics/roadmap", label: "Roadmap audit", description: "The source roadmap status and scope-integrity report." },
  { path: "/api/diagnostics/runtime-connectors", label: "Runtime connector discovery", description: "Passive local discovery; no runtime process is launched." },
] as const;

export type PlaygroundEndpoint = (typeof playgroundEndpoints)[number]["path"];
const playgroundEndpointPaths = new Set<string>(playgroundEndpoints.map((endpoint) => endpoint.path));


export type EngineFabricProfile = "balanced" | "low-memory" | "throughput" | "easy-setup" | "portable" | "minimal-daemons";
export type EngineFabricBackend = {
  id: string; name: string; kind: string; endpoint: string | null; supports_chat: boolean;
  formats: string[]; strengths: string[]; tradeoffs: string[]; best_for: string[];
  status: "ready" | "offline" | "auth_required" | "incompatible" | "setup_required";
  installed: boolean; reachable: boolean; models: string[]; model_count: number; error: string | null; scope: string;
  generation: {
    transport: "in_process" | "ollama_options" | "openai_compatible";
    parameter_mapping: Record<string, string>;
    native_sampling_controls: string[];
    extension_sampling_controls: string[];
    note: string;
  };
};
export type EngineFabricReport = {
  schema_version: number; checked_at: string; host: { os: string; architecture: string };
  profile: EngineFabricProfile; profiles: Array<{ id: EngineFabricProfile; label: string }>;
  backends: EngineFabricBackend[];
  recommendation: { profile: EngineFabricProfile; backend_id: string; name: string; status: "ready" | "setup_required"; reason: string };
  security: { network_scope: string; environment_proxies_used: false; remote_endpoints_configurable: false; credentials_sent: false; model_downloads_started: false; processes_launched: false };
  scope_note: string;
};
export type EngineFabricChatResult = {
  schema_version: number; backend_id: "axiom-native" | "ollama" | "llama-cpp" | "lm-studio" | "vllm" | "llama-cpp-python";
  backend_name: string; model: string; response: string; done: boolean;
  prompt_eval_count: number | null; eval_count: number | null; total_duration_ms: number;
  generated_at: string; execution: "local_loopback" | "in_process_native"; device?: string | null; precision?: string | null;
  privacy: { prompt_returned: false; response_persisted: false; cloud_provider_used: false };
  routing?: {
    strategy: string; profile: EngineFabricProfile; requested_model: string; fallback_used: boolean;
    attempts: Array<{ backend_id: string; backend_name: string; outcome: "failed" | "selected"; reason?: string }>;
    max_attempts: number; scope_note: string;
  };
};


export type EngineFabricBenchmarkResult = {
  schema_version: number; backend_id: EngineFabricChatResult["backend_id"]; backend_name: string; model: string;
  configuration: { prompt_characters: number; max_new_tokens: number; temperature: number; top_p: number; top_k: number; min_p: number; repetition_penalty: number; seed: number | null; stop_sequence_count: number; warmup_runs: number; iterations: number; measured_requests: number };
  measurements: {
    wall_time_ms: number; requests_per_second: number; completion_tokens_known: number; total_completion_tokens: number | null;
    completion_tokens_per_second: number | null; latency_ms: { min: number; median: number; p95: number; max: number };
    requests: Array<{ latency_ms: number; completion_tokens: number | null }>;
  };
  privacy: { prompt_returned: false; responses_returned: false; results_persisted: false }; scope_note: string;
};

export type LocalInferenceStatus = {
  backend: "ollama";
  endpoint: string;
  runtime_installed: boolean;
  status: "online" | "offline";
  models: string[];
  error: string | null;
  checked_at: string;
  service_managed_by_axiom?: boolean;
  service_can_start?: boolean;
  service_can_stop?: boolean;
  service_control_scope?: "loopback_only" | string;
  service_message?: string;
};
export type OllamaPullJob = {
  id: string;
  model: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  updated_at: string;
  progress: string;
  percent: number | null;
  logs: string[];
  error: string | null;
  cancel_requested: boolean;
};
export type OllamaRunningModel = {
  name: string; size_bytes: number | null; size_vram_bytes: number | null;
  size_ram_bytes: number | null; memory_placement: "mixed" | "gpu" | "cpu_or_unreported";
  expires_at: string | null; digest_prefix: string | null; format: string | null;
  family: string | null; parameter_size: string | null; quantization_level: string | null;
};
export type OllamaRunningModels = {
  schema_version: 1; backend: "ollama"; endpoint: string; models: OllamaRunningModel[];
  count: number; scope_note: string;
};
export type OllamaModelDetails = {
  schema_version: 1; model: string; installed: boolean; size_bytes: number | null;
  modified_at: string | null; format: string | null; family: string | null; families: string[];
  parameter_size: string | null; parameter_count: number | null; quantization_level: string | null;
  context_length: number | null; capabilities: string[]; template_configured: boolean;
  license_present: boolean; system_prompt_configured: boolean; scope_note: string;
};
export type OllamaModelRemoval = {
  schema_version: 1; model: string; removed: boolean; model_files_deleted: boolean;
  remaining_models: string[]; scope_note: string;
};
export type OllamaModelUnload = {
  schema_version: 1; model: string; unloaded: boolean; model_files_deleted: false; scope_note: string;
};

export type LocalInferenceResult = {
  request_id?: string;
  model: string;
  response: string;
  done: boolean;
  done_reason?: string | null;
  execution?: "local";
  prompt_eval_count: number | null;
  eval_count: number | null;
  total_duration_ms: number | null;
  generated_at: string;
};
export type LocalInferenceInput = {
  model: string; prompt: string; num_predict: number; temperature: number;
  top_p?: number; top_k?: number; min_p?: number; repetition_penalty?: number; seed?: number;
  num_ctx?: number; stop_sequences?: string[];
};
export type LocalInferenceStreamEvent =
  | { type: "token"; token: string; request_id?: string }
  | { type: "complete"; result: LocalInferenceResult }
  | { type: "error"; message: string; request_id?: string };

export type LocalInferenceBenchmark = {
  request_id: string;
  schema_version: number;
  backend: "ollama";
  endpoint: string;
  model: string;
  configuration: {
    prompt_characters: number;
    num_predict: number;
    temperature: number;
    warmup_runs: number;
    iterations: number;
    concurrency: number;
    measured_requests: number;
  };
  measurements: {
    wall_time_ms: number;
    requests_per_second: number;
    completion_tokens_known: number;
    prompt_tokens_known: number;
    total_completion_tokens: number | null;
    completion_tokens_per_second: number | null;
    latency_ms: { min: number | null; median: number | null; p95: number | null; max: number | null };
    requests: Array<{ latency_ms: number; prompt_tokens: number | null; completion_tokens: number | null; done: boolean }>;
  };
  privacy: { prompt_returned: false; responses_returned: false; results_persisted: false };
  scope_note: string;
};

export type RuntimeTrace = {
  request_id: string;
  operation: "inference.generate" | "inference.benchmark";
  model: string;
  status: "success" | "error";
  started_at: string;
  completed_at: string;
  duration_ms: number;
  prompt_characters: number;
  requested_output_tokens: number;
  temperature: number;
  warmup_runs: number | null;
  iterations: number | null;
  concurrency: number | null;
  measured_requests: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  completion_tokens_known: number | null;
  completion_tokens_per_second: number | null;
  error_code: string | null;
};
export type RuntimeTraceReport = {
  schema_version: number;
  traces: RuntimeTrace[];
  total_retained: number;
  limit: number;
  max_retained: number;
  privacy: { prompt_bodies_stored: false; response_bodies_stored: false; disk_persistence: false };
};
export type AxiomQuantizationCandidate = {
  format: "int4" | "int8" | "bf16" | "fp16" | "fp32";
  nominal_weight_bits: number;
  estimated_weight_gib: number;
  estimated_kv_cache_gib: number;
  estimated_workspace_gib: number;
  estimated_total_gib: number;
  estimated_range_gib: [number, number];
  fits_ram_budget: boolean | null;
  fits_vram_budget: boolean | null;
  native_checkpoint_supported: boolean;
  native_support_note: string | null;
};
export type AxiomQuantizationEstimate = {
  schema_version: number;
  model: {
    name: string; model_type: "llama" | "mistral" | "qwen2" | "qwen3"; architecture: string;
    estimated_parameter_count: number; parameter_count_method: string;
  };
  request: {
    context_length: number; batch_size: number;
    ram_budget_gib: number | null; vram_budget_gib: number | null;
  };
  assumptions: Record<string, unknown>;
  candidates: AxiomQuantizationCandidate[];
  recommendation: {
    format: "bf16" | "fp16" | "fp32" | null;
    status: "fits_budget" | "no_native_candidate_fits";
    note: string;
  };
  scope_note: string;
};

export type AxiomEngineModel = {
  path: string; name: string; model_type: "llama" | "mistral" | "qwen2" | "qwen3";
  architecture: string; weight_format: string; ready_for_native_attempt: boolean;
};
export type AxiomEngineAdapter = {
  path: string; name: string; weight_format: string;
};
export type AxiomEngineAssets = {
  schema_version: number; models: AxiomEngineModel[]; adapters: AxiomEngineAdapter[];
  truncated: boolean; scope_note: string;
};

export type AxiomEngineAdapterDevice = {
  name: "CPU" | "CUDA" | "ROCm" | "Metal/MPS";
  available: boolean; compatible: boolean;
  status: "compatible" | "incompatible" | "unavailable"; reason: string;
};
export type AxiomEngineAdapterCompatibility = {
  schema_version: number;
  base_model: { name: string; model_type: "llama" | "mistral" | "qwen2" | "qwen3" | null; architecture: string | null };
  adapter: { name: string; rank: number | null; alpha: number | null; target_modules: string[] };
  structurally_compatible: boolean;
  compatible_on_any_device: boolean;
  adapter_tensor_count: number;
  checked_layers: number;
  devices: AxiomEngineAdapterDevice[];
  issues: string[];
  scope_note: string;
};

export type AxiomEngineSystemTelemetry = {
  schema_version: 1; sampled_at: string; sample_type: "point_in_time";
  cpu: { logical_cores: number | null; system_percent: number | null; process_percent: number | null };
  memory: { total_bytes: number | null; available_bytes: number | null; used_bytes: number | null; percent: number | null };
  process: { pid: number; rss_bytes: number | null; threads: number | null };
  devices: Array<{ id: string | null; name: string | null; available: boolean; memory_free_gib: number | null; memory_total_gib: number | null }>;
  loaded_runtime: { loaded: boolean; model: string | null; architecture: string | null; device: string | null; memory_mode: "full" | "layer_streaming" | "disk_streaming" | null; precision: string | null; last_generation: Record<string, string | number> | null; device_allocator: AxiomEngineDeviceMemory | null };
  disks: Array<{ label: string; total_bytes: number; free_bytes: number; used_bytes: number }>;
  privacy_scope: { prompt_content_included: false; generated_content_included: false; model_paths_included: false; environment_values_included: false; credentials_included: false; network_access_performed: false };
  scope_note: string;
};
export type AxiomEngineStatus = {
  engine: string; ready: boolean; packages: Record<string, boolean>; versions: Record<string, string | null>;
  minimum_versions?: Record<string, string>; problems?: string[];
  devices?: Array<{ id: "cpu" | "cuda" | "rocm" | "mps" | "xpu"; name: string; available: boolean; reason: string; memory_free_gib: number | null; memory_total_gib: number | null }>;
  recommended_device?: "cpu" | "cuda" | "rocm" | "mps" | "xpu" | null; loaded_device?: string | null; loaded_memory_mode?: "full" | "layer_streaming" | "disk_streaming" | null; loaded_precision?: string | null;
  loaded: boolean; architecture: string | null; model_loading: "local_files_only" | "local_safetensors_only";
  execution_backend?: string; transformers_generation_used?: false; ollama_used?: false;
  supported_model_types?: string[]; supported_weights?: string[]; remote_code: false;
  setup_note: string; scope_note: string; checked_at: string;
};
export type AxiomEngineSampling = {
  temperature: number; top_p: number; top_k: number; min_p: number;
  repetition_penalty: number; seed: number | null;
};
export type AxiomEngineMemoryStats = {
  layer_weight_tensors_staged: number; layer_weight_bytes_staged: number;
  layer_weight_bytes_returned_to_cpu: number; layer_weight_bytes_released: number;
  disk_layer_weight_loads: number; disk_layer_weight_bytes_read: number;
  kv_cache_layer_offloads: number; kv_cache_bytes_offloaded: number;
};
export type AxiomEngineDeviceMemory = {
  backend: string; current_allocated_bytes: number | null; current_reserved_bytes: number | null;
  peak_allocated_bytes: number | null; peak_reserved_bytes: number | null; peak_increment_bytes: number | null;
  process_rss_bytes: number | null; process_rss_delta_bytes: number | null;
  peak_reset_supported: boolean; device_allocator_metrics_supported: boolean;
};
export type AxiomEngineLiveTelemetry = {
  type: "telemetry";
  metrics: {
    stage: "prefill" | "decode" | "complete" | string;
    prompt_tokens: number; completion_tokens: number; prefill_ms: number; decode_ms: number;
    time_to_first_token_ms: number; generation_tokens_per_second: number; elapsed_ms: number;
    stop_reason?: string;
  };
  memory_mode: "full" | "layer_streaming" | "disk_streaming";
  memory_selection?: { requested: string; selected: "full" | "layer_streaming" | "disk_streaming"; reason: string; estimated_checkpoint_weight_bytes?: number; estimated_kv_cache_bytes?: number; estimated_context_tokens?: number; available_system_ram_bytes?: number | null; available_device_memory_bytes?: number | null; estimates_are_advisory?: boolean };
  memory_stats: AxiomEngineMemoryStats;
  device_memory: AxiomEngineDeviceMemory;
};
export type AxiomEngineResult = {
  engine: "axiom-native-engine"; execution_backend: "pytorch-native-axiom";
  transformers_generation_used: false; ollama_used: false;
  model: string; architecture: string; model_type: "llama" | "mistral" | "qwen2" | "qwen3"; prompt_format?: string; context_management?: { chat_history_supplied: boolean; original_message_count: number; original_message_characters: number; messages_dropped_to_fit_context: number; prompt_tokens_before_token_truncation: number; prompt_tokens_used: number; token_truncation_applied: boolean; system_prompt_may_be_truncated: boolean };
  adapter: string | null; device: string; device_selection?: { requested: string; selected: string; reason: string }; precision?: string; precision_selection?: { requested: string; selected: string }; memory_selection?: { requested: string; selected: "full" | "layer_streaming" | "disk_streaming"; reason: string; estimated_checkpoint_weight_bytes?: number; estimated_kv_cache_bytes?: number; estimated_context_tokens?: number; available_system_ram_bytes?: number | null; available_device_memory_bytes?: number | null; estimates_are_advisory?: boolean }; memory_mode?: "full" | "layer_streaming" | "disk_streaming"; memory_stats?: AxiomEngineMemoryStats; device_memory?: AxiomEngineDeviceMemory; checkpoint_format: "safetensors";
  decoder: "axiom-autoregressive-kv-cache"; attention_backend: string;
  prefill_ms: number; decode_ms: number; time_to_first_token_ms: number; generation_tokens_per_second: number;
  sampling: AxiomEngineSampling; stop_sequences: string[];
  stop_reason: "eos_token" | "max_new_tokens" | "stop_sequence" | "cancelled";
  response: string;
  prompt_tokens: number; completion_tokens: number; elapsed_ms: number; generated_at: string; local_only: true;
};

export type AxiomEngineBatchResult = {
  engine: "axiom-native-engine";
  execution_backend: "pytorch-native-axiom";
  transformers_generation_used: false;
  ollama_used: false;
  model: string;
  architecture: string;
  model_type: "llama" | "mistral" | "qwen2" | "qwen3";
  batch_size: number;
  results: Array<{
    index: number;
    prompt_format: string;
    context_management: {
      prompt_tokens_before_truncation: number;
      prompt_tokens_used: number;
      token_truncation_applied: boolean;
    };
    prompt_tokens: number;
    completion_tokens: number;
    stop_reason: string;
    response: string;
  }>;
  prompt_tokens: number;
  completion_tokens: number;
  elapsed_ms: number;
  memory_mode: "full" | "layer_streaming" | "disk_streaming";
  checkpoint_format: "safetensors" | "gguf";
  local_only: true;
};

export type AxiomEngineStreamEvent =
  | { type: "token"; response: string; completion_tokens: number }
  | { type: "complete"; result: AxiomEngineResult }
  | AxiomEngineLiveTelemetry
  | { type: "error"; code: string; message: string };



export type ModelStudioModel = { id: string; author: string; downloads: number; likes: number; pipeline_tag: string; last_modified: string; tags: string[]; url: string };
export type ModelStudioFile = { filename: string; size_bytes: number | null; quantization: string };
export type ModelStudioStatus = { schema_version: number; ollama_installed: boolean; ollama_endpoint: string; download_limit_bytes: number; gguf_directory: string; native_model_root?: string; native_model_download_limit_bytes?: number; native_model_file_limit_bytes?: number; native_pull_available?: boolean; catalogue: string; network_search_is_explicit: boolean };
export type ModelStudioJob = { schema_version: 1; id: string; status: "queued" | "running" | "completed" | "failed"; operation: string; created_at: string; updated_at: string; started_at?: string | null; finished_at?: string | null; config: Record<string, unknown>; progress_bytes: number; total_bytes: number | null; logs: string[]; error: string | null; result: Record<string, unknown> | null };
export type LLMProvider = "axiom-local" | "ollama-local" | "openai" | "anthropic" | "gemini" | "deepseek" | "moonshot" | "openrouter" | "openai-compatible";
export type LLMProviderInfo = {
  id: LLMProvider; label: string; kind: "local" | "cloud"; requires_api_key: boolean;
  supports_model_listing: boolean; privacy: "local-only" | "remote-consent-required"; note: string;
};
export type LLMProviderCatalog = { schema_version: 1; providers: LLMProviderInfo[]; credentials_persisted: false };
export type LLMModelRequest = { provider: LLMProvider; api_key?: string; base_url?: string };
export type LLMModelResult = {
  provider: LLMProvider; models: string[]; model_details?: Array<{ id: string; model_type?: string | null; architecture?: string | null }>;
  count?: number; credential_storage: "not_saved" | "not_applicable"; execution: "local" | "remote"; note?: string;
};
export type LLMChatMessage = { role: "system" | "user" | "assistant"; content: string };
export type LLMChatRequest = LLMModelRequest & {
  model: string; messages: LLMChatMessage[]; max_tokens: number; temperature: number; consent: boolean;
  top_p?: number; top_k?: number; min_p?: number; repetition_penalty?: number; seed?: number | null; stop_sequences?: string[];
  device?: "auto" | "cpu" | "cuda" | "rocm" | "mps" | "xpu";
  memory_mode?: "auto" | "full" | "layer_streaming" | "disk_streaming";
  precision?: "auto" | "fp32" | "fp16" | "bf16";
};
export type LLMChatResult = {
  provider: LLMProvider; model: string; response: string; prompt_tokens: number | null;
  completion_tokens: number | null; credential_storage: "not_saved" | "not_applicable";
  execution: "local" | "remote"; device?: string | null; generated_at: string; precision?: string | null;
  precision_selection?: { requested: string; selected: string } | null;
  context_management?: { chat_history_supplied: boolean; original_message_count: number; original_message_characters: number; messages_dropped_to_fit_context: number; prompt_tokens_before_token_truncation: number; prompt_tokens_used: number; token_truncation_applied: boolean; system_prompt_may_be_truncated: boolean };
};

export type RemoteModelCatalog = { provider: string; base_url: string; models: string[]; count: number; credential_storage: "not_saved"; note: string };
export type RemoteModelResponse = { provider: string; base_url: string; model: string; response: string; prompt_tokens: number | null; completion_tokens: number | null; credential_storage: "not_saved"; generated_at: string };

export type RuntimeConnector = {
  id: string; name: string; kind: "executable" | "python-package"; present: boolean;
  status: "discovered_not_connected" | "not_detected"; potential_capabilities: string[];
  evidence: string; connection_checked: false; note: string;
};
export type RuntimeConnectorReport = {
  schema_version: 1; discovery_mode: "passive_local_discovery"; checked_at: string;
  connectors: RuntimeConnector[]; scope_note: string;
};
export type AgentProfile = {
  id: string; name: string; version: number; task_capabilities: string[];
  created_at: string; updated_at: string; built_in: boolean;
};
export type AgentProfileReport = {
  schema_version: 1; profiles: AgentProfile[]; capability_catalog: string[]; authorization_note: string;
};

export type WorkspaceModel = {
  id: string;
  name: string;
  source: string;
  format: string;
  parameters?: string | null;
  quantization?: string | null;
  architecture?: string | null;
  license?: string | null;
  tags?: string[];
  favorite?: boolean;
  content_sha256?: string | null;
};

export type WorkspaceModelPage = {
  models: WorkspaceModel[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

export type WorkspaceModelRegistryFilters = {
  search?: string;
  favorite_only?: boolean;
  format?: string;
  tag?: string;
  sort_by?: "name" | "source" | "format" | "parameters" | "quantization" | "favorite";
  sort_order?: "asc" | "desc";
};

export type WorkspaceModelCandidate = {
  path: string;
  lineage: { relation: "source" | "derived_from"; parent_manifest_sha256: string | null };
  format: string;
  has_config: boolean;
  weight_file_count: number;
  weight_size_bytes: number;
};

export type WorkspaceModelInspection = {
  path: string;
  architecture: string | null;
  model_type: string | null;
  parameter_count: number | null;
  model_format: string;
  weight_size_bytes: number;
  has_config: boolean;
  has_tokenizer: boolean;
  has_safetensors: boolean;
  has_gguf: boolean;
  has_pytorch_weights: boolean;
  capabilities: string[];
  estimated_vram_gb: number | null;
  status: string;
};

export type ModelMemoryRange = { low: number; high: number };
export type WorkspaceMemoryFit = {
  status: "likely" | "tight" | "unlikely" | "unknown";
  physical_gb: number | null;
  budget_gb: number | null;
  reserve_percent: number;
  required_gb: ModelMemoryRange | null;
};
export type WorkspaceModelCompatibility = {
  schema_version: number;
  model: {
    id: string; name: string; format: string; parameters: string | null;
    parameter_count_billions: number | null; parameter_basis: string | null;
    quantization: string | null; architecture: string | null;
  };
  hardware: {
    os_name: string; architecture: string; cpu_cores: number; ram_gb: number | null;
    gpu_name: string | null; vram_gb: number | null; vram_probe: string; unified_memory_detected: boolean;
  };
  quantization_bits_per_parameter: { low: number; high: number } | null;
  estimated_weight_gb: ModelMemoryRange | null;
  estimated_runtime_memory_gb: ModelMemoryRange | null;
  system_ram_fit: WorkspaceMemoryFit;
  gpu_memory_fit: WorkspaceMemoryFit;
  format_compatibility: { status: "candidate_not_verified" | "unknown"; likely_runtimes: string[]; guidance: string };
  warnings: string[];
  scope_note: string;
};

export type WorkspaceDataset = {
  path: string;
  size_bytes: number;
};

export type DatasetProfileField = {
  name: string;
  occurrences: number;
  null_count: number;
  null_rate: number;
  types: Record<string, number>;
  possibly_sensitive: boolean;
};

export type DatasetScanLimits = {
  max_bytes: number;
  max_rows: number;
  max_line_bytes: number;
};

export type DatasetManifest = {
  schema_version: number;
  generated_at: string;
  path: string;
  lineage: { relation: "source" | "derived_from"; parent_manifest_sha256: string | null };
  file_name: string;
  format: string;
  file_size_bytes: number;
  bytes_hashed: number;
  full_content_hashed: boolean;
  sha256: string | null;
  partial_sha256: string | null;
  status: "complete" | "partial" | "changed_during_read";
  modified_at: string | null;
  limits: { max_bytes: number };
};

export type DatasetCleaningPreview = {
  path: string;
  file_name: string;
  preview_only: boolean;
  output_written: boolean;
  file_size_bytes: number;
  bytes_scanned: number;
  lines_scanned: number;
  kept_rows: number;
  removed_rows: number;
  removed_empty_rows: number;
  removed_invalid_rows: number;
  removed_non_object_rows: number;
  removed_schema_rows: number;
  removed_duplicate_rows: number;
  overlong_rows: number;
  rewritten_rows: number;
  estimated_output_bytes: number;
  truncated: boolean;
  truncation_reasons: string[];
  changes: Array<{ operation: string; rows: number }>;
  warnings: string[];
  limits: DatasetScanLimits;
};

export type DatasetProfile = {
  file_name: string;
  file_size_bytes: number;
  bytes_scanned: number;
  lines_scanned: number;
  valid_rows: number;
  invalid_rows: number;
  empty_rows: number;
  duplicate_rows: number;
  duplicate_key_summary: {
    fields: string[];
    eligible_rows: number;
    duplicate_rows: number;
    rate: number;
    threshold: number | null;
    threshold_exceeded: boolean;
  };
  overlong_rows: number;
  diagnostics: {
    encoding: string;
    encoding_errors: number;
    malformed_json_rows: number;
    non_object_json_rows: number;
    newline_style: string;
    newline_counts: Record<string, number>;
    delimiter: string;
  };
  text_token_lengths: Record<string, { observations: number; histogram: Record<string, number> }>;
  privacy_review_recommended: boolean;
  possibly_sensitive_fields: string[];
  sensitive_field_rules: string[];
  ignored_sensitive_fields: string[];
  pii_scan: {
    status: "completed" | "not_run";
    opted_in: boolean;
    values_scanned: number;
    matches_by_type: Record<string, number>;
    fields_by_type: Record<string, string[]>;
    long_values_skipped: number;
    rows_truncated_by_scan_limit: number;
    max_text_bytes: number;
    max_values_per_row: number;
  };
  truncated: boolean;
  truncation_reasons: string[];
  fields: DatasetProfileField[];
  warnings: string[];
  limits: DatasetScanLimits;
};

export type DatasetQualityResult = {
  datasets: Array<{ path: string; profile: DatasetProfile }>;
  comparison: {
    dataset_count: number;
    datasets: Array<{ path: string; valid_rows: number; truncated: boolean }>;
    has_differences: boolean;
    fields: Array<{
      name: string;
      present_in: string[];
      missing_from: string[];
      types_by_dataset: Record<string, string[]>;
      type_mismatch: boolean;
      possibly_sensitive: boolean;
    }>;
  };
  checks: Record<string, {
    passed: boolean;
    missing_fields: string[];
    type_mismatches: Array<{ field: string; expected: string; observed: string[] }>;
  }>;
  limits: DatasetScanLimits;
};

export type HardwareSnapshot = {
  os_name: string;
  architecture: string;
  cpu_cores: number;
  ram_gb: number;
  gpu_name: string | null;
  vram_gb: number | null;
  cuda_available: boolean;
};

export type RuntimeSnapshot = {
  status: string;
  hardware: HardwareSnapshot;
};

export type TrainingPlan = {
  method: string;
  precision: string;
  lora_rank: number;
  batch_size: number;
  gradient_accumulation: number;
  sequence_length: number;
  learning_rate: number;
  estimated_vram_gb: number | null;
  fits_hardware: boolean | null;
  reason: string;
};

export type FineTuningRuntime = {
  backend: string;
  can_start: boolean;
  packages: Record<string, boolean>;
  versions?: Record<string, string | null>;
  minimum_versions?: Record<string, string>;
  problems?: string[];
  setup_note?: string;
  model_loading: "local_files_only";
  scope_note: string;
};
export type FineTuningConfig = {
  model_path: string; dataset_path: string; output_dir: string; text_field: string;
  max_steps: number; sequence_length: number; learning_rate: number; lora_rank: number;
};
export type FineTuningJob = {
  schema_version: 1; id: string; status: "queued" | "running" | "completed" | "failed";
  created_at: string; updated_at: string; started_at: string | null; finished_at: string | null;
  config: FineTuningConfig; completed_steps: number; loss: number | null; error: string | null;
  logs: string[]; result?: { output_dir: string; adapter_files: string[]; target_modules: string[]; dataset_rows: number; max_steps: number; device?: string; engine?: string; trainable_parameters?: number };
};

export type WorkspaceEvent = {
  action: string;
  detail: string | null;
  timestamp: string;
};

export type WorkspaceExperiment = {
  schema_version: 1;
  id: string;
  name: string;
  created_at: string;
  model: string | null;
  dataset: string | null;
  notes: string | null;
  tags: string[];
  parameter_billions: number;
  requested_method: string;
  status: "planned";
  hardware?: HardwareSnapshot;
  plan: TrainingPlan;
};

export type WorkspaceExperimentPage = {
  experiments: WorkspaceExperiment[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

export type WorkspaceEvaluationRun = {
  schema_version: 1;
  id: string;
  name: string;
  created_at: string;
  model_label: string | null;
  baseline: boolean;
  status: "complete" | "partial";
  execution_status: "precomputed_outputs_only";
  comparison: {
    baseline_run_id: string;
    metric: string;
    baseline_score: number;
    current_score: number;
    delta: number;
    regression_tolerance: number;
    status: "passed" | "regression";
  } | null;
  output_manifest: { format: string; sha256: string; bytes: number };
  report: {
    path: string;
    file_name: string;
    dataset_manifest: {
      sha256: string | null;
      partial_sha256: string | null;
      bytes_hashed: number;
      file_size_bytes: number;
      full_content_hashed: boolean;
    };
    fields: {
      prediction: string;
      reference: string;
      split: string | null;
      selected_split: string | null;
    };
    counts: {
      lines_scanned: number;
      valid_samples: number;
      invalid_json_rows: number;
      non_object_rows: number;
      overlong_rows: number;
      missing_value_rows: number;
      non_string_value_rows: number;
      filtered_split_rows: number;
    };
    metrics: {
      exact_match: number;
      normalized_exact_match: number;
      token_precision_micro: number;
      token_recall_micro: number;
      token_f1_micro: number;
    };
    confidence_intervals: {
      confidence_level: number;
      method: string;
      exact_match: { low: number; high: number };
      normalized_exact_match: { low: number; high: number };
    };
    truncated: boolean;
    truncation_reasons: string[];
    limits: DatasetScanLimits;
    privacy: { raw_values_returned: false; raw_values_persisted: false };
  };
};

export type WorkspaceEvaluationRunPage = {
  runs: WorkspaceEvaluationRun[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

export type WorkflowNodeType =
  | "dataset.profile"
  | "model.inspect"
  | "training.plan"
  | "evaluation.run"
  | "report.export";

export type WorkflowNode = { id: string; type: WorkflowNodeType };
export type WorkflowEdge = { source: string; target: string };
export type WorkspaceWorkflowVersion = {
  revision: number;
  updated_at: string;
  name: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
};
export type WorkspaceWorkflow = {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  schema_version: number;
  revision: number;
  versions: WorkspaceWorkflowVersion[];
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  execution_status: "not_executable";
};
export type WorkflowTemplate = {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
};

export type WorkflowDryRun = {
  workflow_id: string;
  workflow_revision: number;
  schema_version: number;
  status: "planning_only";
  execution_status: "not_executable";
  executed: false;
  job_created: false;
  node_count: number;
  steps: Array<{ node_id: string; type: WorkflowNodeType; state: "not_run" }>;
  unresolved_inputs: Array<{ node_id: string; type: WorkflowNodeType; reason: string }>;
  resource_summary: {
    available_hardware: HardwareSnapshot;
    estimated_training_plan: TrainingPlan | null;
    estimated_vram_gb: number | null;
    fits_hardware: boolean | null;
    parameters_billions: number | null;
    method: string | null;
  };
  reproducibility_manifest: {
    schema_version: number;
    generated_at: string;
    workflow_id: string;
    workflow_revision: number;
    graph_sha256: string;
    seed: number;
    seed_applied: boolean;
    execution_status: "not_executable";
    software: {
      axiom: string;
      python: string;
      platform: string;
      architecture: string;
      dependencies: Record<string, string | null>;
    };
    hardware: HardwareSnapshot;
    planning_settings: {
      parameter_billions: number | null;
      requested_method: string | null;
      selected_method: string | null;
    };
  };
  warnings: string[];
};

export type McpTool = { name: string; description: string };
export type McpToolProfile = { name: string; allowed_tools: string[] };
export type McpToolPolicy = {
  active_profile: string;
  profiles: McpToolProfile[];
  tools: McpTool[];
};

export type OptimizerProfile = {
  number: number;
  key: string;
  name: string;
  description: string;
  context_tokens: number;
  temperature: number;
  preferred_quantization: string;
};

export type OptimizationPlan = {
  profile: OptimizerProfile;
  model: {
    source: string;
    local_path: string;
    parameter_billions: number | null;
    estimated_fp16_gb: number | null;
    parameter_source: string;
    files: number;
    weights: string[];
  };
  system: {
    os: string;
    arch: string;
    cpu_count: number | null;
    ram_gb: number | null;
    gpu_available: boolean | null;
    gpu_name: string | null;
    vram_gb: number | null;
    disk_free_gb: number | null;
  };
  optimization: {
    target_tokens_per_second: number;
    recommended_quantization: string;
    recommended_format: string;
    runtime_bundle: boolean;
    benchmark_required: boolean;
  };
};

export type HardwareStatus = {
  cpu_name: string;
  ram_gb: number | null;
  gpu_name: string;
  vram_gb: number | null;
};

export type WorkspaceDiagnosticCheck = {
  id: string;
  title: string;
  status: "pass" | "warn" | "fail" | "info";
  severity: "info" | "warning" | "error";
  detail: string;
  remediation: string | null;
};

export type WorkspaceDiagnostics = {
  schema_version: number;
  generated_at: string;
  status: "pass" | "warn" | "fail";
  workspace: { name: string; project_config_present: boolean; is_project: boolean };
  summary: { passed: number; warnings: number; failed: number; informational: number; missing_dependencies: string[] };
  checks: WorkspaceDiagnosticCheck[];
};

export type RoadmapAuditItem = {
  id: string;
  status: "implemented" | "in_progress" | "planned" | "unknown";
  status_label: string;
  title: string;
  group: string;
};

export type RoadmapAudit = {
  schema_version: number;
  generated_at: string;
  source: string;
  integrity: {
    ok: boolean;
    expected_items: number;
    parsed_items: number;
    missing_ids: string[];
    duplicate_ids: string[];
    unexpected_ids: string[];
    unknown_status_ids: string[];
  };
  total: number;
  counts: { implemented: number; in_progress: number; planned: number; unknown: number };
  items: RoadmapAuditItem[];
};

type ApiErrorBody = {
  request_id?: unknown;
  detail?: unknown;
  message?: unknown;
  error?: unknown;
};

function messageFromPayload(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;

  const body = payload as ApiErrorBody;
  for (const candidate of [body.detail, body.message, body.error]) {
    if (typeof candidate === "string" && candidate.trim()) return candidate;
    if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
      const nestedMessage = (candidate as { message?: unknown }).message;
      if (typeof nestedMessage === "string" && nestedMessage.trim()) return nestedMessage;
    }
    if (Array.isArray(candidate)) {
      const messages = candidate.flatMap((item) => {
        if (!item || typeof item !== "object") return [];
        const entry = item as { loc?: unknown; msg?: unknown };
        const location = Array.isArray(entry.loc) ? entry.loc.join(".") : "";
        const message = typeof entry.msg === "string" ? entry.msg : "Invalid request";
        return [location ? `${location}: ${message}` : message];
      });
      if (messages.length) return messages.join("; ");
    }
  }

  return fallback;
}

async function requestJson<T>(path: string, init: RequestInit = {}, timeoutMs = 15_000): Promise<T> {
  const requestId = globalThis.crypto?.randomUUID?.()
    ?? `ui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  headers.set("X-Request-ID", requestId);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers,
    });
    const raw = await response.text();
    let payload: unknown = null;
    if (raw) {
      try {
        payload = JSON.parse(raw) as unknown;
      } catch {
        payload = raw;
      }
    }

    if (!response.ok) {
      const bodyRequestId = payload && typeof payload === "object"
        ? (payload as ApiErrorBody).request_id
        : undefined;
      const responseRequestId = response.headers.get("X-Request-ID")
        || (typeof bodyRequestId === "string" ? bodyRequestId : requestId);
      throw new Error(`${messageFromPayload(payload, `AXIOM request failed (${response.status})`)} (request ID ${responseRequestId})`);
    }

    return payload as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(`AXIOM backend did not respond within 15 seconds (request ID ${requestId}).`, { cause: error });
    }
    if (error instanceof TypeError) {
      throw new Error(`AXIOM backend is unavailable. Start the local API and try again (request ID ${requestId}).`, { cause: error });
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function streamLocalInferenceRequest(
  input: LocalInferenceInput,
  onEvent: (event: LocalInferenceStreamEvent) => void,
  signal: AbortSignal,
): Promise<LocalInferenceResult> {
  const requestId = globalThis.crypto?.randomUUID?.()
    ?? ("ui-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 12));
  const response = await fetch(API_BASE + "/api/inference/generate/stream", {
    method: "POST",
    headers: {
      Accept: "application/x-ndjson",
      "Content-Type": "application/json",
      "X-Request-ID": requestId,
    },
    body: JSON.stringify(input),
    signal,
  });
  if (!response.ok) {
    const raw = await response.text();
    let payload: unknown = raw;
    try { payload = raw ? JSON.parse(raw) as unknown : null; } catch { /* retain a safe fallback */ }
    const rid = response.headers.get("X-Request-ID") || requestId;
    throw new Error(messageFromPayload(payload, "AXIOM stream request failed (" + response.status + ")") + " (request ID " + rid + ")");
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("AXIOM streaming response did not expose a readable body.");

  const decoder = new TextDecoder();
  let pending = "";
  let responseText = "";
  let completed: LocalInferenceResult | null = null;
  const consumeLine = (line: string): LocalInferenceResult | null => {
    const trimmed = line.trim();
    if (!trimmed) return null;
    let event: LocalInferenceStreamEvent;
    try { event = JSON.parse(trimmed) as LocalInferenceStreamEvent; }
    catch { throw new Error("AXIOM returned an invalid newline-delimited inference event."); }
    if (event.type === "error") throw new Error(event.message || "Local Ollama generation failed.");
    if (event.type === "token") {
      if (typeof event.token !== "string") throw new Error("AXIOM returned an invalid token delta.");
      responseText += event.token;
      onEvent(event);
      return null;
    }
    if (event.type !== "complete" || !event.result || typeof event.result !== "object") {
      throw new Error("AXIOM returned an unknown local inference stream event.");
    }
    const result = { ...event.result, response: responseText, request_id: event.result.request_id || requestId };
    onEvent(event);
    return result;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (value) pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      for (const line of lines) completed = consumeLine(line) ?? completed;
      if (done) break;
    }
    if (pending.trim()) completed = consumeLine(pending) ?? completed;
  } finally {
    if (!signal.aborted) {
      try { await reader.cancel(); } catch { /* upstream may already have closed */ }
    }
    reader.releaseLock();
  }
  if (signal.aborted) throw new DOMException("Local inference was stopped.", "AbortError");
  if (!completed) throw new Error("Ollama stream ended without a completion event.");
  return completed;
}


async function streamAxiomEngineRequest(
  input: { model_path: string; adapter_path?: string; prompt: string; max_new_tokens: number; temperature: number; top_p: number; top_k: number; min_p: number; repetition_penalty: number; seed: number | null; stop_sequences: string[]; device: "auto" | "cpu" | "cuda" | "rocm" | "mps" | "xpu"; memory_mode: "auto" | "full" | "layer_streaming" | "disk_streaming"; precision?: "auto" | "fp32" | "fp16" | "bf16" },
  onEvent: (event: AxiomEngineStreamEvent) => void,
  signal: AbortSignal,
): Promise<AxiomEngineResult> {
  const requestId = globalThis.crypto?.randomUUID?.()
    ?? `ui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
  const headers = new Headers({
    Accept: "application/x-ndjson",
    "Content-Type": "application/json",
    "X-Request-ID": requestId,
  });
  const response = await fetch(`${API_BASE}/api/engine/generate/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify(input),
    signal,
  });
  if (!response.ok) {
    const raw = await response.text();
    let payload: unknown = raw;
    try { payload = raw ? JSON.parse(raw) as unknown : null; } catch { /* retain raw safe error text */ }
    const rid = response.headers.get("X-Request-ID") || requestId;
    throw new Error(`${messageFromPayload(payload, `AXIOM stream request failed (${response.status})`)} (request ID ${rid})`);
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("AXIOM streaming response did not expose a readable body.");
  const decoder = new TextDecoder();
  let pending = "";
  let completed: AxiomEngineResult | null = null;

  const consumeLine = (line: string): AxiomEngineResult | null => {
    const trimmed = line.trim();
    if (!trimmed) return null;
    let event: AxiomEngineStreamEvent;
    try {
      event = JSON.parse(trimmed) as AxiomEngineStreamEvent;
    } catch {
      throw new Error("AXIOM returned an invalid newline-delimited streaming event.");
    }
    if (event.type === "error") throw new Error(event.message || "AXIOM Native Engine stream failed.");
    onEvent(event);
    return event.type === "complete" ? event.result : null;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (value) pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      for (const line of lines) completed = consumeLine(line) ?? completed;
      if (done) break;
    }
    if (pending.trim()) completed = consumeLine(pending) ?? completed;
  } finally {
    if (!signal.aborted) {
      try { await reader.cancel(); } catch { /* stream may already be closed */ }
    }
    reader.releaseLock();
  }
  if (!completed) throw new Error("AXIOM stream ended without a final completion event.");
  return completed;
}


export const axiomApi = {
  health: () => requestJson<{ status: string; version: string }>("/api/health"),

  models: async () => {
    const result = await axiomApi.modelsPage();
    return result.models;
  },

  modelsPage: (offset = 0, limit = 100, filters: WorkspaceModelRegistryFilters = {}) => {
    const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (filters.search?.trim()) query.set("search", filters.search.trim());
    if (filters.favorite_only) query.set("favorite_only", "true");
    if (filters.format?.trim() && filters.format.trim().toLowerCase() !== "all") query.set("format", filters.format.trim());
    if (filters.tag?.trim()) query.set("tag", filters.tag.trim());
    query.set("sort_by", filters.sort_by ?? "name");
    query.set("sort_order", filters.sort_order ?? "asc");
    return requestJson<WorkspaceModelPage>(`/api/workspace/models?${query.toString()}`);
  },

  discoverModels: () => requestJson<{ candidates: WorkspaceModelCandidate[]; truncated: boolean }>("/api/workspace/models/discover"),

  inspectModel: (path: string) => requestJson<{ inspection: WorkspaceModelInspection }>("/api/workspace/models/inspect", {
    method: "POST",
    body: JSON.stringify({ path }),
  }),

  modelCompatibility: (id: string) =>
    requestJson<{ report: WorkspaceModelCompatibility }>(`/api/workspace/models/${encodeURIComponent(id)}/compatibility`),

  addModel: (model: Omit<WorkspaceModel, "id">) =>
    requestJson<{ model: WorkspaceModel }>("/api/workspace/models", {
      method: "POST",
      body: JSON.stringify(model),
    }),

  updateModel: (id: string, changes: Partial<Omit<WorkspaceModel, "id" | "name" | "source">>) =>
    requestJson<{ model: WorkspaceModel }>(`/api/workspace/models/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(changes),
    }),

  removeModel: (id: string) =>
    requestJson<{ removed: WorkspaceModel; weights_deleted: false }>(`/api/workspace/models/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),

  exportModels: () => requestJson<{ schema_version: number; models: WorkspaceModel[] }>("/api/workspace/models/export"),

  importModels: (payload: { schema_version: number; models: WorkspaceModel[] }) =>
    requestJson<{ imported: number; models: WorkspaceModel[] }>("/api/workspace/models/import", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  datasets: async () => {
    const result = await requestJson<{ datasets: WorkspaceDataset[] }>("/api/workspace/datasets");
    return result.datasets;
  },

  profileDataset: async (
    path: string,
    limits?: DatasetScanLimits,
    options?: {
      duplicate_key_fields?: string[];
      duplicate_rate_threshold?: number | null;
      text_fields?: string[];
      sensitive_field_rules?: string[];
      ignored_sensitive_fields?: string[];
      scan_pii?: boolean;
    },
  ) => {
    const query = new URLSearchParams({ path });
    if (limits) {
      query.set("max_bytes", String(limits.max_bytes));
      query.set("max_rows", String(limits.max_rows));
      query.set("max_line_bytes", String(limits.max_line_bytes));
    }
    options?.duplicate_key_fields?.forEach((field) => query.append("duplicate_key_fields", field));
    options?.text_fields?.forEach((field) => query.append("text_fields", field));
    options?.sensitive_field_rules?.forEach((rule) => query.append("sensitive_field_rules", rule));
    options?.ignored_sensitive_fields?.forEach((field) => query.append("ignored_sensitive_fields", field));
    if (options?.duplicate_rate_threshold !== undefined && options.duplicate_rate_threshold !== null) {
      query.set("duplicate_rate_threshold", String(options.duplicate_rate_threshold));
    }
    if (options?.scan_pii) query.set("scan_pii", "true");
    const result = await requestJson<{ profile: DatasetProfile }>(
      `/api/workspace/dataset-profile?${query.toString()}`,
    );
    return result.profile;
  },

  datasetManifest: async (path: string, maxBytes: number, parentManifestSha256?: string) => {
    const query = new URLSearchParams({ path, max_bytes: String(maxBytes) });
    if (parentManifestSha256) query.set("parent_manifest_sha256", parentManifestSha256);
    const result = await requestJson<{ manifest: DatasetManifest }>(
      `/api/workspace/dataset-manifest?${query.toString()}`,
    );
    return result.manifest;
  },

  previewDatasetCleaning: async (path: string, limits: DatasetScanLimits) => {
    const result = await requestJson<{ preview: DatasetCleaningPreview }>(
      "/api/workspace/dataset-cleaning-preview",
      { method: "POST", body: JSON.stringify({ path, ...limits }) },
    );
    return result.preview;
  },

  datasetQuality: (input: {
    paths: string[];
    required_fields: string[];
    expected_types: Record<string, string>;
    duplicate_key_fields: string[];
    duplicate_rate_threshold: number | null;
    text_fields: string[];
    sensitive_field_rules: string[];
    ignored_sensitive_fields: string[];
    scan_pii: boolean;
    limits: DatasetScanLimits;
  }) => requestJson<DatasetQualityResult>("/api/workspace/dataset-quality", {
    method: "POST",
    body: JSON.stringify({
      paths: input.paths,
      required_fields: input.required_fields,
      expected_types: input.expected_types,
      duplicate_key_fields: input.duplicate_key_fields,
      duplicate_rate_threshold: input.duplicate_rate_threshold,
      text_fields: input.text_fields,
      sensitive_field_rules: input.sensitive_field_rules,
      ignored_sensitive_fields: input.ignored_sensitive_fields,
      scan_pii: input.scan_pii,
      ...input.limits,
    }),
  }),

  fineTuningRuntime: () => requestJson<FineTuningRuntime>("/api/fine-tuning/runtime"),
  fineTuningJobs: (limit = 20) => requestJson<{ jobs: FineTuningJob[] }>(`/api/fine-tuning/jobs?limit=${limit}`),
  fineTuningJob: (id: string) => requestJson<FineTuningJob>(`/api/fine-tuning/jobs/${encodeURIComponent(id)}`),
  startFineTuningJob: (config: FineTuningConfig) => requestJson<FineTuningJob>("/api/fine-tuning/jobs", {
    method: "POST", body: JSON.stringify(config),
  }),

  trainingPlan: async (parameterBillions: number, method: string) => {
    const query = new URLSearchParams({
      parameter_billions: String(parameterBillions),
      method,
    });
    const result = await requestJson<{ plan: TrainingPlan }>(
      `/api/workspace/training-plan?${query.toString()}`,
      { method: "POST" },
    );
    return result.plan;
  },

  workspaceDiagnostics: () => requestJson<WorkspaceDiagnostics>("/api/diagnostics/workspace"),

  roadmapAudit: () => requestJson<RoadmapAudit>("/api/diagnostics/roadmap"),

  downloadSupportBundle: async (): Promise<Blob> => {
    const requestId = globalThis.crypto?.randomUUID?.()
      ?? `ui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15_000);
    try {
      const response = await fetch(`${API_BASE}/api/diagnostics/support-bundle`, {
        headers: { Accept: "application/zip", "X-Request-ID": requestId },
        signal: controller.signal,
        cache: "no-store",
      });
      if (!response.ok) {
        const raw = await response.text();
        let payload: unknown = raw;
        try { payload = raw ? JSON.parse(raw) as unknown : null; } catch { /* keep safe text fallback */ }
        const responseRequestId = response.headers.get("X-Request-ID") || requestId;
        throw new Error(`${messageFromPayload(payload, `Support bundle request failed (${response.status})`)} (request ID ${responseRequestId})`);
      }
      return await response.blob();
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new Error(`Support bundle request timed out (request ID ${requestId}).`, { cause: error });
      }
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  },

  runtime: () => requestJson<RuntimeSnapshot>("/api/workspace/runtime"),

  events: async () => {
    const result = await requestJson<{ events: WorkspaceEvent[] }>("/api/workspace/events");
    return result.events;
  },

  experiments: async (limit = 100) => {
    const result = await axiomApi.experimentsPage({ limit });
    return result.experiments;
  },

  experimentsPage: async (input: { limit?: number; offset?: number; q?: string; tag?: string } = {}) => {
    const query = new URLSearchParams({
      limit: String(input.limit ?? 100),
      offset: String(input.offset ?? 0),
    });
    if (input.q) query.set("q", input.q);
    if (input.tag) query.set("tag", input.tag);
    return requestJson<WorkspaceExperimentPage>(
      `/api/workspace/experiments?${query.toString()}`,
    );
  },

  createExperiment: (experiment: {
    name: string;
    parameter_billions: number;
    method: string;
    plan_snapshot?: TrainingPlan;
    model?: string;
    dataset?: string;
    notes?: string;
    tags?: string[];
  }) =>
    requestJson<{ experiment: WorkspaceExperiment }>("/api/workspace/experiments", {
      method: "POST",
      body: JSON.stringify(experiment),
    }),

  updateExperiment: (id: string, metadata: { name: string; notes: string | null; tags: string[] }) =>
    requestJson<{ experiment: WorkspaceExperiment; plan_immutable: boolean }>(
      `/api/workspace/experiments/${encodeURIComponent(id)}`,
      { method: "PATCH", body: JSON.stringify(metadata) },
    ),

  deleteExperiment: (id: string) =>
    requestJson<{ deleted: WorkspaceExperiment; run_artifacts_deleted: boolean }>(
      `/api/workspace/experiments/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),

  evaluationRunsPage: (limit = 20, offset = 0) =>
    requestJson<WorkspaceEvaluationRunPage>(`/api/evaluation/runs?${new URLSearchParams({
      limit: String(limit),
      offset: String(offset),
    }).toString()}`),

  createEvaluationRun: (input: {
    path: string;
    prediction_field: string;
    reference_field: string;
    split_field?: string;
    split_value?: string;
    model_label?: string;
    baseline?: boolean;
    baseline_run_id?: string;
    regression_metric?: "exact_match" | "normalized_exact_match" | "token_f1_micro";
    regression_tolerance?: number;
  }) => requestJson<{ run: WorkspaceEvaluationRun }>("/api/evaluation/runs", {
    method: "POST",
    body: JSON.stringify(input),
  }),

  deleteEvaluationRun: (id: string) =>
    requestJson<{ deleted: WorkspaceEvaluationRun }>(`/api/evaluation/runs/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),

  workflows: async () => {
    const result = await requestJson<{ workflows: WorkspaceWorkflow[] }>("/api/workflows");
    return result.workflows;
  },

  workflowTemplates: async () => {
    const result = await requestJson<{ templates: WorkflowTemplate[] }>("/api/workflows/templates");
    return result.templates;
  },

  validateWorkflow: (workflow: {
    name: string;
    nodes: WorkflowNode[];
    edges: WorkflowEdge[];
  }) => requestJson<{ valid: boolean; node_count: number; connection_count: number }>(
    "/api/workflows/validate",
    { method: "POST", body: JSON.stringify(workflow) },
  ),

  dryRunWorkflow: (id: string, input: { parameter_billions: number; method: string; seed: number }) =>
    requestJson<{ dry_run: WorkflowDryRun }>(`/api/workflows/${encodeURIComponent(id)}/dry-run`, {
      method: "POST",
      body: JSON.stringify(input),
    }),

  createWorkflow: (workflow: {
    name: string;
    nodes: WorkflowNode[];
    edges: WorkflowEdge[];
  }) => requestJson<{ workflow: WorkspaceWorkflow }>("/api/workflows", {
    method: "POST",
    body: JSON.stringify(workflow),
  }),

  updateWorkflow: (id: string, workflow: {
    name: string;
    nodes: WorkflowNode[];
    edges: WorkflowEdge[];
  }) => requestJson<{ workflow: WorkspaceWorkflow }>(`/api/workflows/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(workflow),
  }),

  duplicateWorkflow: (id: string) =>
    requestJson<{ workflow: WorkspaceWorkflow }>(`/api/workflows/${encodeURIComponent(id)}/duplicate`, {
      method: "POST",
    }),

  workflowVersions: async (id: string) => {
    const result = await requestJson<{ versions: WorkspaceWorkflowVersion[] }>(
      `/api/workflows/${encodeURIComponent(id)}/versions`,
    );
    return result.versions;
  },

  mcpToolPolicy: () => requestJson<McpToolPolicy>("/api/security/tool-permissions"),

  saveMcpToolProfile: (name: string, allowedTools: string[]) =>
    requestJson<{ profile: McpToolProfile }>(
      `/api/security/tool-profiles/${encodeURIComponent(name)}`,
      { method: "PUT", body: JSON.stringify({ allowed_tools: allowedTools }) },
    ),

  createMcpToolProfile: (name: string) =>
    requestJson<{ profile: McpToolProfile }>(
      `/api/security/tool-profiles/${encodeURIComponent(name)}`,
      { method: "POST" },
    ),

  setActiveMcpProfile: (name: string) =>
    requestJson<{ active_profile: string }>("/api/security/active-profile", {
      method: "PUT",
      body: JSON.stringify({ name }),
    }),

  optimizerProfiles: async () => {
    const result = await requestJson<{ profiles: OptimizerProfile[] }>("/api/optimizer/profiles");
    return result.profiles;
  },

  optimizationPlan: (model: string, profile: number, targetTokensPerSecond: number) =>
    requestJson<OptimizationPlan>("/api/optimizer/plan", {
      method: "POST",
      body: JSON.stringify({
        model,
        profile,
        target_tokens_per_second: targetTokensPerSecond,
      }),
    }),

  optimizationHardware: async (): Promise<HardwareStatus> => {
    const runtime = await requestJson<RuntimeSnapshot>("/api/workspace/runtime");
    const hardware = runtime.hardware;
    return {
      cpu_name: `${hardware.os_name} ${hardware.architecture} · ${hardware.cpu_cores} cores`,
      ram_gb: hardware.ram_gb,
      gpu_name: hardware.gpu_name ?? "CPU only (no GPU detected)",
      vram_gb: hardware.vram_gb,
    };
  },

  inferenceStatus: () => requestJson<LocalInferenceStatus>("/api/inference/status"),
  streamLocalInference: (input: LocalInferenceInput, onEvent: (event: LocalInferenceStreamEvent) => void, signal: AbortSignal) => streamLocalInferenceRequest(input, onEvent, signal),
  startLocalOllama: () => requestJson<{ started: boolean; already_running: boolean; service: LocalInferenceStatus }>("/api/inference/service/start", { method: "POST", body: JSON.stringify({}) }),
  stopLocalOllama: () => requestJson<{ stopped: boolean; service: LocalInferenceStatus }>("/api/inference/service/stop", { method: "POST", body: JSON.stringify({}) }),
  runningOllamaModels: () => requestJson<OllamaRunningModels>("/api/inference/models/running"),
  inspectOllamaModel: (model: string) => requestJson<OllamaModelDetails>("/api/inference/models/inspect", { method: "POST", body: JSON.stringify({ model }) }),
  unloadOllamaModel: (model: string) => requestJson<OllamaModelUnload>("/api/inference/models/unload", { method: "POST", body: JSON.stringify({ model }) }),
  removeOllamaModel: (model: string) => requestJson<OllamaModelRemoval>("/api/inference/models/remove", { method: "POST", body: JSON.stringify({ model }) }),
  startOllamaModelPull: (model: string) => requestJson<OllamaPullJob>("/api/inference/models/pull", { method: "POST", body: JSON.stringify({ model }) }),
  ollamaModelPullStatus: (jobId: string) => requestJson<OllamaPullJob>(`/api/inference/models/pull/${encodeURIComponent(jobId)}`),
  cancelOllamaModelPull: (jobId: string) => requestJson<OllamaPullJob>(`/api/inference/models/pull/${encodeURIComponent(jobId)}/cancel`, { method: "POST", body: JSON.stringify({}) }),
  runLocalInference: (input: LocalInferenceInput) =>
    requestJson<LocalInferenceResult>("/api/inference/generate", {
      method: "POST", body: JSON.stringify(input),
    }, 300_000),
  runLocalInferenceBenchmark: (input: { model: string; prompt: string; num_predict: number; temperature: number; warmup_runs: number; iterations: number; concurrency: number }) =>
    requestJson<LocalInferenceBenchmark>("/api/inference/benchmark", {
      method: "POST", body: JSON.stringify(input),
    }, 300_000),
  estimateAxiomQuantization: (input: {
    model_path: string;
    context_length: number;
    batch_size: number;
    ram_budget_gib: number | null;
    vram_budget_gib: number | null;
  }) => requestJson<AxiomQuantizationEstimate>("/api/engine/quantization/estimate", {
    method: "POST", body: JSON.stringify(input),
  }),
  axiomEngineStatus: () => requestJson<AxiomEngineStatus>("/api/engine/status"),
  engineBackends: (profile: EngineFabricProfile = "balanced") => requestJson<EngineFabricReport>("/api/engine/backends?" + new URLSearchParams({ profile })),
  engineBackendChat: (input: { backend_id: EngineFabricChatResult["backend_id"]; model: string; messages: Array<{ role: "system" | "user" | "assistant"; content: string }>; max_new_tokens: number; temperature: number; top_p: number; top_k: number; min_p: number; repetition_penalty: number; seed: number | null; stop_sequences: string[] }) =>
    requestJson<EngineFabricChatResult>("/api/engine/backends/chat", { method: "POST", body: JSON.stringify(input) }, 300_000),
  engineBackendAutoChat: (input: { profile: EngineFabricProfile; model: string; messages: Array<{ role: "system" | "user" | "assistant"; content: string }>; max_new_tokens: number; temperature: number; top_p: number; top_k: number; min_p: number; repetition_penalty: number; seed: number | null; stop_sequences: string[] }) =>
    requestJson<EngineFabricChatResult>("/api/engine/backends/auto-chat", { method: "POST", body: JSON.stringify(input) }, 300_000),
  engineBackendBenchmark: (input: { backend_id: EngineFabricChatResult["backend_id"]; model: string; prompt: string; max_new_tokens: number; warmup_runs: number; iterations: number; temperature: number; top_p: number; top_k: number; min_p: number; repetition_penalty: number; seed: number | null; stop_sequences: string[] }) =>
    requestJson<EngineFabricBenchmarkResult>("/api/engine/backends/benchmark", { method: "POST", body: JSON.stringify(input) }, 300_000),
  axiomEngineTelemetry: () => requestJson<AxiomEngineSystemTelemetry>("/api/engine/telemetry"),
  axiomEngineModels: () => requestJson<AxiomEngineAssets>("/api/engine/models"),
  checkAxiomEngineAdapterCompatibility: (input: { model_path: string; adapter_path: string }) =>
    requestJson<AxiomEngineAdapterCompatibility>("/api/engine/adapters/compatibility", {
      method: "POST", body: JSON.stringify(input),
    }),
  runAxiomEngine: (input: { model_path: string; adapter_path?: string; prompt: string; max_new_tokens: number; temperature: number; top_p: number; top_k: number; min_p: number; repetition_penalty: number; seed: number | null; stop_sequences: string[]; device?: "auto" | "cpu" | "cuda" | "rocm" | "mps" | "xpu"; memory_mode?: "auto" | "full" | "layer_streaming" | "disk_streaming"; precision?: "auto" | "fp32" | "fp16" | "bf16" }) =>
    requestJson<AxiomEngineResult>("/api/engine/generate", {
      method: "POST", body: JSON.stringify(input),
    }, 300_000),
  runAxiomEngineBatch: (input: {
    model_path: string; adapter_path?: string; prompts: string[];
    max_new_tokens: number; temperature: number; top_p: number; top_k: number;
    min_p: number; repetition_penalty: number; seed: number | null; stop_sequences: string[];
    device: "auto" | "cpu" | "cuda" | "rocm" | "mps" | "xpu";
    memory_mode: "auto" | "full" | "layer_streaming" | "disk_streaming";
    precision?: "auto" | "fp32" | "fp16" | "bf16";
  }, signal?: AbortSignal) =>
    requestJson<AxiomEngineBatchResult>("/api/engine/generate/batch", {
      method: "POST", body: JSON.stringify(input), signal,
    }, 300_000),
  streamAxiomEngine: (
    input: { model_path: string; adapter_path?: string; prompt: string; max_new_tokens: number; temperature: number; top_p: number; top_k: number; min_p: number; repetition_penalty: number; seed: number | null; stop_sequences: string[]; device: "auto" | "cpu" | "cuda" | "rocm" | "mps" | "xpu"; memory_mode: "auto" | "full" | "layer_streaming" | "disk_streaming"; precision?: "auto" | "fp32" | "fp16" | "bf16" },
    onEvent: (event: AxiomEngineStreamEvent) => void,
    signal: AbortSignal,
  ) => streamAxiomEngineRequest(input, onEvent, signal),
  unloadAxiomEngine: () =>
    requestJson<{ schema_version: number; unloaded: boolean; loaded: false; scope_note: string }>("/api/engine/unload", {
      method: "POST", body: JSON.stringify({}),
    }),
  modelStudioStatus: () => requestJson<ModelStudioStatus>("/api/models/studio/status"),
  modelStudioSearch: (q: string, limit = 12) => requestJson<{ provider: string; query: string; count: number; results: ModelStudioModel[]; note: string }>(`/api/models/studio/search?${new URLSearchParams({ q, limit: String(limit) })}`),
  modelStudioFiles: (repoId: string) => requestJson<{ repo_id: string; files: ModelStudioFile[]; count: number; url: string; note: string }>(`/api/models/studio/files?${new URLSearchParams({ repo_id: repoId })}`),
  llmProviders: () => requestJson<LLMProviderCatalog>("/api/llm/providers"),
  llmModels: (input: LLMModelRequest) => requestJson<LLMModelResult>("/api/llm/models", { method: "POST", body: JSON.stringify(input) }),
  llmChat: (input: LLMChatRequest) => requestJson<LLMChatResult>("/api/llm/chat", { method: "POST", body: JSON.stringify(input) }, 300_000),
  testRemoteModelConnection: (input: { base_url: string; api_key: string }) => requestJson<RemoteModelCatalog>("/api/models/studio/remote/models", { method: "POST", body: JSON.stringify(input) }),
  remoteModelChat: (input: { base_url: string; api_key: string; model: string; prompt: string; max_tokens: number; temperature: number }) => requestJson<RemoteModelResponse>("/api/models/studio/remote/chat", { method: "POST", body: JSON.stringify(input) }),
  modelStudioJobs: (limit = 20) => requestJson<{ jobs: ModelStudioJob[] }>(`/api/models/studio/jobs?limit=${limit}`),
  modelStudioJob: (id: string) => requestJson<ModelStudioJob>(`/api/models/studio/jobs/${encodeURIComponent(id)}`),
  startModelStudioJob: (config: { operation: "pull" | "pull_native" | "install_gguf" | "convert_gguf"; [key: string]: unknown }) => requestJson<ModelStudioJob>("/api/models/studio/jobs", { method: "POST", body: JSON.stringify(config) }),

  runtimeConnectors: () => requestJson<RuntimeConnectorReport>("/api/diagnostics/runtime-connectors"),
  runtimeTraces: (limit = 50) => requestJson<RuntimeTraceReport>(`/api/diagnostics/runtime-traces?limit=${Math.min(200, Math.max(1, Math.floor(limit)))}`),
  clearRuntimeTraces: () => requestJson<{ cleared: number; remaining: 0; disk_persistence: false }>("/api/diagnostics/runtime-traces", { method: "DELETE" }),
  agentProfiles: () => requestJson<AgentProfileReport>("/api/agent-profiles"),
  createAgentProfile: (input: { name: string; task_capabilities: string[] }) =>
    requestJson<{ profile: AgentProfile }>("/api/agent-profiles", { method: "POST", body: JSON.stringify(input) }),
  updateAgentProfile: (id: string, taskCapabilities: string[]) =>
    requestJson<{ profile: AgentProfile }>(`/api/agent-profiles/${encodeURIComponent(id)}`, {
      method: "PUT", body: JSON.stringify({ task_capabilities: taskCapabilities }),
    }),
  deleteAgentProfile: (id: string) =>
    requestJson<{ deleted_id: string; remaining: number }>(`/api/agent-profiles/${encodeURIComponent(id)}`, { method: "DELETE" }),
  playgroundRead: (endpoint: PlaygroundEndpoint) => {
    if (!playgroundEndpointPaths.has(endpoint)) {
      throw new Error("This endpoint is not in the read-only API Playground allowlist.");
    }
    return requestJson<unknown>(endpoint, { method: "GET" });
  },
};
