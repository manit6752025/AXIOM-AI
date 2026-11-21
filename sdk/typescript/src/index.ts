/** Public TypeScript SDK for the AXIOM API v1. */
export { API_CONTRACT_VERSION, DEFAULT_BASE_URL, AxiomClient } from "./client.js";
export {
  AxiomAPIError,
  AxiomConfigurationError,
  AxiomError,
  AxiomNetworkError,
} from "./errors.js";
export type {
  AxiomErrorDetails,
  AxiomErrorEnvelope,
} from "./errors.js";
export type {
  AxiomAPIContractMetadata,
  AxiomClientOptions,
  AxiomEngineStatus,
  AxiomEngineTelemetry,
  AxiomHealth,
  AxiomKnowledgeBase,
  AxiomPluginManifestRegistry,
  AxiomRequestOptions,
  AxiomResourceTelemetryMetrics,
  AxiomResourceTelemetryRecord,
  AxiomTelemetryHistory,
  AxiomTelemetryHistorySettings,
  AxiomWorkspaceBackupReport,
  AxiomWorkspaceBackupRestoreResult,
  AxiomWorkspaceCatalog,
  AxiomWorkspaceSummary,
  JSONPrimitive,
  JSONValue,
} from "./types.js";
