/** Structured errors returned by the AXIOM HTTP API. */
export class AxiomError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

export class AxiomConfigurationError extends AxiomError {}

export class AxiomNetworkError extends AxiomError {
  readonly requestId: string | null;

  constructor(message: string, requestId: string | null = null, options?: ErrorOptions) {
    super(message, options);
    this.requestId = requestId;
  }
}

export interface AxiomErrorDetails {
  code?: string;
  message?: string;
  details?: unknown;
}

export interface AxiomErrorEnvelope {
  request_id?: string;
  error?: AxiomErrorDetails;
  detail?: string;
}

export class AxiomAPIError extends AxiomError {
  readonly statusCode: number;
  readonly code: string | null;
  readonly requestId: string | null;
  readonly details: unknown;

  constructor(
    message: string,
    statusCode: number,
    code: string | null,
    requestId: string | null,
    details: unknown,
  ) {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}
