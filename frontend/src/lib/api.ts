export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new ApiError(response.status, `${path} failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export type HealthResponse = {
  status: string;
  phase: string;
  app: string;
  disclaimer: string;
};

export type PrivacyResponse = {
  raw_patient_images_centralized: boolean;
  federated_training: boolean;
  differential_privacy_enabled: boolean;
  dp_epsilon_target: number;
  dp_delta: number;
  perfect_privacy_claimed: boolean;
};

export type ExperimentItem = {
  id: string;
  name: string;
  status: string;
};

export type ExperimentsResponse = {
  matrix: ExperimentItem[];
  unrun_display: string;
};

export type FederatedStatusResponse = {
  algorithm: string;
  raw_images_on_wire: boolean;
  training: string;
  clients: Array<{ id: string; name: string; data_root: string }>;
};

export type ModelsResponse = {
  results: string;
};

export const api = {
  health: () => getJson<HealthResponse>("/health"),
  privacy: () => getJson<PrivacyResponse>("/api/privacy"),
  experiments: () => getJson<ExperimentsResponse>("/api/experiments"),
  federatedStatus: () => getJson<FederatedStatusResponse>("/api/federated/status"),
  federatedRounds: () => getJson<{ rounds: unknown[]; status: string }>("/api/federated/rounds"),
  models: () => getJson<ModelsResponse>("/api/models"),
  audit: () => getJson<{ events: unknown[]; note: string }>("/api/audit"),
};
