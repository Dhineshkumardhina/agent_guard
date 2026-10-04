import { apiFetch } from "./client";

export interface HealthResponse {
  status: string;
  service: string;
  project: string;
  version: string;
  environment: string;
}

export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}
