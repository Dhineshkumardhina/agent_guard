import { apiFetch, buildQueryString } from "./client";
import type { AblationResponse, PaginatedResponse } from "../types";

export interface AblationFilters {
  model?: string;
  horizon?: number;
  seed?: number;
  limit?: number;
  offset?: number;
}

export function getAblations(filters: AblationFilters = {}): Promise<PaginatedResponse<AblationResponse>> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<AblationResponse>>(`/api/v1/ablations${qs}`);
}

export function getAblationById(experimentId: string): Promise<AblationResponse> {
  return apiFetch<AblationResponse>(`/api/v1/ablations/${encodeURIComponent(experimentId)}`);
}
