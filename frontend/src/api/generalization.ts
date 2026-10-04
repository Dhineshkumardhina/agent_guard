import { apiFetch, buildQueryString } from "./client";
import type { GeneralizationMetricResponse, PaginatedResponse } from "../types";

export interface GeneralizationFilters {
  dimension?: string;
  model?: string;
  horizon?: number;
  split_type?: string;
  limit?: number;
  offset?: number;
}

export function getGeneralization(filters: GeneralizationFilters = {}): Promise<PaginatedResponse<GeneralizationMetricResponse>> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<GeneralizationMetricResponse>>(`/api/v1/generalization${qs}`);
}

export function getGeneralizationById(experimentId: string): Promise<GeneralizationMetricResponse> {
  return apiFetch<GeneralizationMetricResponse>(`/api/v1/generalization/${encodeURIComponent(experimentId)}`);
}
