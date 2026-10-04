import { apiFetch, buildQueryString } from "./client";
import type { ExperimentResponse, PaginatedResponse } from "../types";

export interface ExperimentFilters {
  status?: string;
  model?: string;
  limit?: number;
  offset?: number;
}

export function getExperiments(filters: ExperimentFilters = {}): Promise<PaginatedResponse<ExperimentResponse>> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<ExperimentResponse>>(`/api/v1/experiments${qs}`);
}

export function getExperimentById(experimentId: string): Promise<ExperimentResponse> {
  return apiFetch<ExperimentResponse>(`/api/v1/experiments/${encodeURIComponent(experimentId)}`);
}
