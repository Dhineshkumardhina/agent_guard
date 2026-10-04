import { apiFetch, buildQueryString } from "./client";
import type { EvaluationResponse, ModelComparisonResponse, PaginatedResponse } from "../types";

export interface EvaluationFilters {
  model?: string;
  horizon?: number;
  experiment?: string;
  dataset_version?: string;
  limit?: number;
  offset?: number;
}

export function getEvaluations(filters: EvaluationFilters = {}): Promise<PaginatedResponse<EvaluationResponse>> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<EvaluationResponse>>(`/api/v1/evaluations${qs}`);
}

export function getEvaluationById(experimentId: string): Promise<EvaluationResponse> {
  return apiFetch<EvaluationResponse>(`/api/v1/evaluations/${encodeURIComponent(experimentId)}`);
}

export function getModelComparison(horizon: number = 1): Promise<ModelComparisonResponse> {
  return apiFetch<ModelComparisonResponse>(`/api/v1/evaluations/comparison?horizon=${horizon}`);
}
