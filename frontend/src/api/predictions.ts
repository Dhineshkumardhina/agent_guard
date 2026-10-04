import { apiFetch, buildQueryString } from "./client";
import type { PredictionResponse, PaginatedResponse } from "../types";

export interface PredictionFilters {
  run_id?: string;
  model?: string;
  horizon?: number;
  limit?: number;
  offset?: number;
}

export function getPredictions(filters: PredictionFilters = {}): Promise<PaginatedResponse<PredictionResponse>> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<PredictionResponse>>(`/api/v1/predictions${qs}`);
}

export function getPredictionById(predictionId: string): Promise<PredictionResponse> {
  return apiFetch<PredictionResponse>(`/api/v1/predictions/${encodeURIComponent(predictionId)}`);
}
