import { apiFetch, buildQueryString } from "./client";
import type { ExplanationResponse, PaginatedResponse } from "../types";

export interface ExplanationFilters {
  run_id?: string;
  model?: string;
  limit?: number;
  offset?: number;
}

export function getExplanations(filters: ExplanationFilters = {}): Promise<PaginatedResponse<ExplanationResponse>> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<ExplanationResponse>>(`/api/v1/explanations${qs}`);
}

export function getExplanationById(explanationId: string): Promise<ExplanationResponse> {
  return apiFetch<ExplanationResponse>(`/api/v1/explanations/${encodeURIComponent(explanationId)}`);
}
