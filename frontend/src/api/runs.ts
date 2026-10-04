import { apiFetch, buildQueryString } from "./client";
import type {
  RunResponse,
  RunDetailResponse,
  EventResponse,
  RunFailureResponse,
  RunGraphResponse,
  PredictionResponse,
  ExplanationResponse,
  PaginatedResponse,
} from "../types";

export interface RunFilters {
  task?: string;
  topology?: string;
  status?: string;
  agent_count?: number;
  start_date?: string;
  end_date?: string;
  limit?: number;
  offset?: number;
}

export interface GraphFilters {
  start_time?: number;
  end_time?: number;
  step_idx?: number;
  snapshot_idx?: number;
}

export function getRuns(filters: RunFilters = {}): Promise<PaginatedResponse<RunResponse>> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<RunResponse>>(`/api/v1/runs${qs}`);
}

export function getRunById(runId: string): Promise<RunDetailResponse> {
  return apiFetch<RunDetailResponse>(`/api/v1/runs/${encodeURIComponent(runId)}`);
}

export function getRunEvents(runId: string, pagination: { limit?: number; offset?: number } = {}): Promise<PaginatedResponse<EventResponse>> {
  const qs = buildQueryString(pagination as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<EventResponse>>(`/api/v1/runs/${encodeURIComponent(runId)}/events${qs}`);
}

export function getRunFailures(runId: string, pagination: { limit?: number; offset?: number } = {}): Promise<PaginatedResponse<RunFailureResponse>> {
  const qs = buildQueryString(pagination as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<RunFailureResponse>>(`/api/v1/runs/${encodeURIComponent(runId)}/failures${qs}`);
}

export function getRunGraph(runId: string, filters: GraphFilters = {}): Promise<RunGraphResponse> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<RunGraphResponse>(`/api/v1/runs/${encodeURIComponent(runId)}/graph${qs}`);
}

export function getRunPredictions(
  runId: string,
  params: { model?: string; horizon?: number; limit?: number; offset?: number } = {}
): Promise<PaginatedResponse<PredictionResponse>> {
  const qs = buildQueryString(params as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<PredictionResponse>>(`/api/v1/runs/${encodeURIComponent(runId)}/predictions${qs}`);
}

export function getRunExplanations(
  runId: string,
  params: { limit?: number; offset?: number } = {}
): Promise<PaginatedResponse<ExplanationResponse>> {
  const qs = buildQueryString(params as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<ExplanationResponse>>(`/api/v1/runs/${encodeURIComponent(runId)}/explanations${qs}`);
}
