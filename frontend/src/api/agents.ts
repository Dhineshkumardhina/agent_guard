import { apiFetch, buildQueryString } from "./client";
import type { AgentResponse, PaginatedResponse } from "../types";

export interface AgentFilters {
  run_id?: string;
  role?: string;
  limit?: number;
  offset?: number;
}

export function getAgents(filters: AgentFilters = {}): Promise<PaginatedResponse<AgentResponse>> {
  const qs = buildQueryString(filters as Record<string, string | number | boolean>);
  return apiFetch<PaginatedResponse<AgentResponse>>(`/api/v1/agents${qs}`);
}

export function getAgentById(agentId: string): Promise<AgentResponse> {
  return apiFetch<AgentResponse>(`/api/v1/agents/${encodeURIComponent(agentId)}`);
}
