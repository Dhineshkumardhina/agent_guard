/**
 * AgentGuard Frontend TypeScript Definitions matching Backend Pydantic Schemas.
 */

// -------------------------------------------------------------
// Common & Pagination
// -------------------------------------------------------------
export interface ErrorDetail {
  code: string;
  message: string;
  details?: unknown;
}

export interface ErrorEnvelope {
  error: ErrorDetail;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

// -------------------------------------------------------------
// Agents
// -------------------------------------------------------------
export interface AgentResponse {
  agent_id: string;
  name: string;
  role: string;
  description?: string;
  creation_time: string;
  run_id?: string;
  status?: string;
}

// -------------------------------------------------------------
// Runs & Failures
// -------------------------------------------------------------
export interface RunFailureResponse {
  id: string;
  failure_id?: string;
  run_id: string;
  step_idx: number;
  failure_level: number;
  severity?: number;
  originating_agent: string;
  agent_id?: string;
  affected_agents: string[];
  failure_type: string;
  description?: string;
  cascaded?: boolean;
  created_at: string;
}

export interface RunResponse {
  id: string;
  run_id?: string;
  experiment_id?: string;
  dataset_id?: string;
  task: string;
  topology: string;
  status: string;
  agent_count: number;
  duration_seconds: number;
  total_steps?: number;
  has_cascading_failure: boolean;
  cascading_failure_step?: number;
  cascade_step?: number;
  failure_count?: number;
  random_seed: number;
  seed?: number;
  created_at: string;
}

export interface RunDetailResponse extends RunResponse {
  agents: AgentResponse[];
  failures: RunFailureResponse[];
}

// -------------------------------------------------------------
// Events
// -------------------------------------------------------------
export interface EventResponse {
  id: string;
  event_id?: string;
  run_id: string;
  step_idx: number;
  timestamp: number;
  source_agent: string;
  sender?: string;
  target_agent: string;
  receiver?: string;
  event_type: string;
  message_length: number;
  token_count: number;
  latency: number;
  latency_ms?: number;
  confidence: number;
  output_quality: number;
  contradiction_score: number;
  is_contradiction?: boolean;
  tool_used?: string;
  tool_success?: boolean;
  tool_error: boolean;
  retry_count: number;
  injected_fault?: string;
  error_type?: string;
  failure_label: number;
  downstream_failure: boolean;
  summary?: string;
  payload?: any;
  metadata_json: Record<string, unknown>;
  created_at: string;
}

// -------------------------------------------------------------
// Temporal Graphs
// -------------------------------------------------------------
export interface GraphSnapshotSchema {
  snapshot_idx: number;
  timestamp: number;
  step_idx: number;
  num_nodes: number;
  num_edges: number;
  nodes: Array<Record<string, unknown>>;
  edges: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
}

export interface RunGraphResponse {
  run_id: string;
  total_snapshots?: number;
  returned_snapshots?: number;
  timestamps?: number[];
  nodes: Array<{
    id: string;
    role?: string;
    is_active?: boolean;
    event_count?: number;
    message_count?: number;
    error_count?: number;
    retry_count?: number;
    average_latency?: number;
    average_confidence?: number;
    average_output_quality?: number;
    contradiction_rate?: number;
    recent_failure_count?: number;
    [key: string]: unknown;
  }>;
  edges: Array<{
    source: string;
    target: string;
    key?: string;
    interaction_count?: number;
    average_latency?: number;
    average_confidence?: number;
    contradiction_rate?: number;
    error_count?: number;
    [key: string]: unknown;
  }>;
  node_features?: Record<string, unknown>;
  edge_features?: Record<string, unknown>;
  temporal_snapshots: GraphSnapshotSchema[];
  window?: {
    start_time?: number;
    end_time?: number;
    step_idx?: number;
    snapshot_idx?: number;
  };
  density?: number;
  diameter?: number;
  average_clustering?: number;
  is_connected?: boolean;
}

// -------------------------------------------------------------
// Predictions
// -------------------------------------------------------------
export interface PredictionResponse {
  prediction_id: string;
  run_id: string;
  model: string;
  timestamp: number;
  step_idx?: number;
  horizon: number;
  predicted_probability: number;
  predicted_label: number;
  threshold: number;
  actual_outcome?: number;
  risk_level?: string;
  lead_time?: number;
  created_at?: string;
}

// -------------------------------------------------------------
// Experiments
// -------------------------------------------------------------
export interface ExperimentResponse {
  experiment_id: string;
  name: string;
  description?: string;
  model?: string;
  dataset_version?: string;
  configuration: Record<string, unknown>;
  seed: number;
  creation_time: string;
  status: string;
  run_count?: number;
}

// -------------------------------------------------------------
// Evaluations & Comparison
// -------------------------------------------------------------
export interface EvaluationResponse {
  experiment_id: string;
  model: string;
  model_family?: string;
  dataset_version: string;
  horizon: number;
  seed?: number;
  threshold?: number;
  precision: number;
  recall: number;
  f1: number;
  auroc: number;
  auprc: number;
  false_positive_rate: number;
  false_alarm_rate: number;
  mean_lead_time: number;
  median_lead_time: number;
  successful_warnings: number;
  warnings_per_trajectory: number;
  brier_score?: number;
  ece?: number;
  sample_count?: number;
  positive_count?: number;
  negative_count?: number;
}

export interface ModelComparisonItem {
  model_name: string;
  model_family: string;
  horizon: number;
  precision: number;
  recall: number;
  f1: number;
  auroc: number;
  auprc: number;
  false_positive_rate: number;
  false_alarm_rate: number;
  mean_lead_time: number;
  median_lead_time: number;
  successful_warnings: number;
  warnings_per_trajectory: number;
  normalized_f1: number;
  normalized_auroc: number;
  normalized_lead_time: number;
}

export interface ModelComparisonResponse {
  horizon: number;
  total_models: number;
  models: ModelComparisonItem[];
  best_model_by_f1: string;
  best_model_by_auroc: string;
  best_model_by_lead_time: string;
}

// -------------------------------------------------------------
// Ablations
// -------------------------------------------------------------
export interface AblationResponse {
  experiment_id: string;
  parent_experiment_id?: string;
  ablation_name: string;
  removed_component: string;
  baseline_model: string;
  horizon: number;
  seed: number;
  dataset_version: string;
  threshold?: number;
  precision: number;
  recall: number;
  f1: number;
  auroc: number;
  auprc: number;
  false_positive_rate: number;
  false_alarm_rate: number;
  mean_lead_time: number;
  median_lead_time: number;
  successful_warnings: number;
  warnings_per_trajectory: number;
  metrics: Record<string, unknown>;
}

// -------------------------------------------------------------
// Generalization
// -------------------------------------------------------------
export interface GeneralizationGapDetail {
  metric: string;
  id_value: number;
  ood_value: number;
  gap: number;
  pct_change?: number;
  p_value?: number;
  statistically_significant?: boolean;
  interpretation?: string;
}

export interface GeneralizationMetricResponse {
  experiment_id: string;
  dimension: string;
  split_type: string;
  model: string;
  horizon: number;
  seed: number;
  dataset_version: string;
  training_configuration: Record<string, unknown>;
  testing_configuration: Record<string, unknown>;
  metrics: {
    precision?: number;
    recall?: number;
    f1?: number;
    auroc?: number;
    auprc?: number;
    false_alarm_rate?: number;
    mean_lead_time?: number;
    sample_count?: number;
    positive_count?: number;
    negative_count?: number;
    [key: string]: unknown;
  };
  generalization_gap?: number;
  gaps: GeneralizationGapDetail[];
}

// -------------------------------------------------------------
// Explainability
// -------------------------------------------------------------
export interface ImportantFeature {
  feature_name: string;
  importance_score: number;
  direction?: string;
}

export interface ImportantAgent {
  agent_id: string;
  importance_score: number;
  role?: string;
  failure_role?: string;
}

export interface ImportantEdge {
  source_agent: string;
  target_agent: string;
  importance_score: number;
  interaction_count?: number;
  contradiction_rate?: number;
}

export interface ImportantEvent {
  step_idx: number;
  timestamp: number;
  source_agent: string;
  target_agent: string;
  importance_score: number;
  summary?: string;
}

export interface ExplanationResponse {
  explanation_id: string;
  run_id: string;
  sample_id?: string;
  case_type?: string;
  predicted_probability: number;
  prediction_horizon: number;
  predicted_label: number;
  true_label?: number;
  threshold: number;
  explanation_method: string;
  model_name: string;
  model_version: string;
  dataset_version: string;
  important_features: ImportantFeature[];
  important_agents: ImportantAgent[];
  important_edges: ImportantEdge[];
  important_events: ImportantEvent[];
  high_level_summary?: string;
  causality_disclaimer?: string;
}
