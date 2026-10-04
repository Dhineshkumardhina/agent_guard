"""Explainability Data Schema - Phase 15.

Defines structured contracts for all explainability layers:
1. FeatureAttribution (Global and local feature importances)
2. AgentAttribution (Node-level multi-agent importance)
3. InteractionAttribution (Edge-level communication importance)
4. TemporalEventAttribution (Historical event importance)
5. PerturbationResult (Counterfactual sensitivity analysis)
6. CaseStudyExplanation (End-to-end case-based explanation object)
7. GlobalImportanceReport (Dataset-wide aggregated rankings)
8. ExplanationStabilityReport (Cross-seed and cross-horizon consistency)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json


CAUSALITY_DISCLAIMER: str = (
    "NOTICE: These explanations reflect model-internal associations, gradient attributions, "
    "and counterfactual sensitivity measurements. They do NOT establish empirical or physical "
    "causality between observed features and multi-agent system failures."
)


@dataclass
class FeatureAttribution:
    """Attribution score for a single behavioral, graph, or temporal feature."""
    feature_name: str
    feature_group: str  # "node", "edge", "temporal", "agent_behavior", "interaction", "reliability"
    importance_score: float  # Absolute or relative importance
    signed_contribution: float  # Positive: increases risk, Negative: decreases risk
    baseline_value: Optional[float] = None
    perturbed_value: Optional[float] = None
    rank: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentAttribution:
    """Attribution score indicating the relative contribution of an agent to predicted risk."""
    agent_id: str
    agent_role: str  # "planner", "researcher", "analyst", "coder", "verifier", etc.
    attribution_score: float  # Normalized [0, 1] relative importance
    delta_risk: float  # P(full) - P(without_agent)
    signed_contribution: float  # Positive: increases risk, Negative: mitigates risk
    event_count: int = 0
    message_count: int = 0
    contradiction_rate: float = 0.0
    retry_count: int = 0
    rank: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionAttribution:
    """Attribution score for a directed communication channel between two agents."""
    source_agent: str
    target_agent: str
    attribution_score: float  # Normalized importance
    delta_risk: float  # P(full) - P(without_edge)
    interaction_count: int = 0
    average_latency: float = 0.0
    contradiction_rate: float = 0.0
    retry_count: int = 0
    signed_contribution: float = 0.0
    rank: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalEventAttribution:
    """Attribution score for a specific chronological interaction event preceding prediction time t."""
    event_id: str
    timestamp: float
    time_before_prediction: float  # t_pred - t_event
    event_type: str  # "message", "tool_call", "retry", "contradiction", etc.
    source_agent: str
    target_agent: str
    importance_score: float  # Leave-one-out delta or attribution
    signed_contribution: float  # Impact on risk
    details: Dict[str, Any] = field(default_factory=dict)
    rank: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbationResult:
    """Results of a controlled counterfactual perturbation / sensitivity analysis."""
    perturbation_type: str  # e.g. "remove_interaction", "reduce_contradiction_50pct", "modify_confidence_plus20"
    target_entity: str  # Agent ID, edge, or feature name
    original_probability: float
    perturbed_probability: float
    delta_probability: float  # perturbed - original
    percent_change: float
    direction: str  # "risk_increased", "risk_decreased", "neutral"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaseStudyExplanation:
    """Complete, self-contained case-based explanation for a single prediction point."""
    explanation_id: str
    case_type: str  # "true_positive_early", "true_positive_late", "false_positive", "false_negative", "high_risk_no_cascade", "low_risk_success"
    sample_id: str
    run_id: str
    topology: str
    task_type: str
    prediction_timestamp: float
    actual_failure_timestamp: Optional[float]
    prediction_horizon: int
    predicted_probability: float
    predicted_label: int
    true_label: int
    threshold: float
    model_name: str
    model_version: str
    dataset_version: str
    seed: int

    # Attributions
    important_agents: List[AgentAttribution] = field(default_factory=list)
    important_interactions: List[InteractionAttribution] = field(default_factory=list)
    important_features: List[FeatureAttribution] = field(default_factory=list)
    important_events: List[TemporalEventAttribution] = field(default_factory=list)
    positive_contributions: List[str] = field(default_factory=list)
    negative_contributions: List[str] = field(default_factory=list)
    perturbation_results: List[PerturbationResult] = field(default_factory=list)

    # Telemetry and signals
    high_level_summary: str = ""
    causality_disclaimer: str = CAUSALITY_DISCLAIMER
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_formatted_text(self) -> str:
        """Format explanation into the standard research case-study presentation."""
        top_agents = "\n".join([f"  - {a.agent_role.capitalize()} ({a.agent_id}): score={a.attribution_score:.3f}" for a in self.important_agents[:3]]) or "  - None"
        top_interactions = "\n".join([f"  - {i.source_agent} -> {i.target_agent}: score={i.attribution_score:.3f}" for i in self.important_interactions[:3]]) or "  - None"
        top_features = "\n".join([f"  - {f.feature_name} ({f.feature_group}): score={f.importance_score:.3f}" for f in self.important_features[:4]]) or "  - None"
        top_events = "\n".join([f"  - t={e.timestamp:.2f}s [{e.event_type}] {e.source_agent}->{e.target_agent} (impact={e.importance_score:.3f})" for e in self.important_events[:3]]) or "  - None"
        pos_contrib = "\n".join([f"  - {c}" for c in self.positive_contributions[:3]]) or "  - None"
        neg_contrib = "\n".join([f"  - {c}" for c in self.negative_contributions[:3]]) or "  - None"

        return f"""================================================================================
EXPLANATION REPORT: {self.case_type.upper().replace('_', ' ')}
Sample ID: {self.sample_id} | Run: {self.run_id} | Horizon K={self.prediction_horizon}
--------------------------------------------------------------------------------
Model: {self.model_name} (v{self.model_version}) | Seed: {self.seed}
Predicted Risk: {self.predicted_probability:.4f} (Threshold: {self.threshold:.2f} -> Class: {self.predicted_label})
Ground Truth: Class {self.true_label} (Failure Time: {self.actual_failure_timestamp if self.actual_failure_timestamp is not None else 'None'})

High-Risk Contributing Factors (+):
{pos_contrib}

Risk-Mitigating Factors (-):
{neg_contrib}

Key Contributing Agents:
{top_agents}

Key Interaction Paths:
{top_interactions}

Key Salient Features:
{top_features}

Relevant Recent Chronological Events (t <= {self.prediction_timestamp:.2f}s):
{top_events}

{self.causality_disclaimer}
================================================================================"""


@dataclass
class GlobalImportanceReport:
    """Aggregated dataset-wide importance rankings across features, agents, interactions, and events."""
    model_name: str
    horizon: int
    dataset_version: str
    sample_count: int
    feature_importance: List[Dict[str, Any]]  # [{"feature": str, "importance": float, "group": str}]
    agent_role_importance: List[Dict[str, Any]]  # [{"role": str, "frequency": int, "mean_importance": float}]
    interaction_importance: List[Dict[str, Any]]  # [{"path": str, "mean_importance": float}]
    event_type_importance: List[Dict[str, Any]]  # [{"event_type": str, "mean_importance": float}]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationStabilityReport:
    """Quantifies the consistency and stability of explanations across seeds and horizons."""
    model_name: str
    seeds_evaluated: List[int]
    horizons_evaluated: List[int]
    feature_rank_correlation_seeds: float  # Mean pairwise Spearman's rho across seeds
    feature_top_k_jaccard_seeds: float  # Mean Jaccard overlap of top-5 features across seeds
    agent_rank_correlation_seeds: float  # Mean pairwise Spearman's rho for agent rankings
    agent_top_k_jaccard_seeds: float
    feature_rank_correlation_horizons: float  # Correlation across horizons
    agent_rank_correlation_horizons: float
    is_stable: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
