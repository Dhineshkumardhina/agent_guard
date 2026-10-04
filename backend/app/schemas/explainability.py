"""Pydantic schemas for explainability, feature attributions, and case studies."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from backend.app.utils.pagination import PaginatedResponse


class ImportantAgent(BaseModel):
    """Agent node importance attribution."""
    agent_id: str = Field(..., description="Agent node identifier")
    importance_score: float = Field(..., description="Normalized importance weight [0, 1]")
    role: Optional[str] = Field(None, description="Agent role")
    failure_role: Optional[str] = Field(None, description="Observed failure behavior role")


class ImportantEdge(BaseModel):
    """Interaction edge importance attribution."""
    source_agent: str = Field(..., description="Source agent of communication")
    target_agent: str = Field(..., description="Target agent of communication")
    importance_score: float = Field(..., description="Normalized importance weight [0, 1]")
    interaction_count: Optional[int] = Field(None, description="Observed interaction count")
    contradiction_rate: Optional[float] = Field(None, description="Observed contradiction rate")


class ImportantFeature(BaseModel):
    """Behavioral feature attribution score."""
    feature_name: str = Field(..., description="Telemetry or behavioral feature name")
    importance_score: float = Field(..., description="Attribution magnitude or contribution")
    direction: Optional[str] = Field(None, description="Direction of risk impact (risk_increasing, risk_reducing)")


class ImportantEvent(BaseModel):
    """Salient telemetry event attribution."""
    step_idx: int = Field(..., description="Discrete step index")
    timestamp: float = Field(..., description="Simulation timestamp")
    source_agent: str = Field(..., description="Source agent")
    target_agent: str = Field(..., description="Target agent")
    importance_score: float = Field(..., description="Event saliency score")
    summary: Optional[str] = Field(None, description="Event description or message summary")


class ExplanationResponse(BaseModel):
    """Schema for prediction explanation and risk attribution."""
    explanation_id: str = Field(..., description="Unique explanation identifier", examples=["exp_true_positive_early_run_0031_agentguard_generalization_v1_s0"])
    run_id: str = Field(..., description="Associated simulation run ID", examples=["run_0031_agentguard_generalization_v1"])
    sample_id: Optional[str] = Field(None, description="Sample ID evaluated")
    case_type: Optional[str] = Field(None, description="Case study classification (true_positive, false_negative, etc.)")
    predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted cascading failure probability")
    prediction_horizon: int = Field(..., description="Horizon k evaluated")
    predicted_label: int = Field(..., description="Binary predicted classification (0: normal, 1: cascade warning)")
    true_label: Optional[int] = Field(None, description="Ground truth outcome label")
    threshold: float = Field(0.5, description="Decision threshold applied")
    explanation_method: str = Field("Integrated Gradients + Subgraph Attribution", description="Explainability attribution method")
    model_name: str = Field(..., description="Model architecture evaluated")
    model_version: str = Field("v1", description="Model checkpoint or architecture version")
    dataset_version: str = Field("agentguard_generalization_v1", description="Dataset version utilized")
    important_features: List[Dict[str, Any]] = Field(default_factory=list, description="Top ranked feature attributions")
    important_agents: List[Dict[str, Any]] = Field(default_factory=list, description="Top ranked agent node attributions")
    important_edges: List[Dict[str, Any]] = Field(default_factory=list, description="Top ranked interaction edge attributions")
    important_events: List[Dict[str, Any]] = Field(default_factory=list, description="Salient chronological event attributions")
    high_level_summary: Optional[str] = Field(None, description="Human-readable synthesis explaining why failure risk increased")
    causality_disclaimer: Optional[str] = Field(
        "Attributions denote predictive statistical associations within the temporal graph, not proven physical causality.",
        description="Scientific causality disclaimer",
    )


class ExplanationListResponse(PaginatedResponse[ExplanationResponse]):
    """Paginated collection of explanation reports."""
    pass
