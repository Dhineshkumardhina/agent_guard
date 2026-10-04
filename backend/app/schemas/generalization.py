"""Pydantic schemas for generalization and robustness evaluation."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from backend.app.utils.pagination import PaginatedResponse


class GeneralizationGapDetail(BaseModel):
    """Generalization performance gap metric between ID and OOD evaluation."""
    metric: str = Field(..., description="Target metric (f1, auroc, auprc, lead_time)")
    id_value: float = Field(..., description="In-distribution metric value")
    ood_value: float = Field(..., description="Out-of-distribution metric value")
    gap: float = Field(..., description="Performance gap (ID - OOD)")
    pct_change: Optional[float] = Field(None, description="Percentage degradation or change")
    p_value: Optional[float] = Field(None, description="Statistical significance p-value")
    statistically_significant: Optional[bool] = Field(None, description="Whether gap exceeds significance alpha")
    interpretation: Optional[str] = Field(None, description="Qualitative interpretation")


class GeneralizationMetricResponse(BaseModel):
    """Schema for individual generalization experiment condition result."""
    experiment_id: str = Field(..., description="Generalization experiment ID", examples=["G1"])
    dimension: str = Field(..., description="Generalization dimension (agent_count, topology, task_type, compound)", examples=["agent_count"])
    split_type: str = Field(..., description="Evaluation split type (in_distribution, out_of_distribution)")
    model: str = Field(..., description="Evaluated model name", examples=["temporal_gnn"])
    horizon: int = Field(..., description="Prediction horizon k", examples=[1])
    seed: int = Field(42, description="Evaluation seed")
    dataset_version: str = Field("agentguard_generalization_v1", description="Dataset version utilized")
    training_configuration: Dict[str, Any] = Field(default_factory=dict, description="Training configuration attributes")
    testing_configuration: Dict[str, Any] = Field(default_factory=dict, description="Testing configuration attributes")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Evaluation performance metrics")
    generalization_gap: Optional[float] = Field(None, description="Primary F1 generalization gap (ID - OOD)")
    gaps: List[GeneralizationGapDetail] = Field(default_factory=list, description="Detailed gaps across multiple metrics")


class GeneralizationListResponse(PaginatedResponse[GeneralizationMetricResponse]):
    """Paginated collection of generalization experiment records."""
    pass
