"""Pydantic schemas for ablation experiments and component attribution."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from backend.app.utils.pagination import PaginatedResponse


class AblationResponse(BaseModel):
    """Schema for individual ablation study condition result."""
    experiment_id: str = Field(..., description="Unique ablation experiment ID", examples=["abl_full_temporal_gnn_k1_s42"])
    parent_experiment_id: Optional[str] = Field(None, description="Reference baseline experiment ID")
    ablation_name: str = Field(..., description="Ablation study designator", examples=["Full Temporal GNN"])
    removed_component: str = Field(..., description="Specific component or feature source ablated", examples=["None (Reference)"])
    baseline_model: str = Field(..., description="Base model architecture", examples=["temporal_gnn"])
    horizon: int = Field(..., description="Prediction horizon k", examples=[1])
    seed: int = Field(..., description="Random seed")
    dataset_version: str = Field("agentguard_dataset_v1", description="Dataset version utilized")
    threshold: Optional[float] = Field(0.5, description="Evaluation threshold")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1: float = Field(..., description="F1 classification score")
    auroc: float = Field(..., description="AUROC metric")
    auprc: float = Field(..., description="AUPRC metric")
    false_positive_rate: float = Field(0.0, description="False positive rate")
    false_alarm_rate: float = Field(0.0, description="False alarm rate")
    mean_lead_time: float = Field(0.0, description="Mean lead time in steps")
    median_lead_time: float = Field(0.0, description="Median lead time in steps")
    successful_warnings: int = Field(0, description="Number of timely warnings")
    warnings_per_trajectory: float = Field(0.0, description="Average warnings per trajectory")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Complete metrics payload")


class AblationListResponse(PaginatedResponse[AblationResponse]):
    """Paginated collection of ablation study records."""
    pass
