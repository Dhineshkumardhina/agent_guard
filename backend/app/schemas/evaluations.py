"""Pydantic schemas for model evaluations, benchmark summaries, and comparative leaderboards."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from backend.app.utils.pagination import PaginatedResponse


class EvaluationResponse(BaseModel):
    """Schema for individual model evaluation metric results."""
    experiment_id: str = Field(..., description="Evaluation or experiment ID", examples=["Temporal GNN_k1"])
    model: str = Field(..., description="Evaluated model name", examples=["Temporal GNN"])
    model_family: Optional[str] = Field(None, description="Model family categorization (rule_based, classical_ml, sequence, static_gnn, temporal_gnn)")
    dataset_version: str = Field("agentguard_dataset_v1", description="Dataset split evaluated on")
    horizon: int = Field(..., description="Prediction horizon k", examples=[1])
    seed: Optional[int] = Field(42, description="Evaluation seed")
    threshold: Optional[float] = Field(0.5, description="Decision threshold applied")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1: float = Field(..., description="F1 classification score")
    auroc: float = Field(..., description="Area under ROC curve")
    auprc: float = Field(..., description="Area under Precision-Recall curve")
    false_positive_rate: float = Field(..., description="False positive rate (FPR)")
    false_alarm_rate: float = Field(..., description="False alarm rate metric")
    mean_lead_time: float = Field(..., description="Mean lead time in steps before cascading failure")
    median_lead_time: float = Field(..., description="Median lead time in steps before cascading failure")
    successful_warnings: int = Field(..., description="Number of true positive timely warnings issued")
    warnings_per_trajectory: float = Field(..., description="Average warning frequency per trajectory")
    brier_score: Optional[float] = Field(None, description="Brier calibration score")
    ece: Optional[float] = Field(None, description="Expected Calibration Error")
    sample_count: Optional[int] = Field(None, description="Number of evaluation samples")
    positive_count: Optional[int] = Field(None, description="Number of positive cascade samples")
    negative_count: Optional[int] = Field(None, description="Number of negative normal samples")


class EvaluationListResponse(PaginatedResponse[EvaluationResponse]):
    """Paginated collection of evaluation results."""
    pass


class ModelComparisonItem(BaseModel):
    """Normalized metrics entry for cross-model comparison."""
    model_name: str = Field(..., description="Model identifier")
    model_family: str = Field(..., description="Architectural family")
    horizon: int = Field(..., description="Prediction horizon k")
    precision: float = Field(..., description="Raw precision")
    recall: float = Field(..., description="Raw recall")
    f1: float = Field(..., description="Raw F1 score")
    auroc: float = Field(..., description="Raw AUROC")
    auprc: float = Field(..., description="Raw AUPRC")
    false_positive_rate: float = Field(..., description="False positive rate")
    false_alarm_rate: float = Field(..., description="False alarm rate")
    mean_lead_time: float = Field(..., description="Mean lead time")
    median_lead_time: float = Field(..., description="Median lead time")
    successful_warnings: int = Field(..., description="Successful warnings count")
    warnings_per_trajectory: float = Field(..., description="Warnings per trajectory")
    normalized_f1: float = Field(..., description="F1 normalized to [0, 1] relative to best model in group")
    normalized_auroc: float = Field(..., description="AUROC normalized relative to group")
    normalized_lead_time: float = Field(..., description="Lead time normalized relative to group")


class ModelComparisonResponse(BaseModel):
    """Normalized model comparison report across architectures."""
    horizon: int = Field(..., description="Prediction horizon k evaluated")
    total_models: int = Field(..., description="Number of models compared")
    models: List[ModelComparisonItem] = Field(default_factory=list, description="List of comparative model metric entries")
    best_model_by_f1: str = Field(..., description="Top model by F1 score at this horizon")
    best_model_by_auroc: str = Field(..., description="Top model by AUROC at this horizon")
    best_model_by_lead_time: str = Field(..., description="Top model by Mean Lead Time at this horizon")
