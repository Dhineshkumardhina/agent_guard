"""Pydantic schemas for predictions, early warnings, and model evaluations."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Schema for requesting a failure prediction inference on a run."""
    run_id: str
    step_idx: int
    model_name: str = "temporal_gnn"
    horizon_k: int = 5


class PredictionResponse(BaseModel):
    """Schema for prediction output and early warning status."""
    id: str
    run_id: str
    step_idx: int
    model_name: str
    horizon_k: int
    predicted_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str  # NORMAL, WATCH, HIGH_RISK, PREDICTED_CASCADE
    ground_truth: Optional[int] = None
    lead_time: Optional[int] = None
    explanation: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelEvaluationSummary(BaseModel):
    """Schema for model evaluation metrics on benchmark test sets."""
    model_name: str
    horizon_k: int
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float
    false_alarm_rate: float
    mean_lead_time: float
    median_lead_time: float
    metrics: Dict[str, Any] = Field(default_factory=dict)
