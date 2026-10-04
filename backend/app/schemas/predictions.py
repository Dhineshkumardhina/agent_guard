"""Pydantic schemas for failure predictions and early warning inferences."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from backend.app.utils.pagination import PaginatedResponse


class PredictionResponse(BaseModel):
    """Schema for individual failure prediction inference."""
    prediction_id: str = Field(..., description="Unique prediction ID", examples=["pred_temporal_gnn_k1_001"])
    run_id: str = Field(..., description="Simulation run ID", examples=["run_0000_agentguard_dataset_v1"])
    model: str = Field(..., description="Model architecture/name", examples=["temporal_gnn"])
    timestamp: float = Field(..., description="Simulation timestamp at inference time")
    step_idx: Optional[int] = Field(None, description="Discrete step index")
    horizon: int = Field(..., description="Prediction horizon k (steps into future)", examples=[1])
    predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted failure risk probability")
    predicted_label: int = Field(..., description="Binary prediction classification (0: normal, 1: failure risk)")
    threshold: float = Field(0.5, description="Decision threshold applied")
    actual_outcome: Optional[int] = Field(None, description="Ground truth failure outcome if available (0 or 1)")
    risk_level: Optional[str] = Field(None, description="Categorical risk tier (NORMAL, WATCH, HIGH_RISK, PREDICTED_CASCADE)")
    lead_time: Optional[int] = Field(None, description="Lead time in steps before cascade onset")
    created_at: Optional[datetime] = Field(None, description="Prediction timestamp")

    model_config = {"from_attributes": True}


class PredictionListResponse(PaginatedResponse[PredictionResponse]):
    """Paginated collection of predictions."""
    pass
