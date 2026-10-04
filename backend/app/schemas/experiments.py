"""Pydantic schemas for experiments, benchmarks, and model configurations."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from backend.app.utils.pagination import PaginatedResponse


class ExperimentResponse(BaseModel):
    """Schema for research experiment details."""
    experiment_id: str = Field(..., description="Unique experiment ID", examples=["exp_classical_20261003_125823"])
    name: str = Field(..., description="Human-readable experiment title")
    description: Optional[str] = Field(None, description="Experiment description or hypothesis")
    model: Optional[str] = Field(None, description="Primary model architecture evaluated")
    dataset_version: Optional[str] = Field("agentguard_dataset_v1", description="Dataset version utilized")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters and pipeline configuration")
    seed: int = Field(42, description="Reproducibility random seed")
    creation_time: datetime = Field(..., description="Experiment initiation timestamp")
    status: str = Field("completed", description="Experiment lifecycle status (created, running, completed, failed)")
    run_count: Optional[int] = Field(0, description="Total number of trajectory runs in experiment")

    model_config = {"from_attributes": True}


class ExperimentListResponse(PaginatedResponse[ExperimentResponse]):
    """Paginated collection of experiments."""
    pass
