"""Pydantic schemas for experiments, datasets, and benchmark suites."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from ml.config.experiment_config import ExperimentConfig


class ExperimentCreate(BaseModel):
    """Schema for initializing a new research experiment."""
    name: str
    description: Optional[str] = None
    config: ExperimentConfig


class ExperimentResponse(BaseModel):
    """Schema for returning experiment details."""
    id: str
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime
    run_count: Optional[int] = 0

    model_config = {"from_attributes": True}
