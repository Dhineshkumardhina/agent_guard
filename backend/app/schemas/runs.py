"""Pydantic schemas for simulation trajectories (runs) and agents."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from backend.app.schemas.events import EventResponse


class AgentSchema(BaseModel):
    """Schema for individual agent details in a run."""
    id: str
    run_id: str
    agent_name: str
    role: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class RunCreate(BaseModel):
    """Schema for initiating or registering a trajectory run."""
    id: Optional[str] = None
    experiment_id: Optional[str] = None
    dataset_id: Optional[str] = None
    task_type: str
    topology: str
    num_agents: int
    random_seed: int = 42


class RunResponse(BaseModel):
    """Schema for trajectory run responses."""
    id: str
    experiment_id: Optional[str] = None
    dataset_id: Optional[str] = None
    task_type: str
    topology: str
    num_agents: int
    duration_seconds: float
    has_cascading_failure: bool
    cascading_failure_step: Optional[int] = None
    random_seed: int
    created_at: datetime
    agents: List[AgentSchema] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class RunDetailResponse(RunResponse):
    """Detailed trajectory response including events and interactions."""
    events: List[EventResponse] = Field(default_factory=list)
