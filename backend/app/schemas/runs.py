"""Pydantic schemas for simulation runs and failures."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from backend.app.schemas.agents import AgentResponse
from backend.app.utils.pagination import PaginatedResponse


class RunFailureResponse(BaseModel):
    """Schema for individual failure occurrences during a run."""
    id: str = Field(..., description="Unique failure record identifier")
    run_id: str = Field(..., description="Parent simulation run ID")
    step_idx: int = Field(..., description="Execution step where failure manifested")
    failure_level: int = Field(..., description="Failure severity level (1, 2, or 3)")
    originating_agent: str = Field(..., description="First agent manifesting failure")
    affected_agents: List[str] = Field(default_factory=list, description="Agents impacted downstream")
    failure_type: str = Field(..., description="Taxonomy classification of failure")
    description: Optional[str] = Field(None, description="Detailed failure description")
    created_at: datetime = Field(..., description="Timestamp recorded")

    model_config = {"from_attributes": True}


class RunResponse(BaseModel):
    """Schema for simulation run overview."""
    id: str = Field(..., description="Unique run identifier", examples=["run_0001_agentguard_generalization_v1"])
    experiment_id: Optional[str] = Field(None, description="Associated experiment ID")
    dataset_id: Optional[str] = Field(None, description="Associated dataset ID")
    task: str = Field(..., description="Task domain (research, coding, analysis, planning)", examples=["research"])
    topology: str = Field(..., description="Agent communication topology", examples=["pipeline"])
    status: str = Field(..., description="Run outcome status (completed, failed)", examples=["completed"])
    agent_count: int = Field(..., description="Number of agents participating in the run", examples=[5])
    duration_seconds: float = Field(..., description="Total execution duration in seconds")
    has_cascading_failure: bool = Field(..., description="Whether a cascading failure occurred")
    cascading_failure_step: Optional[int] = Field(None, description="Step at which cascade triggered")
    random_seed: int = Field(..., description="Reproducibility random seed")
    created_at: datetime = Field(..., description="Run generation timestamp")

    model_config = {"from_attributes": True}


class RunDetailResponse(RunResponse):
    """Detailed simulation run with participating agents and failure metadata."""
    agents: List[AgentResponse] = Field(default_factory=list, description="Agents participating in this run")
    failures: List[RunFailureResponse] = Field(default_factory=list, description="Failures manifested in this run")


class RunListResponse(PaginatedResponse[RunResponse]):
    """Paginated collection of simulation runs."""
    pass


class RunFailureListResponse(PaginatedResponse[RunFailureResponse]):
    """Paginated collection of failure events."""
    pass
