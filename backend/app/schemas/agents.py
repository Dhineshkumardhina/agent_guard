"""Pydantic schemas for Agent entities."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from backend.app.utils.pagination import PaginatedResponse


class AgentResponse(BaseModel):
    """Schema for returning public agent information."""
    agent_id: str = Field(..., description="Unique agent identifier", examples=["planner_1"])
    name: str = Field(..., description="Human-readable agent name", examples=["planner_1"])
    role: str = Field(..., description="Agent role/specialization", examples=["planner"])
    description: Optional[str] = Field(None, description="Descriptive profile of the agent")
    creation_time: datetime = Field(..., description="Agent creation timestamp")
    run_id: Optional[str] = Field(None, description="Simulation run in which agent participated")
    status: Optional[str] = Field("active", description="Agent lifecycle status (active, degraded, failed)")

    model_config = {"from_attributes": True}


class AgentListResponse(PaginatedResponse[AgentResponse]):
    """Paginated collection of agents."""
    pass
