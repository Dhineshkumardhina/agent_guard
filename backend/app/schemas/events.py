"""Pydantic schemas for event models and API endpoints."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from ml.telemetry.schemas import AgentTelemetryEvent


class EventCreate(AgentTelemetryEvent):
    """Schema for recording a new telemetry event."""
    pass


class EventResponse(BaseModel):
    """Schema for serialized event responses."""
    id: str
    run_id: str
    step_idx: int
    timestamp: float
    source_agent: str
    target_agent: str
    event_type: str
    message_length: int
    token_count: int
    latency: float
    confidence: float
    output_quality: float
    contradiction_score: float
    tool_used: Optional[str] = None
    tool_success: Optional[bool] = None
    tool_error: bool = False
    retry_count: int = 0
    injected_fault: Optional[str] = None
    error_type: Optional[str] = None
    failure_label: int = 0
    downstream_failure: bool = False
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}
