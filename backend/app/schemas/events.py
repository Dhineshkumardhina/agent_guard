"""Pydantic schemas for telemetry events and event streams."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from backend.app.utils.pagination import PaginatedResponse


class EventResponse(BaseModel):
    """Schema for individual telemetry communication event."""
    id: str = Field(..., description="Unique event ID")
    run_id: str = Field(..., description="Simulation run ID")
    step_idx: int = Field(..., description="Discrete step index")
    timestamp: float = Field(..., description="Relative timestamp from run start")
    source_agent: str = Field(..., description="Originating agent ID")
    target_agent: str = Field(..., description="Recipient agent ID")
    event_type: str = Field(..., description="Event/message type")
    message_length: int = Field(0, description="Payload length in characters")
    token_count: int = Field(0, description="Estimated token count")
    latency: float = Field(0.0, description="Event round-trip latency in seconds")
    confidence: float = Field(1.0, description="Self-reported confidence metric")
    output_quality: float = Field(1.0, description="Evaluated output quality score")
    contradiction_score: float = Field(0.0, description="Semantic contradiction score")
    tool_used: Optional[str] = Field(None, description="Tool invoked if any")
    tool_success: Optional[bool] = Field(None, description="Whether tool execution succeeded")
    tool_error: bool = Field(False, description="Whether an error occurred")
    retry_count: int = Field(0, description="Retry attempt count")
    injected_fault: Optional[str] = Field(None, description="Synthetic fault type if injected")
    error_type: Optional[str] = Field(None, description="Error classification if present")
    failure_label: int = Field(0, description="Ground truth failure label")
    downstream_failure: bool = Field(False, description="Whether event triggered downstream failure")
    metadata_json: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary event metadata")
    created_at: datetime = Field(..., description="Record creation timestamp")

    model_config = {"from_attributes": True}


class EventListResponse(PaginatedResponse[EventResponse]):
    """Paginated collection of telemetry events."""
    pass
