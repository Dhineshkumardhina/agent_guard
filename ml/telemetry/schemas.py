"""Telemetry and Event Schemas.

Defines the core data contracts for every agent interaction, telemetry record,
fault trace, and failure annotation in the AgentGuard system.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class AgentTelemetryEvent(BaseModel):
    """Structured telemetry event emitted during multi-agent interactions."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str = Field(..., description="Unique trajectory run identifier")
    step_idx: int = Field(..., ge=0, description="Step index within the trajectory")
    timestamp: float = Field(..., description="Relative or epoch timestamp in seconds")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Agents and routing
    source_agent: str = Field(..., description="ID or role of sending agent")
    target_agent: str = Field(..., description="ID or role of receiving agent")
    event_type: str = Field(default="message", description="message, tool_call, validation, delegation")

    # Content attributes (no raw credentials or sensitive keys)
    message: str = Field(default="", description="Sanitized communication payload or summary")
    message_length: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)

    # Performance and quality metrics
    latency: float = Field(default=0.0, ge=0.0, description="Latency in seconds")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Agent self-reported or calibrated confidence")
    output_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Normalized output quality score")
    contradiction_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Semantic contradiction metric with prior context"
    )

    # Tool execution details
    tool_used: Optional[str] = Field(default=None)
    tool_success: Optional[bool] = Field(default=None)
    tool_error: bool = Field(default=False)
    retry_count: int = Field(default=0, ge=0)

    # Fault injection and failure attribution
    injected_fault: Optional[str] = Field(default=None, description="Fault injected at this step, if any")
    error_type: Optional[str] = Field(default=None, description="Categorical error type")
    failure_label: int = Field(
        default=0, description="0: normal, 1: agent failure, 2: interaction failure, 3: cascading failure"
    )
    downstream_failure: bool = Field(
        default=False, description="Flag indicating if this event triggered downstream failures"
    )

    # Topology and structural context
    topology: str = Field(default="pipeline", description="Topology type (pipeline, star, mesh, custom)")
    topology_info: Dict[str, Any] = Field(default_factory=dict)

    # Additional sanitized metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("message", mode="before")
    @classmethod
    def sanitize_message(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        # Ensure no credential-like tokens leak into telemetry
        forbidden = ["api_key", "secret_key", "bearer ", "password", "token="]
        lower_s = s.lower()
        for f in forbidden:
            if f in lower_s:
                s = "[REDACTED_CREDENTIAL]"
                break
        return s

    @field_validator("message_length", mode="before")
    @classmethod
    def compute_length_if_zero(cls, v: int, info) -> int:
        if v and v > 0:
            return v
        msg = info.data.get("message", "") if isinstance(info.data, dict) else ""
        return len(msg)


class TrajectoryMetadata(BaseModel):
    """Metadata summary for an entire simulated or observed multi-agent execution trajectory."""

    run_id: str
    task_type: str
    topology: str
    num_agents: int
    total_events: int
    has_cascading_failure: bool = False
    cascading_failure_step: Optional[int] = None
    injected_faults_count: int = 0
    injected_faults: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    random_seed: int = 42
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
