"""Telemetry and Event Schemas.

Defines the core data contracts for every agent interaction, telemetry record,
fault trace, and failure annotation in the AgentGuard system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Union, Set
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class TelemetryEventType(str, Enum):
    """Supported event categories in the multi-agent telemetry stream."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRY = "retry"
    ERROR = "error"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    DELEGATION = "delegation"
    FAILURE = "failure"


SUPPORTED_EVENT_TYPES: Set[str] = {e.value for e in TelemetryEventType}


class AgentTelemetryEvent(BaseModel):
    """Structured telemetry event emitted during multi-agent interactions."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: Optional[str] = Field(default=None, description="Unique trajectory run identifier")
    step_idx: Optional[int] = Field(default=0, ge=0, description="Step index within the trajectory")
    timestamp: Optional[float] = Field(default=None, description="Relative or epoch timestamp in seconds")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Agents and routing
    source_agent: str = Field(..., description="ID or role of sending agent")
    target_agent: str = Field(..., description="ID or role of receiving agent")
    event_type: str = Field(default="message", description="Type category of event")

    # Content attributes
    message: Optional[str] = Field(default="", description="Sanitized communication payload or summary")
    message_length: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)

    # Performance and quality metrics
    latency: Optional[float] = Field(default=0.0, ge=0.0, description="Latency in seconds")
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Agent self-reported or calibrated confidence")
    output_quality: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Normalized output quality score")
    contradiction_score: Optional[float] = Field(
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
    def compute_length_if_zero(cls, v: Any, info) -> int:
        if v is not None and int(v) > 0:
            return int(v)
        msg = info.data.get("message", "") if isinstance(info.data, dict) else ""
        return len(str(msg)) if msg else 0

    @field_validator("token_count", mode="before")
    @classmethod
    def compute_token_count_if_zero(cls, v: Any, info) -> int:
        if v is not None and int(v) > 0:
            return int(v)
        msg = info.data.get("message", "") if isinstance(info.data, dict) else ""
        length = len(str(msg)) if msg else 0
        return max(1, length // 4) if length > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry event to dictionary."""
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "event_type": self.event_type,
            "message": self.message,
            "message_length": self.message_length,
            "latency": round(self.latency, 4) if self.latency is not None else None,
            "token_count": self.token_count,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "tool_used": self.tool_used,
            "tool_success": self.tool_success,
            "retry_count": self.retry_count,
            "contradiction_score": round(self.contradiction_score, 4) if self.contradiction_score is not None else None,
            "output_quality": round(self.output_quality, 4) if self.output_quality is not None else None,
            "error_type": self.error_type,
            "failure_label": self.failure_label,
            "downstream_failure": self.downstream_failure,
            "metadata": self.metadata,
        }


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
