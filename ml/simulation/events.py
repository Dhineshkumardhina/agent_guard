"""Simulation Event and Communication Abstractions.

Defines structured message contracts exchanged between agents during simulation runs.
Provides a clean, extensible event interface that Phase 3 (telemetry and fault injection) can build upon.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4

from ml.telemetry.schemas import AgentTelemetryEvent


@dataclass
class SimulationMessage:
    """Structured message and event contract exchanged between simulation agents."""

    source_agent: str
    target_agent: str
    timestamp: float
    message: str
    event_type: str = "message"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Identifiers and execution step
    event_id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str = ""
    step_idx: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metrics and quality scores for Phase 3 telemetry extension
    latency: float = 0.0
    confidence: float = 1.0
    output_quality: float = 1.0
    contradiction_score: float = 0.0
    tool_used: Optional[str] = None
    tool_success: Optional[bool] = None
    tool_error: bool = False
    retry_count: int = 0
    injected_fault: Optional[str] = None
    error_type: Optional[str] = None
    failure_label: int = 0
    downstream_failure: bool = False
    topology: str = "pipeline"

    def __post_init__(self) -> None:
        """Sanitize message content and ensure required defaults."""
        if not self.message:
            self.message = ""
        # Redact any accidental credential leak patterns
        for forbidden in ["api_key", "secret_key", "bearer ", "password", "token="]:
            if forbidden in self.message.lower():
                self.message = "[REDACTED_CREDENTIAL]"
                break

    @property
    def message_length(self) -> int:
        """Computed length of message string."""
        return len(self.message)

    @property
    def token_count(self) -> int:
        """Estimated token count (~4 chars per token)."""
        return max(1, len(self.message) // 4) if self.message else 0

    def to_telemetry_event(self) -> AgentTelemetryEvent:
        """Convert simulation message into pydantic AgentTelemetryEvent."""
        return AgentTelemetryEvent(
            event_id=self.event_id,
            run_id=self.run_id,
            step_idx=self.step_idx,
            timestamp=self.timestamp,
            created_at=self.created_at,
            source_agent=self.source_agent,
            target_agent=self.target_agent,
            event_type=self.event_type,
            message=self.message,
            message_length=self.message_length,
            token_count=self.token_count,
            latency=self.latency,
            confidence=self.confidence,
            output_quality=self.output_quality,
            contradiction_score=self.contradiction_score,
            tool_used=self.tool_used,
            tool_success=self.tool_success,
            tool_error=self.tool_error,
            retry_count=self.retry_count,
            injected_fault=self.injected_fault,
            error_type=self.error_type,
            failure_label=self.failure_label,
            downstream_failure=self.downstream_failure,
            topology=self.topology,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation message to clean serializable dictionary."""
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "timestamp": round(self.timestamp, 4),
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "event_type": self.event_type,
            "message": self.message,
            "message_length": self.message_length,
            "token_count": self.token_count,
            "latency": round(self.latency, 4),
            "confidence": round(self.confidence, 4),
            "output_quality": round(self.output_quality, 4),
            "contradiction_score": round(self.contradiction_score, 4),
            "metadata": self.metadata,
        }
