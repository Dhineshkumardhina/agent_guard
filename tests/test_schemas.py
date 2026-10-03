"""Tests for telemetry event schemas and sanitization."""

import pytest
from pydantic import ValidationError
from ml.telemetry.schemas import AgentTelemetryEvent, TrajectoryMetadata


def test_telemetry_event_creation(sample_telemetry_event):
    """Verify event instantiation and required field integrity."""
    assert sample_telemetry_event.run_id == "run_test_001"
    assert sample_telemetry_event.step_idx == 0
    assert sample_telemetry_event.confidence == 0.88
    assert sample_telemetry_event.failure_label == 0


def test_telemetry_credential_sanitization():
    """Ensure sensitive credentials or tokens are automatically redacted."""
    event = AgentTelemetryEvent(
        run_id="run_leak_test",
        step_idx=1,
        timestamp=1.2,
        source_agent="planner",
        target_agent="coder",
        message="Use this bearer secret: Bearer eyJhbGciOi...",
    )
    assert event.message == "[REDACTED_CREDENTIAL]"


def test_telemetry_bounds_validation():
    """Ensure invalid confidence or contradiction out of [0, 1] triggers validation error."""
    with pytest.raises(ValidationError):
        AgentTelemetryEvent(
            run_id="run_err",
            step_idx=0,
            timestamp=0.1,
            source_agent="a",
            target_agent="b",
            confidence=1.5,  # Out of bounds
        )


def test_trajectory_metadata():
    """Verify trajectory summary metadata schema."""
    meta = TrajectoryMetadata(
        run_id="run_001",
        task_type="research",
        topology="pipeline",
        num_agents=5,
        total_events=30,
        has_cascading_failure=True,
        cascading_failure_step=18,
    )
    assert meta.has_cascading_failure is True
    assert meta.cascading_failure_step == 18
