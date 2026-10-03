"""Tests for Telemetry Collection, Rolling Features, and Data Leakage Prevention (Phase 3)."""

from pathlib import Path
import json
import pytest

from ml.telemetry.schemas import (
    AgentTelemetryEvent,
    TelemetryEventType,
    SUPPORTED_EVENT_TYPES,
)
from ml.telemetry.collector import TelemetryCollector
from ml.telemetry.features import (
    compute_event_features,
    compute_rolling_features,
)
from backend.app.database.models import Run, Event as DBEvent


def test_event_validation_all_fields():
    """Verify all 19 mandatory fields are supported, including nullable fields."""
    event = AgentTelemetryEvent(
        event_id="ev_full_001",
        run_id="run_full_001",
        step_idx=0,
        timestamp=0.15,
        source_agent="researcher_1",
        target_agent="analyst_1",
        event_type="message",
        message="Preliminary empirical observations",
        message_length=35,
        latency=0.25,
        token_count=9,
        confidence=0.92,
        tool_used=None,  # Nullable
        tool_success=None,  # Nullable
        retry_count=0,
        contradiction_score=0.04,
        output_quality=0.95,
        error_type=None,  # Nullable
        failure_label=0,
        downstream_failure=False,
    )

    collector = TelemetryCollector(strict_event_types=True)
    assert collector.validate_event(event) is True


def test_all_eleven_event_types_supported():
    """Verify all 11 required event types are recognized and supported."""
    required_types = [
        "agent_start",
        "agent_end",
        "message",
        "tool_call",
        "tool_result",
        "retry",
        "error",
        "timeout",
        "validation",
        "delegation",
        "failure",
    ]
    collector = TelemetryCollector(strict_event_types=True)

    for etype in required_types:
        assert etype in SUPPORTED_EVENT_TYPES
        ev = AgentTelemetryEvent(
            run_id="run_types_test",
            step_idx=0,
            timestamp=0.1,
            source_agent="agent_a",
            target_agent="agent_b",
            event_type=etype,
            message=f"Event of type {etype}",
        )
        assert collector.validate_event(ev) is True


def test_event_validation_errors():
    """Verify validation errors for invalid fields or types."""
    collector = TelemetryCollector(strict_event_types=True)

    # Empty source_agent
    ev_bad_src = AgentTelemetryEvent(
        run_id="run_1",
        source_agent="",
        target_agent="agent_b",
    )
    with pytest.raises(ValueError, match="valid string source_agent"):
        collector.validate_event(ev_bad_src)

    # Empty target_agent
    ev_bad_tgt = AgentTelemetryEvent(
        run_id="run_1",
        source_agent="agent_a",
        target_agent="",
    )
    with pytest.raises(ValueError, match="valid string target_agent"):
        collector.validate_event(ev_bad_tgt)

    # Invalid event_type under strict mode
    ev_bad_type = AgentTelemetryEvent(
        run_id="run_1",
        source_agent="agent_a",
        target_agent="agent_b",
        event_type="unknown_event_type_xyz",
    )
    with pytest.raises(ValueError, match="Invalid event_type"):
        collector.validate_event(ev_bad_type)


def test_event_ordering_and_timestamp_handling():
    """Verify events are assigned timestamps and buffered in strict chronological order."""
    collector = TelemetryCollector(active_run_id="run_order_test")

    # Ingest events out-of-order in timestamp
    e3 = AgentTelemetryEvent(source_agent="a", target_agent="b", timestamp=3.0, step_idx=2)
    e1 = AgentTelemetryEvent(source_agent="a", target_agent="b", timestamp=1.0, step_idx=0)
    e2 = AgentTelemetryEvent(source_agent="a", target_agent="b", timestamp=2.0, step_idx=1)
    e_no_time = AgentTelemetryEvent(source_agent="a", target_agent="b", timestamp=None)

    collector.receive_event(e3)
    collector.receive_event(e1)
    collector.receive_event(e2)
    ingested_no_time = collector.receive_event(e_no_time)

    # Missing timestamp should have been auto-populated monotonically
    assert ingested_no_time.timestamp is not None
    assert ingested_no_time.timestamp >= 3.0

    # Events in collector must be sorted chronologically
    events = collector.events
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)
    assert events[0].timestamp == 1.0
    assert events[1].timestamp == 2.0
    assert events[2].timestamp == 3.0


def test_telemetry_database_persistence(db_session):
    """Verify TelemetryCollector persists events into database."""
    collector = TelemetryCollector(active_run_id="run_persist_test")

    # Ensure parent run exists in database
    run_record = Run(
        id="run_persist_test",
        task_type="research",
        topology="pipeline",
        num_agents=2,
        duration_seconds=1.5,
    )
    db_session.add(run_record)
    db_session.commit()

    collector.receive_event({
        "source_agent": "agent_1",
        "target_agent": "agent_2",
        "message": "Telemetry message 1",
        "timestamp": 0.5,
        "latency": 0.2,
        "confidence": 0.95,
    })
    collector.receive_event({
        "source_agent": "agent_2",
        "target_agent": "agent_1",
        "message": "Telemetry message 2",
        "timestamp": 1.0,
        "latency": 0.3,
        "confidence": 0.98,
    })

    persisted_count = collector.persist_to_db(db_session)
    assert persisted_count == 2

    # Query back from DB
    queried = db_session.query(DBEvent).filter(DBEvent.run_id == "run_persist_test").all()
    assert len(queried) == 2
    assert queried[0].source_agent == "agent_1"
    assert queried[1].source_agent == "agent_2"


def test_telemetry_export_formats(tmp_path: Path):
    """Verify exporting raw events to JSONL and CSV."""
    collector = TelemetryCollector(active_run_id="run_export_test")
    collector.receive_event({
        "source_agent": "planner",
        "target_agent": "coder",
        "timestamp": 0.1,
        "message": "Code specification",
        "confidence": 0.94,
    })
    collector.receive_event({
        "source_agent": "coder",
        "target_agent": "verifier",
        "timestamp": 0.4,
        "message": "Implementation delivery",
        "confidence": 0.97,
    })

    # 1. JSONL Export
    jsonl_file = tmp_path / "events.jsonl"
    collector.export(jsonl_file, format="jsonl")
    assert jsonl_file.exists()
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["source_agent"] == "planner"
    assert lines[1]["target_agent"] == "verifier"

    # 2. CSV Export
    csv_file = tmp_path / "events.csv"
    collector.export(csv_file, format="csv")
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        csv_content = f.read()
    assert "source_agent,target_agent" in csv_content
    assert "planner,coder" in csv_content

    # 3. Parquet Export without pyarrow triggers clean ImportError
    try:
        import pyarrow
        # If pyarrow happens to be installed, test actual export
        parquet_file = tmp_path / "events.parquet"
        collector.export(parquet_file, format="parquet")
        assert parquet_file.exists()
    except ImportError:
        with pytest.raises(ImportError, match="Parquet export requires 'pyarrow'"):
            collector.export(tmp_path / "events.parquet", format="parquet")


def test_rolling_features_calculation():
    """Verify mathematical correctness of all 7 rolling feature utilities."""
    events = [
        AgentTelemetryEvent(
            run_id="run_feat", step_idx=0, timestamp=1.0,
            source_agent="A", target_agent="B", latency=0.2, confidence=0.9,
            tool_error=False, retry_count=0, contradiction_score=0.1,
        ),
        AgentTelemetryEvent(
            run_id="run_feat", step_idx=1, timestamp=2.0,
            source_agent="A", target_agent="B", latency=0.4, confidence=0.8,
            tool_error=True, retry_count=1, contradiction_score=0.3,
        ),
        AgentTelemetryEvent(
            run_id="run_feat", step_idx=2, timestamp=3.0,
            source_agent="B", target_agent="C", latency=0.6, confidence=0.7,
            tool_error=False, retry_count=0, contradiction_score=0.2,
        ),
    ]

    features = compute_rolling_features(events, window_size=2)
    assert len(features) == 3

    # Step 0 (window: [e0])
    f0 = features[0]
    assert f0["rolling_error_rate"] == 0.0
    assert f0["rolling_retry_rate"] == 0.0
    assert f0["rolling_latency"] == 0.2
    assert f0["rolling_confidence"] == 0.9
    assert f0["interaction_frequency"] == 1.0  # Pair (A, B) is 1/1
    assert f0["contradiction_rate"] == 0.1

    # Step 1 (window: [e0, e1])
    f1 = features[1]
    assert f1["rolling_error_rate"] == 0.5  # 1 error out of 2
    assert f1["rolling_retry_rate"] == 0.5  # 1 retry out of 2
    assert f1["rolling_latency"] == 0.3     # (0.2 + 0.4) / 2
    assert f1["rolling_confidence"] == 0.85 # (0.9 + 0.8) / 2
    assert f1["interaction_frequency"] == 1.0  # Pair (A, B) is 2/2
    assert f1["contradiction_rate"] == 0.2  # (0.1 + 0.3) / 2

    # Step 2 (window: [e1, e2], window_size=2)
    f2 = features[2]
    assert f2["rolling_error_rate"] == 0.5  # e1 has error, e2 doesn't -> 1/2
    assert f2["rolling_retry_rate"] == 0.5  # e1 has retry -> 1/2
    assert f2["rolling_latency"] == 0.5     # (0.4 + 0.6) / 2
    assert f2["rolling_confidence"] == 0.75 # (0.8 + 0.7) / 2
    assert f2["interaction_frequency"] == 0.5 # Pair (B, C) is 1/2 in window [e1(A,B), e2(B,C)]
    assert f2["contradiction_rate"] == 0.25 # (0.3 + 0.2) / 2


def test_data_leakage_protection_invariance():
    """Verify that features at timestamp t are strictly invariant to any future events (t' > t)."""
    # Base sequence of 3 events
    history = [
        AgentTelemetryEvent(run_id="run_leak", step_idx=0, timestamp=1.0, source_agent="A", target_agent="B", latency=0.1, confidence=0.9),
        AgentTelemetryEvent(run_id="run_leak", step_idx=1, timestamp=2.0, source_agent="B", target_agent="C", latency=0.2, confidence=0.8),
        AgentTelemetryEvent(run_id="run_leak", step_idx=2, timestamp=3.0, source_agent="C", target_agent="D", latency=0.3, confidence=0.7),
    ]

    # Compute features at step 1
    feats_before = compute_event_features(current_idx=1, history=history, window_size=3)

    # Now append radical future events (errors, extreme latencies, zero confidence) after step 1
    perturbed_history = [
        history[0],
        history[1],
        AgentTelemetryEvent(run_id="run_leak", step_idx=2, timestamp=3.0, source_agent="C", target_agent="D", latency=99.9, confidence=0.0, tool_error=True),
        AgentTelemetryEvent(run_id="run_leak", step_idx=3, timestamp=4.0, source_agent="D", target_agent="E", latency=99.9, confidence=0.0, failure_label=3),
    ]

    # Re-compute features at step 1
    feats_after = compute_event_features(current_idx=1, history=perturbed_history, window_size=3)

    # Features at step 1 MUST BE IDENTICAL — absolutely zero future leakage!
    assert feats_before == feats_after
    assert feats_after["rolling_error_rate"] == 0.0
    assert feats_after["rolling_latency"] == 0.15
    assert feats_after["rolling_confidence"] == 0.85
