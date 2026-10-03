"""Tests for Temporal Graph Construction, Snapshots, Features, and Future Leakage Prevention (Phase 5)."""

from pathlib import Path
import pytest

from ml.telemetry.schemas import AgentTelemetryEvent
from ml.graph.graph_snapshot import GraphSnapshot
from ml.graph.node_features import extract_node_features
from ml.graph.edge_features import extract_edge_features
from ml.graph.graph_window import WindowStrategy, filter_events_by_window
from ml.graph.graph_builder import TemporalGraphBuilder
from ml.graph.serializers import serialize_snapshot, deserialize_snapshot, save_snapshot, load_snapshot


def test_1_empty_event_sequence():
    """Verify builder handles an empty event sequence gracefully."""
    builder = TemporalGraphBuilder()
    snap = builder.build_snapshot(events=[], timestamp=0.0, run_id="empty_run")
    assert snap.num_nodes == 0
    assert snap.num_edges == 0
    assert snap.timestamp == 0.0


def test_2_single_agent():
    """Verify snapshot containing a single agent without interactions."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(
            run_id="run_single",
            step_idx=0,
            timestamp=0.1,
            source_agent="solo_agent",
            target_agent="",
            event_type="agent_start",
        )
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.1, run_id="run_single")
    assert "solo_agent" in snap.nodes
    assert snap.num_edges == 0
    node_feat = snap.get_node_features("solo_agent")
    assert node_feat["event_count"] == 1
    assert node_feat["error_count"] == 0


def test_3_two_agent_interaction():
    """Verify directed interaction between two agents."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(
            run_id="run_two",
            step_idx=0,
            timestamp=0.2,
            source_agent="agent_a",
            target_agent="agent_b",
            event_type="message",
            message="Initial query",
            latency=0.15,
            confidence=0.9,
        )
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.2)
    assert snap.num_nodes == 2
    assert snap.num_edges == 1
    assert snap.has_edge("agent_a", "agent_b") is True
    assert snap.has_edge("agent_b", "agent_a") is False

    edge_feats = snap.get_edge_features("agent_a", "agent_b")
    assert edge_feats["interaction_count"] == 1
    assert edge_feats["average_latency"] == 0.15
    assert edge_feats["average_confidence"] == 0.9


def test_4_multiple_interactions():
    """Verify multi-agent pipeline workflow graph construction."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.1, step_idx=0),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.3, step_idx=1),
        AgentTelemetryEvent(source_agent="C", target_agent="D", timestamp=0.6, step_idx=2),
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.6)
    assert snap.num_nodes == 4
    assert snap.num_edges == 3
    assert snap.has_edge("A", "B") is True
    assert snap.has_edge("B", "C") is True
    assert snap.has_edge("C", "D") is True


def test_5_repeated_interactions():
    """Verify multiple interactions between the same pair accumulate features and preserve timestamps."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.1, step_idx=0, latency=0.1, message="Hello"),
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.5, step_idx=1, latency=0.3, message="World!!"),
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.5)
    edge_feats = snap.get_edge_features("A", "B")
    assert edge_feats["interaction_count"] == 2
    assert edge_feats["average_latency"] == 0.2  # (0.1 + 0.3) / 2
    assert edge_feats["interaction_timestamps"] == [0.1, 0.5]


def test_6_directed_interactions():
    """Verify directed edge asymmetry: A -> B does not imply B -> A."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.2, step_idx=0),
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.2)
    assert snap.has_edge("A", "B") is True
    assert snap.has_edge("B", "A") is False


def test_7_different_event_types():
    """Verify mapping of different event types (tool_call, error, timeout, retry) into features."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.1, step_idx=0, event_type="message"),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.2, step_idx=1, event_type="tool_call", tool_used="query_db"),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.3, step_idx=2, event_type="timeout", tool_error=True),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.4, step_idx=3, event_type="retry", retry_count=2),
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.4)
    node_b = snap.get_node_features("B")
    assert node_b["tool_call_count"] >= 1
    assert node_b["timeout_count"] >= 1
    assert node_b["retry_count"] >= 2
    assert node_b["error_count"] >= 1


def test_8_temporal_ordering():
    """Verify out-of-order events are strictly ordered by timestamp before building snapshot."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.6, step_idx=2),
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.1, step_idx=0),
        AgentTelemetryEvent(source_agent="A", target_agent="C", timestamp=0.4, step_idx=1),
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.5)
    # At t=0.5, event at t=0.6 must NOT be included
    assert snap.has_edge("A", "B") is True
    assert snap.has_edge("A", "C") is True
    assert snap.has_edge("B", "C") is False


def test_9_snapshot_generation():
    """Verify build_snapshots_over_run produces a sequence of chronological snapshots."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.1, step_idx=0),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.3, step_idx=1),
    ]
    snapshots = builder.build_snapshots_over_run(events)
    assert len(snapshots) == 2
    assert snapshots[0].timestamp == 0.1
    assert snapshots[1].timestamp == 0.3
    assert snapshots[0].num_edges == 1
    assert snapshots[1].num_edges == 2


def test_10_sliding_window():
    """Verify event-count windowing filters strictly the last W events."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.1, step_idx=0),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.2, step_idx=1),
        AgentTelemetryEvent(source_agent="C", target_agent="D", timestamp=0.3, step_idx=2),
        AgentTelemetryEvent(source_agent="D", target_agent="E", timestamp=0.4, step_idx=3),
    ]
    # Window size = 2 events
    snap = builder.build_snapshot(events=events, timestamp=0.4, window_size=2, strategy=WindowStrategy.EVENT_COUNT)
    # Only steps 2 and 3 should be in window
    assert snap.has_edge("C", "D") is True
    assert snap.has_edge("D", "E") is True
    assert snap.has_edge("A", "B") is False


def test_11_historical_sequence():
    """Verify build_temporal_history returns G(t-n)...G(t) containing only past and current info."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.1, step_idx=0),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.2, step_idx=1),
        AgentTelemetryEvent(source_agent="C", target_agent="D", timestamp=0.3, step_idx=2),
        AgentTelemetryEvent(source_agent="D", target_agent="E", timestamp=0.4, step_idx=3),
    ]
    history = builder.build_temporal_history(events=events, current_timestamp=0.3, history_length=2)
    assert len(history) == 2
    assert history[-1].timestamp <= 0.3
    # Step 3 (at t=0.4) must not appear anywhere in history
    for snap in history:
        assert snap.timestamp <= 0.3
        assert snap.has_edge("D", "E") is False


def test_12_feature_extraction_tensor_readiness():
    """Verify node_feature_matrix and edge_index for downstream ML/GNN compatibility."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="agent_1", target_agent="agent_2", timestamp=0.2, step_idx=0, latency=0.25, confidence=0.88),
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.2)
    agent_ids, matrix = snap.node_feature_matrix()
    assert agent_ids == ["agent_1", "agent_2"]
    assert len(matrix) == 2
    assert len(matrix[0]) == 11  # 11 numerical node features

    nodes, edges_idx = snap.edge_index()
    assert edges_idx == [(0, 1)]  # agent_1 (idx 0) -> agent_2 (idx 1)


def test_13_serialization_deserialization(tmp_path: Path):
    """Verify exact lossless JSON serialization and disk reloading."""
    builder = TemporalGraphBuilder()
    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.1, step_idx=0, latency=0.2, confidence=0.95),
    ]
    snap = builder.build_snapshot(events=events, timestamp=0.1, run_id="run_serial_test")
    save_file = tmp_path / "snapshot.json"

    save_snapshot(snap, save_file)
    assert save_file.exists()

    loaded = load_snapshot(save_file)
    assert loaded.run_id == snap.run_id
    assert loaded.timestamp == snap.timestamp
    assert loaded.num_nodes == snap.num_nodes
    assert loaded.num_edges == snap.num_edges
    assert loaded.nodes == snap.nodes
    assert loaded.edges == snap.edges


def test_14_no_future_event_leakage_rigorous():
    """CRITICAL TEST: Verify an event at t+10 containing failure info CANNOT appear in snapshot at t."""
    builder = TemporalGraphBuilder()
    
    # Events up to t=5.0 are normal
    past_events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=1.0, step_idx=0, latency=0.1, confidence=0.95, tool_error=False),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=3.0, step_idx=1, latency=0.1, confidence=0.92, tool_error=False),
        AgentTelemetryEvent(source_agent="C", target_agent="D", timestamp=5.0, step_idx=2, latency=0.1, confidence=0.90, tool_error=False),
    ]

    # Future event at t=15.0 (t + 10) contains catastrophic failure
    future_catastrophic_event = AgentTelemetryEvent(
        source_agent="D",
        target_agent="E",
        timestamp=15.0,  # Future!
        step_idx=3,
        latency=99.9,
        confidence=0.01,
        tool_error=True,
        failure_label=3,
        event_type="failure",
        error_type="catastrophic_cascade",
    )

    full_event_stream = past_events + [future_catastrophic_event]

    # Build snapshot at t=5.0
    snap_t5 = builder.build_snapshot(events=full_event_stream, timestamp=5.0, step_idx=2)

    # 1. Future edge must NOT exist
    assert snap_t5.has_edge("D", "E") is False
    assert snap_t5.num_edges == 3  # Exactly the 3 past edges (A->B, B->C, C->D)

    # 2. Node D must have 0 errors and high confidence (future error must NOT leak!)
    node_d = snap_t5.get_node_features("D")
    assert node_d["error_count"] == 0
    assert node_d["recent_failure_count"] == 0
    assert node_d["average_latency"] == 0.0

    # 3. Direct leak attempt: If future event is passed to leakage verifier, it raises ValueError
    from ml.utils.reproducibility import verify_no_future_leakage
    with pytest.raises(ValueError, match="Data Leakage Detected|Temporal Leakage Detected"):
        verify_no_future_leakage(
            current_step=2,
            current_timestamp=5.0,
            used_event_steps=[0, 1, 2, 3],  # Step 3 > 2!
            used_event_timestamps=[1.0, 3.0, 5.0, 15.0],  # 15.0 > 5.0!
        )


def test_15_reproducibility():
    """Verify that given identical events and configuration, builder produces identical graph structures."""
    builder1 = TemporalGraphBuilder()
    builder2 = TemporalGraphBuilder()

    events = [
        AgentTelemetryEvent(source_agent="A", target_agent="B", timestamp=0.2, step_idx=0, latency=0.15),
        AgentTelemetryEvent(source_agent="B", target_agent="C", timestamp=0.4, step_idx=1, latency=0.25),
    ]

    snap1 = builder1.build_snapshot(events=events, timestamp=0.4, run_id="repro_run")
    snap2 = builder2.build_snapshot(events=events, timestamp=0.4, run_id="repro_run")

    assert snap1.to_dict() == snap2.to_dict()
