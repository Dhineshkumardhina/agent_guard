"""Edge Feature Extraction for Temporal Interaction Graphs.

Computes causal, strictly non-future-looking edge features representing directed
interactions between pairs of agents (source -> target) at time t based strictly on events <= t.
"""

from typing import List, Dict, Any, Optional
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.utils.reproducibility import verify_no_future_leakage


def extract_edge_features(
    source_agent: str,
    target_agent: str,
    events: List[AgentTelemetryEvent],
    timestamp: float,
    step_idx: Optional[int] = None,
    total_trajectory_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract directed edge features for communication from source to target at timestamp t.
    
    Args:
        source_agent: Sending agent ID.
        target_agent: Receiving agent ID.
        events: Chronological sequence of all events up to and including timestamp t.
        timestamp: Upper time bound t for the graph snapshot.
        step_idx: Optional upper step index bound for the graph snapshot.
        total_trajectory_time: Duration of time elapsed up to timestamp t.
        
    Returns:
        Dictionary of edge features preserving temporal history.
        
    Raises:
        ValueError: If any event after timestamp t is detected.
    """
    # 1. Filter events strictly by causal bound: timestamp <= t and step <= step_idx
    causal_events: List[AgentTelemetryEvent] = []
    used_steps: List[int] = []
    used_times: List[float] = []

    for ev in events:
        ev_t = ev.timestamp if ev.timestamp is not None else 0.0
        ev_s = ev.step_idx if ev.step_idx is not None else 0
        if ev_t <= timestamp:
            if step_idx is None or ev_s <= step_idx:
                causal_events.append(ev)
                used_steps.append(ev_s)
                used_times.append(ev_t)

    # 2. Strict Data Leakage Assertion
    verify_no_future_leakage(
        current_step=step_idx if step_idx is not None else (max(used_steps) if used_steps else 0),
        current_timestamp=timestamp,
        used_event_steps=used_steps,
        used_event_timestamps=used_times,
    )

    # 3. Filter directed interactions strictly from source to target
    edge_events = [
        e for e in causal_events
        if e.source_agent == source_agent and e.target_agent == target_agent
    ]

    interaction_count = len(edge_events)
    if interaction_count == 0:
        return {
            "source_agent": source_agent,
            "target_agent": target_agent,
            "interaction_count": 0,
            "message_count": 0,
            "average_latency": 0.0,
            "average_message_length": 0.0,
            "average_confidence": 1.0,
            "retry_count": 0,
            "contradiction_rate": 0.0,
            "error_count": 0,
            "timeout_count": 0,
            "last_interaction_time": 0.0,
            "interaction_frequency": 0.0,
            "interaction_timestamps": [],
        }

    # Temporal preservation: retain all interaction timestamps
    interaction_timestamps = [
        round(e.timestamp, 4) for e in edge_events if e.timestamp is not None
    ]
    last_interaction_time = interaction_timestamps[-1] if interaction_timestamps else 0.0

    message_count = len([e for e in edge_events if e.event_type in ("message", "delegation")])

    latencies = [e.latency for e in edge_events if e.latency is not None]
    average_latency = sum(latencies) / len(latencies) if latencies else 0.0

    lengths = [e.message_length for e in edge_events if e.message_length is not None]
    average_message_length = sum(lengths) / len(lengths) if lengths else 0.0

    confidences = [e.confidence for e in edge_events if e.confidence is not None]
    average_confidence = sum(confidences) / len(confidences) if confidences else 1.0

    retry_count = sum(e.retry_count or 0 for e in edge_events) + len([e for e in edge_events if e.event_type == "retry"])

    contradictions = [e.contradiction_score for e in edge_events if e.contradiction_score is not None]
    contradiction_rate = sum(contradictions) / len(contradictions) if contradictions else 0.0

    error_count = len([
        e for e in edge_events
        if e.tool_error
        or (e.failure_label is not None and e.failure_label > 0)
        or e.event_type in ("error", "failure")
        or e.error_type is not None
    ])

    timeout_count = len([
        e for e in edge_events
        if e.event_type == "timeout"
        or (e.error_type and "timeout" in e.error_type.lower())
        or (e.injected_fault and "timeout" in e.injected_fault.lower())
    ])

    # Interaction frequency (interactions per unit time elapsed)
    time_span = total_trajectory_time if total_trajectory_time and total_trajectory_time > 0 else (timestamp if timestamp > 0 else 1.0)
    interaction_frequency = interaction_count / time_span if time_span > 0 else float(interaction_count)

    return {
        "source_agent": source_agent,
        "target_agent": target_agent,
        "interaction_count": interaction_count,
        "message_count": message_count,
        "average_latency": round(average_latency, 4),
        "average_message_length": round(average_message_length, 2),
        "average_confidence": round(average_confidence, 4),
        "retry_count": retry_count,
        "contradiction_rate": round(contradiction_rate, 4),
        "error_count": error_count,
        "timeout_count": timeout_count,
        "last_interaction_time": round(last_interaction_time, 4),
        "interaction_frequency": round(interaction_frequency, 4),
        "interaction_timestamps": interaction_timestamps,
    }
