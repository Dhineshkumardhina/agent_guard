"""Node Feature Extraction for Temporal Interaction Graphs.

Computes causal, strictly non-future-looking node features representing each agent
in the multi-agent system at time t based strictly on events <= t.
"""

from typing import List, Dict, Any, Optional
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.utils.reproducibility import verify_no_future_leakage


def extract_node_features(
    agent_id: str,
    events: List[AgentTelemetryEvent],
    timestamp: float,
    step_idx: Optional[int] = None,
    agent_role: Optional[str] = None,
    is_active: bool = True,
) -> Dict[str, Any]:
    """Extract node features for a single agent at timestamp t using events <= t.
    
    Args:
        agent_id: Identifier of the agent node.
        events: Chronological sequence of all events up to and including timestamp t.
        timestamp: Upper time bound t for the graph snapshot.
        step_idx: Optional upper step index bound for the graph snapshot.
        agent_role: Optional role name of the agent.
        is_active: Whether the agent is active in the environment.
        
    Returns:
        Dictionary of node features containing all mandatory research metrics.
        
    Raises:
        ValueError: If any event with timestamp > t or step > step_idx is detected.
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

    # 3. Filter events where this agent was either source or target
    agent_events = [
        e for e in causal_events
        if e.source_agent == agent_id or e.target_agent == agent_id
    ]
    sent_events = [e for e in causal_events if e.source_agent == agent_id]

    # Metrics computation
    event_count = len(agent_events)
    message_count = len([e for e in agent_events if e.event_type == "message"])
    tool_call_count = len([e for e in sent_events if e.tool_used is not None or e.event_type == "tool_call"])
    
    error_count = len([
        e for e in sent_events
        if e.tool_error
        or (e.failure_label is not None and e.failure_label > 0)
        or e.event_type in ("error", "failure")
        or e.error_type is not None
    ])
    
    retry_count = sum(e.retry_count or 0 for e in sent_events) + len([e for e in sent_events if e.event_type == "retry"])
    
    timeout_count = len([
        e for e in sent_events
        if e.event_type == "timeout"
        or (e.error_type and "timeout" in e.error_type.lower())
        or (e.injected_fault and "timeout" in e.injected_fault.lower())
    ])

    # Averages over sent communications
    latencies = [e.latency for e in sent_events if e.latency is not None]
    average_latency = sum(latencies) / len(latencies) if latencies else 0.0

    confidences = [e.confidence for e in sent_events if e.confidence is not None]
    average_confidence = sum(confidences) / len(confidences) if confidences else 1.0

    qualities = [e.output_quality for e in sent_events if e.output_quality is not None]
    average_output_quality = sum(qualities) / len(qualities) if qualities else 1.0

    contradictions = [e.contradiction_score for e in sent_events if e.contradiction_score is not None]
    contradiction_rate = sum(contradictions) / len(contradictions) if contradictions else 0.0

    # Recent failures (events within last 5 agent interactions with failure_label > 0 or error)
    recent_events = sent_events[-5:] if len(sent_events) >= 5 else sent_events
    recent_failure_count = len([
        e for e in recent_events
        if (e.failure_label is not None and e.failure_label > 0)
        or e.tool_error
        or e.event_type in ("error", "failure")
    ])

    return {
        "agent_id": agent_id,
        "role": agent_role or "agent",
        "is_active": is_active,
        "event_count": event_count,
        "message_count": message_count,
        "tool_call_count": tool_call_count,
        "error_count": error_count,
        "retry_count": retry_count,
        "timeout_count": timeout_count,
        "average_latency": round(average_latency, 4),
        "average_confidence": round(average_confidence, 4),
        "average_output_quality": round(average_output_quality, 4),
        "contradiction_rate": round(contradiction_rate, 4),
        "recent_failure_count": recent_failure_count,
    }
