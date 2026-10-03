"""Rolling Feature Extraction and Temporal Leakage Prevention.

Computes causal, strictly non-future-looking temporal features over multi-agent event streams:
- rolling error rate
- rolling retry rate
- rolling latency
- rolling confidence
- message frequency
- interaction frequency
- contradiction rate

Strictly adheres to causal ordering: features at timestamp t depend ONLY on events <= t.
"""

from typing import List, Dict, Any, Union
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.utils.reproducibility import verify_no_future_leakage


def compute_event_features(
    current_idx: int,
    history: List[AgentTelemetryEvent],
    window_size: int = 5,
) -> Dict[str, float]:
    """Calculate rolling features for an event at current_idx using strictly past/current events.
    
    Args:
        current_idx: Index of target event in history.
        history: Chronologically sorted sequence of AgentTelemetryEvent objects.
        window_size: Maximum number of backward events to include in the rolling window.
        
    Returns:
        Dictionary of computed rolling features.
        
    Raises:
        ValueError: If temporal or step data leakage is detected.
    """
    if current_idx < 0 or current_idx >= len(history):
        raise IndexError(f"current_idx {current_idx} out of range [0, {len(history)})")

    target_event = history[current_idx]
    current_time = target_event.timestamp if target_event.timestamp is not None else 0.0
    current_step = target_event.step_idx if target_event.step_idx is not None else current_idx

    # Window slice: strictly up to current_idx (never index > current_idx)
    start_idx = max(0, current_idx - window_size + 1)
    candidate_window = history[start_idx : current_idx + 1]

    # Explicit causal filter: timestamp must be <= current_time and step <= current_step
    causal_window = [
        e for e in candidate_window
        if (e.timestamp is not None and e.timestamp <= current_time)
        and (e.step_idx is not None and e.step_idx <= current_step)
    ]

    # Scientific rigor: enforce future leakage verification
    used_steps = [e.step_idx if e.step_idx is not None else 0 for e in causal_window]
    used_timestamps = [e.timestamp if e.timestamp is not None else 0.0 for e in causal_window]
    verify_no_future_leakage(
        current_step=current_step,
        current_timestamp=current_time,
        used_event_steps=used_steps,
        used_event_timestamps=used_timestamps,
    )

    n_events = len(causal_window)
    if n_events == 0:
        return {
            "rolling_error_rate": 0.0,
            "rolling_retry_rate": 0.0,
            "rolling_latency": 0.0,
            "rolling_confidence": 1.0,
            "message_frequency": 0.0,
            "interaction_frequency": 0.0,
            "contradiction_rate": 0.0,
        }

    # 1. Rolling error rate
    error_count = sum(
        1 for e in causal_window
        if e.tool_error
        or (e.failure_label is not None and e.failure_label > 0)
        or e.event_type in ("error", "failure")
        or e.error_type is not None
    )
    rolling_error_rate = error_count / n_events

    # 2. Rolling retry rate
    retry_count = sum(
        1 for e in causal_window
        if (e.retry_count is not None and e.retry_count > 0)
        or e.event_type == "retry"
    )
    rolling_retry_rate = retry_count / n_events

    # 3. Rolling latency
    latencies = [e.latency for e in causal_window if e.latency is not None]
    rolling_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # 4. Rolling confidence
    confidences = [e.confidence for e in causal_window if e.confidence is not None]
    rolling_confidence = sum(confidences) / len(confidences) if confidences else 1.0

    # 5. Message frequency (events per unit time in window)
    window_t_start = causal_window[0].timestamp if causal_window[0].timestamp is not None else 0.0
    dt = current_time - window_t_start
    if dt > 0:
        message_frequency = n_events / dt
    else:
        message_frequency = float(n_events)

    # 6. Interaction frequency for the specific (source, target) pair
    src = target_event.source_agent
    tgt = target_event.target_agent
    pair_count = sum(1 for e in causal_window if e.source_agent == src and e.target_agent == tgt)
    interaction_frequency = pair_count / n_events

    # 7. Contradiction rate
    contradictions = [e.contradiction_score for e in causal_window if e.contradiction_score is not None]
    contradiction_rate = sum(contradictions) / len(contradictions) if contradictions else 0.0

    return {
        "rolling_error_rate": round(rolling_error_rate, 4),
        "rolling_retry_rate": round(rolling_retry_rate, 4),
        "rolling_latency": round(rolling_latency, 4),
        "rolling_confidence": round(rolling_confidence, 4),
        "message_frequency": round(message_frequency, 4),
        "interaction_frequency": round(interaction_frequency, 4),
        "contradiction_rate": round(contradiction_rate, 4),
    }


def compute_rolling_features(
    events: List[Union[AgentTelemetryEvent, Any]],
    window_size: int = 5,
) -> List[Dict[str, Any]]:
    """Compute rolling features across an entire chronological event trajectory.
    
    Guarantees causal leakage protection across all steps.
    
    Args:
        events: Chronological sequence of telemetry events.
        window_size: Window size in event steps.
        
    Returns:
        List of feature dictionaries aligned 1:1 with input events.
    """
    # Normalize to AgentTelemetryEvent
    norm_events: List[AgentTelemetryEvent] = []
    for e in events:
        if isinstance(e, AgentTelemetryEvent):
            norm_events.append(e)
        elif hasattr(e, "to_telemetry_event"):
            norm_events.append(e.to_telemetry_event())
        elif isinstance(e, dict):
            norm_events.append(AgentTelemetryEvent(**e))
        else:
            raise TypeError(f"Cannot process event of type {type(e)}")

    # Sort strictly chronologically
    norm_events.sort(key=lambda e: (e.timestamp if e.timestamp is not None else 0.0, e.step_idx or 0))

    feature_records = []
    for i, ev in enumerate(norm_events):
        features = compute_event_features(
            current_idx=i,
            history=norm_events,
            window_size=window_size,
        )
        record = {
            "event_id": ev.event_id,
            "run_id": ev.run_id,
            "step_idx": ev.step_idx,
            "timestamp": ev.timestamp,
            "source_agent": ev.source_agent,
            "target_agent": ev.target_agent,
            "event_type": ev.event_type,
            **features,
        }
        feature_records.append(record)

    return feature_records
