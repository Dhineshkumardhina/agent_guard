"""Windowing and Slicing Strategies for Temporal Graph Construction.

Supports:
- Event-based windows (e.g. last 5, 10, 20 events up to t)
- Time-duration windows (events within [t - delta_t, t])
- Cumulative history windows (all events within [0, t])
- Sliding windows across full trajectories
"""

from typing import List, Optional
from enum import Enum
from ml.telemetry.schemas import AgentTelemetryEvent


class WindowStrategy(str, Enum):
    """Temporal windowing policy for graph snapshot construction."""
    CUMULATIVE = "cumulative"  # All events in [0, t]
    EVENT_COUNT = "event_count"  # Last K events up to t
    TIME_SPAN = "time_span"  # Events in [t - delta_t, t]


def filter_events_by_window(
    events: List[AgentTelemetryEvent],
    target_timestamp: float,
    target_step_idx: Optional[int] = None,
    strategy: WindowStrategy = WindowStrategy.CUMULATIVE,
    window_size: Optional[int] = None,
    time_window: Optional[float] = None,
) -> List[AgentTelemetryEvent]:
    """Filter event sequence using causal window parameters strictly up to target_timestamp.
    
    Args:
        events: Chronologically sorted sequence of telemetry events.
        target_timestamp: The snapshot timestamp t.
        target_step_idx: Optional snapshot step index.
        strategy: Windowing policy.
        window_size: Number of events to include under EVENT_COUNT strategy.
        time_window: Time duration delta_t to include under TIME_SPAN strategy.
        
    Returns:
        List of causal events belonging to the designated window.
    """
    # 1. Base causal slice: events strictly <= target_timestamp (and <= target_step_idx if set)
    causal_events: List[AgentTelemetryEvent] = []
    for ev in events:
        ev_t = ev.timestamp if ev.timestamp is not None else 0.0
        ev_s = ev.step_idx if ev.step_idx is not None else 0
        if ev_t <= target_timestamp:
            if target_step_idx is None or ev_s <= target_step_idx:
                causal_events.append(ev)

    if not causal_events:
        return []

    # 2. Apply window strategy
    if strategy == WindowStrategy.EVENT_COUNT or (window_size is not None and window_size > 0):
        size = window_size or 10
        return causal_events[-size:] if len(causal_events) >= size else causal_events

    elif strategy == WindowStrategy.TIME_SPAN or (time_window is not None and time_window > 0):
        duration = time_window or 1.0
        cutoff = target_timestamp - duration
        return [e for e in causal_events if (e.timestamp or 0.0) >= cutoff]

    # Default: Cumulative
    return causal_events
