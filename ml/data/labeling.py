"""Scientific Failure Labeling Logic for Prediction Horizons.

Formalizes explicit labeling criteria for:
- Binary failure forecasting: P(F(t+k) | observations <= t) for k in {1, 3, 5, 10, 20}
- Multi-level failure taxonomy:
    * Level 0: No failure
    * Level 1: Agent-level failure (isolated local error, tool failure, hallucination)
    * Level 2: Interaction-level failure (pairwise contradiction, loop, routing failure)
    * Level 3: Cascading system-level failure (multi-hop propagation across >= 2 downstream agents)

CRITICAL RESEARCH INTEGRITY RULES:
1. The future window (step_idx in (current_step, current_step + k]) is used ONLY to compute labels.
2. It must NEVER be used to construct input features.
3. Not every injected fault is a cascade. A cascade requires measurable multi-agent propagation.
"""

from typing import List, Tuple, Optional, Any
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.simulation.fault_injection.propagation import FaultPropagationTracker


def compute_prediction_label(
    events: List[Any],
    current_step_idx: int,
    prediction_horizon: int,
    propagation_tracker: Optional[FaultPropagationTracker] = None,
) -> Tuple[int, str, int]:
    """Compute future ground-truth failure label within horizon (t, t + k].
    
    Args:
        events: Chronological sequence of all events in the simulation run.
        current_step_idx: Discrete event step index at current prediction point t.
        prediction_horizon: Number of future interaction steps k to inspect.
        propagation_tracker: Optional audit tracker for fault propagation and cascading status.
        
    Returns:
        Tuple of (label: int, failure_type: str, failure_level: int):
            - label: 1 if any failure occurs in (current_step_idx, current_step_idx + k], else 0.
            - failure_type: Specific fault mode string if failure occurs, else "none".
            - failure_level: 0 (none), 1 (agent), 2 (interaction), 3 (cascading).
    """
    horizon_start = current_step_idx + 1
    horizon_end = current_step_idx + prediction_horizon

    # Future window events: steps strictly in (current_step_idx, current_step_idx + k]
    future_events = [
        e for e in events
        if getattr(e, "step_idx", None) is not None
        and horizon_start <= getattr(e, "step_idx") <= horizon_end
    ]

    if not future_events:
        return 0, "none", 0

    # Inspect future events for failures or injected faults
    has_failure = False
    dominant_fault_type = "none"
    max_failure_level = 0

    for ev in future_events:
        # Check event-level failure indicators
        injected = getattr(ev, "injected_fault", None) or getattr(ev, "error_type", None)
        ev_label = getattr(ev, "failure_label", 0) or 0
        tool_err = getattr(ev, "tool_error", False)
        event_type = getattr(ev, "event_type", "")
        downstream = getattr(ev, "downstream_failure", False)

        is_ev_failure = (
            ev_label > 0
            or injected is not None
            or tool_err
            or event_type in ("error", "failure")
            or downstream
        )

        if is_ev_failure:
            has_failure = True
            if dominant_fault_type == "none" and injected:
                dominant_fault_type = str(injected)
            elif dominant_fault_type == "none" and event_type in ("error", "failure"):
                dominant_fault_type = getattr(ev, "error_type", None) or "execution_error"

            level = ev_label
            if level == 0 and (tool_err or event_type in ("error", "failure")):
                level = 1
            max_failure_level = max(max_failure_level, level)

    if not has_failure:
        return 0, "none", 0

    # Determine cascading criteria:
    # A failure is Level 3 Cascading ONLY IF:
    # 1. Propagation tracker recorded an originating fault
    # 2. Corrupted state reached >= 2 downstream agents (multi-hop propagation)
    # 3. Downstream effect occurs within or impacts the trajectory
    if propagation_tracker is not None:
        affected = propagation_tracker.affected_agents
        origin = propagation_tracker.originating_agent
        downstream_affected = [a for a in affected if a != origin]
        
        # Check if cascading conditions are met
        is_cascade = (
            propagation_tracker.is_cascading
            or len(downstream_affected) >= 2
        ) and any(getattr(e, "downstream_failure", False) for e in future_events)

        if is_cascade:
            max_failure_level = 3
        elif len(downstream_affected) == 1 and max_failure_level < 2:
            max_failure_level = 2
        elif max_failure_level == 0:
            max_failure_level = 1
    else:
        # Without tracker, check if downstream failures exist in future window
        if any(getattr(e, "downstream_failure", False) for e in future_events):
            max_failure_level = max(max_failure_level, 3)
        elif max_failure_level == 0:
            max_failure_level = 1

    return 1, dominant_fault_type or "system_fault", max_failure_level
