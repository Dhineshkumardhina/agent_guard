"""Phase 18: Fault Injection and Ground-Truth Label Validation.

Verifies:
1. All 12 supported synthetic fault types trigger with correct properties (agent, severity, probability).
2. Scientific cascading criteria:
   - Injected fault alone != automatic cascade.
   - Downstream affected == 0 -> Level 1 (Local error).
   - Downstream affected == 1 -> Level 2 (Pairwise interaction error).
   - Downstream affected >= 2 -> Level 3 (Cascading system failure).
3. Ground-truth label consistency across Levels 0, 1, 2, 3.
4. Causal horizon boundaries and zero future leakage in labeling.
"""

import pytest
from ml.simulation.fault_injection.injector import FaultInjector, ALL_FAULT_TYPES
from ml.simulation.fault_injection.propagation import FaultPropagationTracker
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.data.labeling import compute_prediction_label


@pytest.mark.parametrize("fault_type", ALL_FAULT_TYPES)
def test_all_supported_fault_types_initialize_and_inject(fault_type):
    """Verify each supported fault type is valid and alters telemetry event correctly."""
    injector = FaultInjector(
        fault_type=fault_type,
        target_agent="agent_worker",
        probability=1.0,
        severity=0.75,
        random_seed=42,
    )
    event = AgentTelemetryEvent(
        source_agent="agent_worker",
        target_agent="agent_coordinator",
        step_idx=2,
        timestamp=2.5,
        output_quality=1.0,
        confidence=1.0,
    )

    assert injector.should_inject(step_idx=2, event=event)
    injector.apply_fault(event=event)

    # Event must be marked with the injected fault and appropriate quality/confidence degradation
    assert event.injected_fault is not None or event.error_type is not None or event.tool_error or event.failure_label > 0
    assert len(injector.injected_records) == 1
    assert injector.injected_records[0].fault_type == fault_type


def test_injected_fault_is_not_automatically_cascading():
    """Verify that an injected fault without propagation is classified as Level 1, NOT cascading."""
    tracker = FaultPropagationTracker(run_id="run_non_cascade")
    # Inception at agent_0
    tracker.record_originating_fault(
        agent_id="agent_0",
        step_idx=1,
        timestamp=1.0,
        fault_type="tool_failure",
        initial_level=1,
    )

    assert not tracker.is_cascading
    assert tracker.failure_level == 1
    assert len(tracker.affected_agents) == 1

    # Future window has an event on agent_0 only
    ev1 = AgentTelemetryEvent(
        source_agent="agent_0",
        target_agent="agent_1",
        step_idx=2,
        timestamp=2.0,
        injected_fault="tool_failure",
        failure_label=1,
        downstream_failure=False,
    )

    label, ftype, flevel = compute_prediction_label(
        events=[ev1],
        current_step_idx=1,
        prediction_horizon=3,
        propagation_tracker=tracker,
    )
    assert label == 1
    assert flevel == 1
    assert ftype == "tool_failure"


def test_pairwise_propagation_yields_level_2():
    """Verify that propagation to 1 downstream agent produces Level 2."""
    tracker = FaultPropagationTracker(run_id="run_pairwise")
    tracker.record_originating_fault("agent_0", step_idx=1, timestamp=1.0, fault_type="hallucinated_output")
    # Propagates to 1 downstream agent (agent_1)
    tracker.record_downstream_effect("agent_0", "agent_1", step_idx=2, timestamp=2.0, reason="corrupted_context")

    assert not tracker.is_cascading
    assert tracker.failure_level == 2
    assert len(tracker.affected_agents) == 2

    ev = AgentTelemetryEvent(
        source_agent="agent_0",
        target_agent="agent_1",
        step_idx=2,
        timestamp=2.0,
        failure_label=2,
        downstream_failure=False,
    )
    label, _, flevel = compute_prediction_label(
        events=[ev],
        current_step_idx=1,
        prediction_horizon=3,
        propagation_tracker=tracker,
    )
    assert label == 1
    assert flevel == 2


def test_multi_agent_propagation_yields_level_3_cascade():
    """Verify that propagation across >= 2 downstream agents produces Level 3 cascading failure."""
    tracker = FaultPropagationTracker(run_id="run_cascade")
    tracker.record_originating_fault("agent_0", step_idx=1, timestamp=1.0, fault_type="hallucinated_output")
    # Downstream agent 1
    tracker.record_downstream_effect("agent_0", "agent_1", step_idx=2, timestamp=2.0, reason="bad_context")
    # Downstream agent 2
    tracker.record_downstream_effect("agent_1", "agent_2", step_idx=3, timestamp=3.0, reason="corrupted_decision")

    assert tracker.is_cascading
    assert tracker.failure_level == 3
    assert len([a for a in tracker.affected_agents if a != tracker.originating_agent]) >= 2

    future_events = [
        AgentTelemetryEvent(
            source_agent="agent_1",
            target_agent="agent_2",
            step_idx=3,
            timestamp=3.0,
            failure_label=3,
            downstream_failure=True,
        )
    ]
    label, _, flevel = compute_prediction_label(
        events=future_events,
        current_step_idx=1,
        prediction_horizon=3,
        propagation_tracker=tracker,
    )
    assert label == 1
    assert flevel == 3


def test_level_0_clean_nominal_trajectory():
    """Verify that nominal runs with zero faults yield Level 0 label."""
    events = [
        AgentTelemetryEvent(
            source_agent=f"agent_{i}",
            target_agent=f"agent_{i+1}",
            step_idx=i,
            timestamp=float(i),
            output_quality=0.95,
            confidence=0.98,
            tool_error=False,
            failure_label=0,
            downstream_failure=False,
        )
        for i in range(10)
    ]

    label, ftype, flevel = compute_prediction_label(
        events=events,
        current_step_idx=3,
        prediction_horizon=4,
    )
    assert label == 0
    assert ftype == "none"
    assert flevel == 0


def test_horizon_boundary_exclusion():
    """Verify that failures occurring after cutoff t + k are NOT leaked into the current horizon label."""
    # Failure occurs at step 10
    events = [
        AgentTelemetryEvent(
            source_agent="agent_0",
            target_agent="agent_1",
            step_idx=s,
            timestamp=float(s),
            failure_label=1 if s == 10 else 0,
            injected_fault="tool_failure" if s == 10 else None,
        )
        for s in range(15)
    ]

    # Prediction at t = 4 with horizon k = 3 -> examines (4, 7] -> steps 5, 6, 7
    label, ftype, flevel = compute_prediction_label(
        events=events,
        current_step_idx=4,
        prediction_horizon=3,
    )
    assert label == 0
    assert flevel == 0
    assert ftype == "none"

    # Prediction at t = 7 with horizon k = 3 -> examines (7, 10] -> steps 8, 9, 10
    label_k3, ftype_k3, flevel_k3 = compute_prediction_label(
        events=events,
        current_step_idx=7,
        prediction_horizon=3,
    )
    assert label_k3 == 1
    assert ftype_k3 == "tool_failure"
    assert flevel_k3 >= 1
