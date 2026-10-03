"""Tests for Controlled Fault Injection Engine and Cascading Failure Tracking (Phase 4)."""

import pytest
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.simulation.fault_injection.injector import (
    FaultInjector,
    ALL_FAULT_TYPES,
)
from ml.simulation.fault_injection.propagation import (
    FaultPropagationTracker,
    FaultInjectionRecord,
)
from ml.simulation.run import SimulationRun
from backend.app.database.models import (
    Run,
    FaultInjection as DBFaultInjection,
    Failure as DBFailure,
)


@pytest.mark.parametrize("fault_name", ALL_FAULT_TYPES)
def test_every_fault_type_injection(fault_name):
    """Test all 12 fault types: injection happens, event mutated, and audit record created."""
    injector = FaultInjector(
        fault_type=fault_name,
        probability=1.0,
        severity=0.8,
        random_seed=42,
    )

    base_event = AgentTelemetryEvent(
        run_id="run_test_fault",
        step_idx=1,
        timestamp=0.5,
        source_agent="researcher_1",
        target_agent="analyst_1",
        event_type="message",
        message="Valid empirical data",
        confidence=0.95,
        output_quality=1.0,
        latency=0.2,
    )

    assert injector.should_inject(step_idx=1, event=base_event) is True
    mutated = injector.apply_fault(base_event)

    # 1. Event is mutated
    assert mutated.injected_fault is not None
    assert mutated.error_type is not None

    # 2. Injection record is logged
    assert len(injector.injected_records) == 1
    record = injector.injected_records[0]
    assert record.fault_type == injector.fault_type
    assert record.severity == 0.8
    assert record.step_idx == 1
    assert record.target_agent == "researcher_1"


def test_failure_label_classification():
    """Verify Level 1, Level 2, and Level 3 failure labeling criteria."""
    # Level 1: Local agent error (tool failure)
    inj_l1 = FaultInjector(fault_type="tool_failure", severity=0.7)
    ev_l1 = inj_l1.apply_fault(AgentTelemetryEvent(source_agent="A", target_agent="B"))
    assert ev_l1.failure_label == 1

    # Level 2: Interaction error (contradiction)
    inj_l2 = FaultInjector(fault_type="contradictory_output", severity=0.7)
    ev_l2 = inj_l2.apply_fault(AgentTelemetryEvent(source_agent="A", target_agent="B"))
    assert ev_l2.failure_label == 2

    # Level 3: Cascading failure across dependent nodes
    tracker = FaultPropagationTracker(run_id="run_cascade_test")
    tracker.record_originating_fault("Agent_A", step_idx=0, timestamp=0.1, fault_type="hallucinated_output", initial_level=1)
    assert tracker.failure_level == 1
    assert tracker.is_cascading is False

    # Propagate to 1st downstream agent (Level 2)
    tracker.record_downstream_effect("Agent_A", "Agent_B", step_idx=1, timestamp=0.3, reason="Inherited bad context")
    assert tracker.failure_level == 2

    # Propagate to 2nd downstream agent (Level 3 Cascading)
    tracker.record_downstream_effect("Agent_B", "Agent_C", step_idx=2, timestamp=0.6, reason="Failed audit")
    assert tracker.failure_level == 3
    assert tracker.is_cascading is True

    tracker.finalize_outcome("CASCADING_FAILURE")
    assert "Agent_A -> Agent_B -> Agent_C -> Final Failure" == tracker.to_path_string()


def test_propagation_tracking_sequence_and_timestamps():
    """Verify propagation tracking records sequence of affected agents and failure timestamps."""
    tracker = FaultPropagationTracker(run_id="run_track_01")
    tracker.record_originating_fault("Planner", step_idx=0, timestamp=0.1, fault_type="malformed_output")
    tracker.record_downstream_effect("Planner", "Coder", step_idx=1, timestamp=0.4, reason="Cannot parse spec")
    tracker.record_downstream_effect("Coder", "Verifier", step_idx=2, timestamp=0.8, reason="Build failed")

    assert tracker.originating_agent == "Planner"
    assert tracker.affected_agents == ["Planner", "Coder", "Verifier"]
    assert tracker.failure_timestamps == [0.1, 0.4, 0.8]
    assert tracker.to_path_string() == "Planner -> Coder -> Verifier"


def test_fault_injection_reproducibility():
    """Verify identical random seed produces identical injection decisions and outputs."""
    inj1 = FaultInjector(fault_type="tool_timeout", probability=0.5, severity=0.8, random_seed=999)
    inj2 = FaultInjector(fault_type="tool_timeout", probability=0.5, severity=0.8, random_seed=999)

    decisions1 = [inj1.should_inject(i, AgentTelemetryEvent(source_agent="A", target_agent="B")) for i in range(10)]
    decisions2 = [inj2.should_inject(i, AgentTelemetryEvent(source_agent="A", target_agent="B")) for i in range(10)]

    assert decisions1 == decisions2


def test_controlled_fault_free_vs_fault_injected_runs():
    """Verify controlled experiment comparison between normal and fault-injected runs."""
    # 1. Fault-free run
    run_normal = SimulationRun(
        run_id="run_ctrl_normal",
        task_type="research",
        topology="pipeline",
        random_seed=42,
    ).execute()

    assert run_normal.final_status == "COMPLETED"
    assert run_normal.has_cascading_failure is False
    assert run_normal.cascading_failure_step is None

    # 2. Fault-injected run
    injector = FaultInjector(
        fault_type="hallucinated_output",
        target_agent="researcher_1",
        injection_step=1,
        probability=1.0,
        severity=0.9,
        random_seed=42,
    )

    run_fault = SimulationRun(
        run_id="run_ctrl_fault",
        task_type="research",
        topology="pipeline",
        random_seed=42,
        fault_injector=injector,
    ).execute()

    assert run_fault.final_status == "FAILED"
    assert run_fault.has_cascading_failure is True
    assert run_fault.cascading_failure_step == 1
    assert len(run_fault.propagation_tracker.affected_agents) >= 2


def test_fault_database_persistence_integration(db_session):
    """Verify fault injections and failures are stored in the database."""
    injector = FaultInjector(
        fault_type="tool_failure",
        target_agent="researcher_1",
        injection_step=1,
        probability=1.0,
        severity=0.8,
        random_seed=42,
    )

    run = SimulationRun(
        run_id="run_db_fault_001",
        task_type="research",
        topology="pipeline",
        random_seed=42,
        fault_injector=injector,
    ).execute()

    # Persist to DB
    run.save_to_db(db_session)

    # Query back
    stored_run = db_session.query(Run).filter(Run.id == "run_db_fault_001").first()
    assert stored_run.has_cascading_failure is True

    stored_fis = db_session.query(DBFaultInjection).filter(DBFaultInjection.run_id == "run_db_fault_001").all()
    assert len(stored_fis) == 1
    assert stored_fis[0].fault_type == "tool_failure"

    stored_fails = db_session.query(DBFailure).filter(DBFailure.run_id == "run_db_fault_001").all()
    assert len(stored_fails) >= 1
    assert stored_fails[0].failure_level >= 2
