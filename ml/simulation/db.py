"""Database Persistence for Multi-Agent Simulation Runs, Events, Faults, and Failures.

Stores completed runs, agents, events, fault injections, and failure annotations
in the relational database (SQLite or PostgreSQL) without storing unnecessary huge payloads.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from backend.app.database.models import (
    Run,
    Agent as DBAgent,
    Event as DBEvent,
    FaultInjection as DBFaultInjection,
    Failure as DBFailure,
)
from ml.simulation.events import SimulationMessage


def save_simulation_run(
    run_id: str,
    task_type: str,
    topology: str,
    agents: List[Any],
    events: List[SimulationMessage],
    random_seed: int,
    duration_seconds: float,
    session: Session,
    experiment_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    has_cascading_failure: bool = False,
    cascading_failure_step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    fault_injections: Optional[List[Any]] = None,
    failures: Optional[List[Any]] = None,
) -> Run:
    """Persist a completed simulation run and its agents, events, faults, & failures into the database.
    
    Args:
        run_id: Unique trajectory run identifier.
        task_type: Workflow task category.
        topology: Network topology pattern.
        agents: List of Agent domain objects.
        events: List of recorded SimulationMessage objects.
        random_seed: Seed used for reproducibility.
        duration_seconds: Total simulated execution time.
        session: Active SQLAlchemy database session.
        experiment_id: Optional parent experiment identifier.
        dataset_id: Optional associated dataset identifier.
        has_cascading_failure: Flag indicating whether cascading failure occurred.
        cascading_failure_step: Step index where cascade originated.
        metadata: Optional compact metadata.
        fault_injections: Optional list of FaultInjectionRecord objects.
        failures: Optional list of failure dictionaries or objects.
        
    Returns:
        The created SQLAlchemy Run ORM model instance.
    """
    # 1. Create or update Run
    db_run = session.query(Run).filter(Run.id == run_id).first()
    if not db_run:
        db_run = Run(
            id=run_id,
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            task_type=str(task_type),
            topology=str(topology),
            num_agents=len(agents),
            duration_seconds=round(duration_seconds, 4),
            has_cascading_failure=has_cascading_failure,
            cascading_failure_step=cascading_failure_step,
            random_seed=random_seed,
            created_at=datetime.now(timezone.utc),
        )
        session.add(db_run)
    else:
        db_run.duration_seconds = round(duration_seconds, 4)
        db_run.has_cascading_failure = has_cascading_failure
        db_run.cascading_failure_step = cascading_failure_step
        # Clean previous related items for idempotent re-runs
        session.query(DBAgent).filter(DBAgent.run_id == run_id).delete()
        session.query(DBEvent).filter(DBEvent.run_id == run_id).delete()
        session.query(DBFaultInjection).filter(DBFaultInjection.run_id == run_id).delete()
        session.query(DBFailure).filter(DBFailure.run_id == run_id).delete()

    # 2. Persist participating agents (keyed uniquely by run_id + agent_id)
    for agent in agents:
        agent_id_pk = f"{run_id}_{agent.agent_id}"
        role_val = agent.role.value if hasattr(agent.role, "value") else str(agent.role)
        status_val = agent.state.get("status", "active") if isinstance(agent.state, dict) else "active"
        db_agent = DBAgent(
            id=agent_id_pk,
            run_id=run_id,
            agent_name=agent.name,
            role=role_val,
            status=status_val,
            created_at=datetime.now(timezone.utc),
        )
        session.add(db_agent)

    # 3. Persist events
    for ev in events:
        compact_meta = {
            "length": ev.message_length,
            "type": ev.event_type,
        }
        if ev.metadata:
            for k, v in ev.metadata.items():
                if isinstance(v, (int, float, str, bool)):
                    compact_meta[k] = v

        db_event = DBEvent(
            id=ev.event_id,
            run_id=run_id,
            step_idx=ev.step_idx,
            timestamp=round(ev.timestamp, 4),
            source_agent=ev.source_agent,
            target_agent=ev.target_agent,
            event_type=ev.event_type,
            message_length=ev.message_length,
            token_count=ev.token_count,
            latency=round(ev.latency, 4),
            confidence=round(ev.confidence, 4),
            output_quality=round(ev.output_quality, 4),
            contradiction_score=round(ev.contradiction_score, 4),
            tool_used=ev.tool_used,
            tool_success=ev.tool_success,
            tool_error=ev.tool_error,
            retry_count=ev.retry_count,
            injected_fault=ev.injected_fault,
            error_type=ev.error_type,
            failure_label=ev.failure_label,
            downstream_failure=ev.downstream_failure,
            metadata_json=compact_meta,
            created_at=datetime.now(timezone.utc),
        )
        session.add(db_event)

    # 4. Persist fault injections if any
    if fault_injections:
        for fi in fault_injections:
            fi_id = fi.get("injection_id") if isinstance(fi, dict) else getattr(fi, "injection_id", None)
            fi_step = fi.get("step_idx", 0) if isinstance(fi, dict) else getattr(fi, "step_idx", 0)
            fi_target = fi.get("target_agent", "unknown") if isinstance(fi, dict) else getattr(fi, "target_agent", "unknown")
            fi_type = fi.get("fault_type", "unknown") if isinstance(fi, dict) else getattr(fi, "fault_type", "unknown")
            fi_params = fi.get("parameters", {}) if isinstance(fi, dict) else getattr(fi, "parameters", {})

            db_fi = DBFaultInjection(
                id=fi_id or f"{run_id}_fi_{len(session.new)}",
                run_id=run_id,
                step_idx=fi_step,
                target_agent=fi_target,
                fault_type=fi_type,
                parameters=fi_params,
                created_at=datetime.now(timezone.utc),
            )
            session.add(db_fi)

    # 5. Persist failures if any
    if failures:
        for fail in failures:
            fail_id = fail.get("id") or fail.get("failure_id") if isinstance(fail, dict) else getattr(fail, "failure_id", getattr(fail, "id", None))
            fail_step = fail.get("step_idx", 0) if isinstance(fail, dict) else getattr(fail, "step_idx", 0)
            fail_level = fail.get("failure_level", 1) if isinstance(fail, dict) else getattr(fail, "failure_level", 1)
            fail_origin = fail.get("originating_agent", "unknown") if isinstance(fail, dict) else getattr(fail, "originating_agent", "unknown")
            fail_affected = fail.get("affected_agents", []) if isinstance(fail, dict) else getattr(fail, "affected_agents", [])
            fail_type = fail.get("failure_type", "unknown") if isinstance(fail, dict) else getattr(fail, "failure_type", "unknown")
            fail_desc = fail.get("description", "") if isinstance(fail, dict) else getattr(fail, "description", "")

            db_fail = DBFailure(
                id=fail_id or f"{run_id}_fail_{len(session.new)}",
                run_id=run_id,
                step_idx=fail_step,
                failure_level=fail_level,
                originating_agent=fail_origin,
                affected_agents=fail_affected,
                failure_type=fail_type,
                description=fail_desc,
                created_at=datetime.now(timezone.utc),
            )
            session.add(db_fail)

    session.commit()
    session.refresh(db_run)
    return db_run
