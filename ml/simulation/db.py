"""Database Persistence for Multi-Agent Simulation Runs and Events.

Stores completed runs, agents, and events in the relational database
(SQLite or PostgreSQL) created in Phase 1 without storing unnecessary huge payloads.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from backend.app.database.models import Run, Agent as DBAgent, Event as DBEvent
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
    metadata: Optional[Dict[str, Any]] = None,
) -> Run:
    """Persist a completed simulation run and its agents & events into the database.
    
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
        metadata: Optional compact metadata.
        
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
            cascading_failure_step=None,
            random_seed=random_seed,
            created_at=datetime.now(timezone.utc),
        )
        session.add(db_run)
    else:
        db_run.duration_seconds = round(duration_seconds, 4)
        db_run.has_cascading_failure = has_cascading_failure
        # Clean previous agents/events for idempotent re-runs
        session.query(DBAgent).filter(DBAgent.run_id == run_id).delete()
        session.query(DBEvent).filter(DBEvent.run_id == run_id).delete()

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

    # 3. Persist events (sanitized, structured, no unbounded bloat)
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

    session.commit()
    session.refresh(db_run)
    return db_run
