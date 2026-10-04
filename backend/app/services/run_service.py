"""Run service layer managing simulation trajectories, events, and failure instances."""

from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.models import Run, Agent, Event, Failure
from backend.app.schemas.runs import (
    RunResponse,
    RunDetailResponse,
    RunListResponse,
    RunFailureResponse,
    RunFailureListResponse,
)
from backend.app.schemas.events import EventResponse, EventListResponse
from backend.app.schemas.agents import AgentResponse
from backend.app.services.agent_service import _to_agent_response
from backend.app.core.exceptions import RunNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def _to_run_response(run: Run) -> RunResponse:
    """Map SQLAlchemy Run model to RunResponse schema."""
    status = "failed" if run.has_cascading_failure else "completed"
    return RunResponse(
        id=run.id,
        experiment_id=run.experiment_id,
        dataset_id=run.dataset_id,
        task=run.task_type,
        topology=run.topology,
        status=status,
        agent_count=run.num_agents,
        duration_seconds=run.duration_seconds,
        has_cascading_failure=run.has_cascading_failure,
        cascading_failure_step=run.cascading_failure_step,
        random_seed=run.random_seed,
        created_at=run.created_at,
    )


def _to_failure_response(fail: Failure) -> RunFailureResponse:
    """Map SQLAlchemy Failure model to RunFailureResponse schema."""
    affected = fail.affected_agents if isinstance(fail.affected_agents, list) else []
    return RunFailureResponse(
        id=fail.id,
        run_id=fail.run_id,
        step_idx=fail.step_idx,
        failure_level=fail.failure_level,
        originating_agent=fail.originating_agent,
        affected_agents=affected,
        failure_type=fail.failure_type,
        description=fail.description,
        created_at=fail.created_at,
    )


def _to_event_response(ev: Event) -> EventResponse:
    """Map SQLAlchemy Event model to EventResponse schema."""
    return EventResponse(
        id=ev.id,
        run_id=ev.run_id,
        step_idx=ev.step_idx,
        timestamp=ev.timestamp,
        source_agent=ev.source_agent,
        target_agent=ev.target_agent,
        event_type=ev.event_type,
        message_length=ev.message_length,
        token_count=ev.token_count,
        latency=ev.latency,
        confidence=ev.confidence,
        output_quality=ev.output_quality,
        contradiction_score=ev.contradiction_score,
        tool_used=ev.tool_used,
        tool_success=ev.tool_success,
        tool_error=ev.tool_error,
        retry_count=ev.retry_count,
        injected_fault=ev.injected_fault,
        error_type=ev.error_type,
        failure_label=ev.failure_label,
        downstream_failure=ev.downstream_failure,
        metadata_json=ev.metadata_json or {},
        created_at=ev.created_at,
    )


def get_runs(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    task: Optional[str] = None,
    topology: Optional[str] = None,
    status: Optional[str] = None,
    agent_count: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> RunListResponse:
    """Retrieve filtered and paginated simulation trajectories."""
    query = db.query(Run)

    if task:
        query = query.filter(func.lower(Run.task_type) == task.lower())
    if topology:
        query = query.filter(func.lower(Run.topology) == topology.lower())
    if agent_count:
        query = query.filter(Run.num_agents == agent_count)
    if status:
        if status.lower() in ("failed", "failure"):
            query = query.filter(Run.has_cascading_failure == True)  # noqa: E712
        elif status.lower() in ("completed", "success", "normal"):
            query = query.filter(Run.has_cascading_failure == False)  # noqa: E712
    if start_date:
        query = query.filter(Run.created_at >= start_date)
    if end_date:
        query = query.filter(Run.created_at <= end_date)

    total = query.count()
    records = query.order_by(Run.id).offset(offset).limit(limit).all()
    items = [_to_run_response(r) for r in records]
    has_more = (offset + limit) < total

    return RunListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_run_by_id(db: Session, run_id: str) -> RunDetailResponse:
    """Retrieve full run details including agents and failures."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise RunNotFoundException(run_id=run_id)

    agents = db.query(Agent).filter(Agent.run_id == run_id).order_by(Agent.id).all()
    failures = db.query(Failure).filter(Failure.run_id == run_id).order_by(Failure.step_idx).all()

    base_response = _to_run_response(run)
    return RunDetailResponse(
        **base_response.model_dump(),
        agents=[_to_agent_response(a) for a in agents],
        failures=[_to_failure_response(f) for f in failures],
    )


def get_run_events(
    db: Session,
    run_id: str,
    limit: int = 50,
    offset: int = 0,
) -> EventListResponse:
    """Retrieve telemetry events for a specific run."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise RunNotFoundException(run_id=run_id)

    query = db.query(Event).filter(Event.run_id == run_id)
    total = query.count()
    records = query.order_by(Event.step_idx, Event.timestamp).offset(offset).limit(limit).all()
    items = [_to_event_response(e) for e in records]
    has_more = (offset + limit) < total

    return EventListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_run_failures(
    db: Session,
    run_id: str,
    limit: int = 50,
    offset: int = 0,
) -> RunFailureListResponse:
    """Retrieve failure instances recorded during a run."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise RunNotFoundException(run_id=run_id)

    query = db.query(Failure).filter(Failure.run_id == run_id)
    total = query.count()
    records = query.order_by(Failure.step_idx).offset(offset).limit(limit).all()
    items = [_to_failure_response(f) for f in records]
    has_more = (offset + limit) < total

    return RunFailureListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )
