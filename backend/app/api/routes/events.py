"""Telemetry event inspection routes."""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.api.dependencies import get_db, PaginationParams
from backend.app.database.models import Event
from backend.app.schemas.events import EventResponse, EventListResponse
from backend.app.services.run_service import _to_event_response

router = APIRouter(prefix="/events", tags=["Events"])


@router.get(
    "",
    response_model=EventListResponse,
    summary="List Events",
    description="Retrieve paginated telemetry events with optional filtering by run, event type, or agents.",
    responses={
        200: {"description": "Telemetry event stream returned successfully."},
    },
)
def list_events(
    run_id: Optional[str] = Query(None, description="Filter by simulation run ID"),
    event_type: Optional[str] = Query(None, description="Filter by communication event type"),
    source_agent: Optional[str] = Query(None, description="Filter by source agent ID"),
    target_agent: Optional[str] = Query(None, description="Filter by target agent ID"),
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
) -> EventListResponse:
    """Retrieve filtered telemetry events."""
    query = db.query(Event)
    if run_id:
        query = query.filter(Event.run_id == run_id)
    if event_type:
        query = query.filter(func.lower(Event.event_type) == event_type.lower())
    if source_agent:
        query = query.filter(Event.source_agent == source_agent)
    if target_agent:
        query = query.filter(Event.target_agent == target_agent)

    total = query.count()
    records = query.order_by(Event.run_id, Event.step_idx, Event.timestamp).offset(pagination.offset).limit(pagination.limit).all()
    items = [_to_event_response(e) for e in records]
    has_more = (pagination.offset + pagination.limit) < total

    return EventListResponse(
        items=items,
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
        has_more=has_more,
    )
