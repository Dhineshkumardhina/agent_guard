"""Agent inspection routes exposing multi-agent participants."""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, PaginationParams
from backend.app.schemas.agents import AgentResponse, AgentListResponse
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.agent_service import get_agents, get_agent_by_id

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.get(
    "",
    response_model=AgentListResponse,
    summary="List Agents",
    description="Retrieve paginated list of participating agents with optional run_id or role filters.",
    responses={
        200: {"description": "List of agents returned successfully."},
    },
)
def list_agents(
    run_id: Optional[str] = Query(None, description="Filter agents by simulation run ID"),
    role: Optional[str] = Query(None, description="Filter agents by role specialization"),
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
) -> AgentListResponse:
    """Retrieve agents matching query criteria."""
    return get_agents(
        db=db,
        limit=pagination.limit,
        offset=pagination.offset,
        run_id=run_id,
        role=role,
    )


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get Agent Details",
    description="Retrieve detailed metadata for an individual agent by its unique identifier.",
    responses={
        200: {"description": "Agent metadata retrieved."},
        404: {"model": ErrorEnvelope, "description": "Agent not found."},
    },
)
def get_agent(
    agent_id: str,
    db: Session = Depends(get_db),
) -> AgentResponse:
    """Retrieve individual agent metadata."""
    return get_agent_by_id(db=db, agent_id=agent_id)
