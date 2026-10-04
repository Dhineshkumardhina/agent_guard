"""Explainability and risk attribution routes."""

from typing import Optional
from fastapi import APIRouter, Depends, Query

from backend.app.api.dependencies import PaginationParams
from backend.app.schemas.explainability import (
    ExplanationResponse,
    ExplanationListResponse,
)
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.explainability_service import (
    get_explanations,
    get_explanation_by_id,
    get_explanations_by_run,
)

router = APIRouter(prefix="/explanations", tags=["Explainability"])


@router.get(
    "",
    response_model=ExplanationListResponse,
    summary="List Explanations",
    description="Retrieve Phase 15 explainability reports including feature, agent node, and communication edge attributions.",
    responses={
        200: {"description": "List of explanation records returned."},
    },
)
def list_explanations(
    run_id: Optional[str] = Query(None, description="Filter explanations by simulation run ID"),
    model: Optional[str] = Query(None, description="Filter explanations by evaluated model"),
    pagination: PaginationParams = Depends(),
) -> ExplanationListResponse:
    """Retrieve filtered explanation records."""
    return get_explanations(
        run_id=run_id,
        model=model,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get(
    "/{explanation_id}",
    response_model=ExplanationResponse,
    summary="Get Explanation Details",
    description="Retrieve full attribution breakdown and case study synthesis for an individual explanation.",
    responses={
        200: {"description": "Explanation details returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Explanation not found."},
    },
)
def get_explanation(
    explanation_id: str,
) -> ExplanationResponse:
    """Retrieve individual explanation record."""
    return get_explanation_by_id(explanation_id=explanation_id)
