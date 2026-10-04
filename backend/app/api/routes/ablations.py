"""Ablation study inspection routes."""

from typing import Optional
from fastapi import APIRouter, Depends, Query

from backend.app.api.dependencies import PaginationParams
from backend.app.schemas.ablations import AblationResponse, AblationListResponse
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.ablation_service import (
    get_ablations,
    get_ablation_by_id,
)

router = APIRouter(prefix="/ablations", tags=["Ablation Studies"])


@router.get(
    "",
    response_model=AblationListResponse,
    summary="List Ablation Experiments",
    description="Retrieve Phase 13 ablation study conditions evaluating component removal impacts.",
    responses={
        200: {"description": "List of ablation experiment results returned."},
    },
)
def list_ablations(
    model: Optional[str] = Query(None, description="Filter ablations by base model architecture"),
    horizon: Optional[int] = Query(None, description="Filter ablations by prediction horizon k"),
    seed: Optional[int] = Query(None, description="Filter ablations by experimental seed"),
    pagination: PaginationParams = Depends(),
) -> AblationListResponse:
    """Retrieve filtered ablation records."""
    return get_ablations(
        model=model,
        horizon=horizon,
        seed=seed,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get(
    "/{experiment_id}",
    response_model=AblationResponse,
    summary="Get Ablation Details",
    description="Retrieve full metrics and component attribution for a specific ablation condition.",
    responses={
        200: {"description": "Ablation experiment details returned."},
        404: {"model": ErrorEnvelope, "description": "Ablation experiment not found."},
    },
)
def get_ablation(
    experiment_id: str,
) -> AblationResponse:
    """Retrieve individual ablation experiment record."""
    return get_ablation_by_id(experiment_id=experiment_id)
