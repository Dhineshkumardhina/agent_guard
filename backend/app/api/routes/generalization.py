"""Generalization and robustness evaluation routes."""

from typing import Optional
from fastapi import APIRouter, Depends, Query

from backend.app.api.dependencies import PaginationParams
from backend.app.schemas.generalization import (
    GeneralizationMetricResponse,
    GeneralizationListResponse,
)
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.generalization_service import (
    get_generalization,
    get_generalization_by_id,
)

router = APIRouter(prefix="/generalization", tags=["Generalization"])


@router.get(
    "",
    response_model=GeneralizationListResponse,
    summary="List Generalization Evaluations",
    description="Retrieve Phase 14 generalization experiments across agent count, topology, task type, and compound shifts.",
    responses={
        200: {"description": "List of generalization metrics returned."},
    },
)
def list_generalization(
    dimension: Optional[str] = Query(None, description="Filter by shift dimension (agent_count, topology, task_type, compound)"),
    model: Optional[str] = Query(None, description="Filter by model architecture"),
    horizon: Optional[int] = Query(None, description="Filter by prediction horizon k"),
    split_type: Optional[str] = Query(None, description="Filter by split (in_distribution, out_of_distribution)"),
    pagination: PaginationParams = Depends(),
) -> GeneralizationListResponse:
    """Retrieve filtered generalization metrics."""
    return get_generalization(
        dimension=dimension,
        model=model,
        horizon=horizon,
        split_type=split_type,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get(
    "/{experiment_id}",
    response_model=GeneralizationMetricResponse,
    summary="Get Generalization Details",
    description="Retrieve specific generalization experiment metrics, transfer gaps, and statistical significance.",
    responses={
        200: {"description": "Generalization experiment details returned."},
        404: {"model": ErrorEnvelope, "description": "Generalization experiment not found."},
    },
)
def get_generalization_experiment(
    experiment_id: str,
) -> GeneralizationMetricResponse:
    """Retrieve individual generalization experiment record."""
    return get_generalization_by_id(experiment_id=experiment_id)
