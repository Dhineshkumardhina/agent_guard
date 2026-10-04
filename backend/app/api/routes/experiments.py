"""Research experiment and benchmark suite inspection routes."""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, PaginationParams
from backend.app.schemas.experiments import ExperimentResponse, ExperimentListResponse
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.experiment_service import (
    get_experiments,
    get_experiment_by_id,
)

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get(
    "",
    response_model=ExperimentListResponse,
    summary="List Experiments",
    description="Retrieve paginated benchmark experiments and research suites.",
    responses={
        200: {"description": "List of experiments returned successfully."},
    },
)
def list_experiments(
    status: Optional[str] = Query(None, description="Filter experiments by status (created, running, completed, failed)"),
    model: Optional[str] = Query(None, description="Filter experiments by primary model evaluated"),
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
) -> ExperimentListResponse:
    """Retrieve experiments matching query filters."""
    return get_experiments(
        db=db,
        limit=pagination.limit,
        offset=pagination.offset,
        status=status,
        model=model,
    )


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get Experiment Details",
    description="Retrieve full configuration and metadata for an individual research experiment.",
    responses={
        200: {"description": "Experiment configuration returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Experiment not found."},
    },
)
def get_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
) -> ExperimentResponse:
    """Retrieve individual experiment record."""
    return get_experiment_by_id(db=db, experiment_id=experiment_id)
