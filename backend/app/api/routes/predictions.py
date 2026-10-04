"""Prediction and early warning inference routes."""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, PaginationParams
from backend.app.schemas.predictions import PredictionResponse, PredictionListResponse
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.prediction_service import (
    get_predictions,
    get_prediction_by_id,
)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get(
    "",
    response_model=PredictionListResponse,
    summary="List Predictions",
    description="Retrieve paginated model prediction inferences with optional filtering by run, model, or horizon.",
    responses={
        200: {"description": "List of model predictions returned successfully."},
    },
)
def list_predictions(
    run_id: Optional[str] = Query(None, description="Filter predictions by simulation run ID"),
    model: Optional[str] = Query(None, description="Filter predictions by model architecture name"),
    horizon: Optional[int] = Query(None, description="Filter predictions by prediction horizon k"),
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
) -> PredictionListResponse:
    """Retrieve predictions matching query filters."""
    return get_predictions(
        db=db,
        limit=pagination.limit,
        offset=pagination.offset,
        run_id=run_id,
        model=model,
        horizon=horizon,
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionResponse,
    summary="Get Prediction Details",
    description="Retrieve specific model inference record and risk probabilities by prediction identifier.",
    responses={
        200: {"description": "Prediction record returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Prediction not found."},
    },
)
def get_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Retrieve individual prediction record."""
    return get_prediction_by_id(db=db, prediction_id=prediction_id)
