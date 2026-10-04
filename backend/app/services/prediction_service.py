"""Prediction service layer managing inference records and early warning alerts."""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.models import Prediction, Run
from backend.app.schemas.predictions import PredictionResponse, PredictionListResponse
from backend.app.core.exceptions import PredictionNotFoundException, RunNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def _to_prediction_response(pred: Prediction) -> PredictionResponse:
    """Map SQLAlchemy Prediction model to PredictionResponse schema."""
    label = 1 if pred.predicted_probability >= 0.5 else 0
    return PredictionResponse(
        prediction_id=pred.id,
        run_id=pred.run_id,
        model=pred.model_name,
        timestamp=float(pred.step_idx) if pred.step_idx is not None else 0.0,
        step_idx=pred.step_idx,
        horizon=pred.horizon_k,
        predicted_probability=pred.predicted_probability,
        predicted_label=label,
        threshold=0.5,
        actual_outcome=pred.ground_truth,
        risk_level=pred.risk_level,
        lead_time=pred.lead_time,
        created_at=pred.created_at,
    )


def get_predictions(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    run_id: Optional[str] = None,
    model: Optional[str] = None,
    horizon: Optional[int] = None,
) -> PredictionListResponse:
    """Retrieve paginated failure predictions with optional filters."""
    query = db.query(Prediction)
    if run_id:
        query = query.filter(Prediction.run_id == run_id)
    if model:
        query = query.filter(func.lower(Prediction.model_name) == model.lower())
    if horizon:
        query = query.filter(Prediction.horizon_k == horizon)

    total = query.count()
    records = query.order_by(Prediction.id).offset(offset).limit(limit).all()
    items = [_to_prediction_response(p) for p in records]
    has_more = (offset + limit) < total

    return PredictionListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_prediction_by_id(db: Session, prediction_id: str) -> PredictionResponse:
    """Retrieve a single prediction by ID or raise PredictionNotFoundException."""
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise PredictionNotFoundException(prediction_id=prediction_id)
    return _to_prediction_response(pred)


def get_predictions_by_run(
    db: Session,
    run_id: str,
    limit: int = 50,
    offset: int = 0,
    model: Optional[str] = None,
    horizon: Optional[int] = None,
) -> PredictionListResponse:
    """Retrieve all predictions for a specific simulation run."""
    run = db.query(Run).filter(Run.id == run_id).first()
    query = db.query(Prediction).filter(Prediction.run_id == run_id)
    if model:
        query = query.filter(func.lower(Prediction.model_name) == model.lower())
    if horizon:
        query = query.filter(Prediction.horizon_k == horizon)

    total = query.count()
    if not run and total == 0:
        raise RunNotFoundException(run_id=run_id)
    records = query.order_by(Prediction.step_idx, Prediction.horizon_k).offset(offset).limit(limit).all()
    items = [_to_prediction_response(p) for p in records]
    has_more = (offset + limit) < total

    return PredictionListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )
