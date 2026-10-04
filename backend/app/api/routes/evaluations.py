"""Model evaluation and comparative benchmark routes."""

from typing import Optional
from fastapi import APIRouter, Depends, Query

from backend.app.api.dependencies import PaginationParams
from backend.app.schemas.evaluations import (
    EvaluationResponse,
    EvaluationListResponse,
    ModelComparisonResponse,
)
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.evaluation_service import (
    get_evaluations,
    get_evaluation_by_id,
    get_model_comparison,
)

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


@router.get(
    "/comparison",
    response_model=ModelComparisonResponse,
    summary="Model Architecture Comparison",
    description=(
        "Retrieve normalized performance comparison across all 9 canonical models "
        "(Rule-Based, Logistic Regression, Random Forest, XGBoost, LSTM, GRU, GCN, GAT, Temporal GNN) "
        "at a specified prediction horizon."
    ),
    responses={
        200: {"description": "Normalized comparative leaderboard returned successfully."},
    },
)
def compare_models(
    horizon: int = Query(1, ge=1, le=20, description="Prediction horizon k to compare across models"),
) -> ModelComparisonResponse:
    """Retrieve normalized cross-model comparative metrics."""
    return get_model_comparison(horizon=horizon)


@router.get(
    "",
    response_model=EvaluationListResponse,
    summary="List Evaluations",
    description="Retrieve paginated empirical evaluation metric results with multi-attribute filtering.",
    responses={
        200: {"description": "List of evaluation metrics returned successfully."},
    },
)
def list_evaluations(
    model: Optional[str] = Query(None, description="Filter evaluations by model name"),
    horizon: Optional[int] = Query(None, description="Filter evaluations by prediction horizon k"),
    experiment: Optional[str] = Query(None, description="Filter evaluations by experiment ID"),
    dataset_version: Optional[str] = Query(None, description="Filter evaluations by dataset version"),
    pagination: PaginationParams = Depends(),
) -> EvaluationListResponse:
    """Retrieve filtered evaluation metrics."""
    return get_evaluations(
        model=model,
        horizon=horizon,
        experiment=experiment,
        dataset_version=dataset_version,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get(
    "/{experiment_id}",
    response_model=EvaluationResponse,
    summary="Get Evaluation Details",
    description="Retrieve specific model evaluation results for an experiment identifier.",
    responses={
        200: {"description": "Evaluation metrics returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Evaluation not found."},
    },
)
def get_evaluation(
    experiment_id: str,
) -> EvaluationResponse:
    """Retrieve individual evaluation result."""
    return get_evaluation_by_id(experiment_id=experiment_id)
