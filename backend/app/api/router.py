"""Central API Router for AgentGuard v1 endpoints."""

from fastapi import APIRouter

from backend.app.api.routes import (
    health,
    agents,
    runs,
    events,
    graphs,
    predictions,
    experiments,
    evaluations,
    ablations,
    generalization,
    explainability,
)

api_router = APIRouter(prefix="/api/v1")

# Mount route modules
api_router.include_router(agents.router)
api_router.include_router(runs.router)
api_router.include_router(events.router)
api_router.include_router(graphs.router)
api_router.include_router(predictions.router)
api_router.include_router(experiments.router)
api_router.include_router(evaluations.router)
api_router.include_router(ablations.router)
api_router.include_router(generalization.router)
api_router.include_router(explainability.router)

# Model comparison alias under /models/comparison as well
api_router.add_api_route(
    "/models/comparison",
    evaluations.compare_models,
    methods=["GET"],
    response_model=evaluations.ModelComparisonResponse,
    tags=["Evaluations"],
    summary="Model Comparison Alias",
    description="Alias endpoint for cross-model comparative performance evaluation.",
)
