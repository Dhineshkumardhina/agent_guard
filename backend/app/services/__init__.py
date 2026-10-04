"""Services package re-exporting all service operations."""

from backend.app.services.db_seeder import seed_research_database
from backend.app.services.agent_service import get_agents, get_agent_by_id
from backend.app.services.run_service import (
    get_runs,
    get_run_by_id,
    get_run_events,
    get_run_failures,
)
from backend.app.services.graph_service import get_run_temporal_graph
from backend.app.services.prediction_service import (
    get_predictions,
    get_prediction_by_id,
    get_predictions_by_run,
)
from backend.app.services.experiment_service import (
    get_experiments,
    get_experiment_by_id,
)
from backend.app.services.evaluation_service import (
    get_evaluations,
    get_evaluation_by_id,
    get_model_comparison,
)
from backend.app.services.ablation_service import (
    get_ablations,
    get_ablation_by_id,
)
from backend.app.services.generalization_service import (
    get_generalization,
    get_generalization_by_id,
)
from backend.app.services.explainability_service import (
    get_explanations,
    get_explanation_by_id,
    get_explanations_by_run,
)

__all__ = [
    "seed_research_database",
    "get_agents",
    "get_agent_by_id",
    "get_runs",
    "get_run_by_id",
    "get_run_events",
    "get_run_failures",
    "get_run_temporal_graph",
    "get_predictions",
    "get_prediction_by_id",
    "get_predictions_by_run",
    "get_experiments",
    "get_experiment_by_id",
    "get_evaluations",
    "get_evaluation_by_id",
    "get_model_comparison",
    "get_ablations",
    "get_ablation_by_id",
    "get_generalization",
    "get_generalization_by_id",
    "get_explanations",
    "get_explanation_by_id",
    "get_explanations_by_run",
]
