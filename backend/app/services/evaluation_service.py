"""Evaluation and model comparison service exposing empirical research metrics."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.app.core.config import settings
from backend.app.schemas.evaluations import (
    EvaluationResponse,
    EvaluationListResponse,
    ModelComparisonItem,
    ModelComparisonResponse,
)
from backend.app.core.exceptions import EvaluationNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_EVALUATIONS_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_unified_evaluations() -> List[Dict[str, Any]]:
    """Load and cache unified evaluation metrics from results directory."""
    global _EVALUATIONS_CACHE
    if _EVALUATIONS_CACHE is not None:
        return _EVALUATIONS_CACHE

    metrics_path = settings.BASE_DIR / "results" / "evaluation" / "metrics" / "unified_evaluation_metrics.json"
    if not metrics_path.exists():
        logger.warning("Unified evaluations file not found at %s", metrics_path)
        return []

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _EVALUATIONS_CACHE = data
            return data
    except Exception as e:
        logger.error("Failed to load evaluations metrics: %s", e)
        return []


def _to_evaluation_response(entry: Dict[str, Any]) -> EvaluationResponse:
    """Map raw evaluation dictionary to EvaluationResponse schema."""
    return EvaluationResponse(
        experiment_id=entry.get("experiment_id", "unknown"),
        model=entry.get("model_name") or entry.get("model", "unknown"),
        model_family=entry.get("model_family"),
        dataset_version=entry.get("dataset_version", "agentguard_dataset_v1"),
        horizon=int(entry.get("prediction_horizon") or entry.get("horizon", 1)),
        seed=entry.get("seed", 42),
        threshold=float(entry.get("threshold", 0.5)),
        precision=float(entry.get("precision", 0.0)),
        recall=float(entry.get("recall", 0.0)),
        f1=float(entry.get("f1", 0.0)),
        auroc=float(entry.get("auroc", 0.0)),
        auprc=float(entry.get("auprc", 0.0)),
        false_positive_rate=float(entry.get("false_positive_rate", 0.0)),
        false_alarm_rate=float(entry.get("false_alarm_rate", 0.0)),
        mean_lead_time=float(entry.get("mean_lead_time", 0.0)),
        median_lead_time=float(entry.get("median_lead_time", 0.0)),
        successful_warnings=int(entry.get("successful_early_warnings") or entry.get("successful_warnings", 0)),
        warnings_per_trajectory=float(entry.get("warnings_per_trajectory", 0.0)),
        brier_score=entry.get("brier_score"),
        ece=entry.get("ece"),
        sample_count=entry.get("sample_count"),
        positive_count=entry.get("positive_count"),
        negative_count=entry.get("negative_count"),
    )


def get_evaluations(
    model: Optional[str] = None,
    horizon: Optional[int] = None,
    experiment: Optional[str] = None,
    dataset_version: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> EvaluationListResponse:
    """Retrieve filtered and paginated evaluation metrics."""
    data = _load_unified_evaluations()

    filtered = []
    for entry in data:
        mname = entry.get("model_name") or entry.get("model", "")
        h = int(entry.get("prediction_horizon") or entry.get("horizon", 1))
        exp_id = entry.get("experiment_id", "")
        dver = entry.get("dataset_version", "")

        if model and model.lower() not in mname.lower():
            continue
        if horizon and h != horizon:
            continue
        if experiment and experiment.lower() not in exp_id.lower():
            continue
        if dataset_version and dataset_version.lower() not in dver.lower():
            continue

        filtered.append(_to_evaluation_response(entry))

    total = len(filtered)
    items = filtered[offset : offset + limit]
    has_more = (offset + limit) < total

    return EvaluationListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_evaluation_by_id(experiment_id: str) -> EvaluationResponse:
    """Retrieve evaluation metrics for a specific experiment ID."""
    data = _load_unified_evaluations()
    target_clean = experiment_id.lower().replace("-", "_").strip()

    for entry in data:
        eid = entry.get("experiment_id", "").lower().replace("-", "_").strip()
        if eid == target_clean or experiment_id.lower() == entry.get("experiment_id", "").lower():
            return _to_evaluation_response(entry)

    raise EvaluationNotFoundException(identifier=experiment_id)


def get_model_comparison(horizon: int = 1) -> ModelComparisonResponse:
    """Retrieve normalized comparison for all 9 canonical models at specified horizon."""
    data = _load_unified_evaluations()

    # Target 9 canonical models
    target_models = [
        "Rule-Based",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LSTM",
        "GRU",
        "GCN",
        "GAT",
        "Temporal GNN",
    ]

    matched_entries = []
    for entry in data:
        h = int(entry.get("prediction_horizon") or entry.get("horizon", 1))
        if h == horizon:
            mname = entry.get("model_name") or entry.get("model", "")
            for tm in target_models:
                if tm.lower() == mname.lower():
                    matched_entries.append(entry)
                    break

    if not matched_entries:
        # Fallback to any matching horizon or horizon=1
        matched_entries = [e for e in data if int(e.get("prediction_horizon", 1)) == 1]

    # Calculate max values for normalization
    max_f1 = max((float(e.get("f1", 0.0)) for e in matched_entries), default=1.0)
    max_auroc = max((float(e.get("auroc", 0.0)) for e in matched_entries), default=1.0)
    max_lead = max((float(e.get("mean_lead_time", 0.0)) for e in matched_entries), default=1.0)

    items: List[ModelComparisonItem] = []
    for e in matched_entries:
        f1 = float(e.get("f1", 0.0))
        auroc = float(e.get("auroc", 0.0))
        lead = float(e.get("mean_lead_time", 0.0))

        items.append(
            ModelComparisonItem(
                model_name=e.get("model_name") or e.get("model", "unknown"),
                model_family=e.get("model_family") or "benchmark",
                horizon=horizon,
                precision=float(e.get("precision", 0.0)),
                recall=float(e.get("recall", 0.0)),
                f1=f1,
                auroc=auroc,
                auprc=float(e.get("auprc", 0.0)),
                false_positive_rate=float(e.get("false_positive_rate", 0.0)),
                false_alarm_rate=float(e.get("false_alarm_rate", 0.0)),
                mean_lead_time=lead,
                median_lead_time=float(e.get("median_lead_time", 0.0)),
                successful_warnings=int(e.get("successful_early_warnings") or 0),
                warnings_per_trajectory=float(e.get("warnings_per_trajectory", 0.0)),
                normalized_f1=round(f1 / max_f1, 4) if max_f1 > 0 else 0.0,
                normalized_auroc=round(auroc / max_auroc, 4) if max_auroc > 0 else 0.0,
                normalized_lead_time=round(lead / max_lead, 4) if max_lead > 0 else 0.0,
            )
        )

    # Determine best models
    best_f1_model = max(items, key=lambda x: x.f1).model_name if items else "None"
    best_auroc_model = max(items, key=lambda x: x.auroc).model_name if items else "None"
    best_lead_model = max(items, key=lambda x: x.mean_lead_time).model_name if items else "None"

    return ModelComparisonResponse(
        horizon=horizon,
        total_models=len(items),
        models=items,
        best_model_by_f1=best_f1_model,
        best_model_by_auroc=best_auroc_model,
        best_model_by_lead_time=best_lead_model,
    )
