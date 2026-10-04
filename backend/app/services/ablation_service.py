"""Ablation study service layer exposing component attribution and architecture impact."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.app.core.config import settings
from backend.app.schemas.ablations import AblationResponse, AblationListResponse
from backend.app.core.exceptions import AblationNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_ABLATIONS_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_ablation_metrics() -> List[Dict[str, Any]]:
    """Load and cache ablation metrics from Phase 13 results."""
    global _ABLATIONS_CACHE
    if _ABLATIONS_CACHE is not None:
        return _ABLATIONS_CACHE

    ablation_path = settings.BASE_DIR / "results" / "ablation" / "metrics" / "ablation_metrics.json"
    if not ablation_path.exists():
        logger.warning("Ablation metrics file not found at %s", ablation_path)
        return []

    try:
        with open(ablation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _ABLATIONS_CACHE = data
            return data
    except Exception as e:
        logger.error("Failed to load ablation metrics: %s", e)
        return []


def _to_ablation_response(entry: Dict[str, Any]) -> AblationResponse:
    """Map raw ablation metric dictionary to AblationResponse schema."""
    return AblationResponse(
        experiment_id=entry.get("experiment_id", "unknown"),
        parent_experiment_id=entry.get("parent_experiment_id"),
        ablation_name=entry.get("ablation_name", "Unknown Ablation"),
        removed_component=entry.get("removed_component", "None"),
        baseline_model=entry.get("model", "temporal_gnn"),
        horizon=int(entry.get("horizon", 1)),
        seed=int(entry.get("seed", 42)),
        dataset_version=entry.get("dataset_version", "agentguard_dataset_v1"),
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
        successful_warnings=int(entry.get("successful_warnings", 0)),
        warnings_per_trajectory=float(entry.get("warnings_per_trajectory", 0.0)),
        metrics=entry,
    )


def get_ablations(
    model: Optional[str] = None,
    horizon: Optional[int] = None,
    seed: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
) -> AblationListResponse:
    """Retrieve paginated ablation study results with optional filters."""
    data = _load_ablation_metrics()

    filtered = []
    for entry in data:
        mname = entry.get("model", "")
        h = int(entry.get("horizon", 1))
        s = int(entry.get("seed", 42))

        if model and model.lower() not in mname.lower():
            continue
        if horizon is not None and h != horizon:
            continue
        if seed is not None and s != seed:
            continue

        filtered.append(_to_ablation_response(entry))

    total = len(filtered)
    items = filtered[offset : offset + limit]
    has_more = (offset + limit) < total

    return AblationListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_ablation_by_id(experiment_id: str) -> AblationResponse:
    """Retrieve a single ablation experiment by ID."""
    data = _load_ablation_metrics()
    target_clean = experiment_id.lower().strip()

    for entry in data:
        eid = entry.get("experiment_id", "").lower().strip()
        if eid == target_clean or experiment_id.lower() == entry.get("experiment_id", "").lower():
            return _to_ablation_response(entry)

    raise AblationNotFoundException(experiment_id=experiment_id)
