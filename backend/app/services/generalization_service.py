"""Generalization service exposing out-of-distribution evaluation and transfer gaps."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.app.core.config import settings
from backend.app.schemas.generalization import (
    GeneralizationMetricResponse,
    GeneralizationGapDetail,
    GeneralizationListResponse,
)
from backend.app.core.exceptions import GeneralizationNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_GEN_METRICS_CACHE: Optional[List[Dict[str, Any]]] = None
_GEN_GAPS_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_generalization_data():
    """Load and cache Phase 14 generalization metrics and gaps."""
    global _GEN_METRICS_CACHE, _GEN_GAPS_CACHE
    if _GEN_METRICS_CACHE is not None and _GEN_GAPS_CACHE is not None:
        return _GEN_METRICS_CACHE, _GEN_GAPS_CACHE

    gen_dir = settings.BASE_DIR / "results" / "generalization"
    m_path = gen_dir / "metrics" / "generalization_metrics.json"
    g_path = gen_dir / "gaps" / "generalization_gaps.json"

    metrics = []
    gaps = []

    if m_path.exists():
        try:
            with open(m_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error("Error reading generalization metrics: %s", e)

    if g_path.exists():
        try:
            with open(g_path, "r", encoding="utf-8") as f:
                gaps = json.load(f)
        except Exception as e:
            logger.error("Error reading generalization gaps: %s", e)

    _GEN_METRICS_CACHE = metrics
    _GEN_GAPS_CACHE = gaps
    return _GEN_METRICS_CACHE, _GEN_GAPS_CACHE


def _to_generalization_response(entry: Dict[str, Any], all_gaps: List[Dict[str, Any]]) -> GeneralizationMetricResponse:
    """Combine metric entry with corresponding gap details."""
    exp_id = entry.get("experiment_id", "")
    model = entry.get("model", "")
    horizon = int(entry.get("horizon", 1))

    # Find matching gaps
    matched_gaps = []
    primary_gap = None

    for g in all_gaps:
        if g.get("experiment_id") == exp_id and g.get("model") == model and int(g.get("horizon", 1)) == horizon:
            gap_detail = GeneralizationGapDetail(
                metric=g.get("metric", "f1"),
                id_value=float(g.get("id_value", 0.0)),
                ood_value=float(g.get("ood_value", 0.0)),
                gap=float(g.get("gap", 0.0)),
                pct_change=g.get("pct_change"),
                p_value=g.get("p_value"),
                statistically_significant=g.get("statistically_significant"),
                interpretation=g.get("interpretation"),
            )
            matched_gaps.append(gap_detail)
            if g.get("metric") == "f1":
                primary_gap = float(g.get("gap", 0.0))

    # Derive training and testing configurations based on dimension
    dim = entry.get("dimension", "standard")
    training_cfg = {
        "dimension": dim,
        "seed": entry.get("seed", 42),
        "split_type": "in_distribution",
        "dataset_version": entry.get("dataset_version", "agentguard_generalization_v1"),
    }
    testing_cfg = {
        "dimension": dim,
        "split_type": entry.get("split_type", "out_of_distribution"),
        "sample_count": entry.get("sample_count", 0),
        "positive_count": entry.get("positive_count", 0),
        "negative_count": entry.get("negative_count", 0),
    }

    return GeneralizationMetricResponse(
        experiment_id=exp_id,
        dimension=dim,
        split_type=entry.get("split_type", "in_distribution"),
        model=model,
        horizon=horizon,
        seed=int(entry.get("seed", 42)),
        dataset_version=entry.get("dataset_version", "agentguard_generalization_v1"),
        training_configuration=training_cfg,
        testing_configuration=testing_cfg,
        metrics=entry,
        generalization_gap=primary_gap,
        gaps=matched_gaps,
    )


def get_generalization(
    dimension: Optional[str] = None,
    model: Optional[str] = None,
    horizon: Optional[int] = None,
    split_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> GeneralizationListResponse:
    """Retrieve filtered and paginated generalization metrics."""
    metrics, gaps = _load_generalization_data()

    filtered = []
    for entry in metrics:
        dim = entry.get("dimension", "")
        mname = entry.get("model", "")
        h = int(entry.get("horizon", 1))
        stype = entry.get("split_type", "")

        if dimension and dimension.lower() not in dim.lower():
            continue
        if model and model.lower() not in mname.lower():
            continue
        if horizon is not None and h != horizon:
            continue
        if split_type and split_type.lower() not in stype.lower():
            continue

        filtered.append(_to_generalization_response(entry, gaps))

    total = len(filtered)
    items = filtered[offset : offset + limit]
    has_more = (offset + limit) < total

    return GeneralizationListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_generalization_by_id(experiment_id: str) -> GeneralizationMetricResponse:
    """Retrieve generalization results for an experiment ID."""
    metrics, gaps = _load_generalization_data()
    target_clean = experiment_id.lower().strip()

    for entry in metrics:
        eid = entry.get("experiment_id", "").lower().strip()
        if eid == target_clean or experiment_id.lower() == entry.get("experiment_id", "").lower():
            return _to_generalization_response(entry, gaps)

    raise GeneralizationNotFoundException(experiment_id=experiment_id)
