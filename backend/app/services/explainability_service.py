"""Explainability service exposing feature, agent, and edge attributions."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.app.core.config import settings
from backend.app.schemas.explainability import (
    ExplanationResponse,
    ExplanationListResponse,
)
from backend.app.core.exceptions import ExplanationNotFoundException, RunNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_EXPLANATIONS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_explanations() -> Dict[str, Dict[str, Any]]:
    """Load and index all Phase 15 explanation case studies and summaries."""
    global _EXPLANATIONS_CACHE
    if _EXPLANATIONS_CACHE is not None:
        return _EXPLANATIONS_CACHE

    explanations: Dict[str, Dict[str, Any]] = {}
    exp_dir = settings.BASE_DIR / "results" / "explainability"

    # 1. Load case studies files
    cases_dir = exp_dir / "case_studies"
    if cases_dir.exists():
        for cfile in cases_dir.glob("*.json"):
            try:
                with open(cfile, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                eid = cdata.get("explanation_id") or f"exp_{cfile.stem}"
                cdata["explanation_id"] = eid
                explanations[eid] = cdata
            except Exception as e:
                logger.warning("Error reading case study %s: %s", cfile, e)

    # 2. Check case studies summary for any missing entries
    summary_path = exp_dir / "dashboard_data" / "case_studies_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            items = sdata if isinstance(sdata, list) else list(sdata.values())
            for idx, item in enumerate(items):
                eid = item.get("explanation_id") or f"exp_summary_{item.get('run_id')}_{idx}"
                if eid not in explanations:
                    item["explanation_id"] = eid
                    explanations[eid] = item
        except Exception as e:
            logger.warning("Error reading case studies summary: %s", e)

    _EXPLANATIONS_CACHE = explanations
    return _EXPLANATIONS_CACHE


def _to_explanation_response(data: Dict[str, Any]) -> ExplanationResponse:
    """Map raw explanation record to ExplanationResponse schema."""
    edges = data.get("important_edges") or data.get("important_interactions") or []
    return ExplanationResponse(
        explanation_id=data.get("explanation_id", "unknown_exp"),
        run_id=data.get("run_id", "unknown_run"),
        sample_id=data.get("sample_id"),
        case_type=data.get("case_type"),
        predicted_probability=float(data.get("predicted_probability", 0.0)),
        prediction_horizon=int(data.get("prediction_horizon", 1)),
        predicted_label=int(data.get("predicted_label", 0)),
        true_label=data.get("true_label"),
        threshold=float(data.get("threshold", 0.5)),
        explanation_method=data.get("explanation_method", "Integrated Gradients + Subgraph Attribution"),
        model_name=data.get("model_name", "Temporal GNN"),
        model_version=data.get("model_version", "v1"),
        dataset_version=data.get("dataset_version", "agentguard_generalization_v1"),
        important_features=data.get("important_features", []),
        important_agents=data.get("important_agents", []),
        important_edges=edges,
        important_events=data.get("important_events", []),
        high_level_summary=data.get("high_level_summary"),
        causality_disclaimer=data.get(
            "causality_disclaimer",
            "Attributions denote predictive statistical associations within the temporal graph, not proven physical causality.",
        ),
    )


def get_explanations(
    run_id: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> ExplanationListResponse:
    """Retrieve paginated explanations with optional filters."""
    indexed = _load_explanations()
    filtered = []

    for eid, data in indexed.items():
        rid = data.get("run_id", "")
        mname = data.get("model_name", "")

        if run_id and run_id.lower() not in rid.lower():
            continue
        if model and model.lower() not in mname.lower():
            continue

        filtered.append(_to_explanation_response(data))

    total = len(filtered)
    items = filtered[offset : offset + limit]
    has_more = (offset + limit) < total

    return ExplanationListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_explanation_by_id(explanation_id: str) -> ExplanationResponse:
    """Retrieve a single explanation by ID."""
    indexed = _load_explanations()
    target_clean = explanation_id.lower().strip()

    # Exact or fuzzy match
    for eid, data in indexed.items():
        if eid.lower().strip() == target_clean or explanation_id.lower() in eid.lower():
            return _to_explanation_response(data)

    raise ExplanationNotFoundException(identifier=explanation_id)


def get_explanations_by_run(
    run_id: str,
    limit: int = 50,
    offset: int = 0,
) -> ExplanationListResponse:
    """Retrieve all explanations for a specific simulation run."""
    return get_explanations(run_id=run_id, limit=limit, offset=offset)
