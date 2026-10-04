"""Experiment service layer managing research benchmarks and pipeline configurations."""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.models import Experiment, Run
from backend.app.schemas.experiments import ExperimentResponse, ExperimentListResponse
from backend.app.core.exceptions import ExperimentNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def _to_experiment_response(exp: Experiment, run_count: int = 0) -> ExperimentResponse:
    """Map SQLAlchemy Experiment model to ExperimentResponse schema."""
    config = exp.config or {}
    model_name = config.get("model") or (exp.name.split(" Benchmark")[0] if " Benchmark" in exp.name else None)
    ds_version = config.get("dataset_version") or "agentguard_dataset_v1"
    seed = int(config.get("seed", 42))

    return ExperimentResponse(
        experiment_id=exp.id,
        name=exp.name,
        description=exp.description,
        model=model_name,
        dataset_version=ds_version,
        configuration=config,
        seed=seed,
        creation_time=exp.created_at,
        status=exp.status,
        run_count=run_count,
    )


def get_experiments(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    model: Optional[str] = None,
) -> ExperimentListResponse:
    """Retrieve paginated research experiments with optional filtering."""
    query = db.query(Experiment)
    if status:
        query = query.filter(func.lower(Experiment.status) == status.lower())

    total = query.count()
    records = query.order_by(Experiment.id).offset(offset).limit(limit).all()

    items = []
    for exp in records:
        rcount = db.query(Run).filter(Run.experiment_id == exp.id).count()
        exp_resp = _to_experiment_response(exp, run_count=rcount)
        if model:
            if exp_resp.model and model.lower() not in exp_resp.model.lower():
                continue
        items.append(exp_resp)

    has_more = (offset + limit) < total

    return ExperimentListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_experiment_by_id(db: Session, experiment_id: str) -> ExperimentResponse:
    """Retrieve a single experiment by ID or raise ExperimentNotFoundException."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ExperimentNotFoundException(experiment_id=experiment_id)

    rcount = db.query(Run).filter(Run.experiment_id == exp.id).count()
    return _to_experiment_response(exp, run_count=rcount)
