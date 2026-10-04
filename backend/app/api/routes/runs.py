"""Simulation trajectory routes exposing runs, events, and failure manifests."""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, PaginationParams
from backend.app.schemas.runs import (
    RunResponse,
    RunDetailResponse,
    RunListResponse,
    RunFailureResponse,
    RunFailureListResponse,
)
from backend.app.schemas.events import EventListResponse
from backend.app.schemas.predictions import PredictionListResponse
from backend.app.schemas.graphs import RunGraphResponse
from backend.app.schemas.explainability import ExplanationListResponse
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.run_service import (
    get_runs,
    get_run_by_id,
    get_run_events,
    get_run_failures,
)
from backend.app.services.prediction_service import get_predictions_by_run
from backend.app.services.graph_service import get_run_temporal_graph
from backend.app.services.explainability_service import get_explanations_by_run

router = APIRouter(prefix="/runs", tags=["Simulation Runs"])


@router.get(
    "",
    response_model=RunListResponse,
    summary="List Simulation Runs",
    description="Retrieve paginated simulation trajectory runs with multi-attribute filtering.",
    responses={
        200: {"description": "List of simulation runs matching query criteria."},
    },
)
def list_runs(
    task: Optional[str] = Query(None, description="Filter by task domain (research, coding, analysis, planning)"),
    topology: Optional[str] = Query(None, description="Filter by communication topology (pipeline, star, mesh, custom)"),
    status: Optional[str] = Query(None, description="Filter by run outcome status (completed, failed)"),
    agent_count: Optional[int] = Query(None, description="Filter by number of participating agents"),
    start_date: Optional[datetime] = Query(None, description="Filter runs created on or after this timestamp"),
    end_date: Optional[datetime] = Query(None, description="Filter runs created on or before this timestamp"),
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
) -> RunListResponse:
    """Retrieve filtered and paginated simulation trajectories."""
    return get_runs(
        db=db,
        limit=pagination.limit,
        offset=pagination.offset,
        task=task,
        topology=topology,
        status=status,
        agent_count=agent_count,
        start_date=start_date,
        end_date=end_date,
    )


@router.get(
    "/{run_id}",
    response_model=RunDetailResponse,
    summary="Get Run Details",
    description="Retrieve full details for a simulation run including participating agents and failure occurrences.",
    responses={
        200: {"description": "Simulation run details returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Run not found."},
    },
)
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
) -> RunDetailResponse:
    """Retrieve detailed simulation run information."""
    return get_run_by_id(db=db, run_id=run_id)


@router.get(
    "/{run_id}/events",
    response_model=EventListResponse,
    summary="Get Run Events",
    description="Retrieve chronological telemetry communication events recorded during this simulation run.",
    responses={
        200: {"description": "Telemetry event stream returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Run not found."},
    },
)
def get_events_for_run(
    run_id: str,
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
) -> EventListResponse:
    """Retrieve telemetry events for a specific run."""
    return get_run_events(
        db=db,
        run_id=run_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get(
    "/{run_id}/failures",
    response_model=RunFailureListResponse,
    summary="Get Run Failures",
    description="Retrieve cascading or component failures manifested during this simulation run.",
    responses={
        200: {"description": "Failure records returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Run not found."},
    },
)
def get_failures_for_run(
    run_id: str,
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
) -> RunFailureListResponse:
    """Retrieve failure records for a specific run."""
    return get_run_failures(
        db=db,
        run_id=run_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get(
    "/{run_id}/graph",
    response_model=RunGraphResponse,
    summary="Get Run Temporal Graph",
    description="Retrieve temporal graph snapshots, nodes, edges, and behavioral features for a specific run with window filtering.",
    responses={
        200: {"description": "Temporal graph slice returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Run or graph data not found."},
    },
)
def get_graph_for_run(
    run_id: str,
    start_time: Optional[float] = Query(None, description="Start timestamp window bound"),
    end_time: Optional[float] = Query(None, description="End timestamp window bound"),
    step_idx: Optional[int] = Query(None, description="Specific simulation step index"),
    snapshot_idx: Optional[int] = Query(None, description="Specific snapshot index in sequence"),
) -> RunGraphResponse:
    """Retrieve temporal graph slice for this run."""
    return get_run_temporal_graph(
        run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        step_idx=step_idx,
        snapshot_idx=snapshot_idx,
    )


@router.get(
    "/{run_id}/predictions",
    response_model=PredictionListResponse,
    summary="Get Run Predictions",
    description="Retrieve model inference predictions and risk assessments generated for this simulation run.",
    responses={
        200: {"description": "Prediction list returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Run not found."},
    },
)
def get_predictions_for_run(
    run_id: str,
    model: Optional[str] = Query(None, description="Filter predictions by model architecture"),
    horizon: Optional[int] = Query(None, description="Filter predictions by prediction horizon k"),
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
) -> PredictionListResponse:
    """Retrieve predictions for this run."""
    return get_predictions_by_run(
        db=db,
        run_id=run_id,
        limit=pagination.limit,
        offset=pagination.offset,
        model=model,
        horizon=horizon,
    )


@router.get(
    "/{run_id}/explanations",
    response_model=ExplanationListResponse,
    summary="Get Run Explanations",
    description="Retrieve failure-risk explainability attributions and case studies for this simulation run.",
    responses={
        200: {"description": "Explanation records returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Run not found."},
    },
)
def get_explanations_for_run(
    run_id: str,
    pagination: PaginationParams = Depends(),
) -> ExplanationListResponse:
    """Retrieve explanation reports for this run."""
    return get_explanations_by_run(
        run_id=run_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
