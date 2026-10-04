"""Temporal graph inspection routes."""

from typing import Optional
from fastapi import APIRouter, Query

from backend.app.schemas.graphs import RunGraphResponse
from backend.app.schemas.common import ErrorEnvelope
from backend.app.services.graph_service import get_run_temporal_graph

router = APIRouter(prefix="/graphs", tags=["Temporal Graphs"])


@router.get(
    "/{run_id}",
    response_model=RunGraphResponse,
    summary="Get Temporal Graph by Run",
    description="Retrieve nodes, directed communication edges, and temporal graph snapshots for a given run with window filtering.",
    responses={
        200: {"description": "Temporal graph slice returned successfully."},
        404: {"model": ErrorEnvelope, "description": "Run or graph data not found."},
    },
)
def get_graph(
    run_id: str,
    start_time: Optional[float] = Query(None, description="Start timestamp window bound"),
    end_time: Optional[float] = Query(None, description="End timestamp window bound"),
    step_idx: Optional[int] = Query(None, description="Specific simulation step index"),
    snapshot_idx: Optional[int] = Query(None, description="Specific snapshot index in sequence"),
) -> RunGraphResponse:
    """Retrieve temporal graph slice for a specific simulation run."""
    return get_run_temporal_graph(
        run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        step_idx=step_idx,
        snapshot_idx=snapshot_idx,
    )
