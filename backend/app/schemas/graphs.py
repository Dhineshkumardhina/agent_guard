"""Pydantic schemas for temporal graph representations, snapshots, and feature matrices."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GraphSnapshotSchema(BaseModel):
    """Schema for a temporal graph snapshot at a specific point in time."""
    snapshot_idx: int = Field(..., description="Chronological index of this snapshot")
    timestamp: float = Field(..., description="Timestamp of the snapshot")
    step_idx: int = Field(..., description="Discrete simulation step index")
    num_nodes: int = Field(..., description="Number of active agent nodes")
    num_edges: int = Field(..., description="Number of directed communication edges")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="List of nodes with behavioral feature states")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="List of edges with interaction features")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Snapshot metadata")


class RunGraphResponse(BaseModel):
    """Schema for run graph slice response."""
    run_id: str = Field(..., description="Simulation run identifier")
    total_snapshots: int = Field(..., description="Total available snapshots for run")
    returned_snapshots: int = Field(..., description="Number of snapshots returned in current window")
    timestamps: List[float] = Field(default_factory=list, description="Sorted list of timestamps present in response")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Unique nodes participating across the graph slice")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Unique communication edges observed")
    node_features: Dict[str, Any] = Field(default_factory=dict, description="Feature dictionary mapping node_id to feature vector/dict")
    edge_features: Dict[str, Any] = Field(default_factory=dict, description="Feature dictionary mapping edge key to feature vector/dict")
    temporal_snapshots: List[GraphSnapshotSchema] = Field(default_factory=list, description="Sequence of temporal snapshots within requested window")
    window: Dict[str, Any] = Field(default_factory=dict, description="Applied window filter bounds (start_time, end_time, snapshot_idx)")
