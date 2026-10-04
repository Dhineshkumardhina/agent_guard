"""Temporal graph service streaming and slicing temporal graph snapshots."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from backend.app.core.config import settings
from backend.app.schemas.graphs import RunGraphResponse, GraphSnapshotSchema
from backend.app.core.exceptions import GraphNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def _find_graph_history_for_run(run_id: str) -> Optional[List[Dict[str, Any]]]:
    """Locate temporal graph history for a run by streaming JSONL/JSON datasets."""
    # 1. Search processed generalization graph sequences
    data_dir = settings.BASE_DIR / "data" / "processed" / "agentguard_generalization_v1"
    jsonl_files = [
        data_dir / "graph_sequences_val.jsonl",
        data_dir / "graph_sequences_test.jsonl",
        data_dir / "graph_sequences_train.jsonl",
    ]

    for jpath in jsonl_files:
        if not jpath.exists():
            continue
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    # Quick substring check to avoid parsing irrelevant lines
                    if f'"{run_id}"' in line:
                        record = json.loads(line)
                        if record.get("run_id") == run_id:
                            history = record.get("temporal_graph_history")
                            if history:
                                return history
        except Exception as e:
            logger.warning("Error scanning %s for run %s: %s", jpath, run_id, e)

    # 2. Check standalone graph snapshots in data/graphs
    graphs_dir = settings.BASE_DIR / "data" / "graphs"
    if graphs_dir.exists():
        for gfile in graphs_dir.glob("*.json"):
            if run_id in gfile.stem or gfile.stem in run_id:
                try:
                    with open(gfile, "r", encoding="utf-8") as f:
                        content = json.load(f)
                    if isinstance(content, dict):
                        if "temporal_graph_history" in content:
                            return content["temporal_graph_history"]
                        # Standalone snapshot
                        return [{
                            "timestamp": content.get("timestamp", 0.0),
                            "run_id": run_id,
                            "step_idx": content.get("step_idx", 0),
                            "num_nodes": len(content.get("nodes", {})),
                            "num_edges": len(content.get("edges", {})),
                            "nodes": content.get("nodes", {}),
                            "edges": content.get("edges", {}),
                            "metadata": content.get("metadata", {}),
                        }]
                except Exception as e:
                    logger.warning("Error reading %s: %s", gfile, e)

    return None


def get_run_temporal_graph(
    run_id: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    step_idx: Optional[int] = None,
    snapshot_idx: Optional[int] = None,
) -> RunGraphResponse:
    """Retrieve filtered temporal graph representation for a run."""
    raw_history = _find_graph_history_for_run(run_id)
    if not raw_history:
        raise GraphNotFoundException(run_id=run_id)

    total_snapshots = len(raw_history)

    # Apply filters
    filtered_snapshots: List[Dict[str, Any]] = []
    for idx, snap in enumerate(raw_history):
        ts = float(snap.get("timestamp", 0.0))
        s_step = int(snap.get("step_idx", 0))

        if snapshot_idx is not None and idx != snapshot_idx:
            continue
        if step_idx is not None and s_step != step_idx:
            continue
        if start_time is not None and ts < start_time:
            continue
        if end_time is not None and ts > end_time:
            continue

        filtered_snapshots.append(snap)

    # Convert snapshots to schemas and aggregate unique nodes/edges
    snapshots_schema: List[GraphSnapshotSchema] = []
    unique_nodes: Dict[str, Dict[str, Any]] = {}
    unique_edges: Dict[str, Dict[str, Any]] = {}
    timestamps: List[float] = []

    for s_idx, snap in enumerate(filtered_snapshots):
        ts = float(snap.get("timestamp", 0.0))
        timestamps.append(ts)

        # Handle nodes format (dict of node_id -> attrs or list)
        snap_nodes_raw = snap.get("nodes", {})
        snap_nodes_list = []
        if isinstance(snap_nodes_raw, dict):
            for nid, nattrs in snap_nodes_raw.items():
                node_entry = {"id": nid, **(nattrs if isinstance(nattrs, dict) else {})}
                snap_nodes_list.append(node_entry)
                unique_nodes[nid] = node_entry
        elif isinstance(snap_nodes_raw, list):
            for n in snap_nodes_raw:
                nid = n.get("id") or n.get("agent_id") or str(n)
                node_entry = {"id": str(nid), **n}
                snap_nodes_list.append(node_entry)
                unique_nodes[str(nid)] = node_entry

        # Handle edges format (dict of edge_key -> attrs or list)
        snap_edges_raw = snap.get("edges", {})
        snap_edges_list = []
        if isinstance(snap_edges_raw, dict):
            for ekey, eattrs in snap_edges_raw.items():
                attrs = eattrs if isinstance(eattrs, dict) else {}
                src = attrs.get("source_agent") or (ekey.split("->")[0] if "->" in ekey else "unknown")
                tgt = attrs.get("target_agent") or (ekey.split("->")[1] if "->" in ekey else "unknown")
                edge_entry = {"source": src, "target": tgt, "key": ekey, **attrs}
                snap_edges_list.append(edge_entry)
                unique_edges[ekey] = edge_entry
        elif isinstance(snap_edges_raw, list):
            for e in snap_edges_raw:
                src = e.get("source") or e.get("source_agent") or "unknown"
                tgt = e.get("target") or e.get("target_agent") or "unknown"
                ekey = f"{src}->{tgt}"
                edge_entry = {"source": src, "target": tgt, "key": ekey, **e}
                snap_edges_list.append(edge_entry)
                unique_edges[ekey] = edge_entry

        snapshots_schema.append(
            GraphSnapshotSchema(
                snapshot_idx=s_idx,
                timestamp=ts,
                step_idx=int(snap.get("step_idx", 0)),
                num_nodes=len(snap_nodes_list),
                num_edges=len(snap_edges_list),
                nodes=snap_nodes_list,
                edges=snap_edges_list,
                metadata=snap.get("metadata", {}),
            )
        )

    return RunGraphResponse(
        run_id=run_id,
        total_snapshots=total_snapshots,
        returned_snapshots=len(snapshots_schema),
        timestamps=sorted(timestamps),
        nodes=list(unique_nodes.values()),
        edges=list(unique_edges.values()),
        node_features=unique_nodes,
        edge_features=unique_edges,
        temporal_snapshots=snapshots_schema,
        window={
            "start_time": start_time,
            "end_time": end_time,
            "step_idx": step_idx,
            "snapshot_idx": snapshot_idx,
        },
    )
