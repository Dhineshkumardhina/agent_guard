"""Graph Snapshot Serialization and Deserialization.

Supports JSON serialization and file persistence for research workflows,
preserving timestamp, run_id, nodes, edges, node features, edge features, and metadata.
"""

from pathlib import Path
from typing import Dict, Any, Union
import json

from ml.graph.graph_snapshot import GraphSnapshot


def serialize_snapshot(snapshot: GraphSnapshot) -> str:
    """Serialize a GraphSnapshot instance into a JSON string with zero feature loss."""
    return json.dumps(snapshot.to_dict(), indent=2)


def deserialize_snapshot(data: Union[str, Dict[str, Any]]) -> GraphSnapshot:
    """Deserialize JSON string or dictionary into a GraphSnapshot instance."""
    if isinstance(data, str):
        payload = json.loads(data)
    elif isinstance(data, dict):
        payload = data
    else:
        raise TypeError(f"Expected str or dict, got {type(data)}")

    return GraphSnapshot.from_dict(payload)


def save_snapshot(snapshot: GraphSnapshot, filepath: Union[str, Path]) -> Path:
    """Save GraphSnapshot to JSON file on disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialize_snapshot(snapshot))
    return path


def load_snapshot(filepath: Union[str, Path]) -> GraphSnapshot:
    """Load GraphSnapshot from JSON file on disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return deserialize_snapshot(f.read())
