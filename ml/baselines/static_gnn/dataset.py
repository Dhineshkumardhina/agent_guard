"""Graph Dataset Builder and PyTorch Geometric Converter for Static GNNs - Phase 10.

Constructs static graph snapshots G(t) = (V, E, X) for each prediction point t:
Predicts failure within forward horizon k:
P(F(t+k) | G(<= t))

STRICT CAUSALITY GUARANTEE:
Each graph snapshot is constructed using exclusively events and interactions
observed up to and including time t (timestamp <= t and step <= t).
No future events, future graph edges, future node states, or future failure labels
are ever incorporated into G(t).
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data, Dataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = object
    Dataset = object

from ml.graph.graph_snapshot import GraphSnapshot


# 14 standard node features
NODE_FEATURE_NAMES = [
    "event_count",
    "message_count",
    "tool_call_count",
    "error_count",
    "retry_count",
    "timeout_count",
    "average_latency",
    "average_confidence",
    "average_output_quality",
    "contradiction_rate",
    "recent_failure_count",
    "incoming_interactions",
    "outgoing_interactions",
    "is_active",
]

# 10 standard edge features
EDGE_FEATURE_NAMES = [
    "interaction_count",
    "message_count",
    "average_latency",
    "average_message_length",
    "average_confidence",
    "retry_count",
    "contradiction_rate",
    "error_count",
    "timeout_count",
    "interaction_frequency",
]


def snapshot_dict_to_graph_snapshot(snap_dict: Dict[str, Any]) -> GraphSnapshot:
    """Convert snapshot dictionary (from JSONL / PredictionSample) to GraphSnapshot instance."""
    if isinstance(snap_dict, GraphSnapshot):
        return snap_dict
    return GraphSnapshot.from_dict(snap_dict)


def snapshot_to_pyg_data(
    snapshot: Union[GraphSnapshot, Dict[str, Any]],
    label: float = 0.0,
    sample_id: str = "",
    horizon: int = 1,
    topology: str = "unknown",
) -> Any:
    """Convert GraphSnapshot into a PyTorch Geometric Data object.
    
    Args:
        snapshot: GraphSnapshot instance or serialized snapshot dictionary.
        label: Target binary failure label in {0.0, 1.0}.
        sample_id: Unique prediction point identifier.
        horizon: Prediction horizon k.
        topology: Network topology string.
        
    Returns:
        PyG Data instance with x, edge_index, edge_attr, edge_weight, y, and metadata.
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric is required for static GNN dataset creation.")

    if not isinstance(snapshot, GraphSnapshot):
        snapshot = snapshot_dict_to_graph_snapshot(snapshot)

    # 1. Sorted node IDs for deterministic index mapping
    node_ids = sorted(list(snapshot.nodes.keys()))
    if not node_ids:
        # Fallback for empty graph: single dummy agent node
        node_ids = ["agent_0"]

    id_to_idx = {aid: idx for idx, aid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    # Calculate incoming and outgoing interaction counts per node
    incoming_counts: Dict[str, float] = {aid: 0.0 for aid in node_ids}
    outgoing_counts: Dict[str, float] = {aid: 0.0 for aid in node_ids}

    for (u, v), edge_attrs in snapshot.edges.items():
        cnt = float(edge_attrs.get("interaction_count", 1.0))
        if u in outgoing_counts:
            outgoing_counts[u] += cnt
        if v in incoming_counts:
            incoming_counts[v] += cnt

    # 2. Extract node feature matrix X of shape [num_nodes, 14]
    node_matrix: List[List[float]] = []
    for aid in node_ids:
        attrs = snapshot.nodes.get(aid, {})
        row = [
            float(attrs.get("event_count", 0.0)),
            float(attrs.get("message_count", 0.0)),
            float(attrs.get("tool_call_count", 0.0)),
            float(attrs.get("error_count", 0.0)),
            float(attrs.get("retry_count", 0.0)),
            float(attrs.get("timeout_count", 0.0)),
            float(attrs.get("average_latency", 0.0)),
            float(attrs.get("average_confidence", 1.0)),
            float(attrs.get("average_output_quality", 1.0)),
            float(attrs.get("contradiction_rate", 0.0)),
            float(attrs.get("recent_failure_count", 0.0)),
            float(incoming_counts.get(aid, 0.0)),
            float(outgoing_counts.get(aid, 0.0)),
            1.0 if attrs.get("is_active", True) else 0.0,
        ]
        node_matrix.append(row)

    x_tensor = torch.tensor(node_matrix, dtype=torch.float32)

    # 3. Extract edge index and edge attributes
    edge_pairs: List[Tuple[int, int]] = []
    edge_matrix: List[List[float]] = []
    edge_weights: List[float] = []

    for (u, v) in sorted(snapshot.edges.keys()):
        if u in id_to_idx and v in id_to_idx:
            u_idx = id_to_idx[u]
            v_idx = id_to_idx[v]
            edge_pairs.append((u_idx, v_idx))

            e_attrs = snapshot.edges[(u, v)]
            e_row = [
                float(e_attrs.get("interaction_count", 1.0)),
                float(e_attrs.get("message_count", 0.0)),
                float(e_attrs.get("average_latency", 0.0)),
                float(e_attrs.get("average_message_length", 0.0)),
                float(e_attrs.get("average_confidence", 1.0)),
                float(e_attrs.get("retry_count", 0.0)),
                float(e_attrs.get("contradiction_rate", 0.0)),
                float(e_attrs.get("error_count", 0.0)),
                float(e_attrs.get("timeout_count", 0.0)),
                float(e_attrs.get("interaction_frequency", 0.0)),
            ]
            edge_matrix.append(e_row)
            # Scalar weight for GCN: normalized interaction count
            edge_weights.append(max(0.1, float(e_attrs.get("interaction_count", 1.0))))

    if edge_pairs:
        edge_index_tensor = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_matrix, dtype=torch.float32)
        edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        # Empty edge tensor with valid shape [2, 0] and [0, 10]
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32)
        edge_weight_tensor = torch.empty((0,), dtype=torch.float32)

    # Target label tensor [1]
    y_tensor = torch.tensor([float(label)], dtype=torch.float32)

    # Construct PyG Data object
    data = Data(
        x=x_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        edge_weight=edge_weight_tensor,
        y=y_tensor,
        num_nodes=num_nodes,
        sample_id=sample_id,
        run_id=snapshot.run_id,
        timestamp=float(snapshot.timestamp),
        step_idx=int(snapshot.step_idx) if snapshot.step_idx is not None else 0,
        prediction_horizon=int(horizon),
        topology=str(topology),
    )
    return data


def compute_graph_diagnostics(graph_data_list: List[Any]) -> Dict[str, Any]:
    """Compute and format comprehensive structural diagnostics across graph snapshots.
    
    Diagnostics include:
    - Number of graphs
    - Average nodes per graph
    - Average edges per graph
    - Graph density
    - Isolated node count
    - Average node degree
    - Topology distribution
    - Class balance distribution
    """
    if not graph_data_list:
        return {
            "num_graphs": 0,
            "avg_nodes_per_graph": 0.0,
            "avg_edges_per_graph": 0.0,
            "avg_graph_density": 0.0,
            "total_isolated_nodes": 0,
            "avg_node_degree": 0.0,
            "topology_distribution": {},
            "class_distribution": {"positive_count": 0, "negative_count": 0, "positive_ratio": 0.0},
        }

    total_graphs = len(graph_data_list)
    nodes_counts = []
    edges_counts = []
    densities = []
    isolated_counts = 0
    total_node_degrees = 0
    total_nodes_all = 0
    topology_counts: Dict[str, int] = {}
    pos_count = 0
    neg_count = 0

    for g in graph_data_list:
        n = int(g.num_nodes)
        e = int(g.edge_index.size(1)) if hasattr(g, "edge_index") and g.edge_index is not None else 0

        nodes_counts.append(n)
        edges_counts.append(e)

        # Graph density for directed graph
        if n > 1:
            possible_edges = n * (n - 1)
            density = e / float(possible_edges)
        else:
            density = 0.0
        densities.append(density)

        # Degree calculations
        if e > 0 and hasattr(g, "edge_index"):
            degrees = [0] * n
            for edge_idx in range(e):
                src = int(g.edge_index[0, edge_idx])
                dst = int(g.edge_index[1, edge_idx])
                if 0 <= src < n:
                    degrees[src] += 1
                if 0 <= dst < n:
                    degrees[dst] += 1
            isolated_counts += sum(1 for d in degrees if d == 0)
            total_node_degrees += sum(degrees)
        else:
            isolated_counts += n

        total_nodes_all += n

        top = getattr(g, "topology", "unknown")
        topology_counts[top] = topology_counts.get(top, 0) + 1

        label = int(g.y.item()) if hasattr(g.y, "item") else int(g.y[0])
        if label == 1:
            pos_count += 1
        else:
            neg_count += 1

    avg_degree = (total_node_degrees / float(total_nodes_all)) if total_nodes_all > 0 else 0.0
    pos_ratio = (pos_count / float(total_graphs)) if total_graphs > 0 else 0.0

    return {
        "num_graphs": total_graphs,
        "avg_nodes_per_graph": round(float(np.mean(nodes_counts)), 4),
        "avg_edges_per_graph": round(float(np.mean(edges_counts)), 4),
        "avg_graph_density": round(float(np.mean(densities)), 4),
        "total_isolated_nodes": int(isolated_counts),
        "avg_node_degree": round(float(avg_degree), 4),
        "topology_distribution": topology_counts,
        "class_distribution": {
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_ratio": round(pos_ratio, 4),
        },
    }


class StaticGraphDatasetBuilder:
    """Constructs static PyG Graph datasets from prediction samples and graph sequences."""

    def __init__(self) -> None:
        self.node_feature_dim = len(NODE_FEATURE_NAMES)
        self.edge_feature_dim = len(EDGE_FEATURE_NAMES)

    def build_dataset_for_horizon(
        self,
        samples: List[Dict[str, Any]],
        graph_sequences: Dict[str, List[Dict[str, Any]]],
        horizon: int,
    ) -> List[Any]:
        """Build a list of PyG Data objects for a specific prediction horizon K.
        
        Args:
            samples: List of tabular sample dictionaries for this split.
            graph_sequences: Mapping of sample_id -> list of serialized snapshots.
            horizon: Target prediction horizon K in {1, 3, 5, 10, 20}.
            
        Returns:
            List of PyG Data objects.
        """
        filtered_samples = [s for s in samples if int(s.get("prediction_horizon", 1)) == horizon]
        graph_data_list: List[Any] = []

        for s in filtered_samples:
            s_id = s.get("sample_id", "")
            seq = graph_sequences.get(s_id, [])

            if seq:
                # The latest snapshot in sequence history represents the graph at time t: G(t)
                snapshot_dict = seq[-1]
            else:
                # Reconstruct minimal snapshot from sample metadata if sequence is not found
                snapshot_dict = {
                    "timestamp": s.get("timestamp", 0.0),
                    "run_id": s.get("run_id", "unknown"),
                    "step_idx": s.get("step_idx", 0),
                    "nodes": {},
                    "edges": {},
                }

            data = snapshot_to_pyg_data(
                snapshot=snapshot_dict,
                label=float(s.get("label", 0.0)),
                sample_id=s_id,
                horizon=horizon,
                topology=str(s.get("topology", "unknown")),
            )
            graph_data_list.append(data)

        return graph_data_list
