"""GraphSnapshot: Discrete Temporal Interaction Graph Representation G(t) = (V(t), E(t), X(t)).

Provides NetworkX directed graph representation, node/edge attribute storage,
and prediction target readiness without future data leakage.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import networkx as nx


class GraphSnapshot:
    """Represents a discrete snapshot of the multi-agent interaction graph at timestamp t."""

    def __init__(
        self,
        timestamp: float,
        run_id: str,
        step_idx: Optional[int] = None,
        nodes: Optional[Dict[str, Dict[str, Any]]] = None,
        edges: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize graph snapshot.
        
        Args:
            timestamp: The temporal cutoff t of this snapshot.
            run_id: Identifier of the associated simulation trajectory.
            step_idx: Optional step index cutoff.
            nodes: Mapping of agent_id -> node feature dictionary.
            edges: Mapping of (source, target) -> edge feature dictionary.
            metadata: Supplemental dictionary (topology, windowing params, future labels).
        """
        self.timestamp: float = round(timestamp, 4)
        self.run_id: str = run_id
        self.step_idx: Optional[int] = step_idx
        self.nodes: Dict[str, Dict[str, Any]] = nodes or {}
        self.edges: Dict[Tuple[str, str], Dict[str, Any]] = edges or {}
        self.metadata: Dict[str, Any] = metadata or {}

        # Underlying NetworkX directed graph
        self._graph: nx.DiGraph = nx.DiGraph()
        self._sync_networkx()

    def _sync_networkx(self) -> None:
        """Synchronize node and edge features into the internal NetworkX graph."""
        self._graph = nx.DiGraph()
        self._graph.graph["timestamp"] = self.timestamp
        self._graph.graph["run_id"] = self.run_id
        self._graph.graph["step_idx"] = self.step_idx
        for k, v in self.metadata.items():
            self._graph.graph[k] = v

        for node_id, attrs in self.nodes.items():
            self._graph.add_node(node_id, **attrs)

        for (u, v), attrs in self.edges.items():
            self._graph.add_edge(u, v, **attrs)

    @property
    def graph(self) -> nx.DiGraph:
        """Internal NetworkX directed graph instance."""
        return self._graph

    @property
    def num_nodes(self) -> int:
        """Count of active agent nodes in this snapshot."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Count of directed interaction channels in this snapshot."""
        return len(self.edges)

    def to_networkx(self) -> nx.DiGraph:
        """Return a copy of the NetworkX directed graph."""
        return self._graph.copy()

    def has_node(self, agent_id: str) -> bool:
        """Check if an agent node is present."""
        return agent_id in self.nodes

    def has_edge(self, source: str, target: str) -> bool:
        """Check if a directed interaction edge exists from source to target."""
        return (source, target) in self.edges

    def get_node_features(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve features for a specific node."""
        return self.nodes.get(agent_id)

    def get_edge_features(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """Retrieve features for a specific directed edge."""
        return self.edges.get((source, target))

    def node_feature_matrix(self, feature_keys: Optional[List[str]] = None) -> Tuple[List[str], List[List[float]]]:
        """Extract ordered numerical node feature vectors for downstream ML/GNN tensor conversion.
        
        Args:
            feature_keys: Optional list of numerical keys to extract.
            
        Returns:
            Tuple of (ordered_agent_ids, feature_matrix).
        """
        keys = feature_keys or [
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
        ]
        agent_ids = sorted(list(self.nodes.keys()))
        matrix: List[List[float]] = []
        for aid in agent_ids:
            attrs = self.nodes[aid]
            vec = [float(attrs.get(k, 0.0)) for k in keys]
            matrix.append(vec)
        return agent_ids, matrix

    def edge_index(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Extract edge index tuples for GNN/PyG compatibility based on sorted node ordering."""
        agent_ids = sorted(list(self.nodes.keys()))
        id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        edges_idx: List[Tuple[int, int]] = []
        for (u, v) in sorted(self.edges.keys()):
            if u in id_to_idx and v in id_to_idx:
                edges_idx.append((id_to_idx[u], id_to_idx[v]))
        return agent_ids, edges_idx

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot into clean serializable dictionary."""
        # Convert tuple edge keys to string keys "u->v" for JSON safety
        edges_serialized = {
            f"{u}->{v}": attrs
            for (u, v), attrs in self.edges.items()
        }
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "nodes": self.nodes,
            "edges": edges_serialized,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphSnapshot":
        """Reconstruct GraphSnapshot from dictionary."""
        edges_deserialized: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for edge_str, attrs in data.get("edges", {}).items():
            if "->" in edge_str:
                u, v = edge_str.split("->", 1)
                edges_deserialized[(u, v)] = attrs
            elif isinstance(edge_str, (tuple, list)):
                edges_deserialized[(edge_str[0], edge_str[1])] = attrs

        return cls(
            timestamp=data["timestamp"],
            run_id=data["run_id"],
            step_idx=data.get("step_idx"),
            nodes=data.get("nodes", {}),
            edges=edges_deserialized,
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"<GraphSnapshot t={self.timestamp} nodes={self.num_nodes} edges={self.num_edges} run_id='{self.run_id}'>"
