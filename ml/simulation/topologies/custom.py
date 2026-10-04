"""Custom Topology Implementation.

Represents a configurable or user-defined interaction graph topology.
Supports arbitrary directed edge specifications or hierarchical fallback.
"""

from typing import List, Optional, Tuple, Union
import networkx as nx

from ml.config.experiment_config import TopologyType
from ml.simulation.topologies.base import BaseTopology


class CustomTopology(BaseTopology):
    """Custom topology supporting arbitrary user-defined edges or hierarchical DAGs."""

    def __init__(
        self,
        agent_ids: Optional[List[str]] = None,
        custom_edges: Optional[List[Union[Tuple[str, str], List[str]]]] = None,
    ) -> None:
        """Initialize Custom topology.
        
        Args:
            agent_ids: Sequence of agent identifiers.
            custom_edges: Optional sequence of directed (source, target) edge pairs.
        """
        self._custom_edges = custom_edges
        super().__init__(agent_ids=agent_ids)

    @property
    def topology_type(self) -> TopologyType:
        return TopologyType.CUSTOM

    def build_graph(self, agent_ids: List[str]) -> nx.DiGraph:
        """Construct graph using custom edges if provided, or hierarchical fallback."""
        self._validate_agents(agent_ids, min_agents=2)
        self._agent_ids = list(agent_ids)
        self._graph = nx.DiGraph()

        for aid in self._agent_ids:
            self._graph.add_node(aid, role="agent")

        if self._custom_edges:
            for edge in self._custom_edges:
                src, dst = edge[0], edge[1]
                if src in self._agent_ids and dst in self._agent_ids and src != dst:
                    self._graph.add_edge(src, dst, weight=1.0)
        else:
            # Hierarchical / tree fallback: root connects to first half, first half connects to second half, ring back to root
            n = len(self._agent_ids)
            for i in range(n):
                next_i = (i + 1) % n
                self._graph.add_edge(self._agent_ids[i], self._agent_ids[next_i], weight=1.0)
            # Add cross-edge from root to midpoint if n >= 4
            if n >= 4:
                self._graph.add_edge(self._agent_ids[0], self._agent_ids[n // 2], weight=1.0)

        return self._graph
