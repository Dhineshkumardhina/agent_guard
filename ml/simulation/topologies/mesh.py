"""Mesh Topology Implementation.

Represents a distributed peer-to-peer mesh network:
A ↔ B
↕   ↕
C ↔ D
All participating agents are directly interconnected with bidirectional communication channels.
"""

from typing import List, Optional
import networkx as nx

from ml.config.experiment_config import TopologyType
from ml.simulation.topologies.base import BaseTopology


class MeshTopology(BaseTopology):
    """Fully interconnected mesh topology where all distinct agent pairs can communicate."""

    def __init__(self, agent_ids: Optional[List[str]] = None) -> None:
        """Initialize Mesh topology.
        
        Args:
            agent_ids: Sequence of agent identifiers.
        """
        super().__init__(agent_ids=agent_ids)

    @property
    def topology_type(self) -> TopologyType:
        return TopologyType.MESH

    def build_graph(self, agent_ids: List[str]) -> nx.DiGraph:
        """Construct the fully connected bidirectional mesh directed graph."""
        self._validate_agents(agent_ids, min_agents=2)
        self._agent_ids = list(agent_ids)
        self._graph = nx.DiGraph()

        for aid in self._agent_ids:
            self._graph.add_node(aid, role="peer")

        # Add all-to-all bidirectional edges (excluding self-loops)
        for src in self._agent_ids:
            for dst in self._agent_ids:
                if src != dst:
                    self._graph.add_edge(src, dst, weight=1.0)

        return self._graph
