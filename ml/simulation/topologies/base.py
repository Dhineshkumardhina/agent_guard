"""Base Topology Abstraction for Multi-Agent Communication Networks.

Provides NetworkX graph representation, validation rules, and routing verification.
"""

from typing import List, Optional
import networkx as nx

from ml.config.experiment_config import TopologyType
from ml.simulation.interfaces import TopologyInterface


class BaseTopology(TopologyInterface):
    """Base class providing common graph management and routing checks."""

    def __init__(self, agent_ids: Optional[List[str]] = None) -> None:
        """Initialize topology with optional agent IDs list.
        
        Args:
            agent_ids: Sequence of unique agent identifier strings.
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self._agent_ids: List[str] = []
        if agent_ids:
            self.build_graph(agent_ids)

    @property
    def graph(self) -> nx.DiGraph:
        """Underlying NetworkX directed graph."""
        return self._graph

    @property
    def agent_ids(self) -> List[str]:
        """Registered agent IDs in this topology."""
        return list(self._agent_ids)

    def _validate_agents(self, agent_ids: List[str], min_agents: int = 2) -> None:
        """Ensure agent list meets requirements for a valid topology."""
        if not agent_ids:
            raise ValueError("agent_ids list cannot be empty")
        if len(agent_ids) < min_agents:
            raise ValueError(
                f"Topology '{self.topology_type.value}' requires at least {min_agents} agents, "
                f"got {len(agent_ids)}"
            )
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError(f"Agent IDs must be unique, got duplicates in: {agent_ids}")

    def can_communicate(self, source: str, target: str) -> bool:
        """Determine whether source agent is permitted to send a direct message to target.
        
        Args:
            source: Originating agent identifier.
            target: Destination agent identifier.
            
        Returns:
            True if a directed edge exists from source to target in the topology.
        """
        if source not in self._graph or target not in self._graph:
            return False
        if source == target:
            return False
        return self._graph.has_edge(source, target)

    def get_allowed_targets(self, source: str) -> List[str]:
        """Get all permissible recipient agents for a given source agent."""
        if source not in self._graph:
            return []
        return list(self._graph.successors(source))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type='{self.topology_type.value}' agents={len(self._agent_ids)}>"
