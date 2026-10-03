"""Star Topology Implementation.

Represents a centralized coordinator / hub-and-spoke architecture:
       B
       |
    C → A → D
       |
       E
A is the central coordinator hub. Peripheral leaves communicate with and through A.
Leaves do not communicate directly with each other without routing through A.
"""

from typing import List, Optional
import networkx as nx

from ml.config.experiment_config import TopologyType
from ml.simulation.topologies.base import BaseTopology


class StarTopology(BaseTopology):
    """Centralized star topology with a coordinator hub and peripheral leaf agents."""

    def __init__(
        self,
        agent_ids: Optional[List[str]] = None,
        hub_agent_id: Optional[str] = None,
    ) -> None:
        """Initialize Star topology.
        
        Args:
            agent_ids: Sequence of agent IDs. First agent is hub if hub_agent_id is not given.
            hub_agent_id: Optional explicit ID of the central coordinator agent.
        """
        self._hub_agent_id = hub_agent_id
        super().__init__(agent_ids=agent_ids)

    @property
    def topology_type(self) -> TopologyType:
        return TopologyType.STAR

    @property
    def hub_agent_id(self) -> Optional[str]:
        """Identifier of the central coordinator hub agent."""
        return self._hub_agent_id

    @property
    def leaf_agent_ids(self) -> List[str]:
        """Identifiers of the peripheral leaf agents."""
        if not self._hub_agent_id:
            return []
        return [aid for aid in self._agent_ids if aid != self._hub_agent_id]

    def build_graph(self, agent_ids: List[str]) -> nx.DiGraph:
        """Construct the star directed graph with central hub and spokes."""
        self._validate_agents(agent_ids, min_agents=2)
        self._agent_ids = list(agent_ids)
        self._graph = nx.DiGraph()

        # Designate hub
        if self._hub_agent_id is not None:
            if self._hub_agent_id not in self._agent_ids:
                raise ValueError(
                    f"Specified hub_agent_id '{self._hub_agent_id}' not found in agent_ids: {self._agent_ids}"
                )
        else:
            self._hub_agent_id = self._agent_ids[0]

        hub = self._hub_agent_id

        for aid in self._agent_ids:
            role_type = "hub" if aid == hub else "leaf"
            self._graph.add_node(aid, topology_role=role_type)

        # Star links: Hub can talk to all leaves, and all leaves can talk to hub
        for aid in self._agent_ids:
            if aid != hub:
                self._graph.add_edge(hub, aid, weight=1.0)
                self._graph.add_edge(aid, hub, weight=1.0)

        return self._graph
