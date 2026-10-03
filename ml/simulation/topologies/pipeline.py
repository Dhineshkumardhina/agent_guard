"""Pipeline Topology Implementation.

Represents a linear sequential workflow: A → B → C → D.
Information flows directly from each predecessor agent to its immediate successor.
"""

from typing import List, Optional
import networkx as nx

from ml.config.experiment_config import TopologyType
from ml.simulation.topologies.base import BaseTopology


class PipelineTopology(BaseTopology):
    """Linear pipeline topology where agents communicate sequentially in an ordered chain."""

    def __init__(
        self,
        agent_ids: Optional[List[str]] = None,
        bidirectional: bool = False,
    ) -> None:
        """Initialize Pipeline topology.
        
        Args:
            agent_ids: Ordered sequence of agent identifiers: [A, B, C, D, ...].
            bidirectional: If True, allows return messages from successor to predecessor.
        """
        self._bidirectional = bidirectional
        super().__init__(agent_ids=agent_ids)

    @property
    def topology_type(self) -> TopologyType:
        return TopologyType.PIPELINE

    def build_graph(self, agent_ids: List[str]) -> nx.DiGraph:
        """Construct the linear pipeline directed graph: A → B → C → D."""
        self._validate_agents(agent_ids, min_agents=2)
        self._agent_ids = list(agent_ids)
        self._graph = nx.DiGraph()

        for aid in self._agent_ids:
            self._graph.add_node(aid, role="agent")

        for i in range(len(self._agent_ids) - 1):
            src = self._agent_ids[i]
            dst = self._agent_ids[i + 1]
            self._graph.add_edge(src, dst, weight=1.0)
            if self._bidirectional:
                self._graph.add_edge(dst, src, weight=1.0)

        return self._graph

    def get_next_agent(self, current_agent: str) -> Optional[str]:
        """Get immediate successor agent in the pipeline, if any."""
        if current_agent not in self._agent_ids:
            return None
        idx = self._agent_ids.index(current_agent)
        if idx < len(self._agent_ids) - 1:
            return self._agent_ids[idx + 1]
        return None
