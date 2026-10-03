"""Tests for Multi-Agent Communication Topologies (Phase 2)."""

import pytest
from ml.config.experiment_config import TopologyType
from ml.simulation.topologies import (
    BaseTopology,
    PipelineTopology,
    StarTopology,
    MeshTopology,
    create_topology,
    register_topology,
    TOPOLOGY_REGISTRY,
)


def test_pipeline_topology_generation_and_routing():
    """Verify linear pipeline generation: A -> B -> C -> D."""
    agents = ["A", "B", "C", "D"]
    pipeline = PipelineTopology(agent_ids=agents)

    assert pipeline.topology_type == TopologyType.PIPELINE
    assert pipeline.graph.number_of_nodes() == 4
    assert pipeline.graph.number_of_edges() == 3

    # Permitted forward communications
    assert pipeline.can_communicate("A", "B") is True
    assert pipeline.can_communicate("B", "C") is True
    assert pipeline.can_communicate("C", "D") is True

    # Forbidden backward communications (strict forward)
    assert pipeline.can_communicate("B", "A") is False
    assert pipeline.can_communicate("D", "C") is False

    # Forbidden non-adjacent communications
    assert pipeline.can_communicate("A", "C") is False
    assert pipeline.can_communicate("A", "D") is False

    # Successor helper
    assert pipeline.get_next_agent("A") == "B"
    assert pipeline.get_next_agent("D") is None


def test_star_topology_generation_and_routing():
    """Verify star topology: hub A coordinates with leaves B, C, D, E."""
    agents = ["A", "B", "C", "D", "E"]
    star = StarTopology(agent_ids=agents, hub_agent_id="A")

    assert star.topology_type == TopologyType.STAR
    assert star.hub_agent_id == "A"
    assert star.leaf_agent_ids == ["B", "C", "D", "E"]
    assert star.graph.number_of_nodes() == 5
    # 4 leaves * 2 (bidirectional hub-leaf links) = 8 edges
    assert star.graph.number_of_edges() == 8

    # Hub <-> Leaf is permitted
    assert star.can_communicate("A", "B") is True
    assert star.can_communicate("B", "A") is True
    assert star.can_communicate("A", "E") is True
    assert star.can_communicate("E", "A") is True

    # Leaf <-> Leaf is strictly forbidden (must route through hub)
    assert star.can_communicate("B", "C") is False
    assert star.can_communicate("C", "D") is False
    assert star.can_communicate("D", "E") is False


def test_mesh_topology_generation_and_routing():
    """Verify mesh topology: all-to-all bidirectional communication."""
    agents = ["A", "B", "C", "D"]
    mesh = MeshTopology(agent_ids=agents)

    assert mesh.topology_type == TopologyType.MESH
    assert mesh.graph.number_of_nodes() == 4
    # 4 * 3 = 12 directed edges
    assert mesh.graph.number_of_edges() == 12

    # Any distinct pair can communicate
    for u in agents:
        for v in agents:
            if u != v:
                assert mesh.can_communicate(u, v) is True
            else:
                assert mesh.can_communicate(u, v) is False


def test_create_topology_factory():
    """Verify create_topology factory instantiates appropriate topologies."""
    agents = ["node_1", "node_2", "node_3"]
    pipe = create_topology("pipeline", agent_ids=agents)
    assert isinstance(pipe, PipelineTopology)

    star = create_topology("star", agent_ids=agents)
    assert isinstance(star, StarTopology)

    mesh = create_topology("mesh", agent_ids=agents)
    assert isinstance(mesh, MeshTopology)


def test_invalid_topology_handling():
    """Verify error handling on invalid topology specifications."""
    # Unknown topology name
    with pytest.raises(ValueError, match="Invalid or unsupported topology"):
        create_topology("unsupported_ring_topology")

    # Insufficient agents (< 2)
    with pytest.raises(ValueError, match="requires at least 2 agents"):
        PipelineTopology(agent_ids=["only_one_agent"])

    # Duplicate agent IDs
    with pytest.raises(ValueError, match="Agent IDs must be unique"):
        MeshTopology(agent_ids=["agent_1", "agent_1", "agent_2"])

    # Non-existent hub in star topology
    with pytest.raises(ValueError, match="not found in agent_ids"):
        StarTopology(agent_ids=["agent_1", "agent_2"], hub_agent_id="unknown_hub")


def test_extensible_topology_registry():
    """Verify registering a new custom topology pattern."""
    @register_topology("ring")
    class RingTopology(BaseTopology):
        @property
        def topology_type(self):
            return TopologyType.CUSTOM

        def build_graph(self, agent_ids):
            self._validate_agents(agent_ids, min_agents=3)
            self._agent_ids = list(agent_ids)
            import networkx as nx
            self._graph = nx.DiGraph()
            n = len(self._agent_ids)
            for i in range(n):
                self._graph.add_edge(self._agent_ids[i], self._agent_ids[(i + 1) % n])
            return self._graph

    assert "ring" in TOPOLOGY_REGISTRY
    ring = create_topology("ring", agent_ids=["R1", "R2", "R3"])
    assert ring.can_communicate("R1", "R2") is True
    assert ring.can_communicate("R2", "R3") is True
    assert ring.can_communicate("R3", "R1") is True
    assert ring.can_communicate("R1", "R3") is False
