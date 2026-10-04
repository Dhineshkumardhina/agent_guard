"""Multi-agent communication topologies and routing registry."""

from typing import Dict, Type, Union, List, Optional
from ml.config.experiment_config import TopologyType
from ml.simulation.topologies.base import BaseTopology
from ml.simulation.topologies.pipeline import PipelineTopology
from ml.simulation.topologies.star import StarTopology
from ml.simulation.topologies.mesh import MeshTopology
from ml.simulation.topologies.custom import CustomTopology

TOPOLOGY_REGISTRY: Dict[str, Type[BaseTopology]] = {
    "pipeline": PipelineTopology,
    "star": StarTopology,
    "mesh": MeshTopology,
    "custom": CustomTopology,
}


def register_topology(name: str):
    """Decorator to register custom multi-agent topologies."""
    def decorator(cls: Type[BaseTopology]) -> Type[BaseTopology]:
        TOPOLOGY_REGISTRY[name.lower().strip()] = cls
        return cls
    return decorator


def create_topology(
    topology_type: Union[TopologyType, str],
    agent_ids: Optional[List[str]] = None,
    **kwargs,
) -> BaseTopology:
    """Factory method to instantiate a topology by name or enum with validation.
    
    Args:
        topology_type: Topology category (pipeline, star, mesh, or custom).
        agent_ids: Optional list of agent IDs to build graph immediately.
        **kwargs: Additional parameters passed to topology constructor.
        
    Raises:
        ValueError: If topology_type is unrecognized or invalid.
    """
    clean_name = topology_type.value if isinstance(topology_type, TopologyType) else str(topology_type).lower().strip()

    if clean_name not in TOPOLOGY_REGISTRY:
        available = sorted(list(TOPOLOGY_REGISTRY.keys()))
        raise ValueError(
            f"Invalid or unsupported topology: '{clean_name}'. Available: {available}"
        )

    cls = TOPOLOGY_REGISTRY[clean_name]
    return cls(agent_ids=agent_ids, **kwargs)


__all__ = [
    "BaseTopology",
    "PipelineTopology",
    "StarTopology",
    "MeshTopology",
    "TOPOLOGY_REGISTRY",
    "register_topology",
    "create_topology",
]
