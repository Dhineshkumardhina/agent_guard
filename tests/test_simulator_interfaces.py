"""Tests verifying simulator interface contracts and abstract classes."""

import pytest
from ml.simulation.interfaces import (
    AgentInterface,
    TopologyInterface,
    TaskInterface,
    FaultInjectorInterface,
    EnvironmentInterface,
)


def test_agent_interface_cannot_instantiate_directly():
    """Ensure abstract AgentInterface cannot be instantiated without implementation."""
    with pytest.raises(TypeError):
        AgentInterface()  # type: ignore


def test_topology_interface_cannot_instantiate_directly():
    """Ensure abstract TopologyInterface cannot be instantiated without implementation."""
    with pytest.raises(TypeError):
        TopologyInterface()  # type: ignore


def test_task_interface_cannot_instantiate_directly():
    """Ensure abstract TaskInterface cannot be instantiated without implementation."""
    with pytest.raises(TypeError):
        TaskInterface()  # type: ignore


def test_fault_injector_interface_cannot_instantiate_directly():
    """Ensure abstract FaultInjectorInterface cannot be instantiated without implementation."""
    with pytest.raises(TypeError):
        FaultInjectorInterface()  # type: ignore


def test_environment_interface_cannot_instantiate_directly():
    """Ensure abstract EnvironmentInterface cannot be instantiated without implementation."""
    with pytest.raises(TypeError):
        EnvironmentInterface()  # type: ignore
