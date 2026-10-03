"""Multi-agent abstractions and role definitions."""

from ml.simulation.agents.base import Agent
from ml.simulation.agents.roles import (
    Planner,
    Researcher,
    Analyst,
    Coder,
    Verifier,
    Critic,
    DecisionAgent,
    AGENT_ROLE_REGISTRY,
    register_role,
    create_agent,
)

__all__ = [
    "Agent",
    "Planner",
    "Researcher",
    "Analyst",
    "Coder",
    "Verifier",
    "Critic",
    "DecisionAgent",
    "AGENT_ROLE_REGISTRY",
    "register_role",
    "create_agent",
]
