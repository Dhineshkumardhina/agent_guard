"""Tests for Agent Abstraction and Communication (Phase 2)."""

import pytest
from ml.config.experiment_config import AgentRole
from ml.simulation.agents import (
    Agent,
    Planner,
    Researcher,
    Analyst,
    Coder,
    Verifier,
    Critic,
    DecisionAgent,
    create_agent,
    register_role,
    AGENT_ROLE_REGISTRY,
)
from ml.simulation.events import SimulationMessage


def test_agent_base_attributes():
    """Verify base Agent contains all mandatory attributes."""
    agent = Agent(
        agent_id="agent_test_01",
        name="Test Agent",
        role=AgentRole.RESEARCHER,
        state={"status": "idle", "energy": 100},
        memory=[],
        configuration={"timeout": 30, "model": "mock"},
    )
    assert agent.agent_id == "agent_test_01"
    assert agent.name == "Test Agent"
    assert agent.role == AgentRole.RESEARCHER
    assert agent.state["status"] == "idle"
    assert agent.state["energy"] == 100
    assert agent.memory == []
    assert agent.configuration["timeout"] == 30


def test_all_seven_initial_roles_instantiation():
    """Verify all 7 initial roles can be instantiated and have correct roles."""
    roles_classes = [
        (Planner, AgentRole.PLANNER),
        (Researcher, AgentRole.RESEARCHER),
        (Analyst, AgentRole.ANALYST),
        (Coder, AgentRole.CODER),
        (Verifier, AgentRole.VERIFIER),
        (Critic, AgentRole.CRITIC),
        (DecisionAgent, AgentRole.DECISION),
    ]
    for cls, expected_role in roles_classes:
        agent = cls(agent_id=f"test_{expected_role.value}", name=f"Agent {cls.__name__}")
        assert agent.agent_id == f"test_{expected_role.value}"
        assert agent.name == f"Agent {cls.__name__}"
        assert agent.role == expected_role
        assert isinstance(agent.state, dict)
        assert isinstance(agent.memory, list)
        assert isinstance(agent.configuration, dict)


def test_create_agent_factory():
    """Verify create_agent factory instantiates agents by role string or enum."""
    planner = create_agent("planner", agent_id="p1", name="Alpha Planner")
    assert isinstance(planner, Planner)
    assert planner.agent_id == "p1"
    assert planner.name == "Alpha Planner"

    decision = create_agent(AgentRole.DECISION, agent_id="d1")
    assert isinstance(decision, DecisionAgent)
    assert decision.agent_id == "d1"


def test_invalid_agent_handling():
    """Verify error handling on invalid agent instantiation or inputs."""
    # Unknown role
    with pytest.raises(ValueError, match="Unknown agent role"):
        create_agent("non_existent_role_xyz")

    # Empty agent_id
    with pytest.raises(ValueError, match="agent_id must be a non-empty string"):
        Agent(agent_id="", name="Valid Name", role="planner")

    # Empty name
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        Agent(agent_id="valid_id", name="", role="planner")

    # Invalid state assignment type
    agent = Agent(agent_id="valid_id", name="Valid Name", role="planner")
    with pytest.raises(TypeError, match="Agent state must be a dictionary"):
        agent.state = "invalid_non_dict"  # type: ignore


def test_agent_communication_and_memory():
    """Verify agent message creation, receipt, and memory recording."""
    sender = Researcher(agent_id="researcher_1", name="Researcher")
    receiver = Analyst(agent_id="analyst_1", name="Analyst")

    msg = sender.send_message(
        target_agent=receiver.agent_id,
        message="Empirical observations regarding cascading rate.",
        event_type="research_findings",
        timestamp=1.25,
        metadata={"citation": "Paper 2026"},
    )

    assert isinstance(msg, SimulationMessage)
    assert msg.source_agent == "researcher_1"
    assert msg.target_agent == "analyst_1"
    assert msg.timestamp == 1.25
    assert msg.message == "Empirical observations regarding cascading rate."
    assert msg.event_type == "research_findings"
    assert msg.metadata == {"citation": "Paper 2026"}

    # Receiver processes incoming message
    assert len(receiver.memory) == 0
    receiver.receive_event(msg)
    assert len(receiver.memory) == 1
    assert receiver.memory[0].source_agent == "researcher_1"
    assert receiver.state["last_received_from"] == "researcher_1"
    assert receiver.state["step_count"] == 1


def test_agent_credential_sanitization():
    """Verify that sensitive patterns in messages are sanitized."""
    agent = Coder(agent_id="coder_1", name="Coder")
    msg = agent.send_message(
        target_agent="verifier_1",
        message="Using api_key=SECRET_TOKEN_12345 for service",
    )
    assert "SECRET_TOKEN_12345" not in msg.message
    assert msg.message == "[REDACTED_CREDENTIAL]"


def test_agent_reset():
    """Verify reset clears mutated memory and state back to initial."""
    agent = Researcher(
        agent_id="r1",
        name="Researcher",
        state={"status": "active", "step_count": 0},
        memory=[],
    )
    agent.state["status"] = "busy"
    agent.state["step_count"] = 5
    agent.memory.append({"dummy": "event"})

    agent.reset()
    assert agent.state["status"] == "active"
    assert agent.state["step_count"] == 0
    assert len(agent.memory) == 0


def test_extensible_agent_registry():
    """Verify new custom roles can be dynamically registered and instantiated."""
    @register_role("security_auditor")
    class SecurityAuditor(Agent):
        def __init__(self, agent_id="sec_0", name="Security Auditor", **kwargs):
            super().__init__(agent_id=agent_id, name=name, role="security_auditor", **kwargs)

    assert "security_auditor" in AGENT_ROLE_REGISTRY
    sec_agent = create_agent("security_auditor", agent_id="sec_1")
    assert isinstance(sec_agent, SecurityAuditor)
    assert sec_agent.role == "security_auditor"
