"""Agent service layer managing retrieval and metadata presentation."""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.models import Agent
from backend.app.schemas.agents import AgentResponse, AgentListResponse
from backend.app.core.exceptions import AgentNotFoundException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def _to_agent_response(agent: Agent) -> AgentResponse:
    """Map SQLAlchemy Agent model to AgentResponse schema."""
    role_descriptions = {
        "planner": "Task decomposition, sub-goal generation, and execution planning",
        "researcher": "Information retrieval, context extraction, and domain exploration",
        "analyst": "Quantitative analysis, evidence aggregation, and synthesis",
        "verifier": "Consistency verification, constraint checking, and quality auditing",
        "decision": "Consensus formation, multi-agent arbitration, and final outputs",
        "executor": "External tool execution and API interaction",
        "monitor": "Runtime invariant monitoring and anomaly detection",
        "coordinator": "Inter-agent workflow orchestration and synchronization",
    }
    desc = role_descriptions.get(agent.role.lower(), f"Specialized {agent.role} agent in multi-agent execution")
    return AgentResponse(
        agent_id=agent.id,
        name=agent.agent_name,
        role=agent.role,
        description=desc,
        creation_time=agent.created_at,
        run_id=agent.run_id,
        status=agent.status,
    )


def get_agents(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    run_id: Optional[str] = None,
    role: Optional[str] = None,
) -> AgentListResponse:
    """Retrieve paginated agents with optional filtering."""
    query = db.query(Agent)
    if run_id:
        query = query.filter(Agent.run_id == run_id)
    if role:
        query = query.filter(func.lower(Agent.role) == role.lower())

    total = query.count()
    records = query.order_by(Agent.id).offset(offset).limit(limit).all()
    items = [_to_agent_response(a) for a in records]
    has_more = (offset + limit) < total

    return AgentListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_agent_by_id(db: Session, agent_id: str) -> AgentResponse:
    """Retrieve single agent by ID or raise AgentNotFoundException."""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        # Fallback check by agent_name
        agent = db.query(Agent).filter(Agent.agent_name == agent_id).first()
    if not agent:
        raise AgentNotFoundException(agent_id=agent_id)
    return _to_agent_response(agent)
