"""SQLAlchemy Relational Database Models for AgentGuard.

Structured with foreign keys, indexes, and clean separation between
research data (runs, events, graphs, predictions) and system entities.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship

from backend.app.database.session import Base


class User(Base):
    """System user for authentication and access control."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


class Experiment(Base):
    """Research experiment tracking run sets, configs, and evaluations."""
    __tablename__ = "experiments"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    config = Column(JSON, nullable=False, default=dict)
    status = Column(String(32), default="created", index=True)  # created, running, completed, failed
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    runs = relationship("Run", back_populates="experiment", cascade="all, delete-orphan")
    results = relationship("ModelResult", back_populates="experiment", cascade="all, delete-orphan")


class Dataset(Base):
    """Dataset registry for raw and processed trajectory collections."""
    __tablename__ = "datasets"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    split_type = Column(String(32), default="trajectory_level")
    num_trajectories = Column(Integer, default=0, nullable=False)
    num_events = Column(Integer, default=0, nullable=False)
    storage_path = Column(String(512), nullable=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    runs = relationship("Run", back_populates="dataset")


class Run(Base):
    """Individual multi-agent simulation trajectory."""
    __tablename__ = "runs"

    id = Column(String(64), primary_key=True, index=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True, index=True)
    dataset_id = Column(String(64), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True)
    task_type = Column(String(64), nullable=False, index=True)
    topology = Column(String(64), nullable=False, index=True)
    num_agents = Column(Integer, nullable=False)
    duration_seconds = Column(Float, default=0.0)
    has_cascading_failure = Column(Boolean, default=False, index=True)
    cascading_failure_step = Column(Integer, nullable=True)
    random_seed = Column(Integer, nullable=False, default=42)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    experiment = relationship("Experiment", back_populates="runs")
    dataset = relationship("Dataset", back_populates="runs")
    agents = relationship("Agent", back_populates="run", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="run", cascade="all, delete-orphan")
    interactions = relationship("AgentInteraction", back_populates="run", cascade="all, delete-orphan")
    fault_injections = relationship("FaultInjection", back_populates="run", cascade="all, delete-orphan")
    failures = relationship("Failure", back_populates="run", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="run", cascade="all, delete-orphan")


class Agent(Base):
    """Agent entity participating in a specific trajectory run."""
    __tablename__ = "agents"

    id = Column(String(64), primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_name = Column(String(64), nullable=False)
    role = Column(String(64), nullable=False)
    status = Column(String(32), default="active")  # active, failed, degraded
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    run = relationship("Run", back_populates="agents")


class Event(Base):
    """Detailed telemetry record of a single communication event."""
    __tablename__ = "events"

    id = Column(String(64), primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    step_idx = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    source_agent = Column(String(64), nullable=False, index=True)
    target_agent = Column(String(64), nullable=False, index=True)
    event_type = Column(String(32), default="message", nullable=False)
    message_length = Column(Integer, default=0)
    token_count = Column(Integer, default=0)
    latency = Column(Float, default=0.0)
    confidence = Column(Float, default=1.0)
    output_quality = Column(Float, default=1.0)
    contradiction_score = Column(Float, default=0.0)
    tool_used = Column(String(64), nullable=True)
    tool_success = Column(Boolean, nullable=True)
    tool_error = Column(Boolean, default=False)
    retry_count = Column(Integer, default=0)
    injected_fault = Column(String(64), nullable=True)
    error_type = Column(String(64), nullable=True)
    failure_label = Column(Integer, default=0, index=True)
    downstream_failure = Column(Boolean, default=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    run = relationship("Run", back_populates="events")

    __table_args__ = (
        Index("idx_events_run_step", "run_id", "step_idx"),
        Index("idx_events_src_tgt", "source_agent", "target_agent"),
    )


class AgentInteraction(Base):
    """Aggregated interaction pair summary between agents within a run."""
    __tablename__ = "agent_interactions"

    id = Column(String(64), primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    source_agent = Column(String(64), nullable=False)
    target_agent = Column(String(64), nullable=False)
    message_count = Column(Integer, default=1)
    mean_latency = Column(Float, default=0.0)
    mean_contradiction = Column(Float, default=0.0)
    error_count = Column(Integer, default=0)
    last_timestamp = Column(Float, nullable=False)

    run = relationship("Run", back_populates="interactions")

    __table_args__ = (
        Index("idx_interactions_run_pair", "run_id", "source_agent", "target_agent"),
    )


class FaultInjection(Base):
    """Audit log of synthetic faults introduced during a simulation."""
    __tablename__ = "fault_injections"

    id = Column(String(64), primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    step_idx = Column(Integer, nullable=False)
    target_agent = Column(String(64), nullable=False)
    fault_type = Column(String(64), nullable=False)
    parameters = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    run = relationship("Run", back_populates="fault_injections")


class Failure(Base):
    """Recorded failure instances across Level 1, 2, or 3."""
    __tablename__ = "failures"

    id = Column(String(64), primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    step_idx = Column(Integer, nullable=False)
    failure_level = Column(Integer, nullable=False, index=True)
    originating_agent = Column(String(64), nullable=False)
    affected_agents = Column(JSON, default=list)
    failure_type = Column(String(64), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    run = relationship("Run", back_populates="failures")


class Prediction(Base):
    """Model inference records for early failure warning."""
    __tablename__ = "predictions"

    id = Column(String(64), primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    step_idx = Column(Integer, nullable=False)
    model_name = Column(String(64), nullable=False, index=True)
    horizon_k = Column(Integer, nullable=False)
    predicted_probability = Column(Float, nullable=False)
    risk_level = Column(String(32), nullable=False)  # NORMAL, WATCH, HIGH_RISK, PREDICTED_CASCADE
    ground_truth = Column(Integer, nullable=True)
    lead_time = Column(Integer, nullable=True)
    explanation_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    run = relationship("Run", back_populates="predictions")

    __table_args__ = (
        Index("idx_predictions_run_step_model", "run_id", "step_idx", "model_name"),
    )


class ModelResult(Base):
    """Evaluation summary metrics for a given model on an experiment set."""
    __tablename__ = "model_results"

    id = Column(String(64), primary_key=True, index=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String(64), nullable=False, index=True)
    horizon_k = Column(Integer, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1 = Column(Float, nullable=False)
    auroc = Column(Float, nullable=False)
    auprc = Column(Float, nullable=False)
    false_alarm_rate = Column(Float, nullable=False)
    mean_lead_time = Column(Float, nullable=False)
    median_lead_time = Column(Float, nullable=False)
    metrics_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    experiment = relationship("Experiment", back_populates="results")
