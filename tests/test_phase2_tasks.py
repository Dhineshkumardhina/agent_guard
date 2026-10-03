"""Tests for Multi-Agent Task Workflows (Phase 2)."""

import pytest
from ml.config.experiment_config import TaskType, AgentRole
from ml.simulation.tasks import (
    BaseTask,
    ResearchTask,
    CodingTask,
    AnalysisTask,
    PlanningTask,
    create_task,
)
from ml.simulation.events import SimulationMessage


def test_research_task_workflow():
    """Verify Research task workflow roles and deterministic output."""
    task = ResearchTask(seed=42)
    assert task.task_type == TaskType.RESEARCH
    assert task.required_roles == [
        AgentRole.RESEARCHER,
        AgentRole.ANALYST,
        AgentRole.VERIFIER,
        AgentRole.DECISION,
    ]
    assert len(task.workflow_stages) == 4

    # Simulate trajectory of events
    events = [
        SimulationMessage(source_agent="researcher", target_agent="analyst", timestamp=0.2, message="Facts", confidence=0.9),
        SimulationMessage(source_agent="analyst", target_agent="verifier", timestamp=0.5, message="Stats", confidence=0.92),
        SimulationMessage(source_agent="verifier", target_agent="decision", timestamp=0.8, message="Audit", confidence=0.96),
        SimulationMessage(source_agent="decision", target_agent="researcher", timestamp=1.1, message="Conclude", confidence=0.98),
    ]
    assert task.is_complete(events) is True
    assert task.evaluate_success(events) is True

    output = task.generate_final_output(events)
    assert output["status"] == "COMPLETED"
    assert "metrics" in output
    assert output["metrics"]["verified"] is True


def test_coding_task_workflow():
    """Verify Coding task workflow roles and deterministic output."""
    task = CodingTask(seed=42)
    assert task.task_type == TaskType.CODING
    assert task.required_roles == [
        AgentRole.PLANNER,
        AgentRole.CODER,
        AgentRole.VERIFIER,
        AgentRole.CRITIC,
    ]
    assert len(task.workflow_stages) == 4

    events = [
        SimulationMessage(source_agent="planner", target_agent="coder", timestamp=0.2, message="Spec"),
        SimulationMessage(source_agent="coder", target_agent="verifier", timestamp=0.6, message="Code"),
        SimulationMessage(source_agent="verifier", target_agent="critic", timestamp=0.9, message="Tests"),
        SimulationMessage(source_agent="critic", target_agent="planner", timestamp=1.2, message="Review"),
    ]
    assert task.is_complete(events) is True
    output = task.generate_final_output(events)
    assert output["status"] == "COMPLETED"
    assert "code_artifact" in output
    assert output["code_artifact"]["tests_passed"] == 12


def test_analysis_task_workflow():
    """Verify Data Analysis task workflow roles and deterministic output."""
    task = AnalysisTask(seed=42)
    assert task.task_type == TaskType.DATA_ANALYSIS
    assert len(task.workflow_stages) == 4
    events = [
        SimulationMessage(source_agent="researcher", target_agent="analyst", timestamp=0.2, message="Data"),
        SimulationMessage(source_agent="analyst", target_agent="verifier", timestamp=0.5, message="Fit"),
        SimulationMessage(source_agent="verifier", target_agent="decision", timestamp=0.8, message="Audit"),
        SimulationMessage(source_agent="decision", target_agent="researcher", timestamp=1.0, message="Signoff"),
    ]
    output = task.generate_final_output(events)
    assert output["status"] == "COMPLETED"
    assert "statistical_summary" in output
    assert output["statistical_summary"]["auroc"] == 0.892


def test_planning_task_workflow():
    """Verify Planning task workflow roles and deterministic output."""
    task = PlanningTask(seed=42)
    assert task.task_type == TaskType.PLANNING
    assert len(task.workflow_stages) == 4
    events = [
        SimulationMessage(source_agent="planner", target_agent="analyst", timestamp=0.2, message="Plan"),
        SimulationMessage(source_agent="analyst", target_agent="critic", timestamp=0.5, message="Feasibility"),
        SimulationMessage(source_agent="critic", target_agent="decision", timestamp=0.8, message="Risk"),
        SimulationMessage(source_agent="decision", target_agent="planner", timestamp=1.1, message="Authorize"),
    ]
    output = task.generate_final_output(events)
    assert output["status"] == "COMPLETED"
    assert output["plan_approved"] is True


def test_create_task_factory_and_invalid_handling():
    """Verify create_task factory instantiates tasks and rejects invalid ones."""
    res_task = create_task("research")
    assert isinstance(res_task, ResearchTask)

    code_task = create_task("coding")
    assert isinstance(code_task, CodingTask)

    with pytest.raises(ValueError, match="Invalid or unsupported task type"):
        create_task("invalid_task_nonexistent")
