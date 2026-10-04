"""Custom application exceptions and error envelopes for AgentGuard API."""

from typing import Any, Optional


class AgentGuardException(Exception):
    """Base exception for all AgentGuard application errors."""
    pass


class AgentGuardAPIException(AgentGuardException):
    """Base API exception with standardized error response structure."""

    def __init__(
        self,
        status_code: int = 400,
        code: str = "BAD_REQUEST",
        message: str = "An error occurred processing your request.",
        details: Optional[Any] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(self.message)


class AgentNotFoundException(AgentGuardAPIException):
    """Raised when an agent is not found."""
    def __init__(self, agent_id: str):
        super().__init__(
            status_code=404,
            code="AGENT_NOT_FOUND",
            message=f"Agent '{agent_id}' was not found.",
        )


class RunNotFoundException(AgentGuardAPIException):
    """Raised when a simulation run is not found."""
    def __init__(self, run_id: str):
        super().__init__(
            status_code=404,
            code="RUN_NOT_FOUND",
            message=f"The requested simulation run '{run_id}' was not found.",
        )


class PredictionNotFoundException(AgentGuardAPIException):
    """Raised when a prediction record is not found."""
    def __init__(self, prediction_id: str):
        super().__init__(
            status_code=404,
            code="PREDICTION_NOT_FOUND",
            message=f"Prediction '{prediction_id}' was not found.",
        )


class ExperimentNotFoundException(AgentGuardAPIException):
    """Raised when an experiment is not found."""
    def __init__(self, experiment_id: str):
        super().__init__(
            status_code=404,
            code="EXPERIMENT_NOT_FOUND",
            message=f"Experiment '{experiment_id}' was not found.",
        )


class EvaluationNotFoundException(AgentGuardAPIException):
    """Raised when an evaluation record is not found."""
    def __init__(self, identifier: str):
        super().__init__(
            status_code=404,
            code="EVALUATION_NOT_FOUND",
            message=f"Evaluation for '{identifier}' was not found.",
        )


class AblationNotFoundException(AgentGuardAPIException):
    """Raised when an ablation experiment is not found."""
    def __init__(self, experiment_id: str):
        super().__init__(
            status_code=404,
            code="ABLATION_NOT_FOUND",
            message=f"Ablation experiment '{experiment_id}' was not found.",
        )


class GeneralizationNotFoundException(AgentGuardAPIException):
    """Raised when a generalization record is not found."""
    def __init__(self, experiment_id: str):
        super().__init__(
            status_code=404,
            code="GENERALIZATION_NOT_FOUND",
            message=f"Generalization experiment '{experiment_id}' was not found.",
        )


class ExplanationNotFoundException(AgentGuardAPIException):
    """Raised when an explanation is not found."""
    def __init__(self, identifier: str):
        super().__init__(
            status_code=404,
            code="EXPLANATION_NOT_FOUND",
            message=f"Explanation for '{identifier}' was not found.",
        )


class GraphNotFoundException(AgentGuardAPIException):
    """Raised when temporal graph data is not found for a run."""
    def __init__(self, run_id: str):
        super().__init__(
            status_code=404,
            code="GRAPH_NOT_FOUND",
            message=f"Temporal graph data for run '{run_id}' was not found.",
        )


class InvalidParameterException(AgentGuardAPIException):
    """Raised when an invalid query or body parameter is provided."""
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            status_code=400,
            code="INVALID_PARAMETER",
            message=message,
            details=details,
        )


class DatabaseException(AgentGuardAPIException):
    """Raised when a database failure occurs."""
    def __init__(self, message: str = "A database operation failed."):
        super().__init__(
            status_code=500,
            code="DATABASE_ERROR",
            message=message,
        )
