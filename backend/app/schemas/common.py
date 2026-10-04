"""Common response wrappers and error schemas for AgentGuard API."""

from typing import Any, Optional, Dict
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error object conforming to REST standard."""
    code: str = Field(..., description="Machine-readable uppercase error code", examples=["RUN_NOT_FOUND"])
    message: str = Field(..., description="Human-readable description of the error", examples=["The requested simulation run was not found."])
    details: Optional[Any] = Field(default=None, description="Optional diagnostic details or field errors")


class ErrorEnvelope(BaseModel):
    """Standardized error envelope wrapping all API errors."""
    error: ErrorDetail
