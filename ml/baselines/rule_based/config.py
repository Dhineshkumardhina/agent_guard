"""Configuration and Parameter Schemas for Rule-Based Early Warning Baseline.

Defines:
- Risk indicator weights
- Alert thresholds (LOW, MEDIUM, HIGH, and binary decision threshold)
- Warning levels
- Full RuleBaselineConfig specification for reproducible benchmarking
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class WarningLevel(str, Enum):
    """Categorical alert level for early warning signals."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RuleWeights(BaseModel):
    """Configurable weights assigned to individual risk indicators.
    
    Weights govern how much each behavioral anomaly contributes to the overall risk score.
    """
    w_error_rate: float = Field(default=0.25, ge=0.0, description="Weight for recent error rate")
    w_retry_rate: float = Field(default=0.15, ge=0.0, description="Weight for repeated agent retries")
    w_timeout_rate: float = Field(default=0.15, ge=0.0, description="Weight for tool timeouts and drops")
    w_contradiction_rate: float = Field(default=0.15, ge=0.0, description="Weight for inter-agent contradiction")
    w_confidence_degradation: float = Field(default=0.15, ge=0.0, description="Weight for confidence decline")
    w_latency_anomaly: float = Field(default=0.05, ge=0.0, description="Weight for execution latency spikes")
    w_frequency_anomaly: float = Field(default=0.05, ge=0.0, description="Weight for interaction density spikes")
    w_interaction_concentration: float = Field(default=0.05, ge=0.0, description="Weight for bottleneck concentration")

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class RuleThresholds(BaseModel):
    """Configurable thresholds mapping continuous risk scores to alert levels and binary decisions."""
    threshold_low: float = Field(default=0.20, ge=0.0, le=1.0, description="Lower bound for LOW warning")
    threshold_medium: float = Field(default=0.35, ge=0.0, le=1.0, description="Lower bound for MEDIUM warning")
    threshold_high: float = Field(default=0.60, ge=0.0, le=1.0, description="Lower bound for HIGH warning")
    decision_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Binary cutoff: risk_score >= decision_threshold emits warning (1), else (0)"
    )

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class RuleBaselineConfig(BaseModel):
    """Specification for reproducible rule-based early warning detector benchmarking."""
    model_name: str = Field(default="rule_based_early_warning")
    version: str = Field(default="1.0.0")
    description: str = Field(default="Transparent weighted indicator early warning baseline")
    weights: RuleWeights = Field(default_factory=RuleWeights)
    thresholds: RuleThresholds = Field(default_factory=RuleThresholds)
    active_rules: Optional[List[str]] = Field(
        default=None,
        description="List of active rule indicator keys. None implies all rules are active."
    )
    prediction_horizons: List[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
