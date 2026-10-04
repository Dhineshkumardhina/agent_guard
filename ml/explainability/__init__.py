"""Explainability, attention attribution, and failure-risk interpretation module - Phase 15.

Exports:
- FeatureAttribution, AgentAttribution, InteractionAttribution, TemporalEventAttribution
- PerturbationResult, CaseStudyExplanation, GlobalImportanceReport, ExplanationStabilityReport
- CAUSALITY_DISCLAIMER
- ClassicalModelExplainer
- TemporalGNNExplainer
- PerturbationAnalyzer
- ExplainabilityVisualizer
- ExplainabilityRunner
"""

from ml.explainability.schema import (
    FeatureAttribution,
    AgentAttribution,
    InteractionAttribution,
    TemporalEventAttribution,
    PerturbationResult,
    CaseStudyExplanation,
    GlobalImportanceReport,
    ExplanationStabilityReport,
    CAUSALITY_DISCLAIMER,
)
from ml.explainability.classical_explainer import ClassicalModelExplainer
from ml.explainability.temporal_gnn_explainer import TemporalGNNExplainer
from ml.explainability.perturbation import PerturbationAnalyzer
from ml.explainability.visualizer import ExplainabilityVisualizer
from ml.explainability.runner import ExplainabilityRunner

__all__ = [
    "FeatureAttribution",
    "AgentAttribution",
    "InteractionAttribution",
    "TemporalEventAttribution",
    "PerturbationResult",
    "CaseStudyExplanation",
    "GlobalImportanceReport",
    "ExplanationStabilityReport",
    "CAUSALITY_DISCLAIMER",
    "ClassicalModelExplainer",
    "TemporalGNNExplainer",
    "PerturbationAnalyzer",
    "ExplainabilityVisualizer",
    "ExplainabilityRunner",
]
