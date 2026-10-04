"""Comprehensive Research Evaluation Framework - Phase 12."""

from ml.evaluation.schema import (
    UnifiedEvaluationRecord,
    FailureCentricRecord,
    CalibrationBin,
    CurvePoint,
    BootstrapResult,
    PairwiseComparisonRecord,
    SubgroupMetricRecord,
)
from ml.evaluation.metrics import (
    compute_comprehensive_metrics,
    compute_expected_calibration_error,
    compute_calibration_curve_data,
    compute_roc_and_pr_curves,
    compute_lead_time_statistics,
)
from ml.evaluation.failure_centric import FailureCentricEvaluator
from ml.evaluation.uncertainty import (
    trajectory_block_bootstrap_ci,
    paired_trajectory_bootstrap_test,
)
from ml.evaluation.subgroups import SubgroupEvaluator
from ml.evaluation.error_analysis import ErrorAnalyzer
from ml.evaluation.visualizer import EvaluationVisualizer
from ml.evaluation.engine import ComprehensiveEvaluationEngine

__all__ = [
    "UnifiedEvaluationRecord",
    "FailureCentricRecord",
    "CalibrationBin",
    "CurvePoint",
    "BootstrapResult",
    "PairwiseComparisonRecord",
    "SubgroupMetricRecord",
    "compute_comprehensive_metrics",
    "compute_expected_calibration_error",
    "compute_calibration_curve_data",
    "compute_roc_and_pr_curves",
    "compute_lead_time_statistics",
    "FailureCentricEvaluator",
    "trajectory_block_bootstrap_ci",
    "paired_trajectory_bootstrap_test",
    "SubgroupEvaluator",
    "ErrorAnalyzer",
    "EvaluationVisualizer",
    "ComprehensiveEvaluationEngine",
]
