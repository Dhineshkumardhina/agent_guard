"""Classical Machine Learning Baselines Package.

Provides:
- TabularFeatureExtractor (28 agent-level and interaction behavioral features)
- LogisticRegressionBaseline
- RandomForestBaseline
- XGBoostBaseline
- train_and_evaluate_baseline pipeline
- ClassicalMLExperimentRunner
"""

from ml.baselines.classical_ml.features import (
    TabularFeatureExtractor,
    FEATURE_NAMES,
    FEATURE_DOCUMENTATION,
)
from ml.baselines.classical_ml.models import (
    BaseBaselineModel,
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
    HAS_SKLEARN,
    HAS_XGBOOST,
)
from ml.baselines.classical_ml.pipeline import (
    train_and_evaluate_baseline,
    compute_calibration_analysis,
)
from ml.baselines.classical_ml.experiment import ClassicalMLExperimentRunner

__all__ = [
    "TabularFeatureExtractor",
    "FEATURE_NAMES",
    "FEATURE_DOCUMENTATION",
    "BaseBaselineModel",
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "HAS_SKLEARN",
    "HAS_XGBOOST",
    "train_and_evaluate_baseline",
    "compute_calibration_analysis",
    "ClassicalMLExperimentRunner",
]
