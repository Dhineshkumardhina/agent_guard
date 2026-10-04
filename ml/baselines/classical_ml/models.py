"""Classical Machine Learning Baseline Models.

Implements clean, reusable baseline estimators:
1. LogisticRegressionBaseline (linear benchmark with feature scaling and class weighting)
2. RandomForestBaseline (non-linear ensemble with bagging and tree depth controls)
3. XGBoostBaseline (gradient boosted decision trees with scale_pos_weight for class imbalance)

All models adhere to the BaseBaselineModel interface:
- fit(X, y)
- predict(X)
- predict_proba(X)
- get_feature_importances()
- save(filepath)
- load(filepath)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pickle
import numpy as np

# Optional imports with clear dependency status
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class BaseBaselineModel(ABC):
    """Abstract interface for all tabular classical machine learning baselines."""

    def __init__(self, model_name: str, random_state: int = 42) -> None:
        self.model_name = model_name
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "BaseBaselineModel":
        """Train model on tabular feature matrix X and ground-truth binary targets y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary failure warning labels (0 or 1)."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores for the positive failure class [P(y=1)]."""
        pass

    @abstractmethod
    def get_feature_importances(self) -> Dict[str, float]:
        """Extract descriptive feature importance scores or coefficient magnitudes."""
        pass

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize trained model state to disk using pickle."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseBaselineModel":
        """Deserialize trained model state from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)  # nosec
        return model


class LogisticRegressionBaseline(BaseBaselineModel):
    """Logistic Regression baseline with z-score standardization and class weighting."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ) -> None:
        super().__init__(model_name="logistic_regression", random_state=random_state)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight

        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for LogisticRegressionBaseline. "
                "Install it using: pip install scikit-learn"
            )

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "LogisticRegressionBaseline":
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset X.")
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)

        # Handle degenerate single-class edge case in small batches
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.degenerate_class = int(unique_classes[0])
            self.is_fitted = True
            return self

        self.degenerate_class = None
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if len(X) == 0:
            return np.zeros((0,), dtype=np.int32)
        if getattr(self, "degenerate_class", None) is not None:
            return np.full((len(X),), self.degenerate_class, dtype=np.int32)

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if len(X) == 0:
            return np.zeros((0,), dtype=np.float32)
        if getattr(self, "degenerate_class", None) is not None:
            prob = 1.0 if self.degenerate_class == 1 else 0.0
            return np.full((len(X),), prob, dtype=np.float32)

        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        # Return probability of positive class (label=1)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted or getattr(self, "degenerate_class", None) is not None:
            return {name: 0.0 for name in self.feature_names}
        coefs = np.abs(self.model.coef_[0])
        return {
            name: round(float(val), 4)
            for name, val in zip(self.feature_names, coefs)
        }


class RandomForestBaseline(BaseBaselineModel):
    """Random Forest ensemble baseline with depth controls and class weighting."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ) -> None:
        super().__init__(model_name="random_forest", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight

        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for RandomForestBaseline. "
                "Install it using: pip install scikit-learn"
            )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "RandomForestBaseline":
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset X.")
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.degenerate_class = int(unique_classes[0])
            self.is_fitted = True
            return self

        self.degenerate_class = None
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if len(X) == 0:
            return np.zeros((0,), dtype=np.int32)
        if getattr(self, "degenerate_class", None) is not None:
            return np.full((len(X),), self.degenerate_class, dtype=np.int32)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if len(X) == 0:
            return np.zeros((0,), dtype=np.float32)
        if getattr(self, "degenerate_class", None) is not None:
            prob = 1.0 if self.degenerate_class == 1 else 0.0
            return np.full((len(X),), prob, dtype=np.float32)

        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted or getattr(self, "degenerate_class", None) is not None:
            return {name: 0.0 for name in self.feature_names}
        importances = self.model.feature_importances_
        return {
            name: round(float(val), 4)
            for name, val in zip(self.feature_names, importances)
        }


class XGBoostBaseline(BaseBaselineModel):
    """Gradient boosted decision trees baseline using XGBoost."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(model_name="xgboost", random_state=random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight

        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for XGBoostBaseline. "
                "Install it using: pip install xgboost"
            )

        kwargs: Dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "eval_metric": "logloss",
        }
        if self.scale_pos_weight is not None:
            kwargs["scale_pos_weight"] = self.scale_pos_weight

        self.model = xgb.XGBClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "XGBoostBaseline":
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset X.")
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.degenerate_class = int(unique_classes[0])
            self.is_fitted = True
            return self

        self.degenerate_class = None

        # Automatically calculate scale_pos_weight if not explicitly provided
        if self.scale_pos_weight is None:
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            if n_pos > 0:
                self.model.set_params(scale_pos_weight=float(n_neg / n_pos))

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if len(X) == 0:
            return np.zeros((0,), dtype=np.int32)
        if getattr(self, "degenerate_class", None) is not None:
            return np.full((len(X),), self.degenerate_class, dtype=np.int32)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if len(X) == 0:
            return np.zeros((0,), dtype=np.float32)
        if getattr(self, "degenerate_class", None) is not None:
            prob = 1.0 if self.degenerate_class == 1 else 0.0
            return np.full((len(X),), prob, dtype=np.float32)

        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted or getattr(self, "degenerate_class", None) is not None:
            return {name: 0.0 for name in self.feature_names}
        importances = self.model.feature_importances_
        return {
            name: round(float(val), 4)
            for name, val in zip(self.feature_names, importances)
        }
