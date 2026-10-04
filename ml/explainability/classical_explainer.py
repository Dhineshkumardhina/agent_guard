"""Classical Machine Learning Explainer - Phase 15.

Provides Level 1 Global and Local Feature Attribution for classical tabular baselines:
- Logistic Regression (coefficients & linear contributions)
- Random Forest (MDI feature importances & permutation importance)
- XGBoost (Gain-based tree importance & permutation importance)

Adheres strictly to causal time constraints and non-causal interpretation guidelines.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from sklearn.inspection import permutation_importance

from ml.baselines.classical_ml.models import BaseBaselineModel
from ml.baselines.classical_ml.features import FEATURE_NAMES
from ml.explainability.schema import FeatureAttribution, CAUSALITY_DISCLAIMER


class ClassicalModelExplainer:
    """Computes global and local feature importance for classical ML baseline models."""

    def __init__(self, model: BaseBaselineModel, feature_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.feature_names = feature_names or (model.feature_names if hasattr(model, "feature_names") and model.feature_names else FEATURE_NAMES)

    def explain_global_importance(
        self,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        method: str = "auto",
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> List[FeatureAttribution]:
        """Compute global feature importance ranking across the dataset.
        
        Args:
            X_val: Validation feature matrix [N, D] (required for permutation importance).
            y_val: Validation binary targets [N] (required for permutation importance).
            method: "permutation", "model_native", or "auto".
            n_repeats: Number of shuffles for permutation importance.
            random_state: Deterministic random seed.
            
        Returns:
            List of FeatureAttribution objects ranked by importance score.
        """
        attributions: List[FeatureAttribution] = []

        # Determine method
        use_permutation = method == "permutation" or (method == "auto" and X_val is not None and y_val is not None and len(y_val) > 0)

        if use_permutation and X_val is not None and y_val is not None:
            # Permutation Importance
            raw_estimator = getattr(self.model, "model", None)
            scaler = getattr(self.model, "scaler", None)

            # Define scoring wrapper if scaler exists
            if scaler is not None and raw_estimator is not None:
                X_scaled = scaler.transform(X_val)
                perm_res = permutation_importance(
                    raw_estimator,
                    X_scaled,
                    y_val,
                    scoring="roc_auc" if len(np.unique(y_val)) > 1 else "accuracy",
                    n_repeats=n_repeats,
                    random_state=random_state,
                    n_jobs=1,
                )
                importances = perm_res.importances_mean
            elif raw_estimator is not None:
                perm_res = permutation_importance(
                    raw_estimator,
                    X_val,
                    y_val,
                    scoring="roc_auc" if len(np.unique(y_val)) > 1 else "accuracy",
                    n_repeats=n_repeats,
                    random_state=random_state,
                    n_jobs=1,
                )
                importances = perm_res.importances_mean
            else:
                # Fallback to model-native
                native_dict = self.model.get_feature_importances()
                importances = np.array([native_dict.get(fn, 0.0) for fn in self.feature_names])
        else:
            # Model-native importances
            native_dict = self.model.get_feature_importances()
            importances = np.array([native_dict.get(fn, 0.0) for fn in self.feature_names])

        # Normalize scores to [0, 1] relative scale
        max_val = np.max(np.abs(importances)) if len(importances) > 0 and np.max(np.abs(importances)) > 0 else 1.0

        # Sort indices descending
        sorted_indices = np.argsort(-np.abs(importances))

        for rank_idx, idx in enumerate(sorted_indices, start=1):
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            raw_score = float(importances[idx])
            norm_score = float(abs(raw_score) / max_val)

            # Determine feature group
            if "latency" in feat_name or "error" in feat_name or "retry" in feat_name or "timeout" in feat_name:
                grp = "reliability"
            elif "interaction" in feat_name or "unique" in feat_name or "incoming" in feat_name or "outgoing" in feat_name:
                grp = "interaction"
            elif "trend" in feat_name or "rolling" in feat_name or "change" in feat_name:
                grp = "temporal"
            else:
                grp = "agent_behavior"

            attributions.append(
                FeatureAttribution(
                    feature_name=feat_name,
                    feature_group=grp,
                    importance_score=round(norm_score, 4),
                    signed_contribution=round(raw_score, 4),
                    rank=rank_idx,
                )
            )

        return attributions

    def explain_instance(
        self,
        x_sample: np.ndarray,
        base_probability: Optional[float] = None,
        top_k: int = 8,
    ) -> List[FeatureAttribution]:
        """Explain a single prediction point via local linear/perturbation attribution.
        
        Args:
            x_sample: Single sample 1D feature vector [D].
            base_probability: Model predicted probability for this sample.
            top_k: Number of top features to return.
            
        Returns:
            List of FeatureAttribution objects ranked by contribution magnitude.
        """
        x_sample_2d = x_sample.reshape(1, -1)
        if base_probability is None:
            base_prob = float(self.model.predict_proba(x_sample_2d)[0])
        else:
            base_prob = base_probability

        local_attributions: List[FeatureAttribution] = []

        # For Logistic Regression, local attribution is scaled feature value * coefficient
        raw_est = getattr(self.model, "model", None)
        scaler = getattr(self.model, "scaler", None)

        if hasattr(raw_est, "coef_"):
            coefs = raw_est.coef_[0]
            if scaler is not None:
                x_scaled = scaler.transform(x_sample_2d)[0]
            else:
                x_scaled = x_sample
            contributions = x_scaled * coefs
        else:
            # For tree ensembles (RandomForest, XGBoost), use leave-one-feature-out zeroing
            contributions = np.zeros(len(x_sample))
            for i in range(len(x_sample)):
                x_perturbed = x_sample.copy()
                x_perturbed[i] = 0.0  # Counterfactual zeroing
                p_pert = float(self.model.predict_proba(x_perturbed.reshape(1, -1))[0])
                # If zeroing decreases risk, original value was contributing positively to risk
                contributions[i] = base_prob - p_pert

        max_contrib = np.max(np.abs(contributions)) if len(contributions) > 0 and np.max(np.abs(contributions)) > 0 else 1.0
        sorted_indices = np.argsort(-np.abs(contributions))

        for rank_idx, idx in enumerate(sorted_indices[:top_k], start=1):
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            raw_contrib = float(contributions[idx])
            norm_contrib = float(abs(raw_contrib) / max_contrib)

            if "latency" in feat_name or "error" in feat_name or "retry" in feat_name or "timeout" in feat_name:
                grp = "reliability"
            elif "interaction" in feat_name or "unique" in feat_name or "incoming" in feat_name or "outgoing" in feat_name:
                grp = "interaction"
            elif "trend" in feat_name or "rolling" in feat_name or "change" in feat_name:
                grp = "temporal"
            else:
                grp = "agent_behavior"

            local_attributions.append(
                FeatureAttribution(
                    feature_name=feat_name,
                    feature_group=grp,
                    importance_score=round(norm_contrib, 4),
                    signed_contribution=round(raw_contrib, 4),
                    baseline_value=float(x_sample[idx]),
                    rank=rank_idx,
                )
            )

        return local_attributions
