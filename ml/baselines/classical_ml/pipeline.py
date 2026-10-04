"""Training and Evaluation Pipeline for Classical Machine Learning Baselines.

Coordinates:
- Horizon-specific tabular feature extraction
- Model fitting with class weighting
- Evaluation of Precision, Recall, F1, AUROC, AUPRC, FPR, FAR
- Probability calibration analysis (Brier score & calibration curve)
- Early warning lead-time tracking
- Feature importance extraction
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from ml.baselines.classical_ml.features import TabularFeatureExtractor, FEATURE_NAMES
from ml.baselines.classical_ml.models import BaseBaselineModel
from ml.baselines.rule_based.evaluator import (
    compute_classification_metrics,
    compute_lead_time_metrics,
)


def compute_calibration_analysis(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """Assess model probability calibration via Brier score and binned reliability curve.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted failure probabilities in [0.0, 1.0].
        n_bins: Number of probability bins (default: 5).
        
    Returns:
        Dictionary with Brier score, expected calibration error (ECE), and binned statistics.
    """
    if not y_true or len(y_true) != len(y_prob):
        return {"brier_score": 0.0, "expected_calibration_error": 0.0, "bins": []}

    y_t = np.array(y_true, dtype=np.float32)
    y_p = np.array(y_prob, dtype=np.float32)

    # 1. Brier score = Mean squared error of probabilities
    brier_score = float(np.mean((y_p - y_t) ** 2))

    # 2. Binned calibration
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_p >= low) & (y_p <= high if i == n_bins - 1 else y_p < high)
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            mean_pred = float(np.mean(y_p[mask]))
            true_fraction = float(np.mean(y_t[mask]))
            bin_ece = abs(true_fraction - mean_pred) * (bin_count / n_total)
            ece += bin_ece

            bin_data.append({
                "bin_range": [round(low, 2), round(high, 2)],
                "count": bin_count,
                "mean_predicted_prob": round(mean_pred, 4),
                "true_fraction": round(true_fraction, 4),
            })

    return {
        "brier_score": round(brier_score, 4),
        "expected_calibration_error": round(float(ece), 4),
        "bins": bin_data,
    }


def train_and_evaluate_baseline(
    model: BaseBaselineModel,
    train_samples: List[Any],
    test_samples: List[Any],
    horizon: int,
    extractor: Optional[TabularFeatureExtractor] = None,
) -> Dict[str, Any]:
    """Train a baseline model and evaluate on the test split for a given horizon k.
    
    Args:
        model: BaseBaselineModel instance (LogisticRegression, RandomForest, or XGBoost).
        train_samples: Training split prediction samples or flat rows.
        test_samples: Test split prediction samples or flat rows.
        horizon: Target prediction horizon K (e.g. 1, 3, 5, 10, 20).
        extractor: TabularFeatureExtractor instance.
        
    Returns:
        Dictionary containing classification metrics, lead times, calibration, and feature importances.
    """
    ext = extractor or TabularFeatureExtractor()

    # 1. Extract tabular matrices for horizon K
    X_train, y_train, meta_train = ext.extract_matrix_and_labels(train_samples, horizon=horizon)
    X_test, y_test, meta_test = ext.extract_matrix_and_labels(test_samples, horizon=horizon)

    if len(X_train) == 0 or len(X_test) == 0:
        return {
            "horizon": horizon,
            "model_name": model.model_name,
            "metrics": {},
            "lead_time": {},
            "calibration": {},
            "feature_importances": {},
            "predictions": [],
            "error": f"Insufficient samples for horizon k={horizon} (train={len(X_train)}, test={len(X_test)})",
        }

    # 2. Fit model strictly on TRAIN data
    model.fit(X_train, y_train, feature_names=ext.feature_names)

    # 3. Predict on held-out TEST data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 4. Compute classification metrics
    class_metrics = compute_classification_metrics(
        y_true=y_test.tolist(),
        y_pred=y_pred.tolist(),
        y_scores=y_prob.tolist(),
    )

    # 5. Compute probability calibration
    calibration_metrics = compute_calibration_analysis(
        y_true=y_test.tolist(),
        y_prob=y_prob.tolist(),
    )

    # 6. Structure prediction rows for lead-time calculation and disk persistence
    prediction_records = []
    for idx, meta in enumerate(meta_test):
        rec = {
            "sample_id": meta["sample_id"],
            "run_id": meta["run_id"],
            "step_idx": meta["step_idx"],
            "timestamp": meta["timestamp"],
            "prediction_horizon": horizon,
            "true_label": int(y_test[idx]),
            "predicted_label": int(y_pred[idx]),
            "predicted_probability": round(float(y_prob[idx]), 4),
            "model_name": model.model_name,
            # For lead-time calculation interface
            "prediction": int(y_pred[idx]),
            "ground_truth": int(y_test[idx]),
        }
        prediction_records.append(rec)

    # 7. Compute early warning lead times
    lead_time_metrics = compute_lead_time_metrics(prediction_records)

    # 8. Feature importances
    importances = model.get_feature_importances()

    return {
        "horizon": horizon,
        "model_name": model.model_name,
        "train_samples_count": len(X_train),
        "test_samples_count": len(X_test),
        "class_balance_train": {
            "pos": int(np.sum(y_train == 1)),
            "neg": int(np.sum(y_train == 0)),
        },
        "class_balance_test": {
            "pos": int(np.sum(y_test == 1)),
            "neg": int(np.sum(y_test == 0)),
        },
        "metrics": class_metrics,
        "lead_time": lead_time_metrics,
        "calibration": calibration_metrics,
        "feature_importances": importances,
        "predictions": prediction_records,
    }
