"""Experiment Tracking and Multi-Horizon Benchmarking for Classical ML Baselines.

Coordinates running:
- Logistic Regression
- Random Forest
- XGBoost
across horizons k in {1, 3, 5, 10, 20}.

Persists structured results into:
results/baselines/classical_ml/
  ├── logistic_regression/
  ├── random_forest/
  ├── xgboost/
  └── feature_importance/

Strictly avoids overwriting previous experiments.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
import json
from uuid import uuid4

from ml.baselines.classical_ml.features import TabularFeatureExtractor
from ml.baselines.classical_ml.models import (
    BaseBaselineModel,
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
    HAS_SKLEARN,
    HAS_XGBOOST,
)
from ml.baselines.classical_ml.pipeline import train_and_evaluate_baseline
from ml.data.storage import DatasetStorage, HAS_PYARROW

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.parquet as pq


class ClassicalMLExperimentRunner:
    """Manages training, evaluation, and result tracking across classical ML baselines."""

    def __init__(
        self,
        base_results_dir: Union[str, Path] = "results/baselines/classical_ml",
        random_seed: int = 42,
    ) -> None:
        self.base_results_dir = Path(base_results_dir)
        self.random_seed = random_seed
        self.extractor = TabularFeatureExtractor()

    def run_experiment(
        self,
        train_samples: List[Any],
        test_samples: List[Any],
        models_to_run: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        dataset_version: str = "agentguard_dataset_v1",
    ) -> Dict[str, Any]:
        """Execute full benchmark across specified models and prediction horizons.
        
        Args:
            train_samples: Training split samples.
            test_samples: Test split samples.
            models_to_run: List of model keys ('logistic_regression', 'random_forest', 'xgboost').
            horizons: Horizons k to evaluate (default: [1, 3, 5, 10, 20]).
            dataset_version: Semantic dataset version string.
            
        Returns:
            Dictionary containing all experiment results and summary.
        """
        target_models = models_to_run or ["logistic_regression", "random_forest", "xgboost"]
        eval_horizons = horizons or [1, 3, 5, 10, 20]

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_id = f"exp_classical_{timestamp_str}_{uuid4().hex[:6]}"

        experiment_summary: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset_version,
            "random_seed": self.random_seed,
            "models": target_models,
            "horizons": eval_horizons,
            "model_results": {},
            "feature_importance_summary": {},
        }

        # Ensure directories exist
        fi_dir = self.base_results_dir / "feature_importance"
        fi_dir.mkdir(parents=True, exist_ok=True)

        for model_key in target_models:
            model_dir = self.base_results_dir / model_key / experiment_id
            model_dir.mkdir(parents=True, exist_ok=True)

            model_horizon_results: Dict[int, Any] = {}
            all_model_predictions: List[Dict[str, Any]] = []

            for k in eval_horizons:
                model_inst = self._instantiate_model(model_key)
                if model_inst is None:
                    continue

                res = train_and_evaluate_baseline(
                    model=model_inst,
                    train_samples=train_samples,
                    test_samples=test_samples,
                    horizon=k,
                    extractor=self.extractor,
                )

                model_horizon_results[k] = res
                all_model_predictions.extend(res.get("predictions", []))

                # Save model checkpoint
                checkpoint_path = model_dir / f"model_k{k}.pkl"
                if model_inst.is_fitted:
                    model_inst.save(checkpoint_path)

                # Save horizon feature importance
                if res.get("feature_importances"):
                    fi_path = fi_dir / f"{model_key}_k{k}_{experiment_id}.json"
                    with open(fi_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "experiment_id": experiment_id,
                            "model_name": model_key,
                            "horizon": k,
                            "notice": "Descriptive importance only. Not causal importance.",
                            "feature_importances": res["feature_importances"],
                        }, f, indent=2)

            experiment_summary["model_results"][model_key] = model_horizon_results

            # Persist model metrics JSON
            metrics_to_save = {
                "experiment_id": experiment_id,
                "model_name": model_key,
                "dataset_version": dataset_version,
                "random_seed": self.random_seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "horizons": {
                    k: {
                        "metrics": v.get("metrics", {}),
                        "lead_time": v.get("lead_time", {}),
                        "calibration": v.get("calibration", {}),
                        "feature_importances": v.get("feature_importances", {}),
                    }
                    for k, v in model_horizon_results.items()
                },
            }
            with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics_to_save, f, indent=2)

            # Persist model predictions
            self._save_predictions(all_model_predictions, model_dir / "predictions.parquet")

        # Save overarching experiment summary
        summary_path = self.base_results_dir / f"summary_{experiment_id}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            # Clean JSON serialization
            serializable_summary = {
                "experiment_id": experiment_id,
                "timestamp": experiment_summary["timestamp"],
                "dataset_version": dataset_version,
                "models": target_models,
                "horizons": eval_horizons,
                "results_by_model_and_horizon": {
                    m: {
                        k: {
                            "precision": v.get("metrics", {}).get("precision", 0.0),
                            "recall": v.get("metrics", {}).get("recall", 0.0),
                            "f1": v.get("metrics", {}).get("f1", 0.0),
                            "auroc": v.get("metrics", {}).get("auroc", 0.0),
                            "auprc": v.get("metrics", {}).get("auprc", 0.0),
                            "mean_lead_time": v.get("lead_time", {}).get("mean_lead_time", 0.0),
                            "brier_score": v.get("calibration", {}).get("brier_score", 0.0),
                        }
                        for k, v in res_dict.items()
                    }
                    for m, res_dict in experiment_summary["model_results"].items()
                },
            }
            json.dump(serializable_summary, f, indent=2)

        return experiment_summary

    def _instantiate_model(self, model_key: str) -> Optional[BaseBaselineModel]:
        """Instantiate specified baseline model with default configurations."""
        clean = model_key.lower().strip()
        if clean in ("logistic_regression", "lr"):
            if not HAS_SKLEARN:
                raise ImportError("scikit-learn is required for Logistic Regression.")
            return LogisticRegressionBaseline(C=1.0, class_weight="balanced", random_state=self.random_seed)
        elif clean in ("random_forest", "rf"):
            if not HAS_SKLEARN:
                raise ImportError("scikit-learn is required for Random Forest.")
            return RandomForestBaseline(n_estimators=100, max_depth=8, class_weight="balanced", random_state=self.random_seed)
        elif clean in ("xgboost", "xgb"):
            if not HAS_XGBOOST:
                raise ImportError("xgboost is required for XGBoost. Install it via pip install xgboost.")
            return XGBoostBaseline(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=self.random_seed)
        else:
            raise ValueError(f"Unknown baseline model: '{model_key}'. Supported: logistic_regression, random_forest, xgboost")

    def _save_predictions(self, predictions: List[Dict[str, Any]], filepath: Path) -> None:
        """Save prediction rows to Parquet or JSONL."""
        if not predictions:
            return
        if HAS_PYARROW:
            table = pa.Table.from_pylist(predictions)
            pq.write_table(table, filepath)
        else:
            fallback = filepath.with_suffix(".jsonl")
            with open(fallback, "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")
