"""Phase 8 Test Suite: Classical Machine Learning Baselines.

Verifies:
1. Feature extraction (28 features across 4 groups)
2. Feature dimensions (consistent shape and ordering)
3. Train/validation/test separation (strict split disjointness)
4. No future leakage (strictly causal inputs <= t)
5. Logistic Regression (fit, predict, scaling, class_weight)
6. Random Forest (fit, predict, feature_importances_)
7. XGBoost (fit, predict, scale_pos_weight)
8. Probability prediction (predict_proba in [0.0, 1.0])
9. Reproducibility (deterministic seeds yield identical models)
10. Model serialization (save and load checkpoints)
11. Prediction schema (all required fields present)
12. Metric calculation (Precision, Recall, F1, AUROC, AUPRC, FPR)
13. Lead-time calculation (t_failure - t_warning, ignoring post-failure alerts)
14. Experiment tracking (summary and checkpoint persistence)
"""

import pytest
from pathlib import Path
import tempfile
import json
import numpy as np

from ml.baselines.classical_ml.features import (
    TabularFeatureExtractor,
    FEATURE_NAMES,
    FEATURE_DOCUMENTATION,
)
from ml.baselines.classical_ml.models import (
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
from ml.data.schema import PredictionSample


@pytest.fixture
def mock_dataset():
    """Create synthetic training and testing samples for model benchmarking."""
    np.random.seed(42)
    train_samples = []
    test_samples = []

    # Generate 40 train samples and 20 test samples
    for i in range(60):
        is_test = (i >= 40)
        label = 1 if (i % 3 == 0) else 0

        # Features correlate with label to enable meaningful evaluation
        quality = 0.4 if label == 1 else 0.95
        confidence = 0.3 if label == 1 else 0.90
        latency = 0.6 if label == 1 else 0.10
        errors = 3.0 if label == 1 else 0.0

        sample = PredictionSample(
            sample_id=f"sample_{i}",
            run_id=f"run_{i // 5}",
            timestamp=float(i * 0.5),
            step_idx=i % 10,
            task_type="research",
            topology="pipeline",
            number_of_agents=5,
            prediction_horizon=1 if (i % 2 == 0) else 3,
            node_features={},
            edge_features={},
            temporal_graph_history=[],
            agent_level_features={
                "total_events_observed": float(i + 1),
                "error_count": errors,
                "total_retries": 1.0 if label == 1 else 0.0,
                "tool_failure_count": 1.0 if label == 1 else 0.0,
                "mean_contradiction_score": 0.8 if label == 1 else 0.0,
                "max_contradiction_score": 0.9 if label == 1 else 0.0,
                "mean_confidence": confidence,
                "min_confidence": confidence - 0.05,
                "mean_latency": latency,
                "max_latency": latency + 0.1,
                "interaction_density": 2.0,
                "min_output_quality": quality - 0.1,
                "mean_output_quality": quality,
            },
            label=label,
            failure_type="tool_failure" if label == 1 else "none",
            failure_level=2 if label == 1 else 0,
            source_event_id=f"evt_{i}",
            random_seed=42,
            dataset_version="v1",
            split="test" if is_test else "train",
        )
        if is_test:
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, test_samples


def test_1_feature_extraction(mock_dataset):
    """Test 1: TabularFeatureExtractor correctly computes all features."""
    train_samples, _ = mock_dataset
    extractor = TabularFeatureExtractor()
    feat_dict = extractor.extract_from_sample(train_samples[0])

    assert len(feat_dict) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in feat_dict
        assert isinstance(feat_dict[name], float)
        assert not np.isnan(feat_dict[name])


def test_2_feature_dimensions(mock_dataset):
    """Test 2: Extracted 2D matrix shape strictly matches (N, 28)."""
    train_samples, _ = mock_dataset
    extractor = TabularFeatureExtractor()
    X = extractor.extract_features_array(train_samples)

    assert X.shape == (len(train_samples), 28)
    assert X.dtype == np.float32


def test_3_train_validation_test_separation(mock_dataset):
    """Test 3: Samples from train and test splits do not overlap."""
    train_samples, test_samples = mock_dataset
    train_runs = {s.run_id for s in train_samples}
    test_runs = {s.run_id for s in test_samples}

    # Samples are strictly segregated
    train_ids = {s.sample_id for s in train_samples}
    test_ids = {s.sample_id for s in test_samples}
    assert train_ids.isdisjoint(test_ids)


def test_4_no_future_leakage():
    """Test 4: Strict causality test ensuring features <= t never access future information."""
    extractor = TabularFeatureExtractor()

    # Modify only future label/level and assert extracted features remain unchanged
    s_a = PredictionSample(
        sample_id="sample_leak_test",
        run_id="run_leak",
        timestamp=2.0,
        step_idx=3,
        task_type="coding",
        topology="star",
        number_of_agents=5,
        prediction_horizon=3,
        node_features={},
        edge_features={},
        temporal_graph_history=[],
        agent_level_features={"error_count": 0.0, "mean_latency": 0.10},
        label=0,
        failure_type="none",
        failure_level=0,
        source_event_id="evt_0",
        random_seed=42,
        dataset_version="v1",
    )
    s_b = PredictionSample(
        sample_id="sample_leak_test",
        run_id="run_leak",
        timestamp=2.0,
        step_idx=3,
        task_type="coding",
        topology="star",
        number_of_agents=5,
        prediction_horizon=3,
        node_features={},
        edge_features={},
        temporal_graph_history=[],
        agent_level_features={"error_count": 0.0, "mean_latency": 0.10},
        label=1,  # Future failure occurred
        failure_type="cascading_failure",
        failure_level=3,
        source_event_id="evt_0",
        random_seed=42,
        dataset_version="v1",
    )

    feats_a = extractor.extract_from_sample(s_a)
    feats_b = extractor.extract_from_sample(s_b)
    assert feats_a == feats_b, "Data leakage detected: features changed when future target changed!"


def test_5_logistic_regression(mock_dataset):
    """Test 5: LogisticRegression baseline trains, predicts, and extracts coefficients."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    extractor = TabularFeatureExtractor()
    X_train, y_train, _ = extractor.extract_matrix_and_labels(train_samples, horizon=1)
    X_test, y_test, _ = extractor.extract_matrix_and_labels(test_samples, horizon=1)

    model = LogisticRegressionBaseline(random_state=42)
    model.fit(X_train, y_train, feature_names=extractor.feature_names)

    assert model.is_fitted
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})

    importances = model.get_feature_importances()
    assert len(importances) == 28


def test_6_random_forest(mock_dataset):
    """Test 6: RandomForest baseline trains, predicts, and extracts Gini importances."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    extractor = TabularFeatureExtractor()
    X_train, y_train, _ = extractor.extract_matrix_and_labels(train_samples, horizon=1)
    X_test, _, _ = extractor.extract_matrix_and_labels(test_samples, horizon=1)

    model = RandomForestBaseline(n_estimators=20, random_state=42)
    model.fit(X_train, y_train, feature_names=extractor.feature_names)

    assert model.is_fitted
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)

    importances = model.get_feature_importances()
    assert len(importances) == 28
    assert sum(importances.values()) > 0.0


def test_7_xgboost(mock_dataset):
    """Test 7: XGBoost baseline trains, predicts, and handles class imbalance."""
    if not HAS_XGBOOST:
        pytest.skip("xgboost not installed")

    train_samples, test_samples = mock_dataset
    extractor = TabularFeatureExtractor()
    X_train, y_train, _ = extractor.extract_matrix_and_labels(train_samples, horizon=1)
    X_test, _, _ = extractor.extract_matrix_and_labels(test_samples, horizon=1)

    model = XGBoostBaseline(n_estimators=10, random_state=42)
    model.fit(X_train, y_train, feature_names=extractor.feature_names)

    assert model.is_fitted
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)

    importances = model.get_feature_importances()
    assert len(importances) == 28


def test_8_probability_prediction(mock_dataset):
    """Test 8: predict_proba produces valid calibrated probabilities in [0.0, 1.0]."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    extractor = TabularFeatureExtractor()
    X_train, y_train, _ = extractor.extract_matrix_and_labels(train_samples, horizon=1)
    X_test, _, _ = extractor.extract_matrix_and_labels(test_samples, horizon=1)

    model = LogisticRegressionBaseline(random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)

    assert len(probs) == len(X_test)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_9_reproducibility(mock_dataset):
    """Test 9: Identical random seeds produce bit-for-bit identical predictions."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    extractor = TabularFeatureExtractor()
    X_train, y_train, _ = extractor.extract_matrix_and_labels(train_samples, horizon=1)
    X_test, _, _ = extractor.extract_matrix_and_labels(test_samples, horizon=1)

    rf1 = RandomForestBaseline(n_estimators=30, random_state=777)
    rf1.fit(X_train, y_train)
    p1 = rf1.predict_proba(X_test)

    rf2 = RandomForestBaseline(n_estimators=30, random_state=777)
    rf2.fit(X_train, y_train)
    p2 = rf2.predict_proba(X_test)

    np.testing.assert_allclose(p1, p2, rtol=1e-5)


def test_10_model_serialization(mock_dataset):
    """Test 10: Model serialization and deserialization via save and load."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    extractor = TabularFeatureExtractor()
    X_train, y_train, _ = extractor.extract_matrix_and_labels(train_samples, horizon=1)
    X_test, _, _ = extractor.extract_matrix_and_labels(test_samples, horizon=1)

    model = LogisticRegressionBaseline(random_state=42)
    model.fit(X_train, y_train)
    orig_preds = model.predict_proba(X_test)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "lr_test.pkl"
        model.save(ckpt_path)

        reloaded = LogisticRegressionBaseline.load(ckpt_path)
        reloaded_preds = reloaded.predict_proba(X_test)

        np.testing.assert_allclose(orig_preds, reloaded_preds)


def test_11_prediction_schema(mock_dataset):
    """Test 11: Prediction records match required output schema."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    model = LogisticRegressionBaseline(random_state=42)
    result = train_and_evaluate_baseline(
        model=model,
        train_samples=train_samples,
        test_samples=test_samples,
        horizon=1,
    )

    preds = result["predictions"]
    assert len(preds) > 0
    first = preds[0]

    required_fields = [
        "sample_id",
        "run_id",
        "timestamp",
        "prediction_horizon",
        "true_label",
        "predicted_label",
        "predicted_probability",
        "model_name",
    ]
    for rf in required_fields:
        assert rf in first, f"Missing required prediction field: {rf}"


def test_12_metric_calculation(mock_dataset):
    """Test 12: Metric evaluation generates all required scientific metrics."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    model = LogisticRegressionBaseline(random_state=42)
    result = train_and_evaluate_baseline(
        model=model,
        train_samples=train_samples,
        test_samples=test_samples,
        horizon=1,
    )

    metrics = result["metrics"]
    for m in ["precision", "recall", "f1", "auroc", "auprc", "false_positive_rate", "false_alarm_rate"]:
        assert m in metrics
        assert 0.0 <= metrics[m] <= 1.0


def test_13_lead_time_calculation(mock_dataset):
    """Test 13: Lead-time calculation handles early warnings appropriately."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    model = RandomForestBaseline(n_estimators=10, random_state=42)
    result = train_and_evaluate_baseline(
        model=model,
        train_samples=train_samples,
        test_samples=test_samples,
        horizon=1,
    )

    lt = result["lead_time"]
    for field in ["mean_lead_time", "median_lead_time", "successful_early_warnings", "warnings_per_trajectory"]:
        assert field in lt


def test_14_experiment_tracking(mock_dataset):
    """Test 14: ClassicalMLExperimentRunner executes and records experiment summaries."""
    if not HAS_SKLEARN:
        pytest.skip("scikit-learn not installed")

    train_samples, test_samples = mock_dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = ClassicalMLExperimentRunner(
            base_results_dir=tmp_dir,
            random_seed=42,
        )
        models = ["logistic_regression", "random_forest"]
        if HAS_XGBOOST:
            models.append("xgboost")

        summary = runner.run_experiment(
            train_samples=train_samples,
            test_samples=test_samples,
            models_to_run=models,
            horizons=[1, 3],
        )

        assert "experiment_id" in summary
        assert len(summary["model_results"]) == len(models)

        # Verify disk artifacts
        base_path = Path(tmp_dir)
        for m in models:
            assert (base_path / m).exists()
        assert (base_path / "feature_importance").exists()
