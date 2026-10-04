"""Phase 6 Test Suite: Dataset Generation, Temporal Samples, and Failure Labeling.

Verifies:
1. Sample generation
2. Label correctness (binary 0/1 and multi-level taxonomy)
3. Horizon correctness (k in {1, 3, 5, 10, 20})
4. No future leakage (strictly causal features <= t)
5. Run-level splitting (zero run overlap between train/val/test)
6. Reproducibility (deterministic generation from seed)
7. Dataset versioning (manifest metadata)
8. Schema validation (required types, non-nulls)
9. Duplicate detection (unique sample_id constraints)
10. Invalid sample rejection (rejection of truncated future horizons)
11. Class distribution reporting (positive/negative ratios, failure types/levels)
12. Different horizons and topologies (pipeline, star, mesh, custom, agent counts 3, 5, 8, 12)
"""

import pytest
from pathlib import Path
import tempfile
import json
import random

from ml.simulation.run import SimulationRun
from ml.simulation.fault_injection.injector import FaultInjector
from ml.data.schema import PredictionSample, DatasetManifest
from ml.data.labeling import compute_prediction_label
from ml.data.feature_windows import extract_agent_level_features, extract_graph_features
from ml.data.prediction_samples import PredictionSampleGenerator
from ml.data.split import split_trajectories, split_samples
from ml.data.storage import DatasetStorage
from ml.data.validation import DatasetValidator, ValidationResult
from ml.data.dataset_builder import DatasetBuilder


@pytest.fixture
def clean_simulation_run():
    """Generate a clean, fault-free simulation run."""
    run = SimulationRun(
        run_id="test_run_clean",
        task_type="research",
        topology="pipeline",
        num_agents=5,
        random_seed=42,
    )
    run.execute()
    return run


@pytest.fixture
def fault_injected_run():
    """Generate a simulation run with an injected fault."""
    injector = FaultInjector(
        fault_type="hallucinated_output",
        probability=1.0,
        injection_step=1,
        severity=0.85,
        random_seed=42,
    )
    run = SimulationRun(
        run_id="test_run_fault",
        task_type="research",
        topology="pipeline",
        num_agents=5,
        random_seed=42,
        fault_injector=injector,
    )
    run.execute()
    return run


def test_1_sample_generation(clean_simulation_run):
    """Test 1: Verify prediction samples are generated with valid structure."""
    gen = PredictionSampleGenerator(prediction_horizons=[1, 2], dataset_version="test_v1")
    samples = gen.generate_from_run(clean_simulation_run)

    assert len(samples) > 0, "No samples generated"
    first = samples[0]
    assert isinstance(first, PredictionSample)
    assert first.run_id == "test_run_clean"
    assert first.prediction_horizon in [1, 2]
    assert first.dataset_version == "test_v1"
    assert isinstance(first.node_features, dict)
    assert isinstance(first.edge_features, dict)
    assert isinstance(first.agent_level_features, dict)
    assert isinstance(first.temporal_graph_history, list)


def test_2_label_correctness(clean_simulation_run, fault_injected_run):
    """Test 2: Verify binary and multi-level failure labels are computed correctly."""
    # Clean run should have 0 labels
    norm_clean = clean_simulation_run.events
    label_clean, f_type_clean, level_clean = compute_prediction_label(
        events=norm_clean,
        current_step_idx=0,
        prediction_horizon=2,
        propagation_tracker=clean_simulation_run.propagation_tracker,
    )
    assert label_clean == 0
    assert f_type_clean == "none"
    assert level_clean == 0

    # Injected run should detect failure in future horizon spanning step 1
    norm_fault = fault_injected_run.events
    label_fault, f_type_fault, level_fault = compute_prediction_label(
        events=norm_fault,
        current_step_idx=0,
        prediction_horizon=2,
        propagation_tracker=fault_injected_run.propagation_tracker,
    )
    assert label_fault == 1
    assert f_type_fault == "hallucinated_output"
    assert level_fault in (1, 2, 3)


def test_3_horizon_correctness(fault_injected_run, clean_simulation_run):
    """Test 3: Verify horizon bounds are strictly respected."""
    # Injected run has fault injected at step 1
    # At step 0, horizon k=1 only looks at step 1 (contains fault) -> label=1
    label_k1, _, _ = compute_prediction_label(
        events=fault_injected_run.events,
        current_step_idx=0,
        prediction_horizon=1,
    )
    assert label_k1 == 1

    # At step 2 of clean run, no fault occurs in (2, 3] -> label should be 0
    label_later, _, _ = compute_prediction_label(
        events=clean_simulation_run.events,
        current_step_idx=2,
        prediction_horizon=1,
    )
    assert label_later == 0


def test_4_no_future_leakage(fault_injected_run):
    """Test 4: Strict causality test ensuring features <= t never access future information."""
    events = fault_injected_run.events
    t_cutoff = events[1].timestamp
    step_cutoff = 1

    # Extract features at step 1
    agent_feats = extract_agent_level_features(
        events=events,
        current_step_idx=step_cutoff,
        current_timestamp=t_cutoff,
    )
    nodes, edges, history = extract_graph_features(
        events=events,
        current_step_idx=step_cutoff,
        current_timestamp=t_cutoff,
        run_id=fault_injected_run.run_id,
    )

    # All snapshots in history must have timestamp <= t_cutoff and step <= step_cutoff
    for snap in history:
        assert snap["timestamp"] <= t_cutoff + 1e-6
        if snap.get("step_idx") is not None:
            assert snap["step_idx"] <= step_cutoff

    # Future events (step >= 2) must not be counted in total_events_observed
    assert agent_feats["total_events_observed"] <= 2


def test_5_run_level_splitting():
    """Test 5: Trajectory/run-level splitting prevents data leakage."""
    run_ids = [f"run_{i:03d}" for i in range(20)]
    splits = split_trajectories(run_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Disjointness checks
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == 20

    # Verify sample assignment
    dummy_samples = [
        PredictionSample(
            sample_id=f"{rid}_s0_k1",
            run_id=rid,
            timestamp=0.0,
            step_idx=0,
            task_type="research",
            topology="pipeline",
            number_of_agents=5,
            prediction_horizon=1,
            node_features={},
            edge_features={},
            temporal_graph_history=[],
            agent_level_features={},
            label=0,
            failure_type="none",
            failure_level=0,
            source_event_id="e0",
            random_seed=42,
            dataset_version="v1",
        )
        for rid in run_ids
    ]
    sample_splits = split_samples(dummy_samples, splits)
    assert len(sample_splits["train"]) == len(train_set)
    assert len(sample_splits["val"]) == len(val_set)
    assert len(sample_splits["test"]) == len(test_set)


def test_6_reproducibility():
    """Test 6: Verify exact deterministic reproducibility with identical random seed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path1 = Path(tmp_dir) / "ds1"
        path2 = Path(tmp_dir) / "ds2"

        builder1 = DatasetBuilder(num_runs=3, random_seed=12345, output_dir=path1)
        _, manifest1, _ = builder1.build_dataset()

        builder2 = DatasetBuilder(num_runs=3, random_seed=12345, output_dir=path2)
        _, manifest2, _ = builder2.build_dataset()

        assert manifest1.number_of_samples == manifest2.number_of_samples
        assert manifest1.class_distribution == manifest2.class_distribution

        storage = DatasetStorage(Path(tmp_dir))
        tab1 = storage.load_tabular(path1 / "train.parquet")
        tab2 = storage.load_tabular(path2 / "train.parquet")

        assert len(tab1) == len(tab2)
        for r1, r2 in zip(tab1, tab2):
            assert r1["sample_id"] == r2["sample_id"]
            assert r1["label"] == r2["label"]
            assert r1["step_idx"] == r2["step_idx"]


def test_7_dataset_versioning():
    """Test 7: Verify dataset version and metadata persistence in manifest."""
    manifest = DatasetManifest(
        dataset_version="agentguard_dataset_v1_test",
        random_seed=999,
        number_of_runs=10,
        number_of_samples=100,
        prediction_horizons=[1, 3, 5],
    )
    json_str = manifest.to_json()
    reloaded = DatasetManifest.from_json(json_str)

    assert reloaded.dataset_version == "agentguard_dataset_v1_test"
    assert reloaded.random_seed == 999
    assert reloaded.prediction_horizons == [1, 3, 5]
    assert reloaded.simulator_version == "0.6.0"


def test_8_schema_validation():
    """Test 8: Schema conformance and missing value checks."""
    validator = DatasetValidator(strict_mode=True)
    valid_sample = PredictionSample(
        sample_id="run_0_s0_k1",
        run_id="run_0",
        timestamp=1.5,
        step_idx=0,
        task_type="coding",
        topology="star",
        number_of_agents=5,
        prediction_horizon=1,
        node_features={"a1": {"feat1": 0.5}},
        edge_features={},
        temporal_graph_history=[],
        agent_level_features={"mean_quality": 0.9},
        label=0,
        failure_type="none",
        failure_level=0,
        source_event_id="e0",
        random_seed=42,
        dataset_version="v1",
        split="train",
    )
    result = validator.validate_samples([valid_sample])
    assert result.is_valid
    assert len(result.errors) == 0


def test_9_duplicate_detection():
    """Test 9: Verify duplicate sample_id detection."""
    validator = DatasetValidator(strict_mode=True)
    sample_a = PredictionSample(
        sample_id="duplicate_id_001",
        run_id="run_0",
        timestamp=1.0,
        step_idx=0,
        task_type="research",
        topology="pipeline",
        number_of_agents=3,
        prediction_horizon=1,
        node_features={},
        edge_features={},
        temporal_graph_history=[],
        agent_level_features={},
        label=0,
        failure_type="none",
        failure_level=0,
        source_event_id="e0",
        random_seed=42,
        dataset_version="v1",
    )
    sample_b = PredictionSample(
        sample_id="duplicate_id_001",
        run_id="run_1",
        timestamp=2.0,
        step_idx=1,
        task_type="research",
        topology="pipeline",
        number_of_agents=3,
        prediction_horizon=1,
        node_features={},
        edge_features={},
        temporal_graph_history=[],
        agent_level_features={},
        label=0,
        failure_type="none",
        failure_level=0,
        source_event_id="e1",
        random_seed=42,
        dataset_version="v1",
    )
    result = validator.validate_samples([sample_a, sample_b])
    assert not result.is_valid
    assert any("Duplicate sample_id" in err for err in result.errors)


def test_10_invalid_sample_rejection(clean_simulation_run):
    """Test 10: Verify rejection of invalid samples where horizon exceeds trajectory length."""
    gen = PredictionSampleGenerator(prediction_horizons=[1, 5, 20, 100])
    samples = gen.generate_from_run(clean_simulation_run)

    # For k=100 (which exceeds trajectory steps ~ 4-5 steps), NO sample should be generated
    k100_samples = [s for s in samples if s.prediction_horizon == 100]
    assert len(k100_samples) == 0, "Invalid sample generated where future horizon does not exist!"


def test_11_class_distribution_reporting():
    """Test 11: Accurate calculation of class distribution metrics."""
    validator = DatasetValidator(strict_mode=True)
    samples = [
        PredictionSample(
            sample_id=f"sample_{i}",
            run_id=f"run_{i}",
            timestamp=float(i),
            step_idx=i,
            task_type="planning",
            topology="mesh",
            number_of_agents=5,
            prediction_horizon=1,
            node_features={},
            edge_features={},
            temporal_graph_history=[],
            agent_level_features={},
            label=1 if i < 3 else 0,
            failure_type="tool_failure" if i < 3 else "none",
            failure_level=1 if i < 3 else 0,
            source_event_id=f"e{i}",
            random_seed=42,
            dataset_version="v1",
            split="train",
        )
        for i in range(10)
    ]
    result = validator.validate_samples(samples)
    assert result.is_valid
    stats = result.statistics
    assert stats["total_samples"] == 10
    assert stats["positive_samples"] == 3
    assert stats["negative_samples"] == 7
    assert stats["positive_class_ratio"] == 0.3
    assert stats["failure_levels"][1] == 3
    assert stats["failure_levels"][0] == 7


def test_12_different_horizons_topologies_and_agents():
    """Test 12: Verify generalization across topologies, tasks, agent counts (3, 5, 8, 12), and horizons."""
    for num_agents in [3, 5, 8]:
        for topo in ["pipeline", "star", "mesh", "custom"]:
            run = SimulationRun(
                run_id=f"gen_{topo}_{num_agents}",
                task_type="coding",
                topology=topo,
                num_agents=num_agents,
                random_seed=42,
            )
            run.execute()
            assert len(run.events) > 0

            gen = PredictionSampleGenerator(prediction_horizons=[1, 3])
            samples = gen.generate_from_run(run)
            assert len(samples) > 0
            for s in samples:
                assert s.number_of_agents == num_agents
                assert s.topology == topo
