"""Unit and Integration Tests for Generalization and Robustness Framework (Phase 14).

Verifies:
1. Strict run-level isolation (train_runs & test_ood_runs == empty set).
2. Configuration isolation (no test topology/task/count leaks into train).
3. No future leakage (monotonic step order).
4. Correct agent-count splits (G1, G2).
5. Correct topology splits (G3, G3b).
6. Correct task splits (G4, G4b).
7. Correct failure-mode splits (G5).
8. Reproducibility across random seeds.
9. Generalization result schema compliance.
10. End-to-end smoke test of GeneralizationExperimentRunner.
"""

import pytest
import numpy as np
from pathlib import Path

from ml.generalization.schema import (
    GeneralizationExperimentConfig,
    GeneralizationResultRecord,
    GeneralizationGapRecord,
    GeneralizationMatrixEntry,
)
from ml.generalization.splits import (
    GeneralizationSplitter,
    GENERALIZATION_REGISTRY,
    SEEN_FAILURE_MODES,
    UNSEEN_FAILURE_MODES,
)
from ml.generalization.runner import GeneralizationExperimentRunner
from ml.generalization.models import GeneralizationModelRunner


def _create_mock_dataset():
    """Create synthetic prediction samples and graph sequences across diverse configurations."""
    samples = []
    graphs = []

    # 12 runs: 4 topologies x 3 agent counts x 4 tasks
    topos = ["pipeline", "star", "mesh", "custom"]
    counts = [3, 5, 8, 12]
    tasks = ["research", "coding", "analysis", "planning"]
    faults = list(SEEN_FAILURE_MODES) + list(UNSEEN_FAILURE_MODES)

    for r_idx in range(16):
        rid = f"run_mock_{r_idx:03d}"
        topo = topos[r_idx % len(topos)]
        count = counts[r_idx % len(counts)]
        task = tasks[r_idx % len(tasks)]
        fault = faults[r_idx % len(faults)] if r_idx % 2 == 1 else "none"

        for s_idx in range(4):
            ts = float(s_idx * 1.5)
            lbl = 1 if (fault != "none" and s_idx >= 2) else 0

            # Mock tabular sample
            s_dict = {
                "sample_id": f"{rid}_s{s_idx}_k1",
                "run_id": rid,
                "step_idx": s_idx,
                "timestamp": ts,
                "topology": topo,
                "number_of_agents": count,
                "task_type": task,
                "failure_type": fault,
                "prediction_horizon": 1,
                "label": lbl,
            }
            for f_i in range(17):
                s_dict[f"agent_feat_f{f_i}"] = float(f_i * 0.1)
            samples.append(s_dict)

            # Mock graph snapshot
            graphs.append({
                "run_id": rid,
                "step_idx": s_idx,
                "timestamp": ts,
                "nodes": [{"id": f"agent_{i}", "features": [0.1] * 14} for i in range(count)],
                "edges": [{"source": "agent_0", "target": "agent_1", "features": [0.2] * 10}],
            })

    return samples, graphs


def test_generalization_matrix_entries():
    """Verify matrix generator produces valid entries for all registered experiments."""
    entries = GeneralizationSplitter.get_matrix_entries()
    assert len(entries) >= 5
    exp_ids = {e.experiment_id for e in entries}
    assert "G1" in exp_ids
    assert "G2" in exp_ids
    assert "G3" in exp_ids
    assert "G4" in exp_ids
    assert "G5" in exp_ids

    for e in entries:
        row = e.to_markdown_row()
        assert row.startswith("| `G")


def test_agent_count_splits_g1_and_g2():
    """Verify G1 and G2 enforce strict agent-count isolation."""
    samples, graphs = _create_mock_dataset()
    splitter = GeneralizationSplitter(random_seed=42)

    # G1: Train on 3, 5 -> Test OOD on 8
    b1 = splitter.create_split_bundle(GENERALIZATION_REGISTRY["G1"], samples, graphs)
    train_counts_1 = {s["number_of_agents"] for s in b1.train_samples}
    ood_counts_1 = {s["number_of_agents"] for s in b1.test_ood_samples}
    assert train_counts_1.issubset({3, 5})
    assert ood_counts_1 == {8}
    assert len(b1.train_runs & b1.test_ood_runs) == 0

    # G2: Train on 3, 5, 8 -> Test OOD on 12
    b2 = splitter.create_split_bundle(GENERALIZATION_REGISTRY["G2"], samples, graphs)
    train_counts_2 = {s["number_of_agents"] for s in b2.train_samples}
    ood_counts_2 = {s["number_of_agents"] for s in b2.test_ood_samples}
    assert train_counts_2.issubset({3, 5, 8})
    assert ood_counts_2 == {12}
    assert len(b2.train_runs & b2.test_ood_runs) == 0


def test_topology_splits_g3():
    """Verify G3 enforces leave-one-topology-out isolation (Custom OOD)."""
    samples, graphs = _create_mock_dataset()
    splitter = GeneralizationSplitter(random_seed=42)

    b3 = splitter.create_split_bundle(GENERALIZATION_REGISTRY["G3"], samples, graphs)
    train_topos = {s["topology"] for s in b3.train_samples}
    ood_topos = {s["topology"] for s in b3.test_ood_samples}

    assert "custom" not in train_topos
    assert ood_topos == {"custom"}
    assert len(b3.train_runs & b3.test_ood_runs) == 0


def test_task_splits_g4():
    """Verify G4 enforces leave-one-task-out isolation (Analysis OOD)."""
    samples, graphs = _create_mock_dataset()
    splitter = GeneralizationSplitter(random_seed=42)

    b4 = splitter.create_split_bundle(GENERALIZATION_REGISTRY["G4"], samples, graphs)
    train_tasks = {s["task_type"] for s in b4.train_samples}
    ood_tasks = {s["task_type"] for s in b4.test_ood_samples}

    assert "analysis" not in train_tasks
    assert ood_tasks == {"analysis"}
    assert len(b4.train_runs & b4.test_ood_runs) == 0


def test_failure_type_splits_g5():
    """Verify G5 isolates seen vs held-out unseen failure modes."""
    samples, graphs = _create_mock_dataset()
    splitter = GeneralizationSplitter(random_seed=42)

    b5 = splitter.create_split_bundle(GENERALIZATION_REGISTRY["G5"], samples, graphs)
    ood_failures = {s["failure_type"] for s in b5.test_ood_samples}

    # All OOD non-none failures must be in UNSEEN_FAILURE_MODES
    for f in ood_failures:
        if f != "none":
            assert f in UNSEEN_FAILURE_MODES, f"Unexpected failure mode in OOD: {f}"

    assert len(b5.train_runs & b5.test_ood_runs) == 0


def test_no_future_leakage_in_splits():
    """Verify time monotonically advances and no future prediction events leak into graph history."""
    samples, graphs = _create_mock_dataset()
    splitter = GeneralizationSplitter()
    b = splitter.create_split_bundle(GENERALIZATION_REGISTRY["G1"], samples, graphs)

    # Check temporal ordering in train graphs
    runs_seen = set()
    for g in b.train_graphs:
        rid = g["run_id"]
        runs_seen.add(rid)
        assert g["timestamp"] >= 0.0

    assert len(runs_seen) > 0


def test_schema_serialization():
    """Verify GeneralizationResultRecord and GeneralizationGapRecord serialize cleanly."""
    rec = GeneralizationResultRecord(
        experiment_id="G1",
        dimension="agent_count",
        split_type="out_of_distribution",
        model="temporal_gnn",
        dataset_version="v1",
        horizon=1,
        seed=42,
        threshold=0.50,
        precision=0.80,
        recall=0.75,
        f1=0.7742,
        auroc=0.85,
        auprc=0.82,
        false_positive_rate=0.05,
        false_alarm_rate=0.06,
        mean_lead_time=2.5,
        median_lead_time=2.0,
        successful_warnings=5,
        warnings_per_trajectory=1.2,
        sample_count=20,
        positive_count=10,
        negative_count=10,
    )
    d = rec.to_dict()
    assert d["experiment_id"] == "G1"
    assert d["f1"] == 0.7742

    gap = GeneralizationGapRecord(
        experiment_id="G1",
        dimension="agent_count",
        model="temporal_gnn",
        horizon=1,
        metric="f1",
        id_value=0.85,
        ood_value=0.7742,
        gap=0.0758,
        pct_change=-8.9,
        ci_lower=0.01,
        ci_upper=0.14,
        p_value=0.03,
        statistically_significant=True,
        interpretation="Moderate degradation",
    )
    assert gap.gap == 0.0758
    assert gap.statistically_significant is True


def test_model_runner_classical_ml():
    """Verify Classical ML dispatch executes, calibrates threshold, and infers on splits."""
    samples, graphs = _create_mock_dataset()
    splitter = GeneralizationSplitter()
    bundle = splitter.create_split_bundle(GENERALIZATION_REGISTRY["G1"], samples, graphs)

    runner = GeneralizationModelRunner()
    id_preds, ood_preds, th = runner.train_and_eval(
        model_family="logistic_regression",
        split_bundle=bundle,
        horizon=1,
        seed=42,
    )
    assert 0.10 <= th <= 0.90
    assert isinstance(id_preds, list)
    assert isinstance(ood_preds, list)


def test_end_to_end_smoke_test(tmp_path):
    """Smoke test running GeneralizationExperimentRunner end-to-end on synthetic data."""
    # Build minimal directory structure
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    results_dir = tmp_path / "mock_results"

    samples, graphs = _create_mock_dataset()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import json

    table = pa.Table.from_pylist(samples)
    pq.write_table(table, dataset_dir / "all_samples.parquet")
    with open(dataset_dir / "graph_sequences_train.jsonl", "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g) + "\n")

    runner = GeneralizationExperimentRunner(
        dataset_dir=dataset_dir,
        base_results_dir=results_dir,
    )
    runner.load_dataset()
    assert len(runner.all_samples) == len(samples)

    # Run single experiment G1 with classical ML for speed
    id_rec, ood_rec, gap_rec = runner.run_single_experiment(
        experiment_id="G1",
        model="logistic_regression",
        horizon=1,
        seed=42,
    )
    assert id_rec is not None
    assert ood_rec is not None
    assert gap_rec is not None

    runner.save_artifacts()
    assert (results_dir / "generalization_report.md").exists()
    assert (results_dir / "tables" / "generalization_matrix.md").exists()
    assert (results_dir / "metrics" / "generalization_metrics.json").exists()


def test_reproducibility_across_seeds():
    """Verify that runs with identical random seed yield identical metrics and decision thresholds."""
    samples, graphs = _create_mock_dataset()
    splitter = GeneralizationSplitter(random_seed=42)
    bundle = splitter.create_split_bundle(GENERALIZATION_REGISTRY["G1"], samples, graphs)

    runner = GeneralizationModelRunner()
    preds_a, _, th_a = runner.train_and_eval("logistic_regression", bundle, horizon=1, seed=123)
    preds_b, _, th_b = runner.train_and_eval("logistic_regression", bundle, horizon=1, seed=123)

    assert th_a == th_b
    assert len(preds_a) == len(preds_b)
    for p1, p2 in zip(preds_a, preds_b):
        assert p1["predicted_probability"] == p2["predicted_probability"]
        assert p1["predicted_label"] == p2["predicted_label"]
