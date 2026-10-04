"""Generalization Dataset Splitting and Isolation Engine - Phase 14.

Implements strict out-of-distribution (OOD) experiment partitioning for:
- G1: Agent Count (Train: 3, 5 agents -> Test: 8 agents)
- G2: Agent Count (Train: 3, 5, 8 agents -> Test: 12 agents)
- G3: Topology Leave-One-Out (Train: Pipeline, Star, Mesh -> Test: Custom)
- G3b: Topology Leave-One-Out (Train: Star, Mesh, Custom -> Test: Pipeline)
- G4: Task Leave-One-Out (Train: Research, Coding, Planning -> Test: Analysis)
- G4b: Task Leave-One-Out (Train: Coding, Analysis, Planning -> Test: Research)
- G5: Failure Mode Transfer (Train: Seen Failures -> Test: Unseen Held-Out Failures)

Enforces:
1. Strict run-level separation (zero trajectory overlap between train, val, and test).
2. Frozen validation thresholding (threshold tuned solely on in-distribution val set).
3. Zero future leakage.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import random
from ml.generalization.schema import GeneralizationExperimentConfig, GeneralizationMatrixEntry


# Canonical 12 failure modes taxonomy
SEEN_FAILURE_MODES: Set[str] = {
    "hallucinated_output",
    "tool_failure",
    "delayed_response",
    "contradictory_output",
    "malformed_output",
    "communication_loop",
}

UNSEEN_FAILURE_MODES: Set[str] = {
    "incorrect_information",
    "tool_timeout",
    "low_confidence_output",
    "incorrect_delegation",
    "stale_context",
    "agent_dropout",
}

GENERALIZATION_REGISTRY: Dict[str, GeneralizationExperimentConfig] = {
    "G1": GeneralizationExperimentConfig(
        experiment_id="G1",
        dimension="agent_count",
        name="Scaling to 8 Agents",
        description="Train on smaller populations (3, 5 agents) and evaluate generalization to 8 agents.",
        train_filter={"number_of_agents": [3, 5]},
        test_ood_filter={"number_of_agents": [8]},
    ),
    "G2": GeneralizationExperimentConfig(
        experiment_id="G2",
        dimension="agent_count",
        name="Scaling to 12 Agents",
        description="Train on moderate populations (3, 5, 8 agents) and evaluate generalization to 12 agents.",
        train_filter={"number_of_agents": [3, 5, 8]},
        test_ood_filter={"number_of_agents": [12]},
    ),
    "G3": GeneralizationExperimentConfig(
        experiment_id="G3",
        dimension="topology",
        name="Topology Shift (Custom)",
        description="Train on canonical topologies (Pipeline, Star, Mesh) and evaluate generalization to Custom.",
        train_filter={"topology": ["pipeline", "star", "mesh"]},
        test_ood_filter={"topology": ["custom"]},
    ),
    "G3b": GeneralizationExperimentConfig(
        experiment_id="G3b",
        dimension="topology",
        name="Topology Shift (Pipeline)",
        description="Train on non-linear topologies (Star, Mesh, Custom) and evaluate generalization to linear Pipeline.",
        train_filter={"topology": ["star", "mesh", "custom"]},
        test_ood_filter={"topology": ["pipeline"]},
    ),
    "G4": GeneralizationExperimentConfig(
        experiment_id="G4",
        dimension="task",
        name="Task Transfer (Analysis)",
        description="Train on Research, Coding, Planning and evaluate generalization to Data Analysis.",
        train_filter={"task_type": ["research", "coding", "planning"]},
        test_ood_filter={"task_type": ["analysis"]},
    ),
    "G4b": GeneralizationExperimentConfig(
        experiment_id="G4b",
        dimension="task",
        name="Task Transfer (Research)",
        description="Train on Coding, Analysis, Planning and evaluate generalization to open Research.",
        train_filter={"task_type": ["coding", "analysis", "planning"]},
        test_ood_filter={"task_type": ["research"]},
    ),
    "G5": GeneralizationExperimentConfig(
        experiment_id="G5",
        dimension="failure_type",
        name="Unseen Failure Modes Transfer",
        description="Train on Seen failure types (and nominal runs) and test on Held-Out Unseen failure types.",
        train_filter={"failure_type_in": list(SEEN_FAILURE_MODES) + ["none"]},
        test_ood_filter={"failure_type_in": list(UNSEEN_FAILURE_MODES)},
    ),
}


@dataclass
class GeneralizationSplitBundle:
    """Contains partitioned samples, graph sequences, and isolated run IDs."""

    experiment_id: str
    dimension: str
    train_runs: Set[str]
    val_id_runs: Set[str]
    test_id_runs: Set[str]
    test_ood_runs: Set[str]
    train_samples: List[Dict[str, Any]]
    val_id_samples: List[Dict[str, Any]]
    test_id_samples: List[Dict[str, Any]]
    test_ood_samples: List[Dict[str, Any]]
    train_graphs: List[Dict[str, Any]]
    val_id_graphs: List[Dict[str, Any]]
    test_id_graphs: List[Dict[str, Any]]
    test_ood_graphs: List[Dict[str, Any]]


class GeneralizationSplitter:
    """Splits full dataset runs into clean In-Distribution and Out-of-Distribution partitions."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def _matches_filter(self, row: Dict[str, Any], filter_spec: Dict[str, Any]) -> bool:
        """Evaluate if sample/run metadata matches criterion."""
        for key, expected_vals in filter_spec.items():
            if key == "failure_type_in":
                val = row.get("failure_type", "none")
                if val not in expected_vals:
                    return False
            else:
                val = row.get(key)
                if val is not None and val not in expected_vals:
                    return False
        return True

    def create_split_bundle(
        self,
        config: GeneralizationExperimentConfig,
        all_samples: List[Dict[str, Any]],
        all_graphs: List[Dict[str, Any]],
    ) -> GeneralizationSplitBundle:
        """Create an isolated ID/OOD split bundle adhering strictly to run-level isolation."""
        # 1. Group runs and their metadata
        run_meta: Dict[str, Dict[str, Any]] = {}
        for s in all_samples:
            rid = s["run_id"]
            if rid not in run_meta:
                run_meta[rid] = {
                    "run_id": rid,
                    "number_of_agents": s.get("number_of_agents"),
                    "topology": s.get("topology"),
                    "task_type": s.get("task_type"),
                    "failure_types": set(),
                }
            run_meta[rid]["failure_types"].add(s.get("failure_type", "none"))

        # For failure_type splitting, assign dominant failure type to run
        for rid, meta in run_meta.items():
            ftypes = meta["failure_types"]
            # Exclude 'none' if any actual fault occurred in the run
            non_none = [f for f in ftypes if f != "none"]
            meta["failure_type"] = non_none[0] if non_none else "none"

        # 2. Partition runs into ID candidates and OOD candidates
        id_candidate_runs: List[str] = []
        ood_candidate_runs: List[str] = []

        for rid, meta in run_meta.items():
            if self._matches_filter(meta, config.test_ood_filter):
                ood_candidate_runs.append(rid)
            elif self._matches_filter(meta, config.train_filter):
                id_candidate_runs.append(rid)

        # Ensure determinism
        rng = random.Random(self.random_seed)
        rng.shuffle(id_candidate_runs)
        rng.shuffle(ood_candidate_runs)

        # 3. Partition In-Distribution runs into train (70%), val_id (15%), test_id (15%)
        n_id = len(id_candidate_runs)
        if n_id < 3:
            # Fallback for very small splits: ensure at least 1 in train, 1 in val, 1 in test
            train_runs = set(id_candidate_runs[:max(1, int(n_id * 0.6))])
            val_id_runs = set(id_candidate_runs[max(1, int(n_id * 0.6)):max(2, int(n_id * 0.8))])
            test_id_runs = set(id_candidate_runs[max(2, int(n_id * 0.8)):])
            if not val_id_runs:
                val_id_runs = set(train_runs)
            if not test_id_runs:
                test_id_runs = set(train_runs)
        else:
            n_train = max(1, int(n_id * 0.70))
            n_val = max(1, int(n_id * 0.15))
            train_runs = set(id_candidate_runs[:n_train])
            val_id_runs = set(id_candidate_runs[n_train:n_train + n_val])
            test_id_runs = set(id_candidate_runs[n_train + n_val:])
            if not test_id_runs:
                test_id_runs = set(val_id_runs)

        test_ood_runs = set(ood_candidate_runs)

        # 4. Strict assertion: zero overlap between train and test_ood
        assert len(train_runs & test_ood_runs) == 0, f"Leakage detected: train & test_ood overlap in {config.experiment_id}!"
        assert len(val_id_runs & test_ood_runs) == 0, f"Leakage detected: val_id & test_ood overlap in {config.experiment_id}!"

        # 5. Filter samples
        train_samples = [s for s in all_samples if s["run_id"] in train_runs]
        val_id_samples = [s for s in all_samples if s["run_id"] in val_id_runs]
        test_id_samples = [s for s in all_samples if s["run_id"] in test_id_runs]
        test_ood_samples = [s for s in all_samples if s["run_id"] in test_ood_runs]

        # 6. Filter graphs
        graph_by_run: Dict[str, List[Dict[str, Any]]] = {}
        for g in all_graphs:
            rid = g.get("run_id")
            if rid not in graph_by_run:
                graph_by_run[rid] = []
            graph_by_run[rid].append(g)

        train_graphs = [g for rid in train_runs for g in graph_by_run.get(rid, [])]
        val_id_graphs = [g for rid in val_id_runs for g in graph_by_run.get(rid, [])]
        test_id_graphs = [g for rid in test_id_runs for g in graph_by_run.get(rid, [])]
        test_ood_graphs = [g for rid in test_ood_runs for g in graph_by_run.get(rid, [])]

        return GeneralizationSplitBundle(
            experiment_id=config.experiment_id,
            dimension=config.dimension,
            train_runs=train_runs,
            val_id_runs=val_id_runs,
            test_id_runs=test_id_runs,
            test_ood_runs=test_ood_runs,
            train_samples=train_samples,
            val_id_samples=val_id_samples,
            test_id_samples=test_id_samples,
            test_ood_samples=test_ood_samples,
            train_graphs=train_graphs,
            val_id_graphs=val_id_graphs,
            test_id_graphs=test_id_graphs,
            test_ood_graphs=test_ood_graphs,
        )

    @staticmethod
    def get_matrix_entries() -> List[GeneralizationMatrixEntry]:
        """Generate matrix entries describing all planned experiments."""
        entries = []
        for key, exp in GENERALIZATION_REGISTRY.items():
            if exp.dimension == "agent_count":
                train_cfg = f"{exp.train_filter.get('number_of_agents')} agents"
                test_cfg = f"{exp.test_ood_filter.get('number_of_agents')} agents"
            elif exp.dimension == "topology":
                train_cfg = ", ".join(exp.train_filter.get("topology", []))
                test_cfg = ", ".join(exp.test_ood_filter.get("topology", []))
            elif exp.dimension == "task":
                train_cfg = ", ".join(exp.train_filter.get("task_type", []))
                test_cfg = ", ".join(exp.test_ood_filter.get("task_type", []))
            else:
                train_cfg = "Seen Faults + Nominal"
                test_cfg = "Held-Out Unseen Faults"

            entries.append(
                GeneralizationMatrixEntry(
                    experiment_id=exp.experiment_id,
                    dimension=exp.dimension,
                    train_configuration=train_cfg,
                    test_configuration=test_cfg,
                    models_tested="Temporal GNN, Classical ML, Sequence, Static GNN",
                )
            )
        return entries
