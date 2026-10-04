"""End-to-End Dataset Generation and Orchestration Pipeline.

Executes the complete research pipeline:
Simulation
  -> Telemetry
  -> Fault Injection
  -> Temporal Graph
  -> Prediction Points
  -> Labels
  -> Dataset Splits
  -> Validation
  -> Saved Dataset
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import random
from datetime import datetime, timezone

from ml.simulation.run import SimulationRun
from ml.simulation.fault_injection.injector import FaultInjector
from ml.data.schema import PredictionSample, DatasetManifest
from ml.data.prediction_samples import PredictionSampleGenerator
from ml.data.split import split_trajectories, split_samples
from ml.data.storage import DatasetStorage
from ml.data.validation import DatasetValidator, ValidationResult


class DatasetBuilder:
    """Orchestrates multi-agent simulation trajectories, causal labeling, and dataset persistence."""

    def __init__(
        self,
        num_runs: int = 50,
        random_seed: int = 42,
        task_types: Optional[List[str]] = None,
        topologies: Optional[List[str]] = None,
        agent_counts: Optional[List[int]] = None,
        fault_probability: float = 0.40,
        fault_types: Optional[List[str]] = None,
        prediction_horizons: Optional[List[int]] = None,
        dataset_version: str = "agentguard_dataset_v1",
        output_dir: Optional[Union[str, Path]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        self.num_runs = num_runs
        self.random_seed = random_seed
        self.task_types = task_types or ["research", "coding", "analysis", "planning"]
        self.topologies = topologies or ["pipeline", "star", "mesh", "custom"]
        self.agent_counts = agent_counts or [3, 5, 8, 12]
        self.fault_probability = fault_probability
        self.fault_types = fault_types or [
            "hallucinated_output",
            "incorrect_information",
            "tool_failure",
            "tool_timeout",
            "delayed_response",
            "contradictory_output",
        ]
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 20]
        self.dataset_version = dataset_version
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed") / dataset_version
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.sample_generator = PredictionSampleGenerator(
            prediction_horizons=self.prediction_horizons,
            dataset_version=self.dataset_version,
        )
        self.storage = DatasetStorage(self.output_dir.parent)
        self.validator = DatasetValidator(strict_mode=True)

    def generate_runs(self) -> List[SimulationRun]:
        """Execute deterministic multi-agent simulation trajectories."""
        rng = random.Random(self.random_seed)
        runs: List[SimulationRun] = []

        for idx in range(self.num_runs):
            run_seed = self.random_seed + (idx * 1009)
            run_rng = random.Random(run_seed)

            task = self.task_types[idx % len(self.task_types)]
            topology = self.topologies[idx % len(self.topologies)]
            num_agents = self.agent_counts[idx % len(self.agent_counts)]

            # Determine whether fault is injected in this run
            should_inject = run_rng.random() < self.fault_probability
            fault_injector: Optional[FaultInjector] = None

            if should_inject:
                fault_type = self.fault_types[run_rng.randint(0, len(self.fault_types) - 1)]
                fault_injector = FaultInjector(
                    fault_type=fault_type,
                    probability=1.0,
                    injection_step=run_rng.randint(0, 2),
                    severity=round(run_rng.uniform(0.6, 0.95), 2),
                    random_seed=run_seed,
                )

            run = SimulationRun(
                run_id=f"run_{idx:04d}_{self.dataset_version}",
                task_type=task,
                topology=topology,
                num_agents=num_agents,
                random_seed=run_seed,
                fault_injector=fault_injector,
            )
            run.execute()
            runs.append(run)

        return runs

    def build_dataset(self) -> tuple[Path, DatasetManifest, ValidationResult]:
        """Execute end-to-end dataset generation, splitting, validation, and storage.
        
        Returns:
            Tuple of:
                - target_directory: Path where dataset artifacts are saved
                - manifest: Complete DatasetManifest metadata object
                - validation_result: Diagnostic report confirming zero data leakage
        """
        # Step 1: Simulate multi-agent runs
        runs = self.generate_runs()

        # Step 2: Extract prediction samples
        all_samples = self.sample_generator.generate_from_runs(runs)

        # Step 3: Run-level trajectory splitting (zero temporal/run leakage)
        run_ids = [r.run_id for r in runs]
        run_splits = split_trajectories(
            run_ids=run_ids,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
        )
        samples_by_split = split_samples(all_samples, run_splits)

        # Step 4: Validate in-memory samples
        validation_result = self.validator.validate_samples(all_samples)
        if not validation_result.is_valid:
            error_msg = "\n".join(validation_result.errors)
            raise ValueError(f"Dataset validation failed with {len(validation_result.errors)} errors:\n{error_msg}")

        # Step 5: Construct Manifest
        stats = validation_result.statistics
        manifest = DatasetManifest(
            dataset_version=self.dataset_version,
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            simulator_version="0.6.0",
            configuration={
                "num_runs": self.num_runs,
                "random_seed": self.random_seed,
                "fault_probability": self.fault_probability,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
            },
            random_seed=self.random_seed,
            number_of_runs=len(runs),
            number_of_samples=len(all_samples),
            prediction_horizons=self.prediction_horizons,
            topologies=self.topologies,
            tasks=self.task_types,
            agent_counts=self.agent_counts,
            class_distribution=stats,
            split_distribution={
                "train_runs": len(run_splits["train"]),
                "val_runs": len(run_splits["val"]),
                "test_runs": len(run_splits["test"]),
                "train_samples": len(samples_by_split["train"]),
                "val_samples": len(samples_by_split["val"]),
                "test_samples": len(samples_by_split["test"]),
            },
            feature_schema={
                "agent_level_features": list(all_samples[0].agent_level_features.keys()) if all_samples else [],
                "prediction_horizons": self.prediction_horizons,
            },
            storage_format="parquet",
            description="Controlled experimental multi-agent dataset for cascading failure prediction",
        )

        # Step 6: Persist dataset artifacts
        saved_path = self.storage.save_dataset(
            samples_by_split=samples_by_split,
            manifest=manifest,
            output_dir=self.output_dir,
        )

        # Step 7: Verify persisted files on disk
        disk_validation = self.validator.validate_dataset_directory(saved_path)
        if not disk_validation.is_valid:
            error_msg = "\n".join(disk_validation.errors)
            raise ValueError(f"Persisted dataset validation failed:\n{error_msg}")

        return saved_path, manifest, validation_result
