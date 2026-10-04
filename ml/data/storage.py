"""Storage and Serialization Engine for Scientific Datasets.

Persists multi-agent prediction datasets in research-friendly formats:
- Parquet for tabular samples, features, metadata, and labels
- Structured JSONL for temporal graph sequences and snapshot history
- Standardized manifest.json for dataset versioning and provenance tracking
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import json
import os

from ml.data.schema import PredictionSample, DatasetManifest

# Check pyarrow availability
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class DatasetStorage:
    """Manages disk persistence and loading for AgentGuard scientific datasets."""

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base_dir = Path(base_dir)

    def save_dataset(
        self,
        samples_by_split: Dict[str, List[PredictionSample]],
        manifest: DatasetManifest,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Persist complete dataset: splits, tabular parquets, graph sequences, and manifest.
        
        Args:
            samples_by_split: Dictionary mapping "train", "val", "test" to lists of samples.
            manifest: Dataset manifest instance.
            output_dir: Optional override directory.
            
        Returns:
            Path to the saved dataset directory.
        """
        target_dir = Path(output_dir) if output_dir else (self.base_dir / manifest.dataset_version)
        target_dir.mkdir(parents=True, exist_ok=True)

        all_samples: List[PredictionSample] = []
        for split_name in ("train", "val", "test"):
            split_samples = samples_by_split.get(split_name, [])
            all_samples.extend(split_samples)

            # 1. Save split tabular parquet
            tabular_rows = [s.to_tabular_dict() for s in split_samples]
            self._save_tabular(tabular_rows, target_dir / f"{split_name}.parquet")

            # 2. Save split graph sequences
            graph_seq_path = target_dir / f"graph_sequences_{split_name}.jsonl"
            self._save_graph_sequences(split_samples, graph_seq_path)

        # 3. Save combined all_samples.parquet
        all_tabular_rows = [s.to_tabular_dict() for s in all_samples]
        self._save_tabular(all_tabular_rows, target_dir / "all_samples.parquet")

        # 4. Save manifest.json
        manifest_path = target_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json(indent=2))

        return target_dir

    def _save_tabular(self, rows: List[Dict[str, Any]], filepath: Path) -> None:
        """Save flat rows to Parquet file, falling back to JSONL if pyarrow is unavailable."""
        if not rows:
            # Create empty file with appropriate format
            if HAS_PYARROW:
                schema = pa.schema([("sample_id", pa.string())])
                empty_table = pa.Table.from_arrays([pa.array([])], schema=schema)
                pq.write_table(empty_table, filepath)
            else:
                with open(filepath.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
                    pass
            return

        if HAS_PYARROW:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, filepath)
        else:
            fallback_path = filepath.with_suffix(".jsonl")
            with open(fallback_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def _save_graph_sequences(self, samples: List[PredictionSample], filepath: Path) -> None:
        """Save temporal graph sequences in structured JSONL format."""
        with open(filepath, "w", encoding="utf-8") as f:
            for s in samples:
                payload = {
                    "sample_id": s.sample_id,
                    "run_id": s.run_id,
                    "step_idx": s.step_idx,
                    "timestamp": s.timestamp,
                    "prediction_horizon": s.prediction_horizon,
                    "temporal_graph_history": s.temporal_graph_history,
                }
                f.write(json.dumps(payload) + "\n")

    def load_manifest(self, dataset_dir: Union[str, Path]) -> DatasetManifest:
        """Load manifest.json from a dataset directory."""
        path = Path(dataset_dir) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Dataset manifest not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return DatasetManifest.from_json(f.read())

    def load_tabular(self, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load tabular dataset partition into list of dictionaries."""
        path = Path(filepath)
        if not path.exists():
            jsonl_fallback = path.with_suffix(".jsonl")
            if jsonl_fallback.exists():
                with open(jsonl_fallback, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]
            raise FileNotFoundError(f"File not found: {path}")

        if HAS_PYARROW and path.suffix == ".parquet":
            table = pq.read_table(path)
            return table.to_pylist()
        else:
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

    def load_graph_sequences(self, filepath: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
        """Load graph sequences mapping sample_id -> temporal graph history list."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Graph sequences not found at {path}")
        sequences: Dict[str, List[Dict[str, Any]]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    sequences[item["sample_id"]] = item.get("temporal_graph_history", [])
        return sequences
