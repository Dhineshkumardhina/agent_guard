"""Training Engine and Checkpointing for Static Graph Neural Networks - Phase 10.

Coordinates:
- Graph DataLoader iteration across mini-batches of graph snapshots
- Train loop with weighted BCE loss (pos_weight from training set balance)
- Validation loop with early stopping
- Validation-based threshold tuning (freezing optimal threshold before test evaluation)
- Checkpoint persistence with complete metadata and reproducibility details
- Machine-readable training_history.csv generation
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import csv
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch_geometric
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    DataLoader = object

from ml.baselines.rule_based.evaluator import compute_classification_metrics
from ml.baselines.static_gnn.models import get_device


class StaticGNNTrainer:
    """Trains, validates, tunes thresholds, and checkpoints Static GNN models."""

    def __init__(
        self,
        model: Any,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 5,
        device: Optional[Any] = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize Static GNN trainer."""
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required for StaticGNNTrainer.")

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_seed = random_seed
        self.device = device or get_device(verbose=False)
        self.model.to(self.device)

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int = 25,
        pos_weight: Optional[float] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train model with early stopping on validation loss and threshold calibration.
        
        Args:
            train_loader: PyG DataLoader for training graph snapshots.
            val_loader: PyG DataLoader for validation graph snapshots.
            epochs: Maximum training epochs.
            pos_weight: Optional scalar weight for positive class imbalance (N_neg / N_pos).
            checkpoint_dir: Directory to persist model checkpoints and training history.
            checkpoint_metadata: Additional experiment metadata to embed in checkpoint.
            
        Returns:
            Dictionary with training history, best checkpoint path, and frozen threshold.
        """
        # Loss function with positive class weighting
        if pos_weight is not None and pos_weight > 0.0:
            pw_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # ── 1. Training Phase ──
            self.model.train()
            train_losses = []
            for batch in train_loader:
                batch = batch.to(self.device)
                edge_attr = getattr(batch, "edge_attr", None)
                edge_weight = getattr(batch, "edge_weight", None)

                optimizer.zero_grad()
                logits = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    edge_attr=edge_attr,
                    edge_weight=edge_weight,
                )
                labels = batch.y.squeeze() if batch.y.dim() > 1 else batch.y
                # Ensure labels match logits shape
                if labels.shape != logits.shape:
                    labels = labels.view_as(logits)

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # ── 2. Validation Phase ──
            val_loss, val_metrics, _, _ = self._evaluate_loader(val_loader, criterion)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_f1": round(val_metrics.get("f1", 0.0), 4),
                "val_auroc": round(val_metrics.get("auroc", 0.0), 4),
                "val_auprc": round(val_metrics.get("auprc", 0.0), 4),
            })

            # Check early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Load best state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # ── 3. Threshold Calibration on Validation Split ONLY ──
        # Freeze optimal threshold before touching test set
        frozen_threshold = self._select_optimal_threshold(val_loader)

        # ── 4. Save Checkpoint & Training History ──
        ckpt_path = None
        if checkpoint_dir:
            c_dir = Path(checkpoint_dir)
            c_dir.mkdir(parents=True, exist_ok=True)

            # Save training_history.csv
            csv_path = c_dir / "training_history.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_f1", "val_auroc", "val_auprc"])
                writer.writeheader()
                writer.writerows(history)

            # Save model checkpoint
            ckpt_path = c_dir / "checkpoint.pt"
            ckpt_payload = {
                "model_state_dict": self.model.state_dict(),
                "model_name": getattr(self.model, "model_name", "static_gnn"),
                "architecture_config": getattr(self.model, "config", {}),
                "training_config": {
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "epochs_trained": len(history),
                    "best_epoch": best_epoch,
                    "best_val_loss": round(float(best_val_loss), 4),
                    "pos_weight": float(pos_weight) if pos_weight is not None else None,
                    "random_seed": self.random_seed,
                },
                "selected_threshold": frozen_threshold,
                "metadata": checkpoint_metadata or {},
            }
            torch.save(ckpt_payload, ckpt_path)

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "frozen_threshold": frozen_threshold,
            "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        }

    def _evaluate_loader(self, loader: Any, criterion: Any) -> Tuple[float, Dict[str, Any], List[float], List[int]]:
        """Evaluate loss and classification metrics on a DataLoader."""
        self.model.eval()
        losses = []
        all_probs = []
        all_trues = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                edge_attr = getattr(batch, "edge_attr", None)
                edge_weight = getattr(batch, "edge_weight", None)

                logits = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    edge_attr=edge_attr,
                    edge_weight=edge_weight,
                )
                labels = batch.y.squeeze() if batch.y.dim() > 1 else batch.y
                if labels.shape != logits.shape:
                    labels = labels.view_as(logits)

                loss = criterion(logits, labels)
                losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs if isinstance(probs, list) else [probs])
                trues = labels.cpu().numpy().astype(int).tolist()
                all_trues.extend(trues if isinstance(trues, list) else [trues])

        mean_loss = float(np.mean(losses)) if losses else 0.0
        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        metrics = compute_classification_metrics(all_trues, preds, all_probs) if all_trues else {}
        return mean_loss, metrics, all_probs, all_trues

    def _select_optimal_threshold(self, val_loader: Any) -> float:
        """Scan candidate thresholds on validation set to find threshold maximizing F1."""
        self.model.eval()
        all_probs = []
        all_trues = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                edge_attr = getattr(batch, "edge_attr", None)
                edge_weight = getattr(batch, "edge_weight", None)

                logits = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    edge_attr=edge_attr,
                    edge_weight=edge_weight,
                )
                labels = batch.y.squeeze() if batch.y.dim() > 1 else batch.y

                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs if isinstance(probs, list) else [probs])
                trues = labels.cpu().numpy().astype(int).tolist()
                all_trues.extend(trues if isinstance(trues, list) else [trues])

        if not all_trues or sum(all_trues) == 0:
            return 0.50

        best_tau = 0.50
        best_f1 = -1.0

        for tau in np.linspace(0.10, 0.90, 17):
            preds = [1 if p >= tau else 0 for p in all_probs]
            m = compute_classification_metrics(all_trues, preds)
            f1 = m.get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)

        return round(best_tau, 2)
