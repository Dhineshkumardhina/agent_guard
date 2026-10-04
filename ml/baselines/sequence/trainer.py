"""Training Engine and Checkpointing for PyTorch Temporal Baselines.

Coordinates:
- Train loop with weighted BCE loss (pos_weight from training set balance)
- Validation loop with early stopping
- Validation-based threshold tuning (freezing threshold before test evaluation)
- Checkpoint persistence with complete metadata
- Machine-readable training_history.csv generation
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import csv
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ml.baselines.rule_based.evaluator import compute_classification_metrics
from ml.baselines.sequence.models import get_device


class SequenceTrainer:
    """Trains, tunes, and checkpoints temporal sequence models (LSTM / GRU)."""

    def __init__(
        self,
        model: Any,
        lr: float = 0.001,
        patience: int = 5,
        device: Optional[Any] = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize trainer."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for SequenceTrainer.")

        self.model = model
        self.lr = lr
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
            train_loader: DataLoader for training sequence dataset.
            val_loader: DataLoader for validation sequence dataset.
            epochs: Maximum training epochs.
            pos_weight: Optional scalar weight for positive class imbalance.
            checkpoint_dir: Directory to save model checkpoints and training history.
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # ── 1. Training Phase ──
            self.model.train()
            train_losses = []
            for batch_seqs, batch_labels, _ in train_loader:
                batch_seqs = batch_seqs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_seqs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # ── 2. Validation Phase ──
            val_loss, val_metrics, val_probs, val_trues = self._evaluate_loader(val_loader, criterion)

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
                "model_name": getattr(self.model, "model_name", "sequence_model"),
                "architecture_config": {
                    "input_size": getattr(self.model, "input_size", 28),
                    "hidden_size": getattr(self.model, "hidden_size", 64),
                    "num_layers": getattr(self.model, "num_layers", 2),
                    "dropout": getattr(self.model, "dropout_rate", 0.2),
                    "bidirectional": getattr(self.model, "bidirectional", False),
                },
                "training_config": {
                    "lr": self.lr,
                    "epochs_trained": len(history),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "pos_weight": pos_weight,
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
            for batch_seqs, batch_labels, _ in loader:
                batch_seqs = batch_seqs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_seqs)
                loss = criterion(logits, batch_labels)
                losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs if isinstance(probs, list) else [probs])
                all_trues.extend(batch_labels.cpu().numpy().astype(int).tolist())

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
            for batch_seqs, batch_labels, _ in val_loader:
                batch_seqs = batch_seqs.to(self.device)
                logits = self.model(batch_seqs)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs if isinstance(probs, list) else [probs])
                all_trues.extend(batch_labels.numpy().astype(int).tolist())

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
