"""Training Engine and Checkpointing for Temporal Graph Networks - Phase 11.

Executes chronological, trajectory-aware training:
- Resets node memory at the start of each simulation trajectory
- Processes interactions in strict temporal order
- Computes weighted BCE loss (pos_weight from training set balance)
- Validation loop with early stopping
- Validation-based threshold tuning (freezing optimal threshold before test evaluation)
- Checkpoint persistence and machine-readable training_history.csv generation
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import csv
import numpy as np
import torch
import torch.nn as nn

from ml.baselines.rule_based.evaluator import compute_classification_metrics
from ml.baselines.temporal_gnn.dataset import TemporalRunTrajectory
from ml.baselines.temporal_gnn.models import TemporalGraphFailurePredictor, get_device


class TemporalGNNTrainer:
    """Trains, tunes, and checkpoints Temporal Graph Neural Networks."""

    def __init__(
        self,
        model: TemporalGraphFailurePredictor,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 5,
        device: Optional[torch.device] = None,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_seed = random_seed
        self.device = device or get_device(verbose=False)
        self.model.to(self.device)
        self.model.device = self.device

    def train(
        self,
        train_trajectories: List[TemporalRunTrajectory],
        val_trajectories: List[TemporalRunTrajectory],
        epochs: int = 20,
        pos_weight: Optional[float] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train model with early stopping on validation loss and threshold calibration."""
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
            train_losses: List[float] = []

            for traj in train_trajectories:
                # Reset memory between independent trajectories
                self.model.reset_memory()

                inter_idx = 0
                total_inters = len(traj.interactions)

                for pp in traj.prediction_points:
                    target_t = pp.timestamp

                    # Advance interactions up to target_t
                    while inter_idx < total_inters and traj.interactions[inter_idx].timestamp <= target_t:
                        ev = traj.interactions[inter_idx]
                        e_feat = torch.tensor(ev.features, dtype=torch.float32, device=self.device)
                        self.model.process_interaction(
                            source_agent=ev.source_agent,
                            target_agent=ev.target_agent,
                            timestamp=ev.timestamp,
                            edge_features=e_feat,
                        )
                        inter_idx += 1

                    # Compute prediction at target_t
                    optimizer.zero_grad()
                    logit, _ = self.model.predict_at_timestamp(
                        agent_ids=pp.active_agents,
                        node_features_dict=pp.node_features,
                        current_timestamp=target_t,
                    )
                    target_label = torch.tensor([pp.label], dtype=torch.float32, device=self.device)
                    loss = criterion(logit.unsqueeze(0) if logit.dim() == 0 else logit, target_label)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # ── 2. Validation Phase ──
            val_loss, val_metrics, _, _ = self._evaluate_trajectories(val_trajectories, criterion)

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

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # ── 3. Threshold Calibration on Validation Split ONLY ──
        frozen_threshold = self._select_optimal_threshold(val_trajectories)

        # ── 4. Save Checkpoint & Training History ──
        ckpt_path = None
        if checkpoint_dir:
            c_dir = Path(checkpoint_dir)
            c_dir.mkdir(parents=True, exist_ok=True)

            csv_path = c_dir / "training_history.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_f1", "val_auroc", "val_auprc"])
                writer.writeheader()
                writer.writerows(history)

            ckpt_path = c_dir / "checkpoint.pt"
            ckpt_meta = {
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
                **(checkpoint_metadata or {}),
            }
            self.model.save_checkpoint(ckpt_path, optimizer=optimizer, metadata=ckpt_meta)

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "frozen_threshold": frozen_threshold,
            "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        }

    def _evaluate_trajectories(
        self,
        trajectories: List[TemporalRunTrajectory],
        criterion: nn.Module,
    ) -> Tuple[float, Dict[str, Any], List[float], List[int]]:
        """Evaluate loss and metrics over a list of trajectories."""
        self.model.eval()
        losses: List[float] = []
        all_probs: List[float] = []
        all_trues: List[int] = []

        with torch.no_grad():
            for traj in trajectories:
                self.model.reset_memory()
                inter_idx = 0
                total_inters = len(traj.interactions)

                for pp in traj.prediction_points:
                    target_t = pp.timestamp
                    while inter_idx < total_inters and traj.interactions[inter_idx].timestamp <= target_t:
                        ev = traj.interactions[inter_idx]
                        e_feat = torch.tensor(ev.features, dtype=torch.float32, device=self.device)
                        self.model.process_interaction(
                            source_agent=ev.source_agent,
                            target_agent=ev.target_agent,
                            timestamp=ev.timestamp,
                            edge_features=e_feat,
                        )
                        inter_idx += 1

                    logit, prob = self.model.predict_at_timestamp(
                        agent_ids=pp.active_agents,
                        node_features_dict=pp.node_features,
                        current_timestamp=target_t,
                    )
                    target_label = torch.tensor([pp.label], dtype=torch.float32, device=self.device)
                    loss = criterion(logit.unsqueeze(0) if logit.dim() == 0 else logit, target_label)
                    losses.append(loss.item())
                    all_probs.append(prob)
                    all_trues.append(int(pp.label))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        preds = [1 if p >= 0.50 else 0 for p in all_probs]
        metrics = compute_classification_metrics(all_trues, preds, all_probs) if all_trues else {}
        return mean_loss, metrics, all_probs, all_trues

    def _select_optimal_threshold(self, val_trajectories: List[TemporalRunTrajectory]) -> float:
        """Scan candidate thresholds on validation set to find threshold maximizing F1."""
        self.model.eval()
        all_probs: List[float] = []
        all_trues: List[int] = []

        with torch.no_grad():
            for traj in val_trajectories:
                self.model.reset_memory()
                inter_idx = 0
                total_inters = len(traj.interactions)

                for pp in traj.prediction_points:
                    target_t = pp.timestamp
                    while inter_idx < total_inters and traj.interactions[inter_idx].timestamp <= target_t:
                        ev = traj.interactions[inter_idx]
                        e_feat = torch.tensor(ev.features, dtype=torch.float32, device=self.device)
                        self.model.process_interaction(
                            source_agent=ev.source_agent,
                            target_agent=ev.target_agent,
                            timestamp=ev.timestamp,
                            edge_features=e_feat,
                        )
                        inter_idx += 1

                    _, prob = self.model.predict_at_timestamp(
                        agent_ids=pp.active_agents,
                        node_features_dict=pp.node_features,
                        current_timestamp=target_t,
                    )
                    all_probs.append(prob)
                    all_trues.append(int(pp.label))

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
