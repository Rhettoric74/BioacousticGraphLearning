from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .gnn import GNNModel
from .losses import asymmetric_loss, soft_multilabel_cross_entropy, chordal_loss


class GNNTrainer:
    """Full-graph GNN training with loss restricted to selected node indices.

    Each batch supplies raw ``node_features`` [N, feature_dim], ``edge_index``
    [2, E], ``indices`` [B], and ``species_multilabel`` [B, C]. Edges can instead
    be stored in ``model.edge_index``. Coordinates/audio/negative_mask normally
    have B rows too. Full-graph targets are also accepted when N != B; when
    N == B, set batch['targets_are_global']=True if targets are in original
    graph order rather than the order of indices.

    Species-only training is the default. Positive auxiliary weights enable
    separate modality-masked passes; audio reconstruction requires audio_proj.
    ``node_batch_size`` optionally chunks the supervised indices into optimizer
    steps, but each step still forwards the full graph. It is NOT neighborhood
    sampling and does not reduce the graph's activation memory.
    """

    def __init__(self, model: GNNModel, optimizer: torch.optim.Optimizer,
                 device: str = "cuda", weights: tuple = (1., 0., 0.),
                 validation_loader=None, checkpoint_dir: str | Path | None = None,
                 top_k: int = 3, validation_context_length: int = 1,
                 species_location_mask_prob: float = 0.0,
                 checkpoint_metric: str = "macro_auroc",
                 species_loss: str = "asymmetric", node_batch_size: int | None = None):
        if len(weights) != 3 or any(not math.isfinite(float(w)) or w < 0 for w in weights) or not any(weights):
            raise ValueError("weights must contain three finite, nonnegative values, with at least one positive")
        if weights[2] and model.audio_head is None:
            raise ValueError("A positive audio loss weight requires GNNModel(audio_proj=True)")
        if species_loss not in ("asymmetric", "cross_entropy"):
            raise ValueError("species_loss must be asymmetric or cross_entropy")
        if not 0 <= species_location_mask_prob <= 1:
            raise ValueError("species_location_mask_prob must be in [0, 1]")
        if top_k < 0 or (node_batch_size is not None and node_batch_size < 1):
            raise ValueError("top_k must be nonnegative and node_batch_size must be positive")
        self.model, self.optimizer = model.to(device), optimizer
        self.device, self.weights = device, tuple(float(w) for w in weights)
        self.validation_loader = validation_loader
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.top_k, self.validation_context_length = top_k, validation_context_length
        self.species_location_mask_prob = species_location_mask_prob
        self.checkpoint_metric, self.species_loss = checkpoint_metric, species_loss
        self.node_batch_size = node_batch_size
        self.last_checkpoint = None
        self.saved = []
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _target_rows(self, value, indices, num_nodes, name, targets_are_global=False):
        value = torch.as_tensor(value, device=self.device)
        if value.ndim == 0:
            raise ValueError(f"{name} must have a node/sample axis")
        if targets_are_global:
            if value.shape[0] != num_nodes:
                raise ValueError(f"Global {name} must have {num_nodes} rows")
            return value[indices]
        if value.shape[0] == len(indices):
            return value
        if value.shape[0] == num_nodes:
            return value[indices]
        raise ValueError(f"{name} needs {len(indices)} seed rows or {num_nodes} graph rows")

    def step(self, batch: dict):
        """Project raw features once, mask modalities per objective, and optimize."""
        self.model.train()
        x = torch.as_tensor(batch["node_features"], device=self.device,
                            dtype=next(self.model.parameters()).dtype)
        if x.ndim != 2 or x.shape[1] != self.model.feature_dim:
            raise ValueError("node_features must be the raw [N, feature_dim] graph features")
        edge_index = batch.get("edge_index", getattr(self.model, "edge_index", None))
        if edge_index is None:
            raise ValueError("Provide batch['edge_index'] or model.edge_index")
        edge_index = torch.as_tensor(edge_index, device=self.device)
        indices = torch.as_tensor(batch["indices"], device=self.device)
        if indices.dtype == torch.bool:
            if indices.ndim != 1 or len(indices) != len(x):
                raise ValueError("A boolean indices mask must have shape [N]")
            indices = indices.nonzero(as_tuple=False).flatten()
        elif indices.dtype in (torch.int32, torch.int64):
            indices = indices.long().reshape(-1)
        else:
            raise ValueError("indices must be integer IDs or a boolean node mask")
        if not len(indices) or indices.min() < 0 or indices.max() >= len(x):
            raise ValueError("indices must be nonempty and within the supplied graph")
        global_targets = bool(batch.get("targets_are_global", False))
        targets = {}
        if self.weights[0]:
            targets["species"] = self._target_rows(batch["species_multilabel"], indices, len(x),
                                                    "species_multilabel", global_targets).to(dtype=x.dtype)
            expected_classes = self.model.species_head[-1].out_features
            if targets["species"].shape != (len(indices), expected_classes):
                raise ValueError(f"species_multilabel must have {expected_classes} class columns")
            if not torch.isfinite(targets["species"]).all() or ((targets["species"] < 0) | (targets["species"] > 1)).any():
                raise ValueError("Species targets must be finite values in [0, 1]")
        if self.weights[1]:
            if batch.get("coords") is None:
                raise ValueError("A positive location weight requires batch['coords'] in lat/lon degrees")
            targets["coords"] = self._target_rows(batch["coords"], indices, len(x), "coords", global_targets).to(dtype=x.dtype)
            if targets["coords"].shape != (len(indices), 2) or not torch.isfinite(targets["coords"]).all():
                raise ValueError("coords must be finite [B, 2] latitude/longitude degrees")
            if (targets["coords"][:, 0].abs() > 90).any() or (targets["coords"][:, 1].abs() > 180).any():
                raise ValueError("coords must be latitude/longitude degrees")
            if batch.get("negative_mask") is not None:
                negative = self._target_rows(batch["negative_mask"], indices, len(x), "negative_mask", global_targets)
                if negative.shape != (len(indices),):
                    raise ValueError("negative_mask must have one boolean value per seed")
                targets["valid_location"] = ~negative.bool()
        if self.weights[2]:
            if batch.get("audio") is None:
                targets["audio"] = x[indices, :self.model.audio_dim]
            else:
                targets["audio"] = self._target_rows(batch["audio"], indices, len(x), "audio", global_targets).to(dtype=x.dtype)
            if targets["audio"].shape != (len(indices), self.model.audio_dim):
                raise ValueError(f"audio targets must have {self.model.audio_dim} columns")

        size = self.node_batch_size or len(indices)
        sums = {name: 0.0 for name in ("species", "location", "audio", "total")}
        for start in range(0, len(indices), size):
            stop = min(start + size, len(indices))
            selected = indices[start:stop]
            losses = {}
            if self.weights[0]:
                out = self.model(x, edge_index, location_mask_prob=self.species_location_mask_prob)
                logits = out["species"][selected]
                y = targets["species"][start:stop]
                losses["species"] = (asymmetric_loss(logits, y) if self.species_loss == "asymmetric"
                                     else soft_multilabel_cross_entropy(logits, y))
            if self.weights[1]:
                out = self.model(x, edge_index, mask_location=True)
                valid = targets.get("valid_location")
                losses["location"] = self._chordal_loss(
                    out["location_vector"][selected], targets["coords"][start:stop],
                    None if valid is None else valid[start:stop])
            if self.weights[2]:
                out = self.model(x, edge_index, mask_audio=True)
                losses["audio"] = self._audio_loss(out["audio"][selected], targets["audio"][start:stop])
            total = sum(weight * losses[name] for name, weight in zip(
                ("species", "location", "audio"), self.weights) if weight)
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite GNN training loss")
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self.optimizer.step()
            for name in ("species", "location", "audio"):
                if name in losses:
                    sums[name] += (stop - start) * float(losses[name].detach())
            sums["total"] += (stop - start) * float(total.detach())
        return {name: value / len(indices) for name, value in sums.items()}

    def _chordal_loss(self, pred_vector, true_coords, valid_mask=None):
        return chordal_loss(pred_vector, true_coords, valid_mask)

    def _audio_loss(self, pred, target):
        return (1 - F.cosine_similarity(pred, target, dim=-1)).mean() + 0.1 * F.mse_loss(pred, target)

    def fit(self, loader, epochs):
        """Train and use the supplied validation loader for checkpoint selection."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        for epoch in range(1, epochs + 1):
            sums, count = {}, 0
            self.model.train()
            for batch in loader:
                values = self.step(batch)
                raw_indices = torch.as_tensor(batch["indices"])
                n = int(raw_indices.sum()) if raw_indices.dtype == torch.bool else raw_indices.numel()
                for name, value in values.items():
                    sums[name] = sums.get(name, 0.0) + n * value
                count += n
            if not count:
                raise ValueError("The training loader yielded no supervised nodes")
            metrics = {name: value / count for name, value in sums.items()}
            score = None
            if self.validation_loader is not None:
                from .gnn_birdset import evaluate_birdset
                was_training = self.model.training
                self.model.eval()
                try:
                    with torch.no_grad():
                        result = evaluate_birdset(self.model, self.validation_loader, self.device)
                finally:
                    self.model.train(was_training)
                metrics.update({"val_" + name: value for name, value in result.items()})
                if self.checkpoint_metric not in result:
                    raise KeyError(f"Validation did not return metric {self.checkpoint_metric!r}")
                score = float(result[self.checkpoint_metric])

            if self.checkpoint_dir is not None:
                checkpoint = {"epoch": epoch, "checkpoint_metric": self.checkpoint_metric,
                              "checkpoint_score": score, "model_state": self.model.state_dict(),
                              "optimizer_state": self.optimizer.state_dict(),
                              "model_config": self.model.config, "weights": self.weights,
                              "species_loss": self.species_loss}
                self.last_checkpoint = self.checkpoint_dir / "last.pt"
                torch.save(checkpoint, self.last_checkpoint)
                if score is not None and math.isfinite(score) and self.top_k:
                    path = self.checkpoint_dir / f"epoch_{epoch:03d}_{self.checkpoint_metric}_{score:.6f}.pt"
                    torch.save(checkpoint, path)
                    self.saved.append((score, path))
                    self.saved.sort(key=lambda item: item[0], reverse=True)
                    while len(self.saved) > self.top_k:
                        self.saved.pop()[1].unlink(missing_ok=True)
            print("epoch", epoch, metrics)
        return self.saved
