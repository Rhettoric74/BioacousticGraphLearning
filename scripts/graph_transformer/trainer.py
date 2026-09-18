from __future__ import annotations
import torch
from pathlib import Path
import heapq
from .losses import compute_losses


class Trainer:
    def __init__(self, model, optimizer, device="cuda", weights=(1., .1, .1),
                 validation_loader=None, checkpoint_dir=None, top_k=3):
        self.model, self.optimizer, self.device = model.to(device), optimizer, device
        self.weights = weights
        self.validation_loader = validation_loader
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.top_k = top_k
        self.saved = []
        if self.checkpoint_dir: self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        # Each objective masks its own target modality. This avoids trivial
        # identity reconstruction through the input token being predicted.
        species_out = self.model(batch["audio"], batch["location"], batch["species"], mask_species=True)
        location_out = self.model(batch["audio"], batch["location"], batch["species"], mask_species=False, mask_location=True)
        audio_out = self.model(batch["audio"], batch["location"], batch["species"], mask_species=False, mask_audio=True)
        species_loss = compute_losses(species_out, batch, 1., 0., 0.)["species"]
        location_loss = compute_losses(location_out, batch, 0., 1., 0.)["location"]
        audio_loss = compute_losses(audio_out, batch, 0., 0., 1.)["audio"]
        losses = {"species": species_loss, "location": location_loss, "audio": audio_loss,
                  "total": self.weights[0] * species_loss + self.weights[1] * location_loss + self.weights[2] * audio_loss}
        self.optimizer.zero_grad(set_to_none=True); losses["total"].backward(); self.optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items()}

    def fit(self, loader, epochs):
        for epoch in range(1, epochs + 1):
            sums = {}
            for batch in loader:
                for k, v in self.step(batch).items(): sums[k] = sums.get(k, 0) + v
            metrics = {k: round(v / len(loader), 4) for k, v in sums.items()}
            if self.validation_loader is not None:
                from .birdset import evaluate_birdset
                metrics.update({"val_" + k: v for k, v in evaluate_birdset(self.model, self.validation_loader, self.device).items()})
                score = metrics["val_macro_auroc"]
                if self.checkpoint_dir and score == score:
                    path = self.checkpoint_dir / f"epoch_{epoch:03d}_auroc_{score:.6f}.pt"
                    torch.save({"epoch": epoch, "val_macro_auroc": score,
                                "model_state": self.model.state_dict(),
                                "optimizer_state": self.optimizer.state_dict()}, path)
                    self.saved.append((score, path)); self.saved.sort(key=lambda x: x[0], reverse=True)
                    while len(self.saved) > self.top_k:
                        _, old = self.saved.pop(); old.unlink(missing_ok=True)
            print("epoch", epoch, metrics)
        return self.saved
