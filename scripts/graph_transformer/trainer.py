from __future__ import annotations
import torch
from .losses import compute_losses


class Trainer:
    def __init__(self, model, optimizer, device="cuda", weights=(1., .1, .1)):
        self.model, self.optimizer, self.device = model.to(device), optimizer, device
        self.weights = weights

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
            print("epoch", epoch, {k: round(v / len(loader), 4) for k, v in sums.items()})
