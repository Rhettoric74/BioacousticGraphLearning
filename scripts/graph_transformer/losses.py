import torch
import torch.nn.functional as F


def compute_losses(out, batch, species_weight=1.0, location_weight=.1, audio_weight=.1):
    losses = {}
    losses["species"] = F.cross_entropy(out["species"].flatten(0, 1), batch["species"].flatten())
    losses["location"] = F.smooth_l1_loss(out["location"], batch["location"])
    pred, target = out["audio"], batch["audio"]
    losses["audio"] = (1 - F.cosine_similarity(pred, target, dim=-1)).mean() + .1 * F.mse_loss(pred, target)
    losses["total"] = species_weight * losses["species"] + location_weight * losses["location"] + audio_weight * losses["audio"]
    return losses
