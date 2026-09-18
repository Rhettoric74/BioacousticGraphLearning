import torch
import torch.nn.functional as F

def asymmetric_loss(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1.0 - xs_pos
    if clip:
        xs_neg = (xs_neg + clip).clamp(max=1)
    loss = targets * torch.log(xs_pos.clamp_min(1e-8)) + (1-targets) * torch.log(xs_neg.clamp_min(1e-8))
    pt = xs_pos * targets + xs_neg * (1-targets)
    gamma = gamma_pos * targets + gamma_neg * (1-targets)
    return -(loss * (1-pt).pow(gamma)).mean()


def compute_losses(out, batch, species_weight=1.0, location_weight=.1, audio_weight=.1):
    losses = {}
    losses["species"] = asymmetric_loss(out["species"], batch["species_multilabel"])
    losses["location"] = F.smooth_l1_loss(out["location"], batch["location"])
    pred, target = out["audio"], batch["audio"]
    losses["audio"] = (1 - F.cosine_similarity(pred, target, dim=-1)).mean() + .1 * F.mse_loss(pred, target)
    losses["total"] = species_weight * losses["species"] + location_weight * losses["location"] + audio_weight * losses["audio"]
    return losses
