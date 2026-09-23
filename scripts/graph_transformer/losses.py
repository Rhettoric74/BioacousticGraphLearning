import torch
import torch.nn.functional as F

def coordinates_to_unit_vector(coords):
    """Convert [latitude, longitude] degrees to unit Earth-surface vectors."""
    radians = torch.deg2rad(coords)
    lat, lon = radians[..., 0], radians[..., 1]
    return torch.stack((torch.cos(lat) * torch.cos(lon),
                        torch.cos(lat) * torch.sin(lon),
                        torch.sin(lat)), dim=-1)


def chordal_loss(pred_vector, true_coords):
    """Mean squared chordal distance between predicted and true unit vectors."""
    target_vector = coordinates_to_unit_vector(true_coords)
    return (pred_vector - target_vector).square().sum(dim=-1).mean()

def asymmetric_loss(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1.0 - xs_pos
    if clip:
        xs_neg = (xs_neg + clip).clamp(max=1)
    loss = targets * torch.log(xs_pos.clamp_min(1e-8)) + (1-targets) * torch.log(xs_neg.clamp_min(1e-8))
    pt = xs_pos * targets + xs_neg * (1-targets)
    gamma = gamma_pos * targets + gamma_neg * (1-targets)
    # Sum over species classes, then average over examples/tokens.  Averaging
    # over classes makes the loss shrink roughly in proportion to the number
    # of species, which is undesirable for sparse multilabel targets.
    per_token_loss = -(loss * (1-pt).pow(gamma)).sum(dim=-1)
    return per_token_loss.mean()


def compute_losses(out, batch, species_weight=1.0, location_weight=.1, audio_weight=.1):
    losses = {}
    losses["species"] = asymmetric_loss(out["species"], batch["species_multilabel"])
    losses["location"] = chordal_loss(out["location_vector"], batch["coords"])
    pred, target = out["audio"], batch["audio"]
    losses["audio"] = (1 - F.cosine_similarity(pred, target, dim=-1)).mean() + .1 * F.mse_loss(pred, target)
    losses["total"] = species_weight * losses["species"] + location_weight * losses["location"] + audio_weight * losses["audio"]
    return losses
