import torch
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel


_asymmetric_multilabel_loss = AsymmetricLossMultiLabel(
    gamma_neg=4.0, gamma_pos=1.0, clip=0.05
)

def coordinates_to_unit_vector(coords):
    """Convert [latitude, longitude] degrees to unit Earth-surface vectors."""
    radians = torch.deg2rad(coords)
    lat, lon = radians[..., 0], radians[..., 1]
    return torch.stack((torch.cos(lat) * torch.cos(lon),
                        torch.cos(lat) * torch.sin(lon),
                        torch.sin(lat)), dim=-1)


def chordal_loss(pred_vector, true_coords, valid_mask=None):
    """Mean squared chordal distance between predicted and true unit vectors."""
    target_vector = coordinates_to_unit_vector(true_coords)
    values = (pred_vector - target_vector).square().sum(dim=-1)
    if valid_mask is not None:
        valid = valid_mask.to(dtype=values.dtype)
        return (values * valid).sum() / valid.sum().clamp_min(1.0)
    return values.mean()

def asymmetric_loss(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
    if (gamma_neg, gamma_pos, clip) == (4.0, 1.0, 0.05):
        criterion = _asymmetric_multilabel_loss
    else:
        criterion = AsymmetricLossMultiLabel(
            gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip
        )
    # timm expects [N, C]. Flatten batch/sequence/token dimensions while
    # preserving the final species-class dimension.
    if logits.ndim < 2:
        raise ValueError(f"Expected logits with a class dimension, got {logits.shape}")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.to(dtype=logits.dtype).reshape_as(flat_logits)
    return criterion(flat_logits, flat_targets)


def soft_multilabel_cross_entropy(logits, targets):
    """Cross entropy with equal mass assigned to each positive label.

    A sample with labels {a, b} receives target probabilities 0.5 for each;
    an all-zero target contributes no loss because no positive class is known.
    """
    targets = targets.to(dtype=logits.dtype)
    normalizer = targets.sum(dim=-1, keepdim=True)
    normalized_targets = targets / normalizer.clamp_min(1.0)
    loss = -(normalized_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    valid = normalizer.squeeze(-1) > 0
    return loss.masked_select(valid).mean() if valid.any() else logits.sum() * 0.0


def compute_losses(out, batch, species_weight=1.0, location_weight=.1, audio_weight=.1,
                   species_loss="asymmetric"):
    losses = {}
    if species_loss == "asymmetric":
        losses["species"] = asymmetric_loss(out["species"], batch["species_multilabel"])
    elif species_loss == "cross_entropy":
        losses["species"] = soft_multilabel_cross_entropy(
            out["species"], batch["species_multilabel"]
        )
    else:
        raise ValueError(f"Unknown species loss: {species_loss}")
    valid_location = None
    if "negative_mask" in batch:
        valid_location = ~batch["negative_mask"]
    losses["location"] = chordal_loss(out["location_vector"], batch["coords"], valid_location)
    pred, target = out["audio"], batch["audio"]
    losses["audio"] = (1 - F.cosine_similarity(pred, target, dim=-1)).mean() + .1 * F.mse_loss(pred, target)
    losses["total"] = species_weight * losses["species"] + location_weight * losses["location"] + audio_weight * losses["audio"]
    return losses
