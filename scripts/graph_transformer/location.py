"""PyTorch-only Sphere2Vec location encoders.

The ``SphereM`` implementation follows Eq. (10) of Mai et al. (2023), using
the same multi-scale spherical basis as the reference implementation's
``spheremixscale`` encoder. Coordinates are latitude/longitude in degrees.
"""
from __future__ import annotations
import torch
from torch import nn


class Sphere2VecSphereM(nn.Module):
    """Fixed Sphere2Vec-sphereM positional encoder.

    At scale s, r**s controls resolution and the five basis terms are:
    sin(phi_s), cos(phi_s)cos(lambda), cos(phi)cos(lambda_s),
    cos(phi_s)sin(lambda), cos(phi)sin(lambda_s).
    """
    def __init__(self, scales: int = 16, radius: float = 2.0):
        super().__init__()
        if scales < 1 or radius <= 1:
            raise ValueError("scales must be positive and radius must be > 1")
        self.scales, self.radius = scales, radius

    @property
    def output_dim(self):
        return 5 * self.scales

    def forward(self, coords_deg: torch.Tensor) -> torch.Tensor:
        coords = coords_deg.to(dtype=torch.get_default_dtype())
        phi = torch.deg2rad(coords[..., 0]).clamp(-torch.pi / 2, torch.pi / 2)
        lam = torch.deg2rad(coords[..., 1])
        pieces = []
        for s in range(self.scales):
            scale = self.radius ** s
            phi_s, lam_s = phi / scale, lam / scale
            pieces += [torch.sin(phi_s),
                       torch.cos(phi_s) * torch.cos(lam),
                       torch.cos(phi) * torch.cos(lam_s),
                       torch.cos(phi_s) * torch.sin(lam),
                       torch.cos(phi) * torch.sin(lam_s)]
        return torch.stack(pieces, dim=-1)


class GridCellSpatialRelationEncoder(nn.Module):
    """PyTorch implementation of the paper's multi-scale grid-cell encoder.

    This is provided as a comparable Euclidean baseline. ``coords`` are
    latitude/longitude in degrees and are converted to radians before the
    multi-scale sinusoidal encoding.
    """
    def __init__(self, scales: int = 20, radius: float = 2.0):
        super().__init__()
        self.scales, self.radius = scales, radius

    @property
    def output_dim(self): return 4 * self.scales

    def forward(self, coords_deg: torch.Tensor) -> torch.Tensor:
        xy = torch.deg2rad(coords_deg.to(dtype=torch.get_default_dtype()))
        out = []
        for s in range(self.scales):
            z = xy / (self.radius ** s)
            out.extend((torch.sin(z[..., 0]), torch.cos(z[..., 0]),
                        torch.sin(z[..., 1]), torch.cos(z[..., 1])))
        return torch.stack(out, dim=-1)
