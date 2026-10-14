from __future__ import annotations

import torch
from torch import nn


class LinearAudioLocationClassifier(nn.Module):
    """Bias-free linear classifier over audio and Sphere2Vec features."""

    def __init__(self, audio_dim: int, location_dim: int, num_classes: int):
        super().__init__()
        self.audio_dim = audio_dim
        self.location_dim = location_dim
        self.classifier = nn.Linear(audio_dim + location_dim, num_classes, bias=False)

    def forward(self, audio, location):
        return self.classifier(torch.cat((audio, location), dim=-1))
