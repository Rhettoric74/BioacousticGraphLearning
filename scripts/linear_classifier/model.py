from __future__ import annotations

import torch
from torch import nn


class LinearAudioLocationClassifier(nn.Module):
    """Linear classifier over audio and an MLP-transformed location feature."""

    def __init__(self, audio_dim: int, location_dim: int, num_classes: int,
                 location_hidden_dim: int = 512, location_dropout: float = 0.5):
        super().__init__()
        self.audio_dim = audio_dim
        self.location_dim = location_dim
        self.location_hidden_dim = location_hidden_dim
        self.location_features = nn.Sequential(
            nn.Linear(location_dim, location_hidden_dim),
            nn.ReLU(),
            nn.Dropout(location_dropout),
        )
        self.classifier = nn.Linear(audio_dim + location_hidden_dim, num_classes)

    def forward(self, audio, location):
        location = self.location_features(location)
        return self.classifier(torch.cat((audio, location), dim=-1))
