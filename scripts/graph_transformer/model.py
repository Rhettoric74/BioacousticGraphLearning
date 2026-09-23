from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class GraphWalkTransformer(nn.Module):
    def __init__(self, audio_dim, location_dim, num_species, d_model=256, layers=4,
                 heads=8, dropout=.1, max_length=64):
        super().__init__()
        self.audio = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, d_model))
        self.location = nn.Sequential(nn.LayerNorm(location_dim), nn.Linear(location_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.species = nn.Embedding(num_species, d_model)
        self.modality = nn.Embedding(3, d_model)
        self.position = nn.Embedding(max_length * 3, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        # Tie the output classifier to the species-token embeddings. This
        # makes the species embeddings train directly from the species loss
        # and avoids a separate biased classifier representation.
        self.species_head = nn.Linear(d_model, num_species, bias=False)
        self.species_head.weight = self.species.weight
        self.location_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.audio_head = nn.Linear(d_model, audio_dim)

    def forward(self, audio, location, species, mask_species=True, mask_location=False, mask_audio=False):
        b, n, _ = audio.shape
        s = self.species(species)
        if mask_species: s = torch.zeros_like(s)
        a = self.audio(audio); l = self.location(location)
        if mask_audio: a = torch.zeros_like(a)
        if mask_location: l = torch.zeros_like(l)
        x = torch.stack([a, l, s], 2).reshape(b, n * 3, -1)
        mods = torch.arange(3, device=x.device).repeat(n).unsqueeze(0)
        pos = torch.arange(n * 3, device=x.device).unsqueeze(0)
        x = self.encoder(x + self.modality(mods) + self.position(pos))
        x = self.norm(x.reshape(b, n, 3, -1))
        raw_location = self.location_head(x[:, :, 1])
        location_vector = F.normalize(raw_location, dim=-1, eps=1e-8)
        latitude = torch.rad2deg(torch.asin(location_vector[..., 2].clamp(-1.0, 1.0)))
        longitude = torch.rad2deg(torch.atan2(location_vector[..., 1], location_vector[..., 0]))
        coordinates = torch.stack((latitude, longitude), dim=-1)
        return {"species": self.species_head(x[:, :, 2]),
                "coordinates": coordinates,
                "location_vector": location_vector,
                "audio": self.audio_head(x[:, :, 0]), "hidden": x}
