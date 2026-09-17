from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .location import Sphere2VecSphereM


@dataclass
class GraphData:
    row: np.ndarray
    col: np.ndarray
    audio: np.ndarray
    audio_node: np.ndarray
    contexts: np.ndarray
    labels: np.ndarray
    coords: np.ndarray

    @classmethod
    def load(cls, directory: str | Path) -> "GraphData":
        directory = Path(directory)
        with np.load(directory / "graph.npz") as g:
            row, col = g["row"], g["col"]
        with np.load(directory / "node_data.npz") as d:
            return cls(row, col, d["audio_embeddings"], d["audio_embedding_node_index"],
                       d["spatiotemporal_contexts"], d["labels"], d["coordinates_lat_lon"])

    @property
    def num_nodes(self):
        return len(self.labels)

    def adjacency(self):
        out = [[] for _ in range(self.num_nodes)]
        for a, b in zip(self.row, self.col):
            out[int(a)].append(int(b))
        return out


class WalkDataset(Dataset):
    def __init__(self, graph: GraphData, length=8, walks=10000, p=1.0, q=1.0,
                 seed=0, sphere_frequencies=8):
        self.g, self.length, self.rng = graph, length, np.random.default_rng(seed)
        self.adj, self.p, self.q = graph.adjacency(), p, q
        self.walks = walks
        self.location_encoder = Sphere2VecSphereM(sphere_frequencies)
        with torch.no_grad():
            self.loc_features = self.location_encoder(torch.from_numpy(graph.coords)).numpy()
        self.node_audio = [[] for _ in range(graph.num_nodes)]
        for i, n in enumerate(graph.audio_node): self.node_audio[int(n)].append(i)

    def __len__(self): return self.walks

    def _walk(self):
        cur = int(self.rng.integers(self.g.num_nodes)); walk = [cur]
        while len(walk) < self.length:
            choices = self.adj[cur]
            if not choices: break
            if len(walk) == 1 or self.q == 1:
                cur = choices[int(self.rng.integers(len(choices)))]
            else:
                prev = walk[-2]
                weights = np.array([1 / self.p if n == prev else (1 if n in self.adj[prev] else 1 / self.q) for n in choices])
                cur = choices[int(self.rng.choice(len(choices), p=weights / weights.sum()))]
            walk.append(cur)
        return walk + [walk[-1]] * (self.length - len(walk))

    def __getitem__(self, _):
        nodes = self._walk(); audios = []
        for n in nodes:
            indices = self.node_audio[n]
            audios.append(self.g.audio[indices[int(self.rng.integers(len(indices)))]])
        return {"audio": torch.from_numpy(np.stack(audios)).float(),
                "location": torch.from_numpy(self.loc_features[nodes]).float(),
                "coords": torch.from_numpy(self.g.coords[nodes]).float(),
                "species": torch.from_numpy(self.g.labels[nodes]).long(),
                "nodes": torch.tensor(nodes).long()}
