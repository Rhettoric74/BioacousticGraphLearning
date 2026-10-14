from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from graph_transformer.location import Sphere2VecSphereM


class BirdSetLinearDataset(Dataset):
    """BirdSet pickle loader with split-local labels mapped to model classes."""

    def __init__(self, path, subset_labels, num_model_classes, sphere_scales=8):
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        embeddings = data["embeddings"]
        contexts = data["st_context"]
        label_batches = data["labels"]
        feature_count = len(embeddings)
        labels_flat = len(label_batches) != feature_count
        audio_parts, context_parts, labels = [], [], []
        for i, (audio_batch, context_batch) in enumerate(zip(embeddings, contexts)):
            if len(audio_batch) != len(context_batch):
                raise ValueError(f"Feature length mismatch in {path}, batch {i}")
            audio_parts.append(np.asarray(audio_batch))
            context_parts.append(np.asarray(context_batch))
            if not labels_flat:
                if len(audio_batch) != len(label_batches[i]):
                    raise ValueError(f"Label length mismatch in {path}, batch {i}")
                labels.extend(label_batches[i])
        if labels_flat:
            labels = list(label_batches)
        self.audio = np.concatenate(audio_parts).astype(np.float32)
        raw_context = np.concatenate(context_parts).astype(np.float32)
        if len(self.audio) != len(labels):
            raise ValueError(f"Fields have inconsistent lengths in {path}")
        coords = raw_context.reshape(len(raw_context), -1)[:, [1, 0]]
        self.coords = coords.astype(np.float32)
        self.location = Sphere2VecSphereM(sphere_scales)(torch.from_numpy(self.coords)).numpy()
        self.labels = np.zeros((len(labels), num_model_classes), dtype=np.float32)
        for i, row in enumerate(labels):
            for local_index in row:
                local_index = int(local_index)
                if 0 <= local_index < len(subset_labels):
                    model_index = int(subset_labels[local_index])
                    if 0 <= model_index < num_model_classes:
                        self.labels[i, model_index] = 1.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"audio": torch.from_numpy(self.audio[index]),
                "location": torch.from_numpy(self.location[index]),
                "labels": torch.from_numpy(self.labels[index])}
