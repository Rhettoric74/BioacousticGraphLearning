from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from .location import Sphere2VecSphereM


class BirdSetDataset(Dataset):
    """Loads the pickle layout used by the previous BirdSet experiments.

    Each pickle contains batched ``embeddings``, ``st_context`` and ``labels``.
    Labels are lists of local BirdSet class indices. ``subset_labels`` maps
    those local columns into graph-transformer output indices.
    """
    def __init__(self, path, subset_labels, num_model_classes, sphere_scales=8):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.audio = np.concatenate(data["embeddings"], axis=0).astype(np.float32)
        raw_context = np.concatenate(data["st_context"], axis=0).astype(np.float32)
        raw_labels = [label for batch in data["labels"] for label in batch]
        if len(self.audio) != len(raw_labels) or len(self.audio) != len(raw_context):
            raise ValueError(f"BirdSet fields have inconsistent lengths in {path}")
        coords = raw_context.reshape(len(raw_context), -1)[:, :2]
        with torch.no_grad():
            self.location = Sphere2VecSphereM(sphere_scales)(torch.from_numpy(coords)).numpy()
        self.labels = np.zeros((len(raw_labels), num_model_classes), dtype=np.float32)
        for i, labels in enumerate(raw_labels):
            for local_index in labels:
                if 0 <= int(local_index) < len(subset_labels):
                    model_index = int(subset_labels[int(local_index)])
                    if 0 <= model_index < num_model_classes:
                        self.labels[i, model_index] = 1.0

    def __len__(self): return len(self.labels)

    def __getitem__(self, index):
        return {"audio": torch.from_numpy(self.audio[index]),
                "location": torch.from_numpy(self.location[index]),
                "labels": torch.from_numpy(self.labels[index])}


@torch.no_grad()
def evaluate_birdset(model, loader, device="cuda"):
    model.eval(); probabilities, labels = [], []
    for batch in loader:
        audio = batch["audio"].to(device).unsqueeze(1)
        location = batch["location"].to(device).unsqueeze(1)
        species = torch.zeros(audio.shape[:2], dtype=torch.long, device=device)
        logits = model(audio, location, species, mask_species=True)["species"][:, 0]
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    per_class = []
    for c in range(labels.shape[1]):
        if labels[:, c].min() == labels[:, c].max(): continue
        per_class.append(roc_auc_score(labels[:, c], probabilities[:, c]))
    return {"macro_auroc": float(np.mean(per_class)) if per_class else float("nan"),
            "cmAP": float(np.mean([average_precision_score(labels[:, c], probabilities[:, c])
                                    for c in range(labels.shape[1]) if labels[:, c].max() > labels[:, c].min()])) if any(labels[:, c].max() > labels[:, c].min() for c in range(labels.shape[1])) else float("nan"),
            "top1": float(labels[np.arange(len(labels)), probabilities.argmax(1)].mean()),
            "n_valid_classes": len(per_class), "n_samples": len(labels)}
