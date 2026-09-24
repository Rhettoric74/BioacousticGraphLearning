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
        audio_parts, context_parts, raw_labels = [], [], []
        feature_batch_count = len(data["embeddings"])
        label_batches = data["labels"]
        labels_are_flat = len(label_batches) != feature_batch_count
        if labels_are_flat:
            expected_samples = sum(len(batch) for batch in data["embeddings"])
            if len(label_batches) != expected_samples:
                raise ValueError(f"SSW label mismatch in {path}: feature samples={expected_samples}, labels={len(label_batches)}")
            batch_iterator = zip(data["embeddings"], data["st_context"], [None] * feature_batch_count)
        else:
            batch_iterator = zip(data["embeddings"], data["st_context"], label_batches)
        for batch_index, (audio_batch, context_batch, label_batch) in enumerate(batch_iterator):
            if len(audio_batch) != len(context_batch):
                raise ValueError(f"Sample-count mismatch in {path}, batch {batch_index}: embeddings={len(audio_batch)}, st_context={len(context_batch)}")
            audio_parts.append(np.asarray(audio_batch))
            context_parts.append(np.asarray(context_batch))
            if not labels_are_flat:
                if len(audio_batch) != len(label_batch):
                    raise ValueError(f"Sample-count mismatch in {path}, batch {batch_index}: embeddings={len(audio_batch)}, labels={len(label_batch)}")
                raw_labels.extend(label_batch)
        if labels_are_flat:
            raw_labels = list(label_batches)
        self.audio = np.concatenate(audio_parts, axis=0).astype(np.float32)
        raw_context = np.concatenate(context_parts, axis=0).astype(np.float32)
        if len(self.audio) != len(raw_labels) or len(self.audio) != len(raw_context):
            raise ValueError(f"BirdSet fields have inconsistent lengths in {path}")
        coords = raw_context.reshape(len(raw_context), -1)[:, :2]
        with torch.no_grad():
            self.location = Sphere2VecSphereM(sphere_scales)(torch.from_numpy(coords)).numpy()
        self.labels = np.zeros((len(raw_labels), num_model_classes), dtype=np.float32)
        self.subset_labels = np.asarray(subset_labels, dtype=np.int64)
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
    model.eval(); probabilities, labels, full_logits = [], [], []
    for batch in loader:
        audio = batch["audio"].to(device).unsqueeze(1)
        location = batch["location"].to(device).unsqueeze(1)
        species = torch.zeros(audio.shape[:2], dtype=torch.long, device=device)
        logits = model(audio, location, species, mask_species=True)["species"][:, 0]
        full_logits.append(logits.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    full_logits = np.concatenate(full_logits)
    per_class = []
    for c in range(labels.shape[1]):
        if labels[:, c].min() == labels[:, c].max(): continue
        per_class.append(roc_auc_score(labels[:, c], probabilities[:, c]))
    return {"macro_auroc": float(np.mean(per_class)) if per_class else float("nan"),
            "cmAP": float(np.mean([average_precision_score(labels[:, c], probabilities[:, c])
                                    for c in range(labels.shape[1]) if labels[:, c].max() > labels[:, c].min()])) if any(labels[:, c].max() > labels[:, c].min() for c in range(labels.shape[1])) else float("nan"),
            "top1": top1_from_full_logits(full_logits, labels),
            "n_valid_classes": len(per_class), "n_samples": len(labels)}


def top1_from_full_logits(full_logits, labels):
    """Top-1 over all model classes, matching the prior BirdSet evaluator."""
    has_label = labels.sum(axis=1) > 0
    if not has_label.any():
        return float("nan")
    predictions = full_logits.argmax(axis=1)
    correct = labels[np.arange(len(labels)), predictions] > 0
    return float(correct[has_label].mean())


@torch.no_grad()
def evaluate_birdset_context(model, loader, device="cuda", context_length=8):
    """Evaluate consecutive samples jointly as one masked-species sequence."""
    model.eval()
    audio = torch.cat([b["audio"] for b in loader])
    location = torch.cat([b["location"] for b in loader])
    labels = torch.cat([b["labels"] for b in loader]).numpy()
    probabilities, full_logits = [], []
    for start in range(0, len(labels), context_length):
        stop = min(start + context_length, len(labels))
        a = audio[start:stop].to(device).unsqueeze(0)
        l = location[start:stop].to(device).unsqueeze(0)
        species = torch.zeros((1, stop - start), dtype=torch.long, device=device)
        logits = model(a, l, species, mask_species=True)["species"][0]
        full_logits.append(logits.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    full_logits = np.concatenate(full_logits)
    aucs, aps = [], []
    for c in range(labels.shape[1]):
        if labels[:, c].min() == labels[:, c].max():
            continue
        aucs.append(roc_auc_score(labels[:, c], probabilities[:, c]))
        aps.append(average_precision_score(labels[:, c], probabilities[:, c]))
    return {"macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
            "cmAP": float(np.mean(aps)) if aps else float("nan"),
            "top1": top1_from_full_logits(full_logits, labels),
            "n_valid_classes": len(aucs), "n_samples": len(labels),
            "context_length": context_length}
