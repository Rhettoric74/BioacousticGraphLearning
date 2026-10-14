#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from graph_transformer.data import GraphData, WalkDataset
from graph_transformer.birdset import BirdSetDataset
from linear_classifier.model import LinearAudioLocationClassifier
from linear_classifier.metrics import multilabel_metrics


def soft_targets_loss(logits, targets):
    weights = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1.0)
    valid = targets.sum(dim=-1) > 0
    losses = -(weights * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    return losses[valid].mean() if valid.any() else logits.sum() * 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); logits, labels = [], []
    for batch in loader:
        logits.append(model(batch["audio"].to(device), batch["location"].to(device)).cpu().numpy())
        labels.append(batch["labels"].numpy())
    return multilabel_metrics(np.concatenate(logits), np.concatenate(labels))


def make_mapping(split, model_codes):
    from birdset_preparation import load_birdset_data
    data = load_birdset_data(split)
    int2str = data.features["ebird_code"]._int2str
    code_to_model = {code: i for i, code in enumerate(model_codes)}
    return [code_to_model.get(int2str[i], -1) for i in range(len(int2str))]


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("graph_dir")
    p.add_argument("--birdset-dir", required=True)
    p.add_argument("--perch-label-mapping", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints/linear_classifier")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--walks", type=int, default=100000)
    p.add_argument("--walk-length", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--sphere-scales", type=int, default=32)
    p.add_argument("--p", type=float, default=1.0); p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    graph = GraphData.load(a.graph_dir)
    walks = WalkDataset(graph, a.walk_length, a.walks, a.p, a.q, sphere_frequencies=a.sphere_scales)
    loader = DataLoader(walks, a.batch_size, shuffle=False, num_workers=0)
    num_classes = graph.multilabels.shape[1]
    model = LinearAudioLocationClassifier(graph.audio.shape[-1], walks.loc_features.shape[-1], num_classes).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    checkpoint_dir = Path(a.checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(a.perch_label_mapping) as f:
        mapping = json.load(f)
    model_codes = [mapping[str(i)] for i in range(len(mapping))]
    pow_mapping = make_mapping("POW", model_codes)
    pow_data = BirdSetDataset(Path(a.birdset_dir) / "POW.pkl", pow_mapping, num_classes, a.sphere_scales)
    pow_loader = DataLoader(pow_data, 1024, shuffle=False)
    best = -float("inf")
    for epoch in range(1, a.epochs + 1):
        model.train(); total = 0.0
        for batch in loader:
            audio = batch["audio"].to(a.device).reshape(-1, graph.audio.shape[-1])
            location = batch["location"].to(a.device).reshape(-1, walks.loc_features.shape[-1])
            targets = batch["species_multilabel"].to(a.device).reshape(-1, num_classes)
            loss = soft_targets_loss(model(audio, location), targets)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())
        metrics = evaluate(model, pow_loader, a.device)
        print("epoch", epoch, {"loss": round(total / len(loader), 4), **metrics})
        state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
        torch.save(state, checkpoint_dir / "last.pt")
        if metrics["macro_auroc"] > best:
            best = metrics["macro_auroc"]
            torch.save(state, checkpoint_dir / "best.pt")

    for name in ("best", "last"):
        state = torch.load(checkpoint_dir / f"{name}.pt", map_location=a.device, weights_only=True)
        model.load_state_dict(state["model_state"])
        for split in ["HSN", "SNE", "PER", "UHH", "NES", "SSW"]:
            path = Path(a.birdset_dir) / f"{split}.pkl"
            if path.exists():
                dataset = BirdSetDataset(path, make_mapping(split, model_codes), num_classes, a.sphere_scales)
                print(name, split, evaluate(model, DataLoader(dataset, 1024), a.device))


if __name__ == "__main__":
    main()
