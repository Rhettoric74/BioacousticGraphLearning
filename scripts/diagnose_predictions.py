#!/usr/bin/env python3
"""Inspect prediction collapse, confidence, and graph-walk label diversity."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from graph_transformer.data import GraphData, WalkDataset
from graph_transformer.model import GraphWalkTransformer
from graph_transformer.birdset import BirdSetDataset


def quantiles(x):
    x = np.asarray(x)
    return {f"p{q}": float(np.percentile(x, q)) for q in [0, 25, 50, 75, 95, 99, 100]}


def print_examples(logits, targets, names, n=10):
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    top = probs.argmax(axis=1)
    for i in np.linspace(0, len(top) - 1, min(n, len(top)), dtype=int):
        truth = np.flatnonzero(targets[i]).tolist()
        print({"prediction": int(top[i]), "confidence": float(probs[i, top[i]]),
               "truth": truth, "prediction_name": names[top[i]] if names else None})


def report(name, logits, targets, names=None):
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    top = probs.argmax(axis=1)
    has_label = targets.sum(axis=1) > 0
    correct = targets[np.arange(len(targets)), top] > 0
    print(f"\n{name}")
    print("samples", len(targets), "labeled", int(has_label.sum()))
    print("top1", float(correct[has_label].mean()) if has_label.any() else float("nan"))
    print("max_probability", quantiles(probs.max(axis=1)))
    print("positive_probability", quantiles(probs[targets > 0]) if (targets > 0).any() else {})
    print("negative_probability", quantiles(probs[targets == 0]))
    unique, counts = np.unique(top, return_counts=True)
    order = np.argsort(counts)[::-1][:10]
    print("top_predictions", [(int(unique[i]), int(counts[i]), names[int(unique[i])] if names else None) for i in order])
    print_examples(logits, targets, names)


@torch.no_grad()
def graph_diagnostics(model, loader, device, num_classes):
    logits, targets, walk_diversity = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(batch["audio"], batch["location"], batch["species"], mask_species=True)
        logits.append(out["species"].reshape(-1, num_classes).cpu().numpy())
        targets.append(batch["species_multilabel"].reshape(-1, num_classes).cpu().numpy())
        walk_diversity.extend((batch["species_multilabel"].sum(dim=1) > 0).sum(dim=1).cpu().numpy())
    print("\nGraph walk diversity")
    print("species-bearing nodes per walk", quantiles(walk_diversity))
    print("single-species-bearing-node walks", float(np.mean(np.asarray(walk_diversity) <= 1)))
    return np.concatenate(logits), np.concatenate(targets)


@torch.no_grad()
def birdset_logits(model, loader, device, context_length=1):
    audio = torch.cat([b["audio"] for b in loader])
    location = torch.cat([b["location"] for b in loader])
    targets = torch.cat([b["labels"] for b in loader]).numpy()
    outputs = []
    for start in range(0, len(targets), context_length):
        stop = min(start + context_length, len(targets))
        a = audio[start:stop].to(device).unsqueeze(0)
        l = location[start:stop].to(device).unsqueeze(0)
        species = torch.zeros((1, stop - start), dtype=torch.long, device=device)
        outputs.append(model(a, l, species, mask_species=True)["species"][0].cpu().numpy())
    return np.concatenate(outputs), targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("graph_dir"); p.add_argument("checkpoint")
    p.add_argument("--birdset-path", required=True)
    p.add_argument("--perch-label-mapping", required=True)
    p.add_argument("--d-model", type=int, default=768); p.add_argument("--layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=12); p.add_argument("--walks", type=int, default=1000)
    p.add_argument("--walk-length", type=int, default=8); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-length", type=int, default=8); p.add_argument("--device", default="cuda")
    p.add_argument("--sphere-scales", type=int, default=8)
    args = p.parse_args(); device = args.device
    graph = GraphData.load(args.graph_dir)
    walks = WalkDataset(graph, args.walk_length, args.walks, sphere_frequencies=args.sphere_scales)
    loader = DataLoader(walks, args.batch_size, shuffle=False)
    num_species = graph.multilabels.shape[1]
    model = GraphWalkTransformer(graph.audio.shape[-1], walks.loc_features.shape[-1], num_species,
                                 args.d_model, args.layers, args.heads, max_length=args.walk_length).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"]); model.eval()
    with open(args.perch_label_mapping) as f: mapping = json.load(f)
    names = [mapping.get(str(i), str(i)) for i in range(num_species)]
    train_logits, train_targets = graph_diagnostics(model, loader, device, num_species)
    report("graph training walks", train_logits, train_targets, names)

    # Reuse the same mapping convention as train_graph_transformer.py.
    from birdset_preparation import load_birdset_data
    split = load_birdset_data("POW")
    codes = [split.features["ebird_code"]._int2str[i] for i in range(len(split.features["ebird_code"]._int2str))]
    model_codes = [mapping[str(i)] for i in range(len(mapping))]
    code_to_model = {code: i for i, code in enumerate(model_codes)}
    subset_labels = [code_to_model.get(code, -1) for code in codes]
    data = BirdSetDataset(args.birdset_path, subset_labels, num_species, args.sphere_scales)
    test_loader = DataLoader(data, 1024, shuffle=False)
    pow_logits, pow_targets = birdset_logits(model, test_loader, device, args.context_length)
    report(f"POW context length {args.context_length}", pow_logits, pow_targets, names)


if __name__ == "__main__": main()
