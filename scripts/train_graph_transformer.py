#!/usr/bin/env python3
import argparse
import torch
from torch.utils.data import DataLoader
from graph_transformer.data import GraphData, WalkDataset
from graph_transformer.model import GraphWalkTransformer
from graph_transformer.trainer import Trainer

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("graph_dir"); p.add_argument("--epochs", type=int, default=10); p.add_argument("--walks", type=int, default=10000)
    p.add_argument("--walk-length", type=int, default=8); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4); p.add_argument("--heads", type=int, default=8); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--p", type=float, default=1.0); p.add_argument("--q", type=float, default=1.0); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--location-weight", type=float, default=.1); p.add_argument("--audio-weight", type=float, default=.1)
    a = p.parse_args(); g = GraphData.load(a.graph_dir)
    ds = WalkDataset(g, a.walk_length, a.walks, a.p, a.q); loader = DataLoader(ds, a.batch_size, shuffle=False, num_workers=0)
    model = GraphWalkTransformer(g.audio.shape[-1], ds.loc_features.shape[-1], int(g.labels.max()) + 1, a.d_model, a.layers, a.heads, max_length=a.walk_length)
    Trainer(model, torch.optim.AdamW(model.parameters(), lr=a.lr), a.device, (1., a.location_weight, a.audio_weight)).fit(loader, a.epochs)

if __name__ == "__main__": main()
