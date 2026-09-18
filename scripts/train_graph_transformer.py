#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from graph_transformer.data import GraphData, WalkDataset
from graph_transformer.model import GraphWalkTransformer
from graph_transformer.trainer import Trainer
from graph_transformer.birdset import BirdSetDataset

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("graph_dir"); p.add_argument("--epochs", type=int, default=10); p.add_argument("--walks", type=int, default=10000)
    p.add_argument("--walk-length", type=int, default=8); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4); p.add_argument("--heads", type=int, default=8); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--p", type=float, default=1.0); p.add_argument("--q", type=float, default=1.0); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sphere-scales", type=int, default=8)
    p.add_argument("--location-weight", type=float, default=.1); p.add_argument("--audio-weight", type=float, default=.1)
    p.add_argument("--birdset-dir", default=None, help="Directory containing POW.pkl and evaluation split pickles")
    p.add_argument("--perch-label-mapping", default="/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json")
    p.add_argument("--checkpoint-dir", default="checkpoints/graph_transformer")
    p.add_argument("--top-k", type=int, default=3)
    a = p.parse_args(); g = GraphData.load(a.graph_dir)
    ds = WalkDataset(g, a.walk_length, a.walks, a.p, a.q, sphere_frequencies=a.sphere_scales); loader = DataLoader(ds, a.batch_size, shuffle=False, num_workers=0)
    model = GraphWalkTransformer(g.audio.shape[-1], ds.loc_features.shape[-1], int(g.labels.max()) + 1, a.d_model, a.layers, a.heads, max_length=a.walk_length)
    val_loader = None; subset_labels = None
    if a.birdset_dir:
        with open(a.perch_label_mapping) as f:
            perch_mapping = json.load(f)
        model_codes = [perch_mapping[str(i)] for i in range(len(perch_mapping))]
        try:
            from birdset_preparation import load_birdset_data
        except ImportError as exc:
            raise ImportError("birdset_preparation.load_birdset_data must be importable when BirdSet evaluation is enabled") from exc

        def split_mapping(split):
            split_data = load_birdset_data(split)
            int2str = split_data.features["ebird_code"]._int2str
            split_codes = [int2str[i] for i in range(len(int2str))]
            code_to_model = {code: i for i, code in enumerate(model_codes)}
            return [code_to_model.get(code, -1) for code in split_codes]

        subset_labels = split_mapping("POW")
        val = BirdSetDataset(Path(a.birdset_dir) / "POW.pkl", subset_labels, model.species_head.out_features, a.sphere_scales)
        val_loader = DataLoader(val, batch_size=1024, shuffle=False, num_workers=0)
    trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=a.lr), a.device,
                      (1., a.location_weight, a.audio_weight), val_loader,
                      a.checkpoint_dir, a.top_k)
    saved = trainer.fit(loader, a.epochs)
    if val_loader is not None and saved:
        from graph_transformer.birdset import evaluate_birdset
        best = max(saved, key=lambda x: x[0]); checkpoint = torch.load(best[1], map_location=a.device)
        model.load_state_dict(checkpoint["model_state"])
        for split in ["HSN", "SNE", "PER", "UHH", "NES", "SSW"]:
            path = Path(a.birdset_dir) / f"{split}.pkl"
            if path.exists():
                test = BirdSetDataset(path, split_mapping(split), model.species_head.out_features, a.sphere_scales)
                result = evaluate_birdset(model, DataLoader(test, 1024, shuffle=False), a.device)
                print(split, result)

if __name__ == "__main__": main()
