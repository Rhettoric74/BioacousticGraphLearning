#!/usr/bin/env python3
"""
Train and eval GNN (GraphSAGE or GATv2)

Usage:
```
python scripts/train_gnn.py /path/to/GRAPH_DIR \
    --birdset-dir /path/to/BIRDSET_DIR \
    --perch-label-mapping /path/to/perch_v2_label_mapping.json \
    --checkpoint-dir checkpoints/graphsage \
    --gnn-type sage \
    --layers 2 \
    --neighbors 10 10 \
    --batch-size 128 \
    --epochs 20
```
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("graph_dir", type=Path)
    p.add_argument("--birdset-dir", type=Path, help="POW.pkl and test pickles, as in the sample scripts")
    p.add_argument("--perch-label-mapping", type=Path,
                   help="Global model column -> eBird code JSON used by your existing experiments")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/graphsage"))
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=Path, help="Checkpoint to test with --eval-only")
    p.add_argument("--skip-test", action="store_true", help="Train/validate only; do not test afterward")
    p.add_argument("--splits", nargs="+", default=["HSN", "SNE", "PER", "UHH", "NES", "SSW"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128, help="Supervised seed nodes per training step")
    p.add_argument("--steps-per-epoch", type=int, help="Optional random seed batches per epoch; otherwise visit all nodes")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--neighbors", type=int, nargs="+", help="Nearest-hop-first training fanouts; default 10 per layer")
    p.add_argument("--gnn-type", choices=["sage", "gat"], default="sage", help="gat selects GATv2")
    p.add_argument("--aggregate", choices=["mean", "pool", "max", "sum", "lstm", "gcn"], default="mean")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--sphere-scales", type=int, default=8)
    p.add_argument("--audio-only", action="store_true", help="Remove location features; geographic graph remains")
    p.add_argument("--species-loss", choices=["asymmetric", "cross_entropy"], default="asymmetric")
    p.add_argument("--location-weight", type=float, default=0.0)
    p.add_argument("--audio-weight", type=float, default=0.0)
    p.add_argument("--species-location-mask-prob", type=float, default=0.0)
    p.add_argument("--negative-embeddings", type=Path, help="NPY matrix or NPZ containing embeddings")
    p.add_argument("--negative-replacement-probability", type=float, default=0.0)
    p.add_argument("--edge-convention", choices=["row_neighbors", "source_to_target"], default="row_neighbors",
                   help="row_neighbors matches the supplied GraphData.adjacency(): col informs row")
    p.add_argument("--train-nodes", type=Path, help="Optional NPY IDs: induce a training-only reference graph")
    p.add_argument("--eval-batch-size", type=int, default=32, help="Exact L-hop evaluation can need substantial memory")
    p.add_argument("--query-k", type=int, default=None,
                   help="Training reference neighbors per query; 10 for training, saved setting for evaluation")
    p.add_argument("--checkpoint-metric", choices=["macro_auroc", "cmAP", "top1"], default="macro_auroc")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--geolocation", action="store_true", help="Also run isolated audio-only geolocation, if its head was trained")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    a = p.parse_args(argv)
    if a.eval_only and (a.checkpoint is None or a.birdset_dir is None):
        p.error("--eval-only requires --checkpoint and --birdset-dir")
    if a.checkpoint is not None and not a.eval_only:
        p.error("--checkpoint is for --eval-only; resume training is not implemented")
    if any(v < 1 for v in (a.epochs, a.batch_size, a.layers, a.d_model, a.sphere_scales, a.eval_batch_size)):
        p.error("epochs, batch sizes, layers, d-model and sphere-scales must be positive")
    if (a.query_k is not None and a.query_k < 0) or a.top_k < 0 or (a.steps_per_epoch is not None and a.steps_per_epoch < 1):
        p.error("query-k/top-k must be nonnegative and steps-per-epoch must be positive")
    if a.neighbors is None:
        a.neighbors = [10] * a.layers
    if len(a.neighbors) != a.layers or any(n < -1 for n in a.neighbors):
        p.error("--neighbors needs one fanout per layer: -1=all, 0=none, positive=sample")
    if (a.location_weight < 0 or a.audio_weight < 0
            or not 0 <= a.species_location_mask_prob <= 1
            or not 0 <= a.negative_replacement_probability <= 1):
        p.error("Loss weights must be nonnegative and probabilities must be in [0, 1]")
    if a.negative_replacement_probability and a.negative_embeddings is None:
        p.error("Negative replacement requires --negative-embeddings")
    if a.geolocation and not a.eval_only and a.location_weight == 0:
        p.error("--geolocation requires a trained location head: set --location-weight > 0")
    if a.eval_only and a.train_nodes is not None:
        p.error("Evaluation restores reference node IDs from the run; do not pass --train-nodes")
    if not a.eval_only and a.query_k is None:
        a.query_k = 10
    return a


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _indices(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    return value.astype(np.int64, copy=False).reshape(-1)


def _subset_graph(graph, ids):
    """Retain global target columns while remapping node/audio/edge indices."""
    ids = _indices(ids, "train-nodes")
    if not len(ids) or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= graph.num_nodes:
        raise ValueError("train-nodes must be nonempty unique valid original node IDs")
    remap = np.full(graph.num_nodes, -1, dtype=np.int64)
    remap[ids] = np.arange(len(ids))
    edges = (remap[graph.row] >= 0) & (remap[graph.col] >= 0)
    audio = remap[graph.audio_node] >= 0
    contexts = graph.contexts[ids] if len(graph.contexts) == graph.num_nodes else graph.contexts
    return type(graph)(remap[graph.row[edges]], remap[graph.col[edges]],
                       graph.audio[audio], remap[graph.audio_node[audio]], contexts,
                       graph.labels[ids], graph.coords[ids], graph.multilabels[ids],
                       None if graph.is_mixup is None else graph.is_mixup[ids])


def _audio_groups(graph):
    nodes = _indices(graph.audio_node, "audio_embedding_node_index")
    if (len(nodes) != len(graph.audio) or not len(nodes)
            or nodes.min() < 0 or nodes.max() >= graph.num_nodes):
        raise ValueError("Invalid audio-to-node mapping")
    counts = np.bincount(nodes, minlength=graph.num_nodes)
    if (counts == 0).any():
        raise ValueError("Every training reference node needs an audio embedding")
    order = np.argsort(nodes, kind="stable")
    ptr = np.concatenate(([0], np.cumsum(counts)))
    return [order[ptr[n]:ptr[n + 1]] for n in range(graph.num_nodes)]


def _signature(graph, original_ids):
    digest = hashlib.sha256()
    for value in (original_ids, graph.row, graph.col, graph.coords, graph.audio_node):
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(memoryview(value).cast("B"))
    digest.update(str((graph.audio.shape, graph.multilabels.shape)).encode())
    return digest.hexdigest()


def _sample_subgraph(incoming, seeds, fanouts, rng):
    """Sample an incoming computation graph; seed nodes are a fixed prefix.

    A node's incoming edges are sampled when first reached by BFS. This graph
    is then passed through the existing homogeneous GNN, without compiled PyG
    sampling extensions. With all fanouts=-1, seed outputs match full-graph
    inference for node-wise normalization and the same number of GNN layers.
    """
    nodes = [int(n) for n in seeds]
    local = {n: i for i, n in enumerate(nodes)}
    frontier, expanded = nodes.copy(), set()
    sources, targets = [], []
    for fanout in fanouts:
        next_frontier = []
        for target in frontier:
            if target in expanded:
                continue
            expanded.add(target)
            start, stop = incoming.indptr[target:target + 2]
            neighbors = incoming.indices[start:stop]
            if fanout >= 0 and len(neighbors) > fanout:
                neighbors = neighbors[rng.choice(len(neighbors), size=fanout, replace=False)]
            for source in neighbors:
                source = int(source)
                if source not in local:
                    local[source] = len(nodes)
                    nodes.append(source)
                sources.append(local[source]); targets.append(local[target])
                next_frontier.append(source)
        frontier = next_frontier
    return np.asarray(nodes, dtype=np.int64), np.asarray([sources, targets], dtype=np.int64).reshape(2, -1)


class TrainingBatches:
    """Fresh subgraphs and one random audio window per sampled node each step."""
    def __init__(self, graph, incoming, node_audio, location, fanouts,
                 batch_size, steps, seed, audio_only=False, negative=None, negative_probability=0.0):
        self.graph, self.incoming, self.node_audio = graph, incoming, node_audio
        self.location, self.fanouts = location, fanouts
        self.batch_size, self.steps = batch_size, steps
        self.rng = np.random.default_rng(seed)
        self.audio_only, self.negative, self.negative_probability = audio_only, negative, negative_probability

    def __len__(self):
        return self.steps or math.ceil(self.graph.num_nodes / self.batch_size)

    def __iter__(self):
        import torch
        g, rng = self.graph, self.rng
        order = rng.permutation(g.num_nodes) if self.steps is None else None
        for step in range(len(self)):
            if order is None:
                seeds = rng.choice(g.num_nodes, min(self.batch_size, g.num_nodes), replace=False)
            else:
                seeds = order[step * self.batch_size:(step + 1) * self.batch_size]
            nodes, edges = _sample_subgraph(self.incoming, seeds, self.fanouts, rng)
            audio_ids = [self.node_audio[n][int(rng.integers(len(self.node_audio[n])))] for n in nodes]
            audio = np.asarray(g.audio[audio_ids], dtype=np.float32).copy()
            targets = g.multilabels[seeds].astype(np.float32, copy=True)
            negative_mask = rng.random(len(seeds)) < self.negative_probability
            if negative_mask.any():
                selected = np.flatnonzero(negative_mask)
                audio[selected] = self.negative[rng.integers(len(self.negative), size=len(selected))]
                targets[selected] = 0
            features = audio if self.audio_only else np.concatenate((audio, self.location[nodes]), axis=1)
            yield {"node_features": torch.from_numpy(features), "edge_index": torch.from_numpy(edges),
                   "indices": torch.arange(len(seeds)), "species_multilabel": torch.from_numpy(targets),
                   "coords": torch.from_numpy(g.coords[seeds].astype(np.float32)),
                   "audio": torch.from_numpy(audio[:len(seeds)].copy()),
                   "negative_mask": torch.from_numpy(negative_mask)}


def _model_codes(path, num_classes):
    mapping = json.loads(Path(path).read_text())
    if isinstance(mapping, dict):
        try:
            codes = [mapping[str(i)] for i in range(num_classes)]
        except KeyError as exc:
            raise ValueError("Perch mapping does not cover every model class index") from exc
    elif isinstance(mapping, list) and len(mapping) >= num_classes:
        codes = mapping[:num_classes]
    else:
        raise ValueError("Perch mapping must be an index-keyed dictionary or ordered list")
    if not all(isinstance(code, str) for code in codes) or len(set(codes)) != len(codes):
        raise ValueError("Expected unique eBird code strings in model output-column order")
    return codes


def _split_mapping(split, codes):
    try:
        from birdset_preparation import load_birdset_data
    except ImportError as exc:
        raise ImportError("BirdSet evaluation needs birdset_preparation.load_birdset_data, as in your sample scripts") from exc
    data = load_birdset_data(split)
    feature = data.features["ebird_code"]
    if hasattr(feature, "_int2str"):
        split_codes = list(feature._int2str)
    else:
        split_codes = [feature.int2str(i) for i in range(feature.num_classes)]
    lookup = {code: i for i, code in enumerate(codes)}
    result = [lookup.get(code, -1) for code in split_codes]
    if all(index < 0 for index in result):
        raise ValueError(f"No {split} eBird codes map into the model vocabulary")
    return result


def _birdset_loader(args, split, graph, node_audio, edges, codes, scales):
    from torch.utils.data import DataLoader
    from graph_transformer.gnn_birdset import GNNBirdSetDataset
    dataset = GNNBirdSetDataset(
        args.birdset_dir / f"{split}.pkl", _split_mapping(split, codes), graph.multilabels.shape[1],
        node_audio, graph.audio, graph.coords, scales,
        reference_edge_index=edges, query_k=args.query_k)
    return DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)


def _negative_audio(path, dim):
    if path is None:
        return None
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            audio = loaded["embeddings"].astype(np.float32)
        finally:
            loaded.close()
    else:
        audio = np.asarray(loaded, dtype=np.float32)
    if audio.ndim != 2 or not len(audio) or audio.shape[1] != dim or not np.isfinite(audio).all():
        raise ValueError(f"Negative embeddings must be finite, nonempty [M, {dim}]")
    return audio


def main(argv=None):
    a = parse_args(argv)
    import torch
    from graph_transformer.data import GraphData
    from graph_transformer.gnn import GNNModel
    from graph_transformer.gnn_trainer import GNNTrainer
    from graph_transformer.gnn_birdset import (
        _sphere_features, _incoming_graph, _validate_coords, evaluate_birdset, evaluate_geolocation)

    if a.device == "auto":
        a.device = ("cuda" if torch.cuda.is_available() else
                    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    run = None
    checkpoint = None
    if a.eval_only:
        # Only load trusted checkpoints produced by this code.
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        config_path = a.checkpoint.parent / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError("run_config.json must accompany the checkpoint; use the original run directory")
        run = json.loads(config_path.read_text())
    elif (a.checkpoint_dir / "last.pt").exists():
        raise FileExistsError("checkpoint-dir already contains last.pt; choose a new run directory")

    with np.load(a.graph_dir / "node_data.npz", allow_pickle=False) as raw:
        if (raw["labels"] < 0).any() and "multilabels" not in raw.files:
            raise ValueError("Scalar labels -1 need explicit multilabels in node_data.npz")
    g = GraphData.load(a.graph_dir)
    if not g.num_nodes:
        raise ValueError("Training/reference graph is empty")
    if g.audio.ndim != 2 or g.multilabels.ndim != 2 or len(g.multilabels) != g.num_nodes:
        raise ValueError("Expected audio [M, D] and species targets [N, C]")
    if not np.isfinite(g.multilabels).all() or ((g.multilabels < 0) | (g.multilabels > 1)).any():
        raise ValueError("Species targets must be finite values in [0, 1]")
    if (g.labels >= g.multilabels.shape[1]).any():
        raise ValueError("Scalar labels exceed the target class vocabulary")
    _validate_coords(g.coords, "Graph coordinates")
    g.row, g.col = _indices(g.row, "row"), _indices(g.col, "col")
    g.audio_node = _indices(g.audio_node, "audio_embedding_node_index")
    if (len(g.row) != len(g.col) or (len(g.row) and
            (min(g.row.min(), g.col.min()) < 0 or max(g.row.max(), g.col.max()) >= g.num_nodes))):
        raise ValueError("Graph edges contain invalid node IDs")
    original_ids = np.arange(g.num_nodes, dtype=np.int64)
    if a.eval_only and run["reference_subset"]:
        original_ids = np.load(a.checkpoint.parent / "reference_node_ids.npy", allow_pickle=False)
        g = _subset_graph(g, original_ids)
    elif a.train_nodes is not None:
        original_ids = np.load(a.train_nodes, allow_pickle=False)
        g = _subset_graph(g, original_ids)
    signature = _signature(g, original_ids)
    if run is not None and signature != run["graph_signature"]:
        raise ValueError("Reference graph does not match the run's node IDs, topology, coordinates or audio mapping")
    convention = run["edge_convention"] if run is not None else a.edge_convention
    source, target = (g.col, g.row) if convention == "row_neighbors" else (g.row, g.col)
    edges = np.stack((source, target)).astype(np.int64)
    incoming = _incoming_graph(source, target, g.num_nodes)
    node_audio = _audio_groups(g)
    scales = run["sphere_scales"] if run is not None else a.sphere_scales
    location = _sphere_features(g.coords, scales)
    if checkpoint is not None:
        model_config = checkpoint["model_config"]
        if model_config["num_classes"] != g.multilabels.shape[1] or model_config["audio_dim"] != g.audio.shape[1]:
            raise ValueError("Checkpoint and graph audio/class dimensions differ")
        model = GNNModel(**model_config).to(a.device)
        model.load_state_dict(checkpoint["model_state"])
        codes = run["model_codes"]
        if codes is None:
            if a.perch_label_mapping is None:
                raise ValueError("This run has no model codes; supply --perch-label-mapping")
            codes = _model_codes(a.perch_label_mapping, g.multilabels.shape[1])
        if a.geolocation and checkpoint.get("weights", [1, 0, 0])[1] <= 0:
            raise ValueError("This checkpoint's location head was not trained")
    else:
        codes = None
        if a.birdset_dir is not None:
            if a.perch_label_mapping is None:
                raise ValueError("--birdset-dir requires --perch-label-mapping, as in your sample scripts")
            codes = _model_codes(a.perch_label_mapping, g.multilabels.shape[1])
        dim = g.audio.shape[1]
        model_config = dict(feature_dim=dim if a.audio_only else dim + location.shape[1],
                            audio_dim=dim, hidden_dim=a.d_model, num_classes=g.multilabels.shape[1],
                            gnn_type=a.gnn_type, num_heads=a.heads, dropout=a.dropout,
                            aggregate=a.aggregate, audio_proj=a.audio_weight > 0, num_layers=a.layers)
        model = GNNModel(**model_config).to(a.device)
        a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run = {"graph_signature": signature, "reference_subset": a.train_nodes is not None,
               "edge_convention": convention, "sphere_scales": scales, "model_codes": codes,
               "query_k": a.query_k,
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}}
        (a.checkpoint_dir / "run_config.json").write_text(json.dumps(run, indent=2))
        if a.train_nodes is not None:
            np.save(a.checkpoint_dir / "reference_node_ids.npy", original_ids)
        validation_loader = None
        if a.birdset_dir is not None:
            validation_loader = _birdset_loader(a, "POW", g, node_audio, edges, codes, scales)
        batches = TrainingBatches(g, incoming, node_audio, location, a.neighbors,
                                  a.batch_size, a.steps_per_epoch, a.seed, a.audio_only,
                                  _negative_audio(a.negative_embeddings, dim), a.negative_replacement_probability)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        trainer = GNNTrainer(model, optimizer, device=a.device,
                             weights=(1.0, a.location_weight, a.audio_weight),
                             validation_loader=validation_loader, checkpoint_dir=a.checkpoint_dir,
                             top_k=a.top_k, species_location_mask_prob=a.species_location_mask_prob,
                             checkpoint_metric=a.checkpoint_metric, species_loss=a.species_loss)
        print(json.dumps({"nodes": g.num_nodes, "edges": len(source), "classes": g.multilabels.shape[1],
                          "device": a.device, "fanouts": a.neighbors, "steps_per_epoch": len(batches)}), flush=True)
        saved = trainer.fit(batches, a.epochs)
        if saved:
            shutil.copyfile(max(saved, key=lambda item: item[0])[1], a.checkpoint_dir / "best.pt")
        if a.skip_test or a.birdset_dir is None:
            return

    # Test-only mode uses the requested checkpoint; training tests best and last.
    checkpoints = [("checkpoint", a.checkpoint)] if a.eval_only else [
        (name, a.checkpoint_dir / f"{name}.pt") for name in ("best", "last")
        if (a.checkpoint_dir / f"{name}.pt").exists()]
    # Restore the run's retrieval setting unless the user explicitly overrides it.
    if a.eval_only and a.query_k is None:
        a.query_k = run["query_k"]
    for checkpoint_name, path in checkpoints:
        state = torch.load(path, map_location=a.device, weights_only=True)
        model.load_state_dict(state["model_state"])
        results = {}
        for split in a.splits:
            if not (a.birdset_dir / f"{split}.pkl").exists():
                print(f"Skipping {split}: pickle not found", flush=True)
                continue
            loader = _birdset_loader(a, split, g, node_audio, edges, codes, scales)
            metrics = evaluate_birdset(model, loader, a.device)
            results[split] = metrics
            print(checkpoint_name, split, metrics, flush=True)
            if a.geolocation:
                geo = evaluate_geolocation(model, loader, a.device)
                results[split + "_geolocation"] = geo
                print(checkpoint_name, split, "geolocation", geo, flush=True)
        result_path = path.parent / f"{path.stem}_test_results.json"
        result_path.write_text(json.dumps({"checkpoint": str(path), "query_k": a.query_k,
                                          "splits": results}, indent=2))
        print("Results:", result_path, flush=True)


if __name__ == "__main__":
    main()
