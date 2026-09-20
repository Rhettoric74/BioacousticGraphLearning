#!/usr/bin/env python3
"""Load Xeno-Canto/Perch batches, build a geographic KNN graph, and save it.

The output directory contains:
  - graph.npz: directed KNN edges (and distances in kilometres)
  - node_data.npz: unique-node contexts/labels plus all window embeddings and
    an ``audio_embedding_node_index`` mapping each embedding to its node
  - metadata.json: graph construction and feature-shape metadata

By default, latitude and longitude are assumed to be columns 0 and 1 of
``spatiotemporal_contexts``. Singleton dimensions such as ``(N, 1, D)`` are
handled automatically. Use --lat-index/--lon-index when that is not true.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except ImportError:  # NumPy-only batches can still be inspected without torch.
    torch = None


def as_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_coordinates(context: np.ndarray, lat_index: int, lon_index: int) -> np.ndarray:
    """Return an (N, 2) latitude/longitude array from common context shapes."""
    if context.ndim < 2:
        raise ValueError(f"Contexts must have at least two dimensions; got {context.shape}")

    # Common format: one singleton axis per sample, e.g. (N, 1, D).
    if context.ndim == 3 and context.shape[1] == 1:
        context_for_coords = context[:, 0, :]
    # Also support (N, D, 1), although this is less common.
    elif context.ndim == 3 and context.shape[2] == 1:
        context_for_coords = context[:, :, 0]
    else:
        context_for_coords = context

    if context_for_coords.ndim != 2:
        raise ValueError(
            f"Could not reduce contexts with shape {context.shape} to (N, D). "
            "Inspect spatiotemporal_contexts and set the coordinate indices accordingly."
        )
    n_features = context_for_coords.shape[1]
    if not (0 <= lat_index < n_features and 0 <= lon_index < n_features):
        raise IndexError(
            f"Coordinate indices ({lat_index}, {lon_index}) are invalid for "
            f"contexts with {n_features} features after removing singleton dimensions."
        )
    return context_for_coords[:, [lat_index, lon_index]]


def context_features(context: np.ndarray) -> np.ndarray:
    """Remove the singleton context axis and return an (N, D) feature matrix."""
    if context.ndim == 3 and context.shape[1] == 1:
        context = context[:, 0, :]
    elif context.ndim == 3 and context.shape[2] == 1:
        context = context[:, :, 0]
    if context.ndim != 2:
        raise ValueError(f"Could not reduce context shape to (N, D): {context.shape}")
    return context


def load_nodes(input_dir: Path, lat_index: int, lon_index: int,
               context_decimals: int):
    audio_parts, context_parts, label_parts, coords_parts = [], [], [], []
    totals = {"raw": 0, "length_mismatch": 0, "invalid_audio": 0,
              "invalid_context": 0, "invalid_coordinates": 0,
              "invalid_temporal": 0, "valid": 0}
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    for path in files:
        try:
            with path.open("rb") as f:
                batch = pickle.load(f)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        required = {"audio_embeddings", "spatiotemporal_contexts", "labels"}
        if not required.issubset(batch):
            continue

        audio = as_numpy(batch["audio_embeddings"])
        context = as_numpy(batch["spatiotemporal_contexts"])
        labels = as_numpy(batch["labels"]).reshape(-1)
        lengths = (len(audio), len(context), len(labels))
        n = min(lengths)
        totals["raw"] += max(lengths)
        totals["length_mismatch"] += max(lengths) - n
        if len(set(lengths)) != 1:
            print(f"{path.name}: LENGTH MISMATCH audio/context/labels={lengths}; using first {n:,}")
        audio, context, labels = audio[:n], context[:n], labels[:n]
        batch_coords = extract_coordinates(context, lat_index, lon_index)
        context_matrix = context_features(context)
        valid_audio = np.isfinite(audio).all(axis=tuple(range(1, audio.ndim)))
        valid_context = np.isfinite(context_matrix).all(axis=1)
        valid_coordinates = np.isfinite(batch_coords).all(axis=1)
        valid_temporal = np.isfinite(context_matrix[:, 2:]).all(axis=1)
        valid = valid_audio & valid_context & valid_coordinates
        totals["invalid_audio"] += int((~valid_audio).sum())
        totals["invalid_context"] += int((~valid_context).sum())
        totals["invalid_coordinates"] += int((~valid_coordinates).sum())
        totals["invalid_temporal"] += int((~valid_temporal).sum())
        totals["valid"] += int(valid.sum())
        if (~valid).any():
            print(f"{path.name}: raw={n:,}, valid={valid.sum():,}, "
                  f"invalid_audio={(~valid_audio).sum():,}, "
                  f"invalid_context={(~valid_context).sum():,}, "
                  f"invalid_coordinates={(~valid_coordinates).sum():,}, "
                  f"invalid_temporal={(~valid_temporal).sum():,}")
        audio_parts.append(audio[valid].astype(np.float32, copy=False))
        context_parts.append(context[valid].astype(np.float32, copy=False))
        label_parts.append(labels[valid].astype(np.int64, copy=False))
        # Keep coordinates aligned with the filtered node arrays.
        coords_parts.append(batch_coords[valid].astype(np.float32, copy=False))

    if not audio_parts:
        raise RuntimeError("No compatible recording batches were found.")
    print("\nLoading summary:")
    print(f"  raw rows (largest field per batch): {totals['raw']:,}")
    print(f"  rows lost to field-length mismatches: {totals['length_mismatch']:,}")
    print(f"  rows with invalid audio embeddings: {totals['invalid_audio']:,}")
    print(f"  rows with invalid context features: {totals['invalid_context']:,}")
    print(f"  rows with invalid coordinates: {totals['invalid_coordinates']:,}")
    print(f"  rows with invalid temporal features: {totals['invalid_temporal']:,}")
    print(f"  rows retained for deduplication: {totals['valid']:,}")
    audio = np.concatenate(audio_parts)
    context = np.concatenate(context_parts)
    labels = np.concatenate(label_parts)
    coords = np.concatenate(coords_parts)
    # Deduplicate by recording-level metadata rather than potentially stale IDs.
    context_matrix = context_features(context)
    # A recording's full encoded spatiotemporal context should be identical
    # across its sampled windows. Include the species label as an additional
    # safeguard, without relying on potentially stale IDs.
    rounded_context = np.round(context_matrix, context_decimals)
    keys = ((tuple(row), int(label)) for row, label in zip(rounded_context, labels))
    key_to_node, first_indices = {}, []
    node_index = np.empty(len(labels), dtype=np.int64)
    for i, key in enumerate(keys):
        node = key_to_node.setdefault(key, len(first_indices))
        if node == len(first_indices):
            first_indices.append(i)
        node_index[i] = node
    first_indices = np.asarray(first_indices, dtype=np.int64)
    counts = np.bincount(node_index, minlength=len(first_indices))
    print(f"  unique deduplicated nodes: {len(first_indices):,}")
    print(f"  rows merged by deduplication: {len(labels) - len(first_indices):,}")
    print(f"  embeddings per node: min={counts.min()}, mean={counts.mean():.4f}, max={counts.max()}")
    return audio, context[first_indices], labels[first_indices], coords[first_indices], node_index


def build_graph(coords: np.ndarray, k: int, include_self: bool, query_batch_size: int = 32768):
    if len(coords) < 2:
        raise ValueError("At least two valid recordings are required.")
    k_query = min(len(coords), k + (0 if include_self else 1))
    radians = np.deg2rad(coords)
    nn = NearestNeighbors(n_neighbors=k_query, metric="haversine", algorithm="ball_tree")
    nn.fit(radians)
    if include_self:
        keep = slice(0, min(k, k_query))
    else:
        keep = slice(1, min(k + 1, k_query))
    n_keep = min(k, k_query)
    row = np.empty(len(coords) * n_keep, dtype=np.int64)
    col = np.empty_like(row)
    km = np.empty(len(row), dtype=np.float32)
    for start in range(0, len(coords), query_batch_size):
        stop = min(start + query_batch_size, len(coords))
        distances, neighbours = nn.kneighbors(radians[start:stop])
        selected = slice(0, n_keep) if include_self else slice(1, n_keep + 1)
        sl = slice(start * n_keep, stop * n_keep)
        row[sl] = np.repeat(np.arange(start, stop), n_keep)
        col[sl] = neighbours[:, selected].reshape(-1)
        km[sl] = distances[:, selected].reshape(-1) * 6371.0088
        print(f"KNN queries: {stop:,}/{len(coords):,}")
    return row, col, km


def load_mixup_nodes(input_dir: Path, num_classes: int):
    """Load the pickle layout in load_mixup_embeddings and make graph nodes."""
    audio, coords, targets = [], [], []
    for path in sorted(input_dir.glob('*.pkl')):
        with path.open('rb') as f: batch = pickle.load(f)
        if 'audio_embeddings' not in batch or 'labels_sparse' not in batch: continue
        a = as_numpy(batch['audio_embeddings'])
        contexts = batch.get('component_contexts')
        if contexts is None and 'component_contexts_flat' in batch:
            # The compact representation is reconstructed inline.
            flat, lengths, shapes = batch['component_contexts_flat'], batch['component_contexts_lengths'], batch['component_contexts_shapes']
            contexts, pos = [], 0
            for length, shape in zip(lengths, shapes):
                row = []
                for _ in range(length):
                    size = int(np.prod(shape)); row.append(flat[pos:pos+size].reshape(shape)); pos += size
                contexts.append(row)
        if contexts is None: continue
        for i, labels in enumerate(batch['labels_sparse']):
            if i >= len(a) or not np.isfinite(a[i]).all() or not contexts[i]: continue
            c = np.asarray([np.asarray(x).reshape(-1)[:2] for x in contexts[i]], dtype=np.float32)
            if not np.isfinite(c).all(): continue
            y = np.zeros(num_classes, dtype=np.float32)
            for label in np.asarray(labels).reshape(-1):
                if 0 <= int(label) < num_classes: y[int(label)] = 1.0
            audio.append(a[i].astype(np.float32)); coords.append(c.mean(0)); targets.append(y)
    if not audio: return None
    return np.stack(audio), np.stack(coords), np.stack(targets)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--mixup-dir", type=Path, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--k", type=int, default=15, help="Number of outgoing geographic neighbours")
    p.add_argument("--lat-index", type=int, default=0)
    p.add_argument("--lon-index", type=int, default=1)
    p.add_argument("--context-decimals", type=int, default=6,
                   help="Decimal places retained for every context feature during deduplication")
    p.add_argument("--include-self", action="store_true")
    p.add_argument("--query-batch-size", type=int, default=32768,
                   help="Number of nodes queried at once; lower this if KNN construction OOMs")
    args = p.parse_args()
    if args.k < 1:
        p.error("-k must be positive")

    audio, context, labels, coords, audio_node_index = load_nodes(
        args.input_dir, args.lat_index, args.lon_index, args.context_decimals
    )
    num_classes = args.num_classes or int(labels.max()) + 1
    multilabels = np.zeros((len(labels), num_classes), dtype=np.float32)
    multilabels[np.arange(len(labels)), labels] = 1.0
    is_mixup = np.zeros(len(labels), dtype=bool)
    if args.mixup_dir:
        mixed = load_mixup_nodes(args.mixup_dir, num_classes)
        if mixed is None: raise RuntimeError('No usable mixup samples found')
        ma, mc, my = mixed; start = len(coords)
        audio = np.concatenate([audio, ma]); coords = np.concatenate([coords, mc])
        labels = np.concatenate([labels, np.full(len(ma), -1, dtype=np.int64)])
        multilabels = np.concatenate([multilabels, my]); is_mixup = np.concatenate([is_mixup, np.ones(len(ma), dtype=bool)])
        context = np.concatenate([context, np.zeros((len(ma),) + context.shape[1:], dtype=context.dtype)])
        audio_node_index = np.concatenate([audio_node_index, np.arange(start, start + len(ma))])
    row, col, distance_km = build_graph(coords, args.k, args.include_self, args.query_batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "graph.npz", row=row, col=col, distance_km=distance_km,
                        num_nodes=np.array(len(coords), dtype=np.int64))
    payload = {"audio_embeddings": audio, "audio_embedding_node_index": audio_node_index,
               "spatiotemporal_contexts": context, "labels": labels,
               "coordinates_lat_lon": coords, "multilabels": multilabels,
               "is_mixup": is_mixup}
    np.savez_compressed(args.output_dir / "node_data.npz", **payload)
    counts = np.bincount(audio_node_index, minlength=len(coords))
    metadata = {"num_nodes": int(len(coords)), "num_edges": int(len(row)),
                "num_input_windows": int(len(audio)), "num_unique_recordings": int(len(coords)),
                "windows_per_recording_min": int(counts.min()),
                "windows_per_recording_max": int(counts.max()),
                "windows_per_recording_mean": float(counts.mean()), "k": args.k,
                "directed": True, "distance_metric": "great-circle haversine",
                "distance_units": "kilometres", "latitude_context_index": args.lat_index,
                "longitude_context_index": args.lon_index, "self_loops": args.include_self,
                "deduplication_key": "rounded full spatiotemporal context + species label",
                "context_decimals": args.context_decimals,
                "audio_embedding_shape": list(audio.shape), "context_shape": list(context.shape),
                "label_dtype": str(labels.dtype)}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(coords):,} nodes and {len(row):,} directed edges to {args.output_dir}")


if __name__ == "__main__":
    main()
