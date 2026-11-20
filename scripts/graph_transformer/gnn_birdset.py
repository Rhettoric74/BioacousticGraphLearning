from __future__ import annotations

from pathlib import Path
import pickle
import warnings

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import BallTree


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _integer_array(value, name):
    value = _numpy(value)
    if value.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    return value.astype(np.int64, copy=False)


def _validate_coords(coords, name):
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError(f"{name} must be finite [N, 2] latitude/longitude degrees")
    if (np.abs(coords[:, 0]) > 90).any() or (np.abs(coords[:, 1]) > 180).any():
        raise ValueError(f"{name} must be [latitude, longitude] in degrees")


def _sphere_features(coords, scales):
    """Same fixed SphereM basis as the supplied location.py, in float32.

    Kept local so evaluation does not invoke the unconditional debug exception
    in the uploaded encoder. No learned location-encoder parameters are replaced.
    """
    if not isinstance(scales, (int, np.integer)) or scales < 1:
        raise ValueError("sphere_scales must be a positive integer")
    radians = np.deg2rad(np.asarray(coords, dtype=np.float32))
    phi = np.clip(radians[:, 0], -np.pi / 2, np.pi / 2)
    lam = radians[:, 1]
    pieces = []
    for s in range(scales):
        phi_s, lam_s = phi / (2.0 ** s), lam / (2.0 ** s)
        pieces.extend((np.sin(phi_s), np.cos(phi_s) * np.cos(lam),
                       np.cos(phi) * np.cos(lam_s), np.cos(phi_s) * np.sin(lam),
                       np.cos(phi) * np.sin(lam_s)))
    return np.stack(pieces, axis=-1).astype(np.float32)


def _feature_parts(value, name):
    """Accept a flat [N, D] matrix or the existing list of [B_i, D] batches."""
    if isinstance(value, np.ndarray) and value.dtype != object and value.ndim == 2:
        parts = [value]
    else:
        parts = [_numpy(part) for part in value]
        if parts and all(part.ndim == 1 for part in parts):
            parts = [np.stack(parts)]
    if name == "st_context":
        parts = [part.reshape(len(part), -1) if part.ndim >= 2 and len(part) else part
                 for part in parts]
    if not parts or any(part.ndim != 2 or part.shape[1] < 1 for part in parts):
        raise ValueError(f"{name} must be [N, D] or batches of [B, D]")
    if len({part.shape[1] for part in parts}) != 1:
        raise ValueError(f"{name} batches have inconsistent feature dimensions")
    if sum(len(part) for part in parts) == 0:
        raise ValueError(f"{name} contains no samples")
    return parts


def _label_set(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return [int(value)]
    if isinstance(value, (list, tuple, np.ndarray)):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return _label_set(value.item())
        values = list(value)
        if all(isinstance(v, (int, np.integer))
               and not isinstance(v, (bool, np.bool_)) for v in values):
            return [int(v) for v in values]
    return None


def _flatten_labels(value, batch_sizes):
    """Resolve nesting rather than using only the number of outer label lists."""
    values = list(value)
    total = sum(batch_sizes)
    flat = [_label_set(v) for v in values]
    if len(values) == total and all(v is not None for v in flat):
        return flat
    if len(values) == len(batch_sizes):
        result = []
        for batch, size in zip(values, batch_sizes):
            if not isinstance(batch, (list, tuple, np.ndarray)) or len(batch) != size:
                break
            sets = [_label_set(v) for v in batch]
            if any(v is None for v in sets):
                break
            result.extend(sets)
        else:
            return result
    raise ValueError("labels must be flat per-sample class-index lists or matching batches of those lists")


class GNNBirdSetDataset(torch.utils.data.Dataset):

    def __init__(self, path, subset_labels, num_model_classes,
                 node_audio, audio_embeddings, coords, sphere_scales=8, *,
                 reference_edge_index=None, reference_k=10, query_k=10,
                 edge_convention="source_to_target"):
        if num_model_classes < 1:
            raise ValueError("num_model_classes must be positive")
        if any(not isinstance(k, (int, np.integer)) or k < 0 for k in (reference_k, query_k)):
            raise ValueError("reference_k and query_k must be nonnegative integers")
        if edge_convention not in ("source_to_target", "row_neighbors"):
            raise ValueError("edge_convention must be source_to_target or row_neighbors")
        mapping = _integer_array(subset_labels, "subset_labels")
        if mapping.ndim != 1 or not len(mapping) or ((mapping < -1) | (mapping >= num_model_classes)).any():
            raise ValueError("subset_labels must be a nonempty 1D mapping into [-1, num_model_classes)")
        with open(Path(path), "rb") as f:
            data = pickle.load(f)
        audio_parts = _feature_parts(data["embeddings"], "embeddings")
        context_parts = _feature_parts(data["st_context"], "st_context")
        self.audio = np.concatenate(audio_parts).astype(np.float32)
        context = np.concatenate(context_parts).astype(np.float32)
        if len(context) != len(self.audio) or context.shape[1] < 2:
            raise ValueError("BirdSet st_context must have one row per sample and at least lon/lat columns")
        if not np.isfinite(self.audio).all():
            raise ValueError("BirdSet embeddings must be finite")
        self.coords = context[:, [1, 0]].copy()
        _validate_coords(self.coords, "BirdSet coordinates")
        self.location = _sphere_features(self.coords, sphere_scales)
        raw_labels = _flatten_labels(data["labels"], [len(part) for part in audio_parts])
        self.labels = np.zeros((len(self.audio), num_model_classes), dtype=np.float32)
        self.subset_labels = mapping
        self.n_unmapped_label_occurrences = 0
        for i, local_labels in enumerate(raw_labels):
            for local in local_labels:
                if not 0 <= local < len(mapping):
                    raise ValueError(f"BirdSet local class {local} is outside subset_labels")
                model_class = int(mapping[local])
                if model_class >= 0:
                    self.labels[i, model_class] = 1
                else:
                    self.n_unmapped_label_occurrences += 1

        self.reference_coords = _numpy(coords).astype(np.float32)
        _validate_coords(self.reference_coords, "Reference coordinates")
        if not len(self.reference_coords):
            raise ValueError("At least one training reference node is required")
        self.reference_audio = _numpy(audio_embeddings)
        if (self.reference_audio.ndim != 2 or self.reference_audio.shape[1] != self.audio.shape[1]
                or self.reference_audio.dtype.kind not in "fiu"):
            raise ValueError("Training and BirdSet audio embedding dimensions must match")
        if len(node_audio) != len(self.reference_coords):
            raise ValueError("node_audio and reference coordinates have different node counts")
        audio_ids = []
        for node, ids in enumerate(node_audio):
            ids = _integer_array(ids, f"node_audio[{node}]")
            if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(self.reference_audio):
                raise ValueError(f"Reference node {node} needs valid, nonempty audio-row indices")
            audio_ids.append(ids)
        self.node_audio = audio_ids
        self.sphere_scales = sphere_scales
        self.reference_k, self.query_k = int(reference_k), int(query_k)
        self.edge_convention = edge_convention
        self.edge_index = reference_edge_index  # Reference-node IDs only, never batch-local query IDs.
        self._tree = None
        self._fallback_graph = None
        self._warned_fallback = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"audio": torch.from_numpy(self.audio[index]),
                "location": torch.from_numpy(self.location[index]),
                "coords": torch.from_numpy(self.coords[index]),
                "labels": torch.from_numpy(self.labels[index])}

    def _reference_tree(self):
        if self._tree is None:
            self._tree = BallTree(np.deg2rad(self.reference_coords), metric="haversine")
        return self._tree

    def query_neighbors(self, coords):
        k = min(self.query_k, len(self.reference_coords))
        if k == 0:
            return np.empty((len(coords), 0), dtype=np.int64)
        return self._reference_tree().query(np.deg2rad(coords), k=k, return_distance=False)

    def reference_graph(self, model):
        edges = self.edge_index
        convention = self.edge_convention
        if edges is None:
            edges = getattr(model, "edge_index", None)
            convention = "source_to_target"  # model edges already follow PyG's contract.
        r = len(self.reference_coords)
        if edges is not None:
            edges = _integer_array(edges, "reference_edge_index")
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError("reference_edge_index must have shape [2, E]")
            if edges.size and (edges.min() < 0 or edges.max() >= r):
                raise ValueError("Reference edges must index only the supplied training reference nodes")
            source, target = edges if convention == "source_to_target" else edges[::-1]
            return _incoming_graph(source, target, r)
        if not self._warned_fallback:
            warnings.warn("No training reference edges supplied: rebuilding geographic KNN "
                          f"with reference_k={self.reference_k}. Pass reference_edge_index "
                          "to evaluate on the exact training graph.", UserWarning, stacklevel=2)
            self._warned_fallback = True
        if self._fallback_graph is None:
            k = min(self.reference_k, r - 1)
            sources, targets = [], []
            if k:
                neighbors = self._reference_tree().query(
                    np.deg2rad(self.reference_coords), k=k + 1, return_distance=False)
                for target, choices in enumerate(neighbors):
                    choices = choices[choices != target][:k]
                    sources.extend(choices)
                    targets.extend([target] * len(choices))
            self._fallback_graph = _incoming_graph(sources, targets, r)
        return self._fallback_graph

    def reference_features(self, nodes):
        # Materialize only the reference nodes needed by this batch's L-hop graph.
        audio = np.stack([self.reference_audio[self.node_audio[int(n)]].mean(axis=0, dtype=np.float32)
                          for n in nodes]).astype(np.float32)
        if not np.isfinite(audio).all():
            raise ValueError("Reference audio embeddings contain non-finite values")
        return audio, _sphere_features(self.reference_coords[nodes], self.sphere_scales)


def _incoming_graph(source, target, num_nodes):
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    # SAGE uses its root term; GATv2 / the GCN-style wrapper add self internally.
    keep = source != target
    source, target = source[keep], target[keep]
    order = np.argsort(target, kind="stable")
    ptr = np.concatenate(([0], np.cumsum(np.bincount(target, minlength=num_nodes))))
    # Preserve repeated edges and per-destination order, as the model does.
    # COO construction or sum_duplicates would silently change their weights;
    # sorting source IDs would also change the optional LSTM aggregation order.
    return csr_matrix((np.ones(len(source), dtype=np.float32), source[order], ptr),
                      shape=(num_nodes, num_nodes))


def _query_subgraph(reference_graph, query_neighbors, depth):
    """Exact incoming L-hop closure; first B nodes are always the B queries.

    Return virtual node IDs and local source -> target edges. Virtual reference
    IDs are [0, R); query IDs are [R, R+B). No edge can have a query as source.
    """
    r = reference_graph.shape[0]
    nodes = list(range(r, r + len(query_neighbors)))
    local = {node: i for i, node in enumerate(nodes)}
    frontier, expanded = nodes.copy(), set()
    sources, targets = [], []
    for _ in range(depth):
        next_frontier = []
        for target in frontier:
            if target in expanded:
                continue
            expanded.add(target)
            if target >= r:
                neighbors = query_neighbors[target - r]
            else:
                start, stop = reference_graph.indptr[target:target + 2]
                neighbors = reference_graph.indices[start:stop]
            for source in neighbors:
                source = int(source)
                if source not in local:
                    local[source] = len(nodes)
                    nodes.append(source)
                sources.append(local[source])
                targets.append(local[target])
                next_frontier.append(source)
        frontier = next_frontier
    return np.asarray(nodes, dtype=np.int64), np.asarray([sources, targets], dtype=np.int64).reshape(2, -1)


def _root_dataset(loader):
    dataset = loader.dataset
    while not isinstance(dataset, GNNBirdSetDataset) and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if not isinstance(dataset, GNNBirdSetDataset):
        raise TypeError("Use GNNBirdSetDataset (or a Subset of it) for reference-graph evaluation")
    return dataset


def _model_features(model, audio, location):
    audio = np.asarray(audio, dtype=np.float32)
    location = np.asarray(location, dtype=np.float32)
    d, l = audio.shape[1], location.shape[1]
    if model.audio_dim != d:
        raise ValueError(f"Model audio_dim={model.audio_dim}, but Perch dimension is {d}; "
                         "set audio_dim explicitly for concatenated audio + location inputs")
    if model.feature_dim == d:
        return audio
    if model.feature_dim == d + l:
        return np.concatenate((audio, location), axis=-1)
    raise ValueError(f"Model feature_dim must be {d} (audio) or {d + l} (audio + SphereM)")


def _haversine_km(pred_coords, true_coords):
    pred, true = np.deg2rad(pred_coords), np.deg2rad(true_coords)
    dlat = pred[:, 0] - true[:, 0]
    dlon = np.arctan2(np.sin(pred[:, 1] - true[:, 1]), np.cos(pred[:, 1] - true[:, 1]))
    a = np.sin(dlat / 2) ** 2 + np.cos(pred[:, 0]) * np.cos(true[:, 0]) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def top1_from_full_logits(full_logits, labels):
    has_label = labels.sum(axis=1) > 0
    if not has_label.any():
        return float("nan")
    correct = labels[np.arange(len(labels)), full_logits.argmax(axis=1)] > 0
    return float(correct[has_label].mean())


def _classification_metrics(logits, labels):
    if logits.ndim != 2 or logits.shape != labels.shape or not len(labels):
        raise ValueError("Predictions and labels must have matching nonempty [N, C] shapes")
    if not np.isfinite(logits).all():
        raise ValueError("Model produced non-finite species logits")
    probabilities = expit(logits)
    aucs, aps = [], []
    for c in range(labels.shape[1]):
        if labels[:, c].min() == labels[:, c].max():
            continue
        aucs.append(roc_auc_score(labels[:, c], probabilities[:, c]))
        aps.append(average_precision_score(labels[:, c], probabilities[:, c]))
    return {"macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
            "cmAP": float(np.mean(aps)) if aps else float("nan"),
            "top1": top1_from_full_logits(logits, labels),
            "n_valid_classes": len(aucs), "n_samples": len(labels)}


@torch.no_grad()
def evaluate_birdset(model, loader, device="cuda"):
    """Predict each query from itself and its incoming training-reference context.

    Works for shuffled/batched loaders: each batch gets a freshly indexed exact
    L-hop subgraph, and only its first B query outputs enter the metrics.
    """
    dataset = _root_dataset(loader)
    reference_graph = dataset.reference_graph(model)
    depth = int(model.num_layers)
    if depth < 1:
        raise ValueError("GNN depth must be positive")
    was_training = model.training
    model.eval()
    full_logits, labels = [], []
    try:
        dtype = next(model.parameters()).dtype
        for batch in loader:
            audio, location, coords = (_numpy(batch[key]) for key in ("audio", "location", "coords"))
            _validate_coords(coords, "Query coordinates")
            if not len(audio):
                continue
            neighbors = dataset.query_neighbors(coords)
            nodes, edges = _query_subgraph(reference_graph, neighbors, depth)
            b = len(audio)
            if len(nodes) > b:
                ref_audio, ref_location = dataset.reference_features(nodes[b:])
                audio = np.concatenate((audio, ref_audio))
                location = np.concatenate((location, ref_location))
            features = _model_features(model, audio, location)
            out = model(torch.as_tensor(features, dtype=dtype, device=device),
                        torch.as_tensor(edges, dtype=torch.long, device=device))
            full_logits.append(out["species"][:b].detach().float().cpu().numpy())
            labels.append(_numpy(batch["labels"]))
    finally:
        model.train(was_training)
    if not labels:
        raise ValueError("Evaluation loader yielded no samples")
    return _classification_metrics(np.concatenate(full_logits), np.concatenate(labels))


@torch.no_grad()
def evaluate_geolocation(model, loader, device="cuda"):
    """Audio-only coordinate prediction; the location head must have been trained.

    Query location features are zero and the graph has no geographic edges.
    True coordinates are used ONLY to score predictions, never for retrieval or
    feature construction. This is an isolated-node baseline, not geographic
    graph-conditioned reconstruction. Classification uses geographic context in
    evaluate_birdset; such context would leak the target in this task.
    """
    was_training = model.training
    model.eval()
    predicted, truth = [], []
    try:
        dtype = next(model.parameters()).dtype
        for batch in loader:
            audio = _numpy(batch["audio"])
            if not len(audio):
                continue
            # Preserve the trained input width while hiding all location columns.
            location = np.zeros((len(audio), model.feature_dim - model.audio_dim), dtype=np.float32)
            features = _model_features(model, audio, location)
            out = model(torch.as_tensor(features, dtype=dtype, device=device),
                        torch.empty((2, 0), dtype=torch.long, device=device), mask_location=True)
            predicted.append(out["coordinates"].detach().float().cpu().numpy())
            truth.append(_numpy(batch["coords"]))
    finally:
        model.train(was_training)
    if not truth:
        raise ValueError("Geolocation loader yielded no samples")
    predicted, truth = np.concatenate(predicted), np.concatenate(truth)
    _validate_coords(truth, "True coordinates")
    if predicted.shape != truth.shape or not np.isfinite(predicted).all():
        raise ValueError("Coordinate predictions must be finite [N, 2]")
    errors = _haversine_km(predicted, truth)
    return {"median_error_km": float(np.median(errors)), "mean_error_km": float(np.mean(errors)),
            "p90_error_km": float(np.percentile(errors, 90)), "n_samples": len(errors)}
