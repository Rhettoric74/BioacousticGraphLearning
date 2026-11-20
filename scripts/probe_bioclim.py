#!/usr/bin/env python3
"""Linear-probe frozen location representations against BioClim rasters.

Rasters must use the standard geographic CRS (EPSG:4326). Coordinates in the
graph are [latitude, longitude], while rasterio samples [longitude, latitude].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import rasterio

from graph_transformer.data import GraphData
from graph_transformer.location import Sphere2VecSphereM
from graph_transformer.model import GraphWalkTransformer


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("graph_dir", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--rasters", nargs="+", type=Path, required=True,
                   help="BioClim GeoTIFF files, one variable per file")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--sphere-scales", type=int, default=8)
    p.add_argument("--walk-length", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-samples", type=int, default=100000)
    p.add_argument("--test-latitude-bin", type=float, default=10.0)
    p.add_argument("--test-longitude-bin", type=float, default=10.0)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sample_bioclim(rasters, coords):
    values = []
    for path in rasters:
        with rasterio.open(path) as src:
            if src.crs is not None and src.crs.to_epsg() != 4326:
                raise ValueError(f"{path} must be in EPSG:4326, got {src.crs}")
            samples = np.asarray(list(src.sample([(float(lon), float(lat))
                                                   for lat, lon in coords])))
            nodata = src.nodata
            if samples.ndim != 2 or samples.shape[1] != 1:
                raise ValueError(f"Expected one-band raster: {path}")
            values.append(samples[:, 0])
    return np.stack(values, axis=1).astype(np.float32)


def spatial_split(coords, lat_bin, lon_bin):
    cells = np.floor(np.column_stack((coords[:, 0] / lat_bin,
                                      coords[:, 1] / lon_bin))).astype(np.int64)
    unique = np.unique(cells, axis=0)
    rng = np.random.default_rng(0)
    rng.shuffle(unique)
    n_train = max(1, int(0.7 * len(unique)))
    n_val = max(1, int(0.15 * len(unique)))
    cell_keys = {tuple(cell): i for i, cell in enumerate(unique)}
    ids = np.asarray([cell_keys[tuple(cell)] for cell in cells])
    return ids < n_train, (ids >= n_train) & (ids < n_train + n_val), ids >= n_train + n_val


@torch.no_grad()
def extract_location_features(model, location, batch_size, device):
    model.eval()
    output = []
    for start in range(0, len(location), batch_size):
        x = torch.from_numpy(location[start:start + batch_size]).to(device)
        output.append(model.location(x).cpu().numpy())
    return np.concatenate(output, axis=0)


def main():
    a = parse_args()
    graph = GraphData.load(a.graph_dir)
    rng = np.random.default_rng(0)
    n = min(a.max_samples, graph.num_nodes)
    indices = rng.choice(graph.num_nodes, n, replace=False)
    coords = np.asarray(graph.coords[indices], dtype=np.float32)
    targets = sample_bioclim(a.rasters, coords)
    valid = np.isfinite(targets).all(axis=1) & np.isfinite(coords).all(axis=1)
    coords, targets = coords[valid], targets[valid]

    location_encoder = Sphere2VecSphereM(a.sphere_scales)
    with torch.no_grad():
        location = location_encoder(torch.from_numpy(coords)).numpy()
    model = GraphWalkTransformer(
        graph.audio.shape[1], location.shape[1], graph.multilabels.shape[1],
        d_model=a.d_model, layers=a.layers, heads=a.heads,
        max_length=a.walk_length,
    ).to(a.device)
    checkpoint = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    features = extract_location_features(model, location, a.batch_size, a.device)

    train, val, test = spatial_split(coords, a.test_latitude_bin, a.test_longitude_bin)
    if not train.any() or not val.any() or not test.any():
        raise ValueError("Spatial split produced an empty partition; adjust geographic bin sizes")
    x_scaler = StandardScaler().fit(features[train])
    y_scaler = StandardScaler().fit(targets[train])
    x = x_scaler.transform(features)
    y = y_scaler.transform(targets)
    probe = Ridge(alpha=a.ridge_alpha)
    probe.fit(x[train], y[train])
    predictions = probe.predict(x)

    results = {"n_samples": int(len(y)), "n_variables": int(y.shape[1]),
               "train_samples": int(train.sum()), "val_samples": int(val.sum()),
               "test_samples": int(test.sum())}
    for name, mask in (("train", train), ("val", val), ("test", test)):
        results[name] = {
            "mean_r2": float(r2_score(y[mask], predictions[mask], multioutput="uniform_average")),
            "mean_mae_standardized": float(mean_absolute_error(y[mask], predictions[mask])),
            "per_variable_r2": [float(v) for v in r2_score(y[mask], predictions[mask], multioutput="raw_values")],
        }
    print(results)
    if a.output:
        import json
        a.output.write_text(json.dumps(results, indent=2))
        print("saved", a.output)


if __name__ == "__main__":
    main()
