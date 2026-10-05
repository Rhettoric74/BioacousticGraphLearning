#!/usr/bin/env python3
"""Precompute Perch embeddings for a directory of audio files.

The Perch package/API differs across HPC environments. Supply an encoder
adapter as ``module:function``. It receives a list of audio paths and must
return an array/tensor shaped (batch, embedding_dim).
"""
import argparse
import importlib
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio_dir", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--encoder", required=True, help="Python module:function batch encoder")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--metadata-csv", type=Path, nargs="*", default=None,
                   help="FSD50K dev.csv/eval.csv files; filters audio using fname and labels")
    p.add_argument("--exclude-label", action="append", default=[],
                   help="Exclude files containing this FSD50K label; repeat as needed")
    p.add_argument("--extensions", nargs="+", default=[".wav", ".flac", ".ogg", ".mp3"])
    args = p.parse_args()
    module_name, function_name = args.encoder.split(":", 1)
    encode = getattr(importlib.import_module(module_name), function_name)
    paths = sorted(p for ext in args.extensions for p in args.audio_dir.rglob(f"*{ext}"))
    if args.metadata_csv:
        metadata = pd.concat([pd.read_csv(path) for path in args.metadata_csv], ignore_index=True)
        by_name = {path.stem: path for path in paths}
        excluded = {label.casefold() for label in args.exclude_label}
        selected = []
        for _, row in metadata.iterrows():
            labels = {x.strip().casefold() for x in str(row.get("labels", "")).split(",")}
            if excluded.intersection(labels):
                continue
            fname = str(row["fname"])
            selected_path = by_name.get(Path(fname).stem)
            if selected_path is not None:
                selected.append(selected_path)
        paths = sorted(set(selected))
    if not paths: raise FileNotFoundError(f"No audio files found in {args.audio_dir}")
    parts, metadata = [], []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        embeddings = np.asarray(encode([str(x) for x in batch_paths]), dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != len(batch_paths):
            raise ValueError(f"Encoder returned {embeddings.shape}; expected ({len(batch_paths)}, D)")
        parts.append(embeddings)
        metadata.extend(str(x.relative_to(args.audio_dir)) for x in batch_paths)
        print(f"{min(start + len(batch_paths), len(paths))}/{len(paths)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=np.concatenate(parts), paths=np.asarray(metadata))


if __name__ == "__main__": main()
