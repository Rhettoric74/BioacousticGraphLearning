#!/usr/bin/env python3
"""Stream FSD50K from Hugging Face and precompute Perch embeddings.

The encoder adapter supplied with --encoder must expose:

    encode(samples) -> array/tensor of shape (batch, embedding_dim)

Each sample is the Hugging Face audio dictionary with ``array`` and
``sampling_rate`` fields. This keeps the Perch-specific API outside this
repository because Perch installations differ across clusters.
"""
import argparse
import importlib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--encoder", required=True, help="Python module:function accepting audio dictionaries")
    p.add_argument("--dataset", default="CLAPv2/FSD50K")
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--exclude-label", action="append", default=[],
                   help="Case-insensitive label to exclude; repeat as needed")
    args = p.parse_args()

    from datasets import load_dataset
    module_name, function_name = args.encoder.split(":", 1)
    encode = getattr(importlib.import_module(module_name), function_name)
    excluded = {x.casefold() for x in args.exclude_label}
    embeddings, ids = [], []
    total = 0

    for split in args.splits:
        stream = load_dataset(args.dataset, split=split, streaming=True)
        batch, batch_ids = [], []
        for index, row in enumerate(stream):
            if args.max_samples is not None and total >= args.max_samples:
                break
            label_text = str(row.get("text", row.get("raw_text", "")))
            labels = {x.strip().casefold() for x in label_text.split(",")}
            if excluded.intersection(labels):
                continue
            batch.append(row["audio"])
            batch_ids.append(f"{split}:{row.get('index', index)}")
            if len(batch) == args.batch_size:
                out = np.asarray(encode(batch), dtype=np.float32)
                if out.ndim != 2 or len(out) != len(batch):
                    raise ValueError(f"Encoder returned {out.shape}; expected ({len(batch)}, D)")
                embeddings.append(out); ids.extend(batch_ids); total += len(batch)
                print(f"processed {total}", flush=True)
                batch, batch_ids = [], []
        if batch:
            out = np.asarray(encode(batch), dtype=np.float32)
            if out.ndim != 2 or len(out) != len(batch):
                raise ValueError(f"Encoder returned {out.shape}; expected ({len(batch)}, D)")
            embeddings.append(out); ids.extend(batch_ids); total += len(batch)
            print(f"processed {total}", flush=True)

    if not embeddings:
        raise RuntimeError("No FSD50K samples remained after filtering")
    np.savez_compressed(args.output, embeddings=np.concatenate(embeddings), ids=np.asarray(ids))
    print(f"saved {total} embeddings to {args.output}")


if __name__ == "__main__": main()
