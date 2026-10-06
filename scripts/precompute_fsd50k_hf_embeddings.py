#!/usr/bin/env python3
"""Stream FSD50K from Hugging Face and precompute Perch 2.0 embeddings."""
import argparse
import numpy as np
import librosa

PERCH_URL = (
    "https://www.kaggle.com/models/google/"
    "bird-vocalization-classifier/tensorFlow2/perch_v2/2"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--dataset", default="CLAPv2/FSD50K")
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--exclude-label", action="append", default=[],
                   help="Case-insensitive label to exclude; repeat as needed")
    p.add_argument("--sample-rate", type=int, default=32000)
    p.add_argument("--clip-seconds", type=float, default=5.0)
    args = p.parse_args()

    from datasets import load_dataset
    import tensorflow as tf
    import tensorflow_hub as hub
    model = hub.load(PERCH_URL)
    clip_samples = int(args.sample_rate * args.clip_seconds)
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
            audio = row["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            if int(audio["sampling_rate"]) != args.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=int(audio["sampling_rate"]), target_sr=args.sample_rate)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
            if len(waveform) < clip_samples:
                waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
            else:
                waveform = waveform[:clip_samples]
            peak = np.max(np.abs(waveform)) if len(waveform) else 0.0
            if peak > 0:
                waveform = waveform / peak * 0.25
            batch.append(waveform)
            batch_ids.append(f"{split}:{row.get('index', index)}")
            if len(batch) == args.batch_size:
                waveform_batch = tf.convert_to_tensor(np.stack(batch), dtype=tf.float32)
                out = model.signatures["serving_default"](inputs=waveform_batch)["embedding"].numpy().astype(np.float32)
                if out.ndim != 2 or len(out) != len(batch):
                    raise ValueError(f"Encoder returned {out.shape}; expected ({len(batch)}, D)")
                embeddings.append(out); ids.extend(batch_ids); total += len(batch)
                print(f"processed {total}", flush=True)
                batch, batch_ids = [], []
        if batch:
            waveform_batch = tf.convert_to_tensor(np.stack(batch), dtype=tf.float32)
            out = model.signatures["serving_default"](inputs=waveform_batch)["embedding"].numpy().astype(np.float32)
            if out.ndim != 2 or len(out) != len(batch):
                raise ValueError(f"Encoder returned {out.shape}; expected ({len(batch)}, D)")
            embeddings.append(out); ids.extend(batch_ids); total += len(batch)
            print(f"processed {total}", flush=True)

    if not embeddings:
        raise RuntimeError("No FSD50K samples remained after filtering")
    np.savez_compressed(args.output, embeddings=np.concatenate(embeddings), ids=np.asarray(ids))
    print(f"saved {total} embeddings to {args.output}")


if __name__ == "__main__": main()
