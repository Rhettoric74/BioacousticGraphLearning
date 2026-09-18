"""Small, self-contained BirdSet utilities used by graph-transformer evaluation.

Audio decoding and Perch embedding generation are intentionally omitted from
the training path because this project consumes the already-created pickle
files. The loader is retained here to recover BirdSet's eBird label ordering
for validation and test-set label alignment.
"""
from __future__ import annotations

import math
from datasets import Audio, load_dataset


MIN_YEAR, MAX_YEAR = 2005, 2025
HUGGINGFACE_CACHE = "/scratch/e1583377/huggingface/"


def load_birdset_data(subset="NES", default_lat_lon=None, cache_dir=HUGGINGFACE_CACHE):
    """Load a BirdSet subset's test_5s split with decoded 32-kHz audio metadata."""
    dataset = load_dataset("DBD-research-group/BirdSet", subset, cache_dir=cache_dir)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
    if default_lat_lon is not None:
        default_lat, default_lon = default_lat_lon

        def fill_missing_coordinates(example):
            if example["lat"] is None or math.isnan(example["lat"]):
                example["lat"] = default_lat
            if example["long"] is None or math.isnan(example["long"]):
                example["long"] = default_lon
            return example

        dataset["test_5s"] = dataset["test_5s"].map(fill_missing_coordinates)
    return dataset["test_5s"]


def encode_time(time, date, min_year=MIN_YEAR, max_year=MAX_YEAR):
    """Encode local time and representative recording date cyclically."""
    hour = int(time[:time.index(":")]) + int(time[time.index(":") + 1:]) / 60
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    year, month, day = date.split("-")
    day_of_year = 30.4167 * int(month) + int(day)
    day_sin = math.sin(2 * math.pi * day_of_year / 365)
    day_cos = math.cos(2 * math.pi * day_of_year / 365)
    year_norm = (int(year) - min_year) / (max_year - min_year)
    return [hour_sin, hour_cos, day_sin, day_cos, year_norm]
