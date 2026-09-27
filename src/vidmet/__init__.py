"""Benchmark datasets for frozen-feature evaluation and fine-tuning.

All loaders read the tar-of-JPEG layout produced by ``scripts/data/benchmarks``
and return fixed-length clips (``target_length`` frames).
"""

from vidmet.datasets import (
    ARIDDataset,
    Diving48Dataset,
    HMDB51Dataset,
    IARDDataset,
    JesterDataset,
    SSv2Dataset,
    VideoLoadError,
)

__all__ = [
    "ARIDDataset",
    "Diving48Dataset",
    "HMDB51Dataset",
    "IARDDataset",
    "JesterDataset",
    "SSv2Dataset",
    "VideoLoadError",
]
