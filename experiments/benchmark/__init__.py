"""
Drift Detection Benchmark Package

A comprehensive benchmark framework for evaluating concept drift detection methods.
Refactored from MultiDetectors_Evaluation_DetectionOnly.ipynb
"""

from .config import (
    STREAM_SIZE,
    N_RUNS,
    RANDOM_SEEDS,
    CHUNK_SIZE,
    OVERLAP,
    WINDOW_METHODS,
    STREAMING_METHODS,
)

__version__ = "1.0.0"
__all__ = [
    "STREAM_SIZE",
    "N_RUNS",
    "RANDOM_SEEDS",
    "CHUNK_SIZE",
    "OVERLAP",
    "WINDOW_METHODS",
    "STREAMING_METHODS",
]

