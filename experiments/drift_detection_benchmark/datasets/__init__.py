"""
Dataset generation module for drift detection benchmarks.
"""

from .catalog import DATASET_CATALOG, get_enabled_datasets
from .generators import generate_drift_stream

__all__ = [
    "DATASET_CATALOG",
    "get_enabled_datasets",
    "generate_drift_stream",
]

