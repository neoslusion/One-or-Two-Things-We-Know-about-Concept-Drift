"""
Windowing utilities for drift detection benchmark.
"""

import numpy as np


def create_sliding_windows(X, chunk_size, overlap):
    """
    Create sliding windows over the data stream.

    Args:
        X: Input data array of shape (n_samples, n_features)
        chunk_size: Size of each window
        overlap: Number of overlapping samples between consecutive windows

    Returns:
        windows: List of window arrays
        indices: List of center indices for each window
    """
    shift = chunk_size - overlap
    windows = []
    indices = []

    for i in range(0, len(X) - chunk_size + 1, shift):
        windows.append(X[i:i+chunk_size])
        indices.append(i + chunk_size // 2)  # Center of window

    return windows, indices
