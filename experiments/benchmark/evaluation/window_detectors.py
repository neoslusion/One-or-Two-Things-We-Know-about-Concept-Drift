"""
Window-based drift detector evaluation.

Contains the evaluate_drift_detector function for evaluating window-based
drift detection methods including:
- D3, DAWIDD, MMD, KS
- ShapeDD variants (buffer-based approach)
# - MMD_ADW, ShapeDD_ADW_MMD
# Note: MMD_ADW and ShapeDD_ADW_MMD were previously named MMD_OW and ShapeDD_OW_MMD,
# respectively. The renaming reflects the "Adaptive Window" (ADW) approach
# described in Bharti et al. (2023) for clarity and consistency.
"""

import time
import numpy as np
from collections import deque

import sys
import os

from core.detectors.shape_dd import shape, shape_mmdagg
from core.detectors.d3 import d3
from core.detectors.dawidd import dawidd
from core.detectors.mmd import mmd
from core.detectors.mmd_variants import mmd_adw, shapedd_adw_mmd, shape_with_wmmd
from core.detectors.ks import ks

from ..config import SHAPE_L1, SHAPE_L2, SHAPE_N_PERM, COOLDOWN, CHUNK_SIZE, OVERLAP
from ..utils import create_sliding_windows
from .metrics import calculate_detection_metrics_enhanced


def evaluate_drift_detector(
    method_name, X, true_drifts, chunk_size=None, overlap=None, verbose=False
):
    """
    Unified sliding window evaluation for ALL drift detectors.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE  # e.g., 300
    if overlap is None:
        overlap = OVERLAP  # e.g., 150

    start_time = time.time()
    detections = []
    last_detection = -(10**9)

    # Create sliding windows (same for all methods)
    windows, window_centers = create_sliding_windows(X, chunk_size, overlap)

    if verbose:
        print(
            f"  {method_name}: {len(windows)} windows, size={chunk_size}, overlap={overlap}"
        )

    for window_idx, (window, center_idx) in enumerate(zip(windows, window_centers)):
        try:
            # === ShapeDD Methods ===
            if method_name == "ShapeDD":
                # Apply ShapeDD to window, get minimum p-value
                shp_results = shape(window, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM)
                min_pvalue = shp_results[:, 2].min()
                trigger = min_pvalue < 0.05

            elif method_name == "ShapeDD_MMDAgg":
                shp_results = shape_mmdagg(
                    window, SHAPE_L1, SHAPE_L2, n_bandwidths=10, alpha=0.05
                )
                min_pvalue = shp_results[:, 2].min()
                trigger = min_pvalue < 0.05

            elif method_name == "ShapeDD_ADW_MMD":
                pattern_score, mmd_max = shapedd_adw_mmd(
                    window, l1=SHAPE_L1, l2=SHAPE_L2, gamma="auto"
                )
                trigger = pattern_score > 0.5

            elif method_name == "ShapeDD_WMMD":
                # Use default n_perm=500 for validation to be fast but valid
                shp_results = shape_with_wmmd(window, SHAPE_L1, SHAPE_L2, n_perm=500)
                min_pvalue = shp_results[:, 2].min()
                trigger = min_pvalue < 0.05

            # === Baseline Methods (unchanged) ===
            elif method_name == "D3":
                score = d3(window)
                trigger = score < 0.25

            elif method_name == "DAWIDD":
                _, p_value = dawidd(window, "rbf")
                trigger = p_value < 0.05

            elif method_name == "MMD":
                stat, p_value = mmd(window)
                trigger = p_value < 0.05

            elif method_name == "KS":
                p_value = ks(window)
                trigger = p_value < 0.05

            elif method_name == "MMD_ADW":
                stat, threshold = mmd_adw(window, gamma="auto")
                trigger = stat > threshold

            else:
                trigger = False

            # Record detection with cooldown
            if trigger and (center_idx - last_detection >= COOLDOWN):
                detections.append(center_idx)
                last_detection = center_idx

        except Exception as e:
            if verbose:
                print(f"    Window {window_idx} failed: {e}")

    # Calculate metrics
    end_time = time.time()
    metrics = calculate_detection_metrics_enhanced(detections, true_drifts, len(X))

    return {
        "method": method_name,
        "detections": detections,
        "stream_size": len(X),
        "runtime_s": end_time - start_time,
        **metrics,
    }
