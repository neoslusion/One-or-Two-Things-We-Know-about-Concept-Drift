"""
Window-based drift detector evaluation.

Contains the evaluate_drift_detector function for evaluating window-based
drift detection methods including:

Baseline Methods:
- MMD: Maximum Mean Discrepancy (standard) - Gretton et al. (2012)
- KS: Kolmogorov-Smirnov test (classical non-parametric)

ShapeDD Variants:
- ShapeDD: Original shape-based detection with permutation test
- ShapeDD_WMMD: ShapeDD + Weighted MMD with permutation (rigorous, slow)
- ShapeDD_WMMD_PROPER: ShapeDD + Weighted MMD with asymptotic p-value (RECOMMENDED)

Unified System:
- SE_CDT: Unified detector-classifier (detection + drift type classification)

Legacy/Deprecated (not recommended for benchmark):
- D3: Deep learning-based (different paradigm)
- DAWIDD: Distribution-Aware Window-based method
- MMD_WMMD: Just Weighted MMD without ShapeDD
- ShapeDD_WMMD_HEURISTIC: Old heuristic version (superseded by PROPER)

Note on Weighted MMD:
- Uses Inverse Density Weighting (IDW): w_i ∝ 1/sqrt(kernel_density)
- Points in sparse regions (boundaries) get higher weights
- Improves sensitivity to distribution changes
"""

import time
import numpy as np
from collections import deque

import sys
import os

from core.detectors.shape_dd import shape
from core.detectors.d3 import d3
from core.detectors.dawidd import dawidd
from core.detectors.mmd import mmd
from core.detectors.mmd_variants import (
    mmd_adw,                  # Weighted MMD split test
    shapedd_adw_mmd,          # Legacy heuristic (deprecated)
    shape_with_wmmd,          # Permutation test version
    shapedd_adw_mmd_proper,   # PROPER design with asymptotic p-value
)
from core.detectors.ks import ks
from core.detectors.se_cdt import SE_CDT  # Unified detector-classifier

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

    start_time = time.process_time()
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
            # === RECOMMENDED Methods ===
            
            if method_name == "ShapeDD":
                # Original ShapeDD with permutation test (baseline)
                shp_results = shape(window, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM)
                min_pvalue = shp_results[:, 2].min()
                trigger = min_pvalue < 0.05

            elif method_name == "ShapeDD_WMMD":
                # ShapeDD + Weighted MMD with permutation test
                # Statistically rigorous but SLOW (~500ms per window)
                shp_results = shape_with_wmmd(window, SHAPE_L1, SHAPE_L2, n_perm=500)
                min_pvalue = shp_results[:, 2].min()
                trigger = min_pvalue < 0.05
            
            elif method_name in ("ShapeDD_WMMD_PROPER", "ShapeDD_ADW_PROPER"):
                # RECOMMENDED: ShapeDD + Weighted MMD with asymptotic p-value
                # Fast (~5ms) + has statistical p-value
                is_drift, positions, _, p_values = shapedd_adw_mmd_proper(
                    window, l1=SHAPE_L1, l2=SHAPE_L2, alpha=0.05
                )
                trigger = is_drift
            
            elif method_name == "SE_CDT":
                # SE-CDT: Unified Detector-Classifier
                # Returns both detection AND drift type classification
                detector = SE_CDT(
                    window_size=SHAPE_L1, 
                    l2=SHAPE_L2, 
                    alpha=0.05, 
                    use_proper=True
                )
                result = detector.monitor(window)
                trigger = result.is_drift

            # === Baseline Methods ===
            
            elif method_name == "MMD":
                # Standard MMD (Gretton et al., 2012)
                stat, p_value = mmd(window)
                trigger = p_value < 0.05

            elif method_name == "KS":
                # Kolmogorov-Smirnov test (classical)
                p_value = ks(window)
                trigger = p_value < 0.05

            elif method_name in ("MMD_WMMD", "MMD_ADW"):
                # Weighted MMD without ShapeDD (ablation study)
                stat, threshold = mmd_adw(window, gamma="auto")
                trigger = stat > threshold

            # === Legacy/Deprecated Methods ===
            
            elif method_name in ("ShapeDD_WMMD_HEURISTIC", "ShapeDD_ADW_MMD"):
                # DEPRECATED: Old heuristic version (use PROPER instead)
                pattern_score, mmd_max = shapedd_adw_mmd(
                    window, l1=SHAPE_L1, l2=SHAPE_L2, gamma="auto"
                )
                trigger = pattern_score > 0.5

            elif method_name == "D3":
                # Deep learning-based (different paradigm)
                score = d3(window)
                trigger = score < 0.25

            elif method_name == "DAWIDD":
                # Distribution-Aware Window-based
                _, p_value = dawidd(window, "rbf")
                trigger = p_value < 0.05

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
    end_time = time.process_time()
    metrics = calculate_detection_metrics_enhanced(detections, true_drifts, len(X))

    return {
        "method": method_name,
        "detections": detections,
        "stream_size": len(X),
        "runtime_s": end_time - start_time,
        **metrics,
    }
