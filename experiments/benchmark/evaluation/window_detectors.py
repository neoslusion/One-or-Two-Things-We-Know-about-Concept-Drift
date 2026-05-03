"""
Window-based drift detector evaluation.

Contains the evaluate_drift_detector function for evaluating window-based
drift detection methods including:

Baseline Methods:
- MMD: Maximum Mean Discrepancy (standard) - Gretton et al. (2012)
- KS:  Kolmogorov-Smirnov test (classical non-parametric)

Other Window-based Methods:
- D3:     Deep-learning-based discriminative drift detector
- DAWIDD: Distribution-Aware Window-based detector
- IDW_MMD: Stand-alone Inverse-Density-Weighted MMD (ablation)

ShapeDD Variants:
- ShapeDD:     Original shape-based detection with permutation test
- ShapeDD_IDW: ShapeDD + IDW-MMD with asymptotic p-value (RECOMMENDED, fast)

Unified System (this thesis' contribution):
- SE_CDT: Unified detector + classifier
          (detection + drift type classification in a single pass)
"""

import time

from core.detectors.shape_dd import shape
from core.detectors.d3 import d3
from core.detectors.dawidd import dawidd
from core.detectors.mmd import mmd
from core.detectors.mmd_variants import (
    wmmd_gamma,               # IDW-MMD with Gamma-approximation p-value
    shapedd_idw_mmd_proper,   # PROPER ShapeDD + IDW-MMD with asymptotic p-value
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

    NOTE ON TIMING
    --------------
    ``runtime_s`` in the returned dict is measured with ``time.process_time()``,
    which counts the CPU time consumed by *this process only* (user + system).
    In a parallel benchmark (joblib multiprocessing) each worker is a separate
    process, so the value is unaffected by co-running workers and gives a fair
    algorithmic comparison.  It is *not* wall-clock elapsed time.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE  # e.g., 300
    if overlap is None:
        overlap = OVERLAP  # e.g., 150

    start_time = time.process_time()
    detections = []
    # SE-CDT also predicts a drift type per detection.  ``predicted_labels``
    # is parallel to ``detections`` (None for methods that only detect, e.g.
    # baselines).  Lower-cased to match the ground-truth ``event_labels``
    # vocabulary in `data.generators.benchmark_generators._VALID_DRIFT_LABELS`.
    predicted_labels: list = []
    last_detection = -(10**9)

    # Create sliding windows (same for all methods)
    windows, window_centers = create_sliding_windows(X, chunk_size, overlap)

    if verbose:
        print(
            f"  {method_name}: {len(windows)} windows, size={chunk_size}, overlap={overlap}"
        )

    # ------------------------------------------------------------------ #
    # Pre-initialise stateful / reusable detector objects ONCE before the  #
    # window loop so we avoid redundant construction overhead per window.   #
    # ------------------------------------------------------------------ #
    _se_cdt = None
    if method_name == "SE_CDT":
        _se_cdt = SE_CDT(
            window_size=SHAPE_L1,
            l2=SHAPE_L2,
            alpha=0.05,
            use_proper=True,
        )

    for window_idx, (window, center_idx) in enumerate(zip(windows, window_centers)):
        # Reset per-window prediction; only SE_CDT populates this.
        window_pred_label: str | None = None
        try:
            # === RECOMMENDED Methods ===
            
            if method_name == "ShapeDD":
                # Original ShapeDD with permutation test (baseline)
                shp_results = shape(window, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM)
                min_pvalue = shp_results[:, 2].min()
                trigger = min_pvalue < 0.05

            elif method_name == "ShapeDD_IDW":
                # ShapeDD + IDW-MMD with asymptotic p-value (fast, recommended)
                is_drift, _positions, _, _p_values = shapedd_idw_mmd_proper(
                    window, l1=SHAPE_L1, l2=SHAPE_L2, alpha=0.05
                )
                trigger = is_drift

            elif method_name == "SE_CDT":
                # SE-CDT: Unified Detector-Classifier.
                # Returns both detection AND drift type classification.
                # The detector object is created ONCE before this loop (see above).
                result = _se_cdt.monitor(window)
                trigger = result.is_drift
                if trigger:
                    sub = (result.subcategory or "Unknown").lower()
                    window_pred_label = sub if sub != "unknown" else None

            # === Baseline Methods ===

            elif method_name == "MMD":
                # Standard MMD (Gretton et al., 2012)
                _stat, p_value = mmd(window)
                trigger = p_value < 0.05

            elif method_name == "KS":
                # Kolmogorov-Smirnov test (classical)
                p_value = ks(window)
                trigger = p_value < 0.05

            elif method_name == "IDW_MMD":
                # Stand-alone IDW-MMD with Gamma-approximation p-value (ablation)
                _, p_val = wmmd_gamma(window, len(window)//2, weight_method="variance_reduction")
                trigger = p_val < 0.05

            elif method_name == "D3":
                # Deep-learning-based discriminative detector
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
                predicted_labels.append(window_pred_label)
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
        # Per-detection drift type predictions (populated by SE_CDT only).
        # ``predicted_labels[i]`` corresponds to ``detections[i]``.
        "predicted_labels": predicted_labels,
        "stream_size": len(X),
        "runtime_s": end_time - start_time,
        **metrics,
    }
