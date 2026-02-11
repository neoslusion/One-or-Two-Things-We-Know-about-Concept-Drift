"""
[DEPRECATED] Drift Type Classifier for Streaming Systems

WARNING: This module is DEPRECATED and no longer used in the current SE-CDT-Stream system.
The SE-CDT module (core/detectors/se_cdt.py) has replaced this KS-distance-based classifier
with a standard MMD-based approach that provides better drift type classification.

This file is kept for historical reference only. Do NOT use in production.

Original description:
Classifies concept drift into types: sudden, incremental, gradual, recurrent, or blip.
Based on the methodology from ConceptDrift_Pipeline.ipynb.
Classification uses temporal tracking of distribution changes (KS distance) to identify patterns.
"""

import warnings
warnings.warn(
    "drift_type_classifier.py is deprecated. Use core/detectors/se_cdt.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from scipy.stats import ks_2samp
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class DriftTypeConfig:
    """Configuration for drift type classification."""
    w_ref: int = 200           # reference window size before drift
    w_basic: int = 50          # basic comparison window size
    step: int = 10             # sliding step for tracking
    grow_step: int = 20        # growth step when expanding window after drift
    sudden_len_thresh: int = 60     # threshold to distinguish sudden vs progressive
    stabilize_delta: float = 0.02    # stability criterion (|Δdistance| < threshold)
    stabilize_patience: int = 3      # consecutive stable checks required
    recur_sim_thresh: float = 0.15   # r(t) ~ 0 => similar to pre (similarity threshold)
    recur_min_len: int = 120         # minimum duration to be considered recurrent
    blip_max_len: int = 60           # if similarity is very short -> blip
    smoothing_k: int = 3             # smoothing for tracking curve
    noise_guard: int = 30            # noise filter at the start after drift
    noise_min_gap: float = 0.08      # minimum difference from pre in early phase


@dataclass
class DriftTypeResult:
    """Result of drift type classification."""
    idx: int                      # drift detection index
    category: str                 # 'TCD' or 'PCD' (transient vs progressive)
    subcategory: str              # 'sudden'|'gradual'|'incremental'|'recurrent'|'blip'|'undetermined'
    drift_length: int             # samples from detection to stabilization
    stability_index: int          # index where drift stabilized
    early_gap: float              # KS distance in early phase (noise verification)
    note: str = ""                # additional notes about classification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov distance between two distributions.

    Returns the KS statistic: sup_x |F_a(x) - F_b(x)|
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    stat, _ = ks_2samp(a, b)
    return float(stat)


def roll_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Simple moving average with window size k."""
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode='valid')


def _to_1d_series(X: np.ndarray, prefer_scores: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert multi-dimensional data to 1D for classification.

    Priority:
    1. If prefer_scores provided → use directly
    2. If X is already 1D → use as-is
    3. If X is multi-dimensional:
       - Use PCA(n_components=1) if sklearn available
       - Otherwise use mean across features
    """
    if prefer_scores is not None:
        return np.asarray(prefer_scores).ravel()

    X = np.asarray(X)
    if X.ndim == 1:
        return X

    if X.ndim == 2 and X.shape[1] == 1:
        return X.ravel()

    # Multi-dimensional: try PCA, fallback to mean
    try:
        from sklearn.decomposition import PCA
        if X.shape[0] > 1:
            pca = PCA(n_components=1)
            return pca.fit_transform(X).ravel()
    except ImportError:
        pass

    # Fallback: mean across features
    return np.mean(X, axis=1)


def classify_drift_type_1d(
    x: np.ndarray,
    drift_idx: int,
    cfg: DriftTypeConfig
) -> DriftTypeResult:
    """
    Classify a single drift point in a 1D series.

    Algorithm Steps:
    1. Extract reference window (pre-drift)
    2. Noise guard - verify significant change
    3. Find stabilization point by growing window
    4. Build tracking curve r(t) over time
    5. Classify based on patterns in r(t)

    Parameters
    ----------
    x : ndarray of shape (n,)
        1D time series data
    drift_idx : int
        Index where drift was detected (t0)
    cfg : DriftTypeConfig
        Configuration parameters

    Returns
    -------
    DriftTypeResult
        Classification result with drift type and metadata
    """
    n = len(x)
    t0 = drift_idx

    # === Step 0: Reference Window ===
    a0 = max(0, t0 - cfg.w_ref)
    if t0 - a0 < cfg.w_basic:
        # Insufficient pre-drift data
        return DriftTypeResult(
            idx=t0,
            category='undetermined',
            subcategory='undetermined',
            drift_length=0,
            stability_index=t0,
            early_gap=0.0,
            note="Insufficient pre-drift history"
        )

    pre_ref = x[a0:t0]

    # === Step 1: Noise Guard ===
    end_guard = min(n, t0 + cfg.noise_guard)
    if end_guard <= t0:
        # No post-drift data
        return DriftTypeResult(
            idx=t0,
            category='undetermined',
            subcategory='undetermined',
            drift_length=0,
            stability_index=t0,
            early_gap=0.0,
            note="No post-drift data available"
        )

    early_b = x[t0:end_guard]
    early_gap = ks_distance(pre_ref[-cfg.w_basic:], early_b)

    if early_gap < cfg.noise_min_gap:
        # Change too small - likely noise
        return DriftTypeResult(
            idx=t0,
            category='undetermined',
            subcategory='undetermined',
            drift_length=0,
            stability_index=t0,
            early_gap=early_gap,
            note=f"Early gap {early_gap:.4f} < threshold {cfg.noise_min_gap}"
        )

    # === Step 2: Find Stabilization Point ===
    distances = []
    deltas = []
    stabilized = False
    patience = 0
    t_end = t0 + cfg.w_basic

    while t_end <= n:
        if t_end - t0 < cfg.w_basic:
            t_end += cfg.grow_step
            continue

        post_win = x[t0:t_end]
        if len(post_win) < cfg.w_basic:
            break

        d = ks_distance(pre_ref[-cfg.w_basic:], post_win[-cfg.w_basic:])
        distances.append(d)

        if len(distances) > 1:
            delta = abs(distances[-1] - distances[-2])
            deltas.append(delta)

            if delta < cfg.stabilize_delta:
                patience += 1
            else:
                patience = 0

            if patience >= cfg.stabilize_patience:
                stabilized = True
                break

        t_end += cfg.grow_step

    # If not stabilized, use end of available data
    if not stabilized:
        t_end = min(n, t_end)

    drift_length = max(1, t_end - t0)

    # === Step 3: TCD vs PCD ===
    category = 'TCD' if drift_length <= cfg.sudden_len_thresh else 'PCD'

    # === Step 4: Build Tracking Curve r(t) ===
    r_times = []
    r_vals = []

    for t in range(t0, n - cfg.w_basic + 1, cfg.step):
        cur = x[t:t + cfg.w_basic]
        if len(cur) < cfg.w_basic:
            break
        r_vals.append(ks_distance(pre_ref[-cfg.w_basic:], cur))
        r_times.append(t)

    if len(r_vals) == 0:
        # Insufficient data for tracking
        subcategory = 'sudden' if category == 'TCD' else 'undetermined'
        return DriftTypeResult(
            idx=t0,
            category=category,
            subcategory=subcategory,
            drift_length=drift_length,
            stability_index=t_end,
            early_gap=early_gap,
            note="Insufficient data for tracking curve"
        )

    r_vals = np.array(r_vals)

    # Smoothing
    if len(r_vals) >= cfg.smoothing_k:
        r_vals_smooth = roll_mean(r_vals, cfg.smoothing_k)
    else:
        r_vals_smooth = r_vals

    # === Step 5: Subcategory Classification ===
    subcategory = 'undetermined'

    # Check for Recurrent / Blip
    similar_mask = r_vals_smooth < cfg.recur_sim_thresh

    if np.any(similar_mask):
        # Find consecutive segments
        segments = []
        in_segment = False
        start = 0

        for i, is_similar in enumerate(similar_mask):
            if is_similar and not in_segment:
                start = i
                in_segment = True
            elif not is_similar and in_segment:
                segments.append(i - start)
                in_segment = False

        if in_segment:
            segments.append(len(similar_mask) - start)

        if segments:
            longest_segment = max(segments) * cfg.step  # Convert to sample count

            if longest_segment >= cfg.recur_min_len:
                subcategory = 'recurrent'
            elif longest_segment <= cfg.blip_max_len:
                subcategory = 'blip'

    # If not recurrent/blip, classify as sudden/gradual/incremental
    if subcategory == 'undetermined':
        if category == 'TCD':
            subcategory = 'sudden'
        else:
            # Analyze monotonicity
            if len(r_vals_smooth) > 2:
                diff = np.diff(r_vals_smooth)
                sign_changes = np.sum((diff[1:] * diff[:-1]) < 0)

                med = np.median(r_vals_smooth)
                crossings = np.sum((r_vals_smooth[:-1] - med) * (r_vals_smooth[1:] - med) < 0)

                if sign_changes <= 2 and crossings <= 2:
                    subcategory = 'incremental'
                else:
                    subcategory = 'gradual'
            else:
                subcategory = 'gradual'

    return DriftTypeResult(
        idx=t0,
        category=category,
        subcategory=subcategory,
        drift_length=drift_length,
        stability_index=t_end,
        early_gap=early_gap,
        note=f"Stabilized={stabilized}, tracking_points={len(r_vals)}"
    )


def classify_drift_at_detection(
    X: np.ndarray,
    drift_idx: int,
    cfg: Optional[DriftTypeConfig] = None,
    prefer_scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Main entry point for drift type classification.

    Handles multi-dimensional data by converting to 1D, then classifies drift type.

    Parameters
    ----------
    X : ndarray of shape (n, d) or (n,)
        Stream data (multi-dimensional or 1D)
    drift_idx : int
        Index where drift was detected
    cfg : DriftTypeConfig, optional
        Configuration parameters (uses defaults if None)
    prefer_scores : ndarray of shape (n,), optional
        Pre-computed 1D projection/scores to use instead of PCA

    Returns
    -------
    dict
        Classification result as dictionary with keys:
        - idx: drift index
        - category: 'TCD' or 'PCD'
        - subcategory: drift type
        - drift_length: samples to stabilization
        - stability_index: stabilization point
        - early_gap: initial KS distance
        - note: additional info

    Example
    -------
    >>> X = np.random.randn(1000, 5)  # 1000 samples, 5 features
    >>> result = classify_drift_at_detection(X, drift_idx=500)
    >>> print(result['subcategory'])  # 'sudden', 'incremental', etc.
    """
    if cfg is None:
        cfg = DriftTypeConfig()

    # Convert to 1D
    x = _to_1d_series(X, prefer_scores)

    # Classify
    result = classify_drift_type_1d(x, drift_idx, cfg)

    return result.to_dict()


# Convenience function for streaming systems
def classify_from_buffer(
    buffer: np.ndarray,
    drift_idx_in_buffer: int,
    cfg: Optional[DriftTypeConfig] = None
) -> Dict[str, Any]:
    """
    Classify drift from a circular buffer.

    This is a convenience wrapper for streaming systems that maintain
    a sliding window buffer.

    Parameters
    ----------
    buffer : ndarray
        Circular buffer containing stream data
    drift_idx_in_buffer : int
        Index of drift within the buffer (not global index)
    cfg : DriftTypeConfig, optional
        Configuration parameters

    Returns
    -------
    dict
        Classification result
    """
    return classify_drift_at_detection(buffer, drift_idx_in_buffer, cfg)
