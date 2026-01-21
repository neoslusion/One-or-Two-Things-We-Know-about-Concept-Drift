"""
Drift Type Detection (Drift Type DD)

Classifies concept drift into types based on temporal patterns in KS distance.

Drift Types:
- sudden: Abrupt change (TCD - Transient Concept Drift)
- incremental: Monotonic progression (PCD - Progressive Concept Drift)
- gradual: Oscillating change (PCD)
- recurrent: Returns to previous distribution (PCD)
- blip: Temporary spike that reverts quickly (noise-like)

Method: Tracks KS distance over time and analyzes temporal patterns.
Based on ConceptDrift_Pipeline.ipynb methodology.
"""

import numpy as np
from scipy.stats import ks_2samp


def ks_distance_1d(a, b):
    """
    Compute KS distance between two 1D distributions.

    Parameters
    ----------
    a, b : array-like
        Two samples to compare

    Returns
    -------
    float
        KS statistic (sup_x |F_a(x) - F_b(x)|)
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    stat, _ = ks_2samp(a, b)
    return float(stat)


def roll_mean(x, k):
    """Simple moving average with window size k."""
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode='valid')


def to_1d_series(X, prefer_scores=None):
    """
    Convert multi-dimensional data to 1D series.

    Priority:
    1. Use prefer_scores if provided
    2. Use X directly if already 1D
    3. Use PCA if available (tries sklearn)
    4. Fallback to mean across features

    Parameters
    ----------
    X : array-like
        Data (1D or 2D)
    prefer_scores : array-like, optional
        Pre-computed 1D projection

    Returns
    -------
    array
        1D series
    """
    if prefer_scores is not None:
        return np.asarray(prefer_scores).ravel()

    X = np.asarray(X)
    if X.ndim == 1:
        return X

    if X.ndim == 2 and X.shape[1] == 1:
        return X.ravel()

    # Try PCA
    try:
        from sklearn.decomposition import PCA
        if X.shape[0] > 1:
            pca = PCA(n_components=1)
            return pca.fit_transform(X).ravel()
    except ImportError:
        pass

    # Fallback: mean across features
    return np.mean(X, axis=1)


def drift_type_classify_1d(x, drift_idx, w_ref=200, w_basic=50, step=10, grow_step=20,
                            sudden_len_thresh=60, stabilize_delta=0.02, stabilize_patience=3,
                            recur_sim_thresh=0.15, recur_min_len=120, blip_max_len=60,
                            smoothing_k=3, noise_guard=30, noise_min_gap=0.08):
    """
    Classify drift type in a 1D time series.

    Algorithm:
    1. Extract reference window (pre-drift stable period)
    2. Noise guard: verify change is significant
    3. Find stabilization point (when KS distance stops changing)
    4. Build tracking curve r(t): KS distance over time
    5. Classify based on patterns in r(t)

    Parameters
    ----------
    x : array-like, shape (n,)
        1D time series
    drift_idx : int
        Index where drift was detected (t0)
    w_ref : int, default=200
        Reference window size (pre-drift)
    w_basic : int, default=50
        Basic comparison window size
    step : int, default=10
        Sliding step for tracking curve
    grow_step : int, default=20
        Growth step when expanding post-drift window
    sudden_len_thresh : int, default=60
        Threshold to distinguish TCD vs PCD (samples)
    stabilize_delta : float, default=0.02
        Stability criterion (|Δdistance| < threshold)
    stabilize_patience : int, default=3
        Consecutive stable checks required
    recur_sim_thresh : float, default=0.15
        Similarity threshold for recurrent drift (r(t) ≈ 0)
    recur_min_len : int, default=120
        Minimum duration for recurrent classification
    blip_max_len : int, default=60
        Maximum duration for blip classification
    smoothing_k : int, default=3
        Smoothing window for tracking curve
    noise_guard : int, default=30
        Noise filter at start after drift
    noise_min_gap : float, default=0.08
        Minimum KS difference to confirm real drift

    Returns
    -------
    result : dict
        - drift_idx: int, detection index
        - category: str, 'TCD' or 'PCD' (Transient vs Progressive)
        - subcategory: str, 'sudden'|'incremental'|'gradual'|'recurrent'|'blip'|'undetermined'
        - drift_length: int, samples to stabilization
        - stability_index: int, where drift stabilized
        - early_gap: float, initial KS distance (for noise verification)
        - tracking_curve: dict, r(t) values and times
        - note: str, additional info

    Example
    -------
    >>> x = np.concatenate([np.random.randn(500), np.random.randn(500) + 2])
    >>> result = drift_type_classify_1d(x, drift_idx=500)
    >>> print(result['subcategory'])  # 'sudden'
    >>> print(result['drift_length'])  # ~60 (abrupt change)
    """
    n = len(x)
    t0 = drift_idx

    # Step 0: Reference Window
    a0 = max(0, t0 - w_ref)
    if t0 - a0 < w_basic:
        return {
            'drift_idx': t0,
            'category': 'undetermined',
            'subcategory': 'undetermined',
            'drift_length': 0,
            'stability_index': t0,
            'early_gap': 0.0,
            'tracking_curve': {'times': [], 'r_vals': []},
            'note': 'insufficient_pre_drift_history'
        }

    pre_ref = x[a0:t0]

    # Step 1: Noise Guard
    end_guard = min(n, t0 + noise_guard)
    if end_guard <= t0:
        return {
            'drift_idx': t0,
            'category': 'undetermined',
            'subcategory': 'undetermined',
            'drift_length': 0,
            'stability_index': t0,
            'early_gap': 0.0,
            'tracking_curve': {'times': [], 'r_vals': []},
            'note': 'no_post_drift_data'
        }

    early_b = x[t0:end_guard]
    early_gap = ks_distance_1d(pre_ref[-w_basic:], early_b)

    if early_gap < noise_min_gap:
        return {
            'drift_idx': t0,
            'category': 'undetermined',
            'subcategory': 'undetermined',
            'drift_length': 0,
            'stability_index': t0,
            'early_gap': early_gap,
            'tracking_curve': {'times': [], 'r_vals': []},
            'note': f'early_gap={early_gap:.4f} < noise_min_gap={noise_min_gap}'
        }

    # Step 2: Find Stabilization Point
    distances = []
    deltas = []
    stabilized = False
    patience = 0
    t_end = t0 + w_basic

    while t_end <= n:
        if t_end - t0 < w_basic:
            t_end += grow_step
            continue

        post_win = x[t0:t_end]
        if len(post_win) < w_basic:
            break

        d = ks_distance_1d(pre_ref[-w_basic:], post_win[-w_basic:])
        distances.append(d)

        if len(distances) > 1:
            delta = abs(distances[-1] - distances[-2])
            deltas.append(delta)

            if delta < stabilize_delta:
                patience += 1
            else:
                patience = 0

            if patience >= stabilize_patience:
                stabilized = True
                break

        t_end += grow_step

    # If not stabilized, use end of available data
    if not stabilized:
        t_end = min(n, t_end)

    drift_length = max(1, t_end - t0)

    # Step 3: TCD vs PCD
    category = 'TCD' if drift_length <= sudden_len_thresh else 'PCD'

    # Step 4: Build Tracking Curve r(t)
    r_times = []
    r_vals = []

    for t in range(t0, n - w_basic + 1, step):
        cur = x[t:t + w_basic]
        if len(cur) < w_basic:
            break
        r_vals.append(ks_distance_1d(pre_ref[-w_basic:], cur))
        r_times.append(t)

    if len(r_vals) == 0:
        subcategory = 'sudden' if category == 'TCD' else 'undetermined'
        return {
            'drift_idx': t0,
            'category': category,
            'subcategory': subcategory,
            'drift_length': drift_length,
            'stability_index': t_end,
            'early_gap': early_gap,
            'tracking_curve': {'times': r_times, 'r_vals': r_vals},
            'note': 'insufficient_data_for_tracking'
        }

    r_vals = np.array(r_vals)

    # Smoothing
    if len(r_vals) >= smoothing_k:
        r_vals_smooth = roll_mean(r_vals, smoothing_k)
    else:
        r_vals_smooth = r_vals

    # Step 5: Subcategory Classification
    subcategory = 'undetermined'

    # Check for Recurrent / Blip
    similar_mask = r_vals_smooth < recur_sim_thresh

    if np.any(similar_mask):
        # Find consecutive segments where r(t) ≈ 0 (similar to pre-drift)
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
            longest_segment = max(segments) * step  # Convert to sample count

            if longest_segment >= recur_min_len:
                subcategory = 'recurrent'
            elif longest_segment <= blip_max_len:
                subcategory = 'blip'

    # If not recurrent/blip, classify as sudden/gradual/incremental
    if subcategory == 'undetermined':
        if category == 'TCD':
            subcategory = 'sudden'
        else:
            # Analyze monotonicity of tracking curve
            if len(r_vals_smooth) > 2:
                diff = np.diff(r_vals_smooth)
                sign_changes = np.sum((diff[1:] * diff[:-1]) < 0)

                med = np.median(r_vals_smooth)
                crossings = np.sum((r_vals_smooth[:-1] - med) * (r_vals_smooth[1:] - med) < 0)

                # Incremental: monotonic (few sign changes, few median crossings)
                # Gradual: oscillating (many sign changes, many crossings)
                if sign_changes <= 2 and crossings <= 2:
                    subcategory = 'incremental'
                else:
                    subcategory = 'gradual'
            else:
                subcategory = 'gradual'

    return {
        'drift_idx': t0,
        'category': category,
        'subcategory': subcategory,
        'drift_length': drift_length,
        'stability_index': t_end,
        'early_gap': early_gap,
        'tracking_curve': {
            'times': r_times,
            'r_vals': r_vals.tolist() if isinstance(r_vals, np.ndarray) else r_vals,
            'r_vals_smooth': r_vals_smooth.tolist() if isinstance(r_vals_smooth, np.ndarray) else r_vals_smooth
        },
        'note': f'stabilized={stabilized}, tracking_points={len(r_vals)}'
    }


def drift_type_classify(X, drift_idx, prefer_scores=None, **kwargs):
    """
    Main entry point for drift type classification.

    Handles multi-dimensional data by converting to 1D, then classifies.

    Parameters
    ----------
    X : array-like, shape (n, d) or (n,)
        Stream data (multi-dimensional or 1D)
    drift_idx : int
        Index where drift was detected
    prefer_scores : array-like, optional
        Pre-computed 1D projection to use instead of PCA
    **kwargs : dict
        Additional parameters for drift_type_classify_1d()
        (w_ref, w_basic, step, grow_step, etc.)

    Returns
    -------
    dict
        Classification result with keys:
        - drift_idx: detection index
        - category: 'TCD' or 'PCD'
        - subcategory: drift type
        - drift_length: samples to stabilization
        - stability_index: stabilization point
        - early_gap: initial KS distance
        - tracking_curve: r(t) data
        - note: additional info

    Example
    -------
    >>> X = np.random.randn(1000, 5)  # 1000 samples, 5 features
    >>> result = drift_type_classify(X, drift_idx=500)
    >>> print(result['subcategory'])  # 'sudden', 'incremental', etc.
    >>> print(result['category'])     # 'TCD' or 'PCD'
    """
    # Convert to 1D
    x = to_1d_series(X, prefer_scores)

    # Classify
    result = drift_type_classify_1d(x, drift_idx, **kwargs)

    return result


def drift_type_from_buffer(buffer, drift_idx_in_buffer, **kwargs):
    """
    Convenience function for streaming systems with circular buffers.

    Parameters
    ----------
    buffer : array-like
        Circular buffer containing stream data
    drift_idx_in_buffer : int
        Index of drift within the buffer (not global index)
    **kwargs : dict
        Additional parameters

    Returns
    -------
    dict
        Classification result
    """
    return drift_type_classify(buffer, drift_idx_in_buffer, **kwargs)


def drift_type_batch(X, drift_indices, **kwargs):
    """
    Classify multiple drift points in batch.

    Parameters
    ----------
    X : array-like
        Full data stream
    drift_indices : list of int
        List of drift detection indices
    **kwargs : dict
        Additional parameters

    Returns
    -------
    list of dict
        Classification results for each drift point
    """
    results = []
    for drift_idx in drift_indices:
        result = drift_type_classify(X, drift_idx, **kwargs)
        results.append(result)
    return results
