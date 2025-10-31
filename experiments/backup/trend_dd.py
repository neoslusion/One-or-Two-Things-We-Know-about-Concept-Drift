"""
Trend-Based Drift Verification (Trend DD)

Distinguishes subtle real drift from noise by analyzing temporal trends in statistical distances.

Key Insight:
- Real drift shows directional trends (monotonic increase in KS distance)
- Noise shows random oscillation (no trend, low R²)

Methods:
- trend_verify_linear(): Linear regression on KS distance history
- trend_verify_cusum(): Cumulative sum for detecting persistent shifts
- trend_verify_multi_scale(): Multi-window comparison for accumulation patterns
"""

import numpy as np
from scipy.stats import ks_2samp, linregress


def ks_distance_multivariate(X_ref, X_curr):
    """
    Calculate average KS distance across all features.

    Parameters
    ----------
    X_ref : array-like, shape (n_ref, n_features)
        Reference window
    X_curr : array-like, shape (n_curr, n_features)
        Current window

    Returns
    -------
    float
        Average KS statistic across features
    """
    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)
    if X_curr.ndim == 1:
        X_curr = X_curr.reshape(-1, 1)

    n_features = X_ref.shape[1]
    ks_distances = []

    for feat_idx in range(n_features):
        ks_stat, _ = ks_2samp(X_ref[:, feat_idx], X_curr[:, feat_idx])
        ks_distances.append(ks_stat)

    return np.mean(ks_distances)


def trend_verify_linear(ks_history, slope_threshold=0.003, r2_threshold=0.6):
    """
    Verify drift by analyzing linear trend in KS distance history.

    Real drift shows positive slope with high R² (good linear fit).
    Noise shows near-zero slope with low R² (no pattern).

    Parameters
    ----------
    ks_history : array-like, shape (n_windows,)
        History of KS distances over time (at least 5 values recommended)
    slope_threshold : float, default=0.003
        Minimum slope to consider as drift (per window unit)
    r2_threshold : float, default=0.6
        Minimum R² for linear fit to confirm trend

    Returns
    -------
    result : dict
        - is_drift: bool, whether drift is confirmed
        - confidence: float [0, 1], confidence in detection
        - slope: float, slope of linear trend
        - r_squared: float, R² of linear fit
        - trend_pvalue: float, p-value for slope significance
        - ks_increase: float, total increase in KS distance
        - reason: str, explanation of decision

    Example
    -------
    Drift case:
    >>> ks_hist = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    >>> result = trend_verify_linear(ks_hist)
    >>> result['is_drift']  # True
    >>> result['r_squared']  # ~0.98 (excellent fit)

    Noise case:
    >>> ks_hist = [0.08, 0.07, 0.09, 0.08, 0.10, 0.09, 0.08, 0.09, 0.08, 0.10]
    >>> result = trend_verify_linear(ks_hist)
    >>> result['is_drift']  # False
    >>> result['r_squared']  # ~0.01 (no fit)
    """
    ks_history = np.array(ks_history)

    if len(ks_history) < 3:
        return {
            'is_drift': False,
            'confidence': 0.0,
            'slope': 0.0,
            'r_squared': 0.0,
            'trend_pvalue': 1.0,
            'ks_increase': 0.0,
            'reason': 'insufficient_history'
        }

    # Fit linear regression
    x = np.arange(len(ks_history))
    slope, intercept, r_value, p_value, std_err = linregress(x, ks_history)
    r_squared = r_value ** 2

    # Calculate KS increase
    ks_increase = ks_history[-1] - ks_history[0]

    # Decision logic
    is_drift = (slope > slope_threshold) and (r_squared > r2_threshold)

    # Confidence score (how much we exceed thresholds)
    slope_ratio = slope / slope_threshold if slope_threshold > 0 else 0
    r2_ratio = r_squared / r2_threshold if r2_threshold > 0 else 0
    confidence = min(1.0, (slope_ratio * r2_ratio) / 2.0)

    # Reason string
    if is_drift:
        reason = f"clear_trend: slope={slope:.5f}, R²={r_squared:.3f}"
    elif slope <= 0:
        reason = f"no_trend: slope={slope:.5f} (not increasing)"
    elif r_squared < r2_threshold:
        reason = f"weak_trend: R²={r_squared:.3f} < {r2_threshold} (noisy pattern)"
    else:
        reason = f"subtle_trend: slope={slope:.5f} < {slope_threshold} (below threshold)"

    return {
        'is_drift': is_drift,
        'confidence': confidence,
        'slope': slope,
        'r_squared': r_squared,
        'trend_pvalue': p_value,
        'ks_increase': ks_increase,
        'current_ks': ks_history[-1],
        'baseline_ks': ks_history[0],
        'reason': reason
    }


def trend_verify_cusum(ks_history, baseline_ks=None, allowance=0.02, threshold=0.10):
    """
    Verify drift using Cumulative Sum (CUSUM) control chart.

    Detects small but persistent shifts by accumulating deviations from baseline.
    Real drift accumulates positive deviations.
    Noise oscillates around baseline, doesn't accumulate.

    Parameters
    ----------
    ks_history : array-like, shape (n_windows,)
        History of KS distances
    baseline_ks : float, optional
        Baseline KS distance (if None, uses first value)
    allowance : float, default=0.02
        Slack parameter k (half of noise range)
    threshold : float, default=0.10
        Threshold h for detection (cumulative deviation limit)

    Returns
    -------
    result : dict
        - is_drift: bool, whether cumsum exceeds threshold
        - cumsum: float, current cumulative sum
        - cumsum_history: array, cumsum at each step
        - threshold: float, detection threshold used
        - reason: str, explanation

    Example
    -------
    Drift case (persistent increase):
    >>> ks_hist = [0.06, 0.08, 0.09, 0.11, 0.12, 0.14]
    >>> result = trend_verify_cusum(ks_hist, baseline_ks=0.06)
    >>> result['is_drift']  # True (cumsum grows)

    Noise case (oscillating):
    >>> ks_hist = [0.06, 0.08, 0.07, 0.09, 0.06, 0.08]
    >>> result = trend_verify_cusum(ks_hist, baseline_ks=0.06)
    >>> result['is_drift']  # False (cumsum resets)
    """
    ks_history = np.array(ks_history)

    if baseline_ks is None:
        baseline_ks = ks_history[0] if len(ks_history) > 0 else 0.0

    cumsum_history = []
    cumsum = 0.0

    for ks_val in ks_history:
        deviation = ks_val - baseline_ks
        # Only accumulate if deviation exceeds allowance
        cumsum = max(0, cumsum + deviation - allowance)
        cumsum_history.append(cumsum)

    is_drift = cumsum > threshold

    reason = f"cumsum={cumsum:.4f} {'>' if is_drift else '<='} threshold={threshold:.4f}"

    return {
        'is_drift': is_drift,
        'cumsum': cumsum,
        'cumsum_history': np.array(cumsum_history),
        'threshold': threshold,
        'baseline_ks': baseline_ks,
        'current_ks': ks_history[-1],
        'reason': reason
    }


def trend_verify_multi_scale(X, drift_idx, window_sizes=[250, 500, 1000]):
    """
    Verify drift by comparing across multiple time scales.

    Real drift shows increasing effect with longer windows (accumulation).
    Noise shows no pattern across scales.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Full data stream
    drift_idx : int
        Index where drift was detected
    window_sizes : list of int
        Different window sizes to compare (default: [250, 500, 1000])

    Returns
    -------
    result : dict
        - is_drift: bool, whether accumulation pattern is detected
        - slope: float, slope of KS vs window_size
        - ks_values: array, KS distance for each window size
        - pattern: str, 'accumulating' or 'flat'
        - reason: str, explanation

    Example
    -------
    Drift case (accumulation):
    >>> ks_values = [0.10, 0.14, 0.20]  # increases with window size
    >>> slope = 0.000133  # positive
    >>> is_drift = True

    Noise case (no pattern):
    >>> ks_values = [0.10, 0.09, 0.08]  # random or decreasing
    >>> slope = -0.000027  # near zero or negative
    >>> is_drift = False
    """
    n = len(X)

    # Ensure drift_idx allows for smallest window before it
    min_window = min(window_sizes)
    if drift_idx < min_window:
        return {
            'is_drift': False,
            'slope': 0.0,
            'ks_values': [],
            'pattern': 'insufficient_data',
            'reason': f'drift_idx={drift_idx} < min_window={min_window}'
        }

    ks_values = []
    valid_window_sizes = []

    for win_size in window_sizes:
        # Reference: [drift_idx - win_size, drift_idx]
        # Current: [drift_idx, drift_idx + win_size]

        if drift_idx < win_size:
            continue  # Skip if not enough pre-drift data

        if drift_idx + win_size > n:
            continue  # Skip if not enough post-drift data

        X_ref = X[drift_idx - win_size:drift_idx]
        X_curr = X[drift_idx:drift_idx + win_size]

        ks_dist = ks_distance_multivariate(X_ref, X_curr)
        ks_values.append(ks_dist)
        valid_window_sizes.append(win_size)

    if len(ks_values) < 2:
        return {
            'is_drift': False,
            'slope': 0.0,
            'ks_values': np.array(ks_values),
            'window_sizes': np.array(valid_window_sizes),
            'pattern': 'insufficient_windows',
            'reason': f'only {len(ks_values)} valid windows'
        }

    # Calculate slope of KS vs window size
    ks_values = np.array(ks_values)
    valid_window_sizes = np.array(valid_window_sizes)

    # Slope: change in KS per sample
    slope = (ks_values[-1] - ks_values[0]) / (valid_window_sizes[-1] - valid_window_sizes[0])

    # Decision: positive slope indicates accumulation (drift)
    # Threshold: 0.0001 per sample (for 1000 samples, expect 0.1 KS increase)
    is_drift = slope > 0.0001

    pattern = 'accumulating' if is_drift else 'flat'

    reason = f"slope={slope:.6f} per sample, KS: {ks_values[0]:.3f} → {ks_values[-1]:.3f}"

    return {
        'is_drift': is_drift,
        'slope': slope,
        'ks_values': ks_values,
        'window_sizes': valid_window_sizes,
        'pattern': pattern,
        'reason': reason
    }


def trend_verify_integrated(X_ref, ks_history=None, X_stream=None, drift_idx=None,
                            method='linear', **kwargs):
    """
    Integrated trend verification with automatic method selection.

    This is a convenience function that routes to the appropriate verification method.

    Parameters
    ----------
    X_ref : array-like
        Reference window (stable pre-drift data)
    ks_history : array-like, optional
        Pre-computed KS distance history (for 'linear' or 'cusum' methods)
    X_stream : array-like, optional
        Full data stream (for 'multi_scale' method)
    drift_idx : int, optional
        Drift detection index (for 'multi_scale' method)
    method : str, default='linear'
        Verification method: 'linear', 'cusum', 'multi_scale', or 'ensemble'
    **kwargs : dict
        Additional parameters for specific methods

    Returns
    -------
    result : dict
        Verification result with 'is_drift' and other method-specific fields

    Examples
    --------
    # Linear trend verification
    >>> result = trend_verify_integrated(X_ref, ks_history=[0.08, 0.10, 0.12, 0.14])

    # CUSUM verification
    >>> result = trend_verify_integrated(X_ref, ks_history=[0.06, 0.08, 0.11],
    ...                                   method='cusum')

    # Multi-scale verification
    >>> result = trend_verify_integrated(X_ref, X_stream=X, drift_idx=500,
    ...                                   method='multi_scale')

    # Ensemble (all methods, majority vote)
    >>> result = trend_verify_integrated(X_ref, ks_history=ks_hist,
    ...                                   X_stream=X, drift_idx=500,
    ...                                   method='ensemble')
    """
    if method == 'linear':
        if ks_history is None:
            raise ValueError("ks_history required for linear method")
        return trend_verify_linear(ks_history, **kwargs)

    elif method == 'cusum':
        if ks_history is None:
            raise ValueError("ks_history required for cusum method")
        return trend_verify_cusum(ks_history, **kwargs)

    elif method == 'multi_scale':
        if X_stream is None or drift_idx is None:
            raise ValueError("X_stream and drift_idx required for multi_scale method")
        return trend_verify_multi_scale(X_stream, drift_idx, **kwargs)

    elif method == 'ensemble':
        # Run all methods and combine results
        results = {}
        votes = 0
        total = 0

        if ks_history is not None:
            linear_result = trend_verify_linear(ks_history, **kwargs.get('linear_kwargs', {}))
            cusum_result = trend_verify_cusum(ks_history, **kwargs.get('cusum_kwargs', {}))
            results['linear'] = linear_result
            results['cusum'] = cusum_result
            votes += int(linear_result['is_drift']) + int(cusum_result['is_drift'])
            total += 2

        if X_stream is not None and drift_idx is not None:
            multi_scale_result = trend_verify_multi_scale(X_stream, drift_idx,
                                                           **kwargs.get('multi_scale_kwargs', {}))
            results['multi_scale'] = multi_scale_result
            votes += int(multi_scale_result['is_drift'])
            total += 1

        if total == 0:
            raise ValueError("Insufficient parameters for ensemble method")

        # Majority vote
        is_drift = votes >= (total / 2)
        confidence = votes / total

        return {
            'is_drift': is_drift,
            'confidence': confidence,
            'votes': votes,
            'total': total,
            'individual_results': results,
            'reason': f"{votes}/{total} methods agree on drift"
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'cusum', 'multi_scale', or 'ensemble'")
