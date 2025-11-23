"""
Optimally-Weighted MMD (OW-MMD) for Drift Detection

Based on: Bharti et al., ICML 2023
"Optimally-weighted Estimators of the Maximum Mean Discrepancy"
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel


def rbf_kernel_ow(X, Y, gamma='auto'):
    """
    RBF kernel computation with median heuristic.
    """
    if gamma == 'auto':
        all_data = np.vstack([X, Y])
        distances = cdist(all_data, all_data, metric='euclidean')
        distances = distances[distances > 0]
        if len(distances) > 0:
            gamma = 1.0 / (2 * np.median(distances)**2)
        else:
            gamma = 1.0

    distances_sq = cdist(X, Y, metric='sqeuclidean')
    return np.exp(-gamma * distances_sq)


def compute_optimal_weights(kernel_matrix, method='variance_reduction'):
    """
    Compute optimal weights for kernel evaluations.
    """
    n = kernel_matrix.shape[0]

    if method == 'uniform':
        return np.ones((n, n)) / (n * n)

    elif method == 'variance_reduction':
        K_off = kernel_matrix.copy()
        np.fill_diagonal(K_off, 0)

        k_sums = np.sum(K_off, axis=1)
        k_sums = np.maximum(k_sums, 1e-10)

        inv_weights = 1.0 / np.sqrt(k_sums)
        weights = np.outer(inv_weights, inv_weights)
        np.fill_diagonal(weights, 0)
        weights = weights / np.sum(weights)

        return weights

    else:
        return np.ones((n, n)) / (n * n)


def mmd_ow(X, s=None, gamma='auto', weight_method='variance_reduction'):
    """
    Compute Optimally-Weighted MMD between two halves of X.

    Args:
        X: Data matrix (n_samples, n_features)
        s: Split point (default: half)
        gamma: RBF kernel bandwidth
        weight_method: Weighting strategy

    Returns:
        mmd_value: OW-MMD statistic
        threshold: Adaptive threshold (using mean + 3*std heuristic)
    """
    if s is None:
        s = int(X.shape[0] / 2)

    # Split data
    X_ref = X[:s]
    X_test = X[s:]

    m, n = X_ref.shape[0], X_test.shape[0]

    # Compute kernel matrices
    K_XX = rbf_kernel_ow(X_ref, X_ref, gamma)
    K_YY = rbf_kernel_ow(X_test, X_test, gamma)
    K_XY = rbf_kernel_ow(X_ref, X_test, gamma)

    # Compute optimal weights
    W_XX = compute_optimal_weights(K_XX, weight_method)
    W_YY = compute_optimal_weights(K_YY, weight_method)
    W_XY = np.ones((m, n)) / (m * n)

    # Weighted MMD computation
    term1 = np.sum(W_XX * K_XX)
    term2 = np.sum(W_YY * K_YY)
    term3 = np.sum(W_XY * K_XY)

    mmd_squared = term1 + term2 - 2 * term3
    mmd_value = np.sqrt(max(0, mmd_squared))

    # Simple threshold heuristic
    # In practice, this should be calibrated on null distribution
    threshold = 0.1

    return mmd_value, threshold


def shapedd_ow_mmd(X, l1=50, l2=150, gamma='auto'):
    """
    ShapeDD-OW-MMD Hybrid: Use OW-MMD statistics with geometric analysis.

    This computes a sequence of OW-MMD values over sliding windows,
    then looks for geometric patterns (triangles, peaks, zero-crossings).

    Args:
        X: Data stream (n_samples, n_features)
        l1: Reference window size
        l2: Test window size
        gamma: RBF kernel parameter

    Returns:
        pattern_score: Geometric pattern strength
        mmd_max: Maximum MMD in sequence
    """
    n_samples = len(X)
    mmd_sequence = []

    # Slide through stream
    step_size = 25
    for ref_start in range(0, n_samples - l1 - l2, step_size):
        ref_end = ref_start + l1
        test_start = ref_end
        test_end = test_start + l2

        if test_end > n_samples:
            break

        # Extract windows
        ref_window = X[ref_start:ref_end]
        test_window = X[test_start:test_end]

        # Compute OW-MMD
        window_combined = np.vstack([ref_window, test_window])
        mmd_val, _ = mmd_ow(window_combined, s=l1, gamma=gamma)
        mmd_sequence.append(mmd_val)

    if len(mmd_sequence) < 5:
        return 0.0, 0.0

    mmd_array = np.array(mmd_sequence)

    # Normalize
    mmd_min, mmd_max_val = mmd_array.min(), mmd_array.max()
    if mmd_max_val - mmd_min < 1e-10:
        return 0.0, mmd_max_val

    mmd_norm = (mmd_array - mmd_min) / (mmd_max_val - mmd_min)

    # Check for triangle pattern
    peak_idx = np.argmax(mmd_norm)

    # Pattern score based on:
    # 1. Peak position (middle is good)
    # 2. Rise before peak
    # 3. Fall after peak

    pattern_score = 0.0

    # Peak in middle region
    if 0.2 < peak_idx / len(mmd_norm) < 0.8:
        pattern_score += 0.3

    # Rising before peak
    if peak_idx > 1:
        rise = mmd_norm[:peak_idx]
        if len(rise) > 0 and np.sum(np.diff(rise) > 0) / len(rise) > 0.5:
            pattern_score += 0.35

    # Falling after peak
    if peak_idx < len(mmd_norm) - 2:
        fall = mmd_norm[peak_idx:]
        if len(fall) > 0 and np.sum(np.diff(fall) < 0) / len(fall) > 0.5:
            pattern_score += 0.35

    return pattern_score, mmd_max_val
