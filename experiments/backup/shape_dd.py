"""
ShapeDD: Shape-based Drift Detection using MMD.

This module provides drift detection methods based on the "triangle shape property"
of MMD statistics around drift points.

Functions:
    shape           - Original ShapeDD algorithm (baseline)
    shape_adaptive_v2 - Improved version with adaptive gamma and thresholding
    shape_snr_adaptive - SNR-aware hybrid that auto-selects strategy
    
Helper Functions:
    benjamini_hochberg_correction - FDR correction for multiple testing
    estimate_snr_robust - Robust SNR estimation for strategy selection
"""

import numpy as np
from mmd import mmd
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from sklearn.metrics.pairwise import pairwise_distances
from scipy.ndimage import uniform_filter1d


def shape(X, l1, l2, n_perm):
    """
    Original ShapeDD drift detection algorithm.
    
    Detects drift by finding "triangle peaks" in the MMD statistic curve.
    When a drift occurs, the MMD between sliding windows forms a characteristic
    triangular shape with a peak at the drift location.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int
        Half-window size for shape statistic computation
    l2 : int
        Window size for MMD statistical test
    n_perm : int
        Number of permutations for MMD p-value estimation
    
    Returns:
    --------
    res : array-like, shape (n_samples, 3)
        [:, 0] - Shape statistic value
        [:, 1] - MMD statistic  
        [:, 2] - p-value (< 0.05 indicates significant drift)
    """
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    K = apply_kernel(X, metric="rbf")
    W = np.zeros((n-2*l1, n))
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    shape_stat = np.convolve(stat, w)
    shape_prime = shape_stat[1:] * shape_stat[:-1] 
    
    res = np.zeros((n, 3))
    res[:, 2] = 1  # Default p-value = 1 (no drift)
    
    for pos in np.where(shape_prime < 0)[0]:
        if shape_stat[pos] > 0:
            res[pos, 0] = shape_stat[pos]
            a, b = max(0, pos-int(l2/2)), min(n, pos+int(l2/2))
            res[pos, 1:] = mmd(X[a:b], pos-a, n_perm)
    return res


def shape_adaptive_v2(X, l1, l2, n_perm, sensitivity='medium'):
    """
    Improved ShapeDD with adaptive gamma selection and optimized thresholding.
    
    Improvements over original shape():
    1. Adaptive RBF gamma using Scott's rule for better kernel bandwidth
    2. Minimal smoothing (window=3) to preserve drift sharpness
    3. Percentile-based threshold that adapts to signal strength
    4. Conditional FDR correction (only when detection density < 3%)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int
        Half-window size for shape statistic
    l2 : int
        Window size for MMD test
    n_perm : int
        Number of permutations for MMD test
    sensitivity : str, default='medium'
        Detection sensitivity:
        - 'none'      : No filtering, test all candidates
        - 'ultrahigh' : Very aggressive, catches weak drifts
        - 'high'      : Aggressive, good for subtle drifts
        - 'medium'    : Balanced (default)
        - 'low'       : Conservative, high precision
    
    Returns:
    --------
    res : array-like, shape (n_samples, 3)
        [:, 0] - Shape statistic
        [:, 1] - MMD statistic
        [:, 2] - p-value
    """
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    n = X.shape[0]
    
    # Adaptive gamma using Scott's rule
    n_sample = min(1000, n)
    X_sample = X[:n_sample]
    d = X.shape[1]
    
    data_std = np.std(X_sample, axis=0).mean()
    if data_std > 0:
        scott_factor = (n_sample ** (-1.0 / (d + 4)))
        sigma = data_std * scott_factor
        gamma = 1.0 / (2 * sigma**2)
    else:
        # Fallback: median heuristic
        distances = pairwise_distances(X_sample, metric='euclidean')
        distances_flat = distances[distances > 0]
        if len(distances_flat) > 0:
            median_dist = np.median(distances_flat)
            gamma = 1.0 / (2 * median_dist**2)
        else:
            gamma = 1.0

    K = apply_kernel(X, metric="rbf", gamma=gamma)
    W = np.zeros((n-2*l1, n))
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    # Minimal smoothing to preserve drift sharpness
    stat_smooth = uniform_filter1d(stat, size=3, mode='nearest')
    
    shape_stat = np.convolve(stat_smooth, w)
    shape_prime = shape_stat[1:] * shape_stat[:-1]
    
    res = np.zeros((n, 3))
    res[:, 2] = 1
    
    potential_peaks = np.where(shape_prime < 0)[0]
    
    # Adaptive threshold based on sensitivity
    if sensitivity == 'none':
        threshold = 0
    else:
        positive_shapes = shape_stat[shape_stat > 0]
        if len(positive_shapes) > 0:
            baseline = np.percentile(positive_shapes, 10)
            sensitivity_multipliers = {
                'low': 1.2,
                'medium': 0.8,
                'high': 0.5,
                'ultrahigh': 0.25
            }
            multiplier = sensitivity_multipliers.get(sensitivity, 0.8)
            percentile_threshold = baseline * multiplier
            
            # Absolute minimum floor to catch weak drifts
            noise_floor = np.percentile(positive_shapes, 5)
            absolute_minimum = noise_floor * 0.4
            threshold = min(percentile_threshold, absolute_minimum)
            
            # Safety: don't go below 20% of noise floor
            if threshold < noise_floor * 0.2:
                threshold = noise_floor * 0.2
        else:
            threshold = 0

    # Test candidates with MMD
    p_values = []
    positions = []
    
    for pos in potential_peaks:
        if shape_stat[pos] > threshold:
            res[pos, 0] = shape_stat[pos]
            a, b = max(0, pos-int(l2/2)), min(n, pos+int(l2/2))
            mmd_result = mmd(X[a:b], pos-a, n_perm)
            res[pos, 1:] = mmd_result
            p_values.append(mmd_result[1])
            positions.append(pos)

    # Conditional FDR: only apply when detection density < 3%
    detection_density = len(p_values) / n if n > 0 else 0
    
    if len(p_values) > 1 and detection_density < 0.03 and sensitivity != 'none':
        p_values_array = np.array(p_values)
        alpha_values = {
            'low': 0.01,
            'medium': 0.08,
            'high': 0.15,
            'ultrahigh': 0.25
        }
        alpha = alpha_values.get(sensitivity, 0.08)
        
        significant_indices = benjamini_hochberg_correction(p_values_array, alpha=alpha)
        significant_set = set(significant_indices)
        
        for i, pos in enumerate(positions):
            if i not in significant_set:
                res[pos, 0] = 0
                res[pos, 1] = 0
                res[pos, 2] = 1.0

    return res


def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction for multiple hypothesis testing.
    
    Controls False Discovery Rate at level alpha.
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values from multiple tests
    alpha : float, default=0.05
        Target FDR level
    
    Returns:
    --------
    significant_indices : array
        Indices of tests that remain significant after correction
    """
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    m = len(p_values)
    
    for k in range(m-1, -1, -1):
        if sorted_p[k] <= (k+1) / m * alpha:
            return sorted_indices[:k+1]
    
    return np.array([])


def estimate_snr_robust(X, window_size=200, n_samples=5, method='mmd'):
    """
    Robust SNR estimation for drift detection strategy selection.
    
    Estimates Signal-to-Noise Ratio by comparing:
    - Signal: MMD between non-overlapping windows (distribution shift)
    - Noise: MMD variance under null hypothesis (random splits)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    window_size : int, default=200
        Size of windows for MMD computation
    n_samples : int, default=5
        Number of window pairs to sample
    method : str, default='mmd'
        Distance metric ('mmd' or 'energy')
    
    Returns:
    --------
    snr : float
        Estimated signal-to-noise ratio
        - snr > 0.01: High SNR, can use aggressive detection
        - snr <= 0.01: Low SNR, use conservative detection
    """
    from sklearn.metrics.pairwise import rbf_kernel
    
    n = X.shape[0]
    if n < window_size * 3:
        return 0.5  # Insufficient data
    
    # Estimate signal (distribution shift between windows)
    signal_estimates = []
    for _ in range(n_samples):
        start1 = np.random.randint(0, n - 2*window_size)
        start2 = start1 + window_size
        
        window1 = X[start1:start1+window_size]
        window2 = X[start2:start2+window_size]
        
        if method == 'mmd':
            K_11 = rbf_kernel(window1, window1).mean()
            K_22 = rbf_kernel(window2, window2).mean()
            K_12 = rbf_kernel(window1, window2).mean()
            mmd_squared = K_11 + K_22 - 2*K_12
            signal_estimates.append(max(0, mmd_squared))
        elif method == 'energy':
            from scipy.spatial.distance import cdist
            D_12 = cdist(window1, window2).mean()
            D_11 = cdist(window1, window1).mean()
            D_22 = cdist(window2, window2).mean()
            energy_dist = 2*D_12 - D_11 - D_22
            signal_estimates.append(max(0, energy_dist))
    
    signal_variance = np.median(signal_estimates)
    
    # Estimate noise (variance under null hypothesis)
    noise_estimates = []
    for _ in range(n_samples * 2):
        start = np.random.randint(0, n - window_size)
        window = X[start:start+window_size]
        
        # Random split simulates null hypothesis
        perm = np.random.permutation(window_size)
        split = window_size // 2
        window1 = window[perm[:split]]
        window2 = window[perm[split:]]
        
        if method == 'mmd':
            K_11 = rbf_kernel(window1, window1).mean()
            K_22 = rbf_kernel(window2, window2).mean()
            K_12 = rbf_kernel(window1, window2).mean()
            mmd_squared = K_11 + K_22 - 2*K_12
            noise_estimates.append(max(0, mmd_squared))
    
    noise_variance = np.var(noise_estimates)
    
    if noise_variance > 1e-10:
        snr = signal_variance / noise_variance
    else:
        snr = 0.0
    
    return snr


def shape_snr_adaptive(X, l1=50, l2=150, n_perm=2500, snr_threshold=0.010,
                       high_snr_sensitivity='medium', low_snr_method='original'):
    """
    SNR-Aware Hybrid Drift Detector.
    
    Automatically selects detection strategy based on estimated Signal-to-Noise Ratio:
    - High SNR: Uses shape_adaptive_v2 for maximum recall
    - Low SNR: Uses original shape for maximum precision
    
    This addresses the fundamental precision-recall tradeoff:
    - Aggressive thresholds: High recall, risk of false positives
    - Conservative thresholds: High precision, risk of missed detections
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int, default=50
        Half-window size for shape statistic
    l2 : int, default=150
        Window size for MMD test
    n_perm : int, default=2500
        Number of permutations for MMD test
    snr_threshold : float, default=0.010
        Threshold for strategy selection
        - snr > threshold: Use aggressive detection
        - snr <= threshold: Use conservative detection
    high_snr_sensitivity : str, default='medium'
        Sensitivity level for high-SNR regime
    low_snr_method : str, default='original'
        Method for low-SNR regime ('original' or 'adaptive_none')
    
    Returns:
    --------
    res : array-like, shape (n_samples, 3)
        [:, 0] - Shape statistic
        [:, 1] - MMD statistic
        [:, 2] - p-value
    """
    # Estimate SNR
    estimated_snr = estimate_snr_robust(X, window_size=min(200, len(X) // 10), method='mmd')
    print(f"  [SNR-Adaptive] Estimated SNR: {estimated_snr:.3f}")

    # Select strategy based on SNR
    if estimated_snr > snr_threshold:
        print(f"  [SNR-Adaptive] Strategy: AGGRESSIVE (shape_adaptive_v2, sensitivity={high_snr_sensitivity})")
        print(f"  [SNR-Adaptive] Rationale: High SNR detected - can use aggressive threshold")
        return shape_adaptive_v2(X, l1, l2, n_perm, sensitivity=high_snr_sensitivity)
    else:
        if low_snr_method == 'original':
            print(f"  [SNR-Adaptive] Strategy: CONSERVATIVE (original ShapeDD)")
            print(f"  [SNR-Adaptive] Rationale: Low SNR detected - prioritize precision over recall")
            return shape(X, l1, l2, n_perm)
        else:
            print(f"  [SNR-Adaptive] Strategy: MODERATE (shape_adaptive_v2, no filtering)")
            print(f"  [SNR-Adaptive] Rationale: Low SNR detected - use adaptive but no FDR filtering")
            return shape_adaptive_v2(X, l1, l2, n_perm, sensitivity='none')
