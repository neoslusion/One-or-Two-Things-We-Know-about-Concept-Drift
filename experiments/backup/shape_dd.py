"""
ShapeDD: Shape-based Drift Detection using MMD.

This module provides drift detection methods based on the "triangle shape property"
of MMD statistics around drift points.

Functions:
    shape           - Original ShapeDD algorithm (baseline)
    shape_mmdagg    - [NEW] Aggregated MMD with multiple bandwidths (JMLR 2023)
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

def shape_mmdagg(X, l1=50, l2=150, n_bandwidths=10, alpha=0.05):
    """
    ShapeDD with Aggregated MMD (MMDAgg) - uses multiple kernel bandwidths.
    Schrab et al., "MMD Aggregated Two-Sample Test" (JMLR 2023)
    
    The key insight: Instead of using a single kernel bandwidth (which may be
    suboptimal), aggregate MMD statistics over multiple bandwidths and use
    the minimum p-value with Bonferroni correction.
    
    Improvements over original ShapeDD:
    1. Multiple bandwidths: Tests across range of kernel scales
    2. Adaptive: No need to tune kernel bandwidth parameter
    3. Robust: Bonferroni correction controls Type I error
    4. Fast: Uses analytical threshold instead of permutation test
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int, default=50
        Half-window size for shape statistic computation
    l2 : int, default=150
        Window size for MMD statistical test
    n_bandwidths : int, default=10
        Number of kernel bandwidths to aggregate over
    alpha : float, default=0.05
        Significance level (before Bonferroni correction)
    
    Returns:
    --------
    res : array-like, shape (n_samples, 3)
        [:, 0] - Shape statistic value
        [:, 1] - Maximum MMD statistic (over bandwidths)
        [:, 2] - Corrected p-value (< 0.05 indicates significant drift)
    
    References:
    -----------
    Schrab, A., et al. (2023). "MMD Aggregated Two-Sample Test."
    Journal of Machine Learning Research, 24(194), 1-72.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    from scipy.stats import norm
    
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    n = X.shape[0]
    
    # Step 1: Compute bandwidth range using median heuristic
    n_sample = min(500, n)
    sample_idx = np.random.choice(n, n_sample, replace=False)
    X_sample = X[sample_idx]
    
    from scipy.spatial.distance import pdist
    distances = pdist(X_sample, metric='euclidean')
    if len(distances) > 0 and np.any(distances > 0):
        median_dist = np.median(distances[distances > 0])
    else:
        median_dist = 1.0
    
    # Create bandwidth range: [0.1 * median, 10 * median] on log scale
    sigma_range = np.logspace(np.log10(0.1 * median_dist), 
                               np.log10(10 * median_dist), 
                               n_bandwidths)
    gamma_range = 1.0 / (2 * sigma_range**2)
    
    # Step 2: Compute shape statistic with middle bandwidth
    gamma_default = gamma_range[len(gamma_range)//2]
    K = rbf_kernel(X, X, gamma=gamma_default)
    W = np.zeros((n-2*l1, n))
    
    for i in range(n-2*l1):
        W[i, i:i+2*l1] = w
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)
    
    # Minimal smoothing to preserve drift sharpness
    stat_smooth = uniform_filter1d(stat, size=3, mode='nearest')
    
    shape_stat = np.convolve(stat_smooth, w)
    shape_prime = shape_stat[1:] * shape_stat[:-1]
    
    res = np.zeros((n, 3))
    res[:, 2] = 1  # Default p-value = 1 (no drift)
    
    # Step 3: At each peak, compute aggregated MMD test
    potential_peaks = np.where(shape_prime < 0)[0]
    
    for pos in potential_peaks:
        if shape_stat[pos] > 0:
            res[pos, 0] = shape_stat[pos]
            
            # Extract window around peak
            a, b = max(0, pos - int(l2/2)), min(n, pos + int(l2/2))
            window = X[a:b]
            split_point = pos - a
            
            if split_point < 5 or len(window) - split_point < 5:
                continue
                
            X1 = window[:split_point]
            X2 = window[split_point:]
            
            # Compute MMD for each bandwidth
            mmd_values = []
            p_values = []
            
            for gamma in gamma_range:
                # Compute unbiased MMD^2 estimate
                K11 = rbf_kernel(X1, X1, gamma=gamma)
                K22 = rbf_kernel(X2, X2, gamma=gamma)
                K12 = rbf_kernel(X1, X2, gamma=gamma)
                
                n1, n2 = len(X1), len(X2)
                
                # Unbiased MMD^2 (remove diagonal terms)
                if n1 > 1 and n2 > 1:
                    mmd2 = (np.sum(K11) - np.trace(K11)) / (n1 * (n1-1)) + \
                           (np.sum(K22) - np.trace(K22)) / (n2 * (n2-1)) - \
                           2 * np.mean(K12)
                else:
                    mmd2 = 0
                
                mmd_values.append(max(0, mmd2))
                
                # Asymptotic p-value using Gaussian approximation
                var_mmd = 4 * (np.var(K12) + 1e-10)
                z_score = mmd2 / np.sqrt(var_mmd / min(n1, n2) + 1e-10)
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
                p_values.append(p_value)
            
            # Aggregate: minimum p-value with Bonferroni correction
            min_p = min(p_values) * n_bandwidths
            min_p = min(min_p, 1.0)
            
            # Store best MMD and corrected p-value
            best_idx = np.argmin(p_values)
            res[pos, 1] = mmd_values[best_idx]
            res[pos, 2] = min_p
    
    return res
