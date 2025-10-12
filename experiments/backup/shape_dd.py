import numpy as np
from mmd import mmd
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from sklearn.metrics.pairwise import pairwise_distances
from scipy.ndimage import uniform_filter1d

def shape(X, l1, l2, n_perm):
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    K = apply_kernel(X, metric="rbf")
    W = np.zeros( (n-2*l1,n) )
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    shape = np.convolve(stat,w)
    shape_prime = shape[1:]*shape[:-1] 
    
    res = np.zeros((n,3))
    res[:,2] = 1
    for pos in np.where(shape_prime < 0)[0]:
        if shape[pos] > 0:
            res[pos,0] = shape[pos]
            a,b = max(0,pos-int(l2/2)),min(n,pos+int(l2/2))
            res[pos,1:] = mmd(X[a:b], pos-a, n_perm)
    return res

def shape_adaptive(X, l1, l2, n_perm):
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    
    # IMPROVEMENT 1: GAMMA CHOICE BASE ON DATA
    # Adaptive bandwidth selection using Scott's rule
    # Scott's rule: sigma = std * n^(-1/(d+4))
    n_sample = min(1000, n)
    X_sample = X[:n_sample]
    d = X.shape[1]
    
    # Calculate data-driven bandwidth
    data_std = np.std(X_sample, axis=0).mean()
    if data_std > 0:
        # Scott's rule for bandwidth selection
        scott_factor = (n_sample ** (-1.0 / (d + 4)))
        sigma = data_std * scott_factor
        gamma = 1.0 / (2 * sigma**2)
    else:
        # Fallback: use median distance heuristic
        distances = pairwise_distances(X_sample, metric='euclidean')
        distances_flat = distances[distances > 0]
        if len(distances_flat) > 0:
            median_dist = np.median(distances_flat)
            gamma = 1.0 / (2 * median_dist**2)
        else:
            gamma = 1.0  # Default fallback
    
    K = apply_kernel(X, metric="rbf", gamma=gamma)
    W = np.zeros((n-2*l1, n))
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    # IMPROVEMENT 2: SMOOTHING THE DRIFT POINTS
    # Adaptive smoothing: reduce window for smaller l1 to maintain responsiveness
    # Use sqrt scaling instead of linear to avoid over-smoothing
    smooth_window = max(3, int(np.sqrt(l1)))
    stat_smooth = uniform_filter1d(stat, size=smooth_window, mode='nearest')

    shape = np.convolve(stat_smooth, w)
    shape_prime = shape[1:]*shape[:-1] 
    
    res = np.zeros((n,3))
    res[:,2] = 1

    # IMPROVEMENT 3: THRESHOLD TO FILTER THE LOW VALUES
    # Improved peak filtering with statistical threshold
    potential_peaks = np.where(shape_prime < 0)[0]
    
    # Adaptive threshold based on shape distribution
    # Use mean + k*std where k depends on desired false positive rate
    # For FPR ~ 0.05, use k = 1.645 (z-score for 95% confidence)
    shape_mean = np.mean(shape)
    shape_std = np.std(shape)
    # Scale threshold with log(n) to control FPR
    log_n_factor = np.log(max(n/1000, 1))
    # threshold = shape_mean + (0.05 + 0.02 * log_n_factor) * shape_std
    threshold = shape_mean + (0.015) * shape_std
    
    # Collect all p-values first
    p_values = []
    positions = []

    for pos in potential_peaks:
        # Apply statistical threshold to filter noise peaks
        # if shape[pos] > threshold:
        if shape[pos] > 0:
            res[pos,0] = shape[pos]
            a, b = max(0, pos-int(l2/2)), min(n, pos+int(l2/2))
            res[pos,1:] = mmd(X[a:b], pos-a, n_perm)
            # _, p_val = mmd(X[a:b], pos-a, n_perm)
            # p_values.append(p_val)
            # positions.append(pos)
    
    # IMPROVEMENT 4: APPLY BENJAMI HOCHBERG TO LOWER FPR
    # TODO: Uncomment to enable FDR correction
    # Apply FDR correction
    # if len(p_values) > 0:
    #     p_values = np.array(p_values)
    #     significant_indices = benjamini_hochberg_correction(p_values, alpha=0.05)
        
    #     for idx in significant_indices:
    #         pos = positions[idx]
    #         res[pos,0] = shape[pos]
    #         res[pos,1] = p_values[idx]
    
    return res

def benjamini_hochberg_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR control"""
    
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    m = len(p_values)
    
    # Find largest k such that P(k) <= (k/m) * alpha
    for k in range(m-1, -1, -1):
        if sorted_p[k] <= (k+1) / m * alpha:
            # Reject hypotheses 0, 1, ..., k
            significant_indices = sorted_indices[:k+1]
            return significant_indices
    
    return np.array([])  # No rejections

def shape_fully_adaptive(X, sensitivity='medium', min_window=30, max_window=None, 
                         n_perm=1000, alpha=0.05, enable_fdr=True):
    """
    Fully adaptive ShapeDD that automatically determines window parameters.
    
    Philosophy similar to ADWIN:
    - Adapts window sizes based on data characteristics
    - No fixed parameters required
    - Automatically adjusts sensitivity based on noise estimates
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data stream
    sensitivity : str, default='medium'
        Detection sensitivity: 'low', 'medium', 'high'
        Controls trade-off between detection speed and false positive rate
    min_window : int, default=30
        Minimum window size (safety constraint)
    max_window : int, optional
        Maximum window size (default: n/10)
    n_perm : int, default=1000
        Number of permutations (adaptive: reduces for large n)
    alpha : float, default=0.05
        Significance level
    enable_fdr : bool, default=True
        Apply Benjamini-Hochberg FDR correction
        
    Returns
    -------
    res : ndarray of shape (n_samples, 4)
        Column 0: Shape statistic value
        Column 1: MMD statistic
        Column 2: p-value
        Column 3: FDR-adjusted significance (0=not sig, 1=sig)
    """
    n, d = X.shape
    
    # ========================================================================
    # ADAPTIVE PARAMETER SELECTION
    # ========================================================================
    
    # 1. ADAPTIVE l1 (detection window) based on data length and dimensionality
    # Rule: l1 should capture "local context" but be small enough to detect changes
    # Heuristic: l1 = c * n^(1/3) / log(d+1)
    # Rationale: 
    # - n^(1/3): sub-linear scaling with data length
    # - 1/log(d+1): smaller windows for high-dim (curse of dimensionality)
    
    sensitivity_factors = {'low': 1.5, 'medium': 1.0, 'high': 0.7}
    c = sensitivity_factors.get(sensitivity, 1.0)
    
    l1_base = int(c * (n ** (1/3)) / np.log(d + 2))
    l1 = max(min_window, min(l1_base, n // 10))
    
    # 2. ADAPTIVE l2 (MMD window) based on l1
    # Rule: l2 should be larger than l1 to capture drift context
    # Heuristic: l2 = 2.5 * l1 (empirically validated range: 2-3x)
    l2 = min(int(2.5 * l1), n // 5)
    
    # 3. ADAPTIVE n_perm based on data size (computational efficiency)
    # Reduce permutations for large datasets
    if n > 2000:
        n_perm = min(n_perm, 500)
    elif n > 5000:
        n_perm = min(n_perm, 250)
    
    # 4. ADAPTIVE GAMMA (RBF bandwidth) using median heuristic + Scott's rule
    # Sample for efficiency
    sample_size = min(500, n)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # Compute pairwise distances
    distances = pairwise_distances(X_sample, metric='euclidean')
    distances_flat = distances[np.triu_indices_from(distances, k=1)]
    
    if len(distances_flat) > 0:
        # Median heuristic (robust to outliers)
        median_dist = np.median(distances_flat)
        
        # Scott's rule adjustment for dimensionality
        scott_factor = (sample_size ** (-1.0 / (d + 4)))
        
        # Combined: median heuristic * Scott's adjustment
        sigma = median_dist * scott_factor
        gamma = 1.0 / (2 * sigma**2) if sigma > 0 else 1.0
    else:
        gamma = 1.0
    
    # ========================================================================
    # CORE SHAPEDD COMPUTATION (as original)
    # ========================================================================
    
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    K = apply_kernel(X, metric="rbf", gamma=gamma)
    W = np.zeros((n-2*l1, n))
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w
    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)
    
    # ADAPTIVE SMOOTHING: based on estimated noise level
    # Estimate noise from high-frequency components
    stat_diff = np.diff(stat)
    noise_estimate = np.median(np.abs(stat_diff)) / 0.6745  # MAD estimator
    
    # Adaptive smoothing window: more smoothing for noisy data
    if noise_estimate > 0:
        # Larger smooth_window for noisier data
        smooth_window = max(3, min(int(np.sqrt(l1) * (1 + noise_estimate)), l1))
    else:
        smooth_window = max(3, int(np.sqrt(l1)))
    
    stat_smooth = uniform_filter1d(stat, size=smooth_window, mode='nearest')
    
    shape = np.convolve(stat_smooth, w)
    shape_prime = shape[1:] * shape[:-1]
    
    # ========================================================================
    # PEAK DETECTION WITH ADAPTIVE THRESHOLD
    # ========================================================================
    
    potential_peaks = np.where(shape_prime < 0)[0]
    
    # ADAPTIVE THRESHOLD based on distribution of shape values
    # Use robust statistics (median, MAD) instead of mean/std
    shape_median = np.median(shape)
    shape_mad = np.median(np.abs(shape - shape_median))
    
    # Threshold = median + k * MAD, where k depends on sensitivity
    threshold_multipliers = {'low': 3.0, 'medium': 2.0, 'high': 1.5}
    k = threshold_multipliers.get(sensitivity, 2.0)
    threshold = shape_median + k * shape_mad
    
    # ========================================================================
    # MMD TESTING AT DETECTED PEAKS
    # ========================================================================
    
    res = np.zeros((n, 4))
    res[:, 2] = 1.0  # Default p-value
    res[:, 3] = 0.0  # Default: not significant
    
    p_values = []
    positions = []
    mmd_stats = []

    for pos in potential_peaks:
        if shape[pos] > threshold:  # FIXED: Actually use threshold
            res[pos, 0] = shape[pos]
            
            # Adaptive window around drift point
            a = max(0, pos - int(l2/2))
            b = min(n, pos + int(l2/2))
            
            mmd_stat, p_val = mmd(X[a:b], pos-a, n_perm)
            
            res[pos, 1] = mmd_stat
            res[pos, 2] = p_val
            
            p_values.append(p_val)
            positions.append(pos)
            mmd_stats.append(mmd_stat)
    
    # ========================================================================
    # FDR CORRECTION (Benjamini-Hochberg)
    # ========================================================================
    
    if enable_fdr and len(p_values) > 0:
        p_values_array = np.array(p_values)
        significant_indices = benjamini_hochberg_correction(p_values_array, alpha=alpha)
        
        # Mark significant detections
        for idx in significant_indices:
            pos = positions[idx]
            res[pos, 3] = 1.0  # Mark as FDR-significant
    else:
        # Without FDR, use raw p-value threshold
        for i, pos in enumerate(positions):
            if p_values[i] < alpha:
                res[pos, 3] = 1.0
    
    return res
