import numpy as np
from scipy.spatial.distance import cdist, pdist

def rbf_kernel(X, Y, gamma='auto'):
    """
    RBF (Gaussian) kernel: k(x,y) = exp(-gamma * ||x-y||^2)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        First set of samples
    Y : array-like, shape (m_samples, n_features)
        Second set of samples
    gamma : str or float, default='auto'
        Bandwidth parameter. 'auto' uses median heuristic.
    
    Returns:
    --------
    K : array-like, shape (n_samples, m_samples)
        Kernel matrix
    """
    if gamma == 'auto':
        # Median heuristic for bandwidth selection
        all_data = np.vstack([X, Y])
        distances = cdist(all_data, all_data, metric='euclidean')
        distances = distances[distances > 0]
        if len(distances) > 0:
            gamma = 1.0 / (2 * np.median(distances)**2)
        else:
            gamma = 1.0
    
    distances_sq = cdist(X, Y, metric='sqeuclidean')
    return np.exp(-gamma * distances_sq)


def compute_gamma_median_heuristic(X):
    """
    Compute gamma using median heuristic.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data samples
    
    Returns:
    --------
    gamma : float
        Kernel bandwidth parameter
    """
    # Subsample for efficiency if large
    n = len(X)
    n_sample = min(1000, n)
    if n > n_sample:
        idx = np.random.choice(n, n_sample, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    
    distances = pdist(X_sample, metric='euclidean')
    if len(distances) > 0 and np.any(distances > 0):
        median_dist = np.median(distances[distances > 0])
        return 1.0 / (2 * median_dist**2)
    return 1.0

def compute_optimal_weights(K, method: str = 'variance_reduction'):
    """
    Compute optimal weights for MMD estimation.
    
    The key insight from Bharti et al. (2023) is that variance-optimal weights
    are inversely proportional to local kernel density. Points in sparse regions
    (distribution boundaries) get higher weights, improving sensitivity to drift.
    
    Parameters:
    -----------
    K : array-like, shape (n, n)
        Kernel matrix
    method : str, default='variance_reduction'
        Weighting strategy:
        - 'uniform': Standard V-statistic (equal weights)
        - 'variance_reduction': Optimal for variance minimization
        - 'adaptive': Density-based weighting
    
    Returns:
    --------
    W : array-like, shape (n, n)
        Weight matrix with zeros on diagonal, normalized to sum to 1
    """
    n = K.shape[0]
    
    if method == 'uniform':
        # Standard V-statistic (equal weights)
        W = np.ones((n, n)) / (n * n)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)
    
    elif method == 'variance_reduction':
        # Variance-optimal weights (Bharti et al., 2023)
        # w_i proportional to 1/sqrt(sum_j K(x_i, x_j))
        K_off = K.copy()
        np.fill_diagonal(K_off, 0)
        
        k_sums = np.sum(K_off, axis=1)
        k_sums = np.maximum(k_sums, 1e-10)  # Numerical stability
        
        inv_weights = 1.0 / np.sqrt(k_sums)
        W = np.outer(inv_weights, inv_weights)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)
    
    elif method == 'adaptive':
        # Adaptive density-based weighting
        k_density = np.sum(K, axis=1)
        k_density = k_density / np.sum(k_density)
        
        inv_density = 1.0 / (k_density + 1e-10)
        inv_density = inv_density / np.sum(inv_density)
        
        W = np.outer(inv_density, inv_density)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)
    
    else:
        raise ValueError(f"Unknown weight method: {method}")


def compute_ow_mmd_squared(X, Y, gamma=None, weight_method='variance_reduction'):
    """
    Compute Optimally-Weighted MMD² between X and Y.
    
    Returns MMD² directly (can be negative due to variance reduction).
    This is the correct statistic for hypothesis testing.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Reference samples
    Y : array-like, shape (m_samples, n_features)
        Test samples
    gamma : str or float, default='auto'
        RBF kernel bandwidth
    weight_method : str, default='variance_reduction'
        Weighting strategy
    
    Returns:
    --------
    mmd_squared : float
        OW-MMD² statistic (can be negative)
    """
    m, n = X.shape[0], Y.shape[0]
    
    # Compute kernel matrices
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    
    # Compute optimal weights
    W_XX = compute_optimal_weights(K_XX, weight_method)
    W_YY = compute_optimal_weights(K_YY, weight_method)
    W_XY = np.ones((m, n)) / (m * n)  # Cross-term uses uniform weights
    
    # Weighted MMD² computation: E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
    term1 = np.sum(W_XX * K_XX)
    term2 = np.sum(W_YY * K_YY)
    term3 = np.sum(W_XY * K_XY)
    
    return term1 + term2 - 2 * term3


def compute_ow_mmd(X, Y, gamma='auto', weight_method='variance_reduction'):
    """
    Compute Optimally-Weighted MMD between X and Y.
    
    Convenience wrapper that returns sqrt of MMD² (always non-negative).
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Reference samples
    Y : array-like, shape (m_samples, n_features)
        Test samples
    gamma : str or float, default='auto'
        RBF kernel bandwidth
    weight_method : str, default='variance_reduction'
        Weighting strategy
    
    Returns:
    --------
    mmd_value : float
        OW-MMD statistic (non-negative)
    """
    mmd_sq = compute_ow_mmd_squared(X, Y, gamma, weight_method)
    return np.sqrt(max(0, mmd_sq))


def mmd_ow(X, s=None, gamma='auto', weight_method='variance_reduction'):
    """
    Compute OW-MMD between two halves of X (split-based interface).
    
    This is the basic OW-MMD computation with a fixed threshold.
    For proper hypothesis testing, use mmd_ow_permutation() instead.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data window
    s : int, optional
        Split point (default: half)
    gamma : str or float, default='auto'
        RBF kernel bandwidth
    weight_method : str, default='variance_reduction'
        Weighting strategy
    
    Returns:
    --------
    mmd_value : float
        OW-MMD statistic
    threshold : float
        Fixed heuristic threshold (0.1)
    """
    if s is None:
        s = len(X) // 2
    
    X_ref = X[:s]
    X_test = X[s:]
    
    mmd_value = compute_ow_mmd(X_ref, X_test, gamma, weight_method)
    
    # Fixed threshold heuristic (for quick detection)
    # Use mmd_ow_permutation() for proper p-value
    threshold = 0.1
    
    return mmd_value, threshold


# =============================================================================
# SECTION 4: PERMUTATION-BASED HYPOTHESIS TESTING
# =============================================================================

def mmd_ow_permutation(X, s=None, n_perm=500, gamma='auto', weight_method='variance_reduction'):
    """
    Optimally-Weighted MMD with permutation test for p-value.
    
    This enables FAIR COMPARISON with standard MMD by using the same
    hypothesis testing framework while leveraging OW-MMD's variance-optimal
    weighting for better sample efficiency.
    
    H0: X[:s] and X[s:] come from the same distribution
    H1: X[:s] and X[s:] come from different distributions
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data window to test for drift
    s : int, optional
        Split point (default: half)
    n_perm : int, default=500
        Number of permutations for p-value estimation
    gamma : str or float, default='auto'
        RBF kernel bandwidth
    weight_method : str, default='variance_reduction'
        Weighting strategy
    
    Returns:
    --------
    mmd_value : float
        Observed OW-MMD statistic
    p_value : float
        Permutation-based p-value (reject H0 if p < 0.05)
    
    Example:
    --------
    >>> mmd_val, p_val = mmd_ow_permutation(window, n_perm=500)
    >>> if p_val < 0.05:
    ...     print("Drift detected!")
    
    References:
    -----------
    Bharti et al. (2023). "Optimally-weighted Estimators of the Maximum Mean
    Discrepancy for Likelihood-Free Inference." ICML 2023.
    """
    n = X.shape[0]
    if s is None:
        s = n // 2
    
    # Compute gamma once using median heuristic
    if gamma == 'auto':
        gamma_val = compute_gamma_median_heuristic(X)
    else:
        gamma_val = gamma
    
    def compute_statistic(X_data, split_point):
        """Compute OW-MMD² statistic for given split."""
        X_ref = X_data[:split_point]
        X_test = X_data[split_point:]
        
        if len(X_ref) < 2 or len(X_test) < 2:
            return 0.0
        
        return compute_ow_mmd_squared(X_ref, X_test, gamma_val, weight_method)
    
    # Step 1: Compute observed statistic
    mmd_obs = compute_statistic(X, s)
    
    # Step 2: Permutation test
    count_greater = 0
    for _ in range(n_perm):
        perm_idx = np.random.permutation(n)
        X_perm = X[perm_idx]
        mmd_perm = compute_statistic(X_perm, s)
        
        if mmd_perm >= mmd_obs:
            count_greater += 1
    
    # Step 3: Compute p-value with continuity correction
    p_value = (count_greater + 1) / (n_perm + 1)
    
    # Return MMD value (sqrt for interpretability) and p-value
    mmd_value = np.sqrt(max(0, mmd_obs))
    
    return mmd_value, p_value


# =============================================================================
# SECTION 5: SHAPEDD + OW-MMD INTEGRATION (MAIN DETECTOR)
# =============================================================================

def shape_ow_mmd(X, l1, l2, n_perm=500):
    """
    Original ShapeDD algorithm with OW-MMD for statistical testing.
    
    This is the RECOMMENDED function for drift detection. It follows the
    original ShapeDD algorithm (convolution-based pattern detection) but
    replaces the standard MMD with OW-MMD + permutation test.
    
    Algorithm:
        1. Compute kernel matrix K on entire data stream
        2. Apply matched filter (triangle pattern) via convolution
        3. Find zero-crossings (peaks in shape statistic)
        4. At each peak, run OW-MMD permutation test
    
    This enables fair comparison with standard ShapeDD:
        - Same pattern detection algorithm
        - Same permutation-based hypothesis testing
        - Only difference: optimal weights in MMD computation
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int
        Half-window size for shape statistic computation
    l2 : int
        Window size for OW-MMD statistical test at peaks
    n_perm : int, default=500
        Number of permutations for p-value estimation
    
    Returns:
    --------
    res : array-like, shape (n_samples, 3)
        [:, 0] - Shape statistic value (geometric pattern strength)
        [:, 1] - OW-MMD statistic at this position
        [:, 2] - p-value (< 0.05 indicates significant drift)
    
    Example:
    --------
    >>> results = shape_ow_mmd(data_stream, l1=50, l2=150, n_perm=500)
    >>> drift_points = np.where(results[:, 2] < 0.05)[0]
    >>> print(f"Detected {len(drift_points)} drift points")
    
    References:
    -----------
    ShapeDD: Based on triangle shape property of MMD around drift points
    OW-MMD: Bharti et al. (2023). "Optimally-weighted Estimators of the MMD." ICML.
    """
    from sklearn.metrics.pairwise import pairwise_kernels
    
    n = X.shape[0]
    
    # Validate inputs
    if n < 2 * l1 + 1:
        raise ValueError(f"Stream too short: {n} < 2*l1+1 = {2*l1+1}")
    
    # Step 1: Create matched filter (same as original ShapeDD)
    w = np.array(l1 * [1.0] + l1 * [-1.0]) / float(l1)
    
    # Step 2: Compute kernel matrix with adaptive gamma
    gamma = compute_gamma_median_heuristic(X)
    K = pairwise_kernels(X, metric="rbf", gamma=gamma)
    
    # Step 3: Apply matched filter via matrix multiplication
    W = np.zeros((n - 2*l1, n))
    for i in range(n - 2*l1):
        W[i, i:i + 2*l1] = w
    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)
    
    # Step 4: Convolve with matched filter to find triangle patterns
    shape_stat = np.convolve(stat, w)
    
    # Step 5: Find zero-crossings (peaks in shape curve)
    shape_prime = shape_stat[1:] * shape_stat[:-1]
    
    # Initialize results
    res = np.zeros((n, 3))
    res[:, 2] = 1.0  # Default p-value = 1 (no drift)
    
    # Step 6: At each peak, run OW-MMD permutation test
    for pos in np.where(shape_prime < 0)[0]:
        if shape_stat[pos] > 0:
            res[pos, 0] = shape_stat[pos]
            
            # Extract window around peak for statistical test
            a = max(0, pos - l2 // 2)
            b = min(n, pos + l2 // 2)
            window = X[a:b]
            split_point = pos - a
            
            # Skip if window too small for meaningful test
            if split_point < 5 or len(window) - split_point < 5:
                continue
            
            # Run OW-MMD permutation test (key difference from original ShapeDD)
            mmd_val, p_val = mmd_ow_permutation(
                window, s=split_point, n_perm=n_perm, gamma=gamma
            )
            
            res[pos, 1] = mmd_val
            res[pos, 2] = p_val
    
    return res

def shapedd_ow_mmd(X, l1=50, l2=150, gamma='auto', mode='simple'):
    """
    LEGACY: ShapeDD-OW-MMD Hybrid with heuristic pattern detection.
    
    NOTE: This is a fast heuristic version. For proper statistical testing,
    use shape_ow_mmd() instead.
    
    This computes OW-MMD values over sliding windows and analyzes geometric
    patterns (triangles, peaks) using heuristics instead of permutation tests.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int, default=50
        Reference window size
    l2 : int, default=150
        Test window size
    gamma : str or float, default='auto'
        RBF kernel bandwidth
    mode : str, default='simple'
        Pattern detection mode: 'simple' or 'enhanced'
    
    Returns:
    --------
    pattern_score : float
        Geometric pattern strength (0.0 to 1.0)
    mmd_max : float
        Maximum OW-MMD value in sequence
    """
    n_samples = len(X)
    
    # Minimum size for pattern detection
    min_size = l1 + l2 + 125
    
    if n_samples < min_size:
        # Fallback: single OW-MMD test
        min_required = min(l1, l2)
        if n_samples < 2 * min_required:
            return 0.0, 0.0
        
        if n_samples >= l1 + l2:
            split_point = l1
        else:
            ratio = l1 / (l1 + l2)
            split_point = max(min_required, int(n_samples * ratio))
        
        mmd_val, _ = mmd_ow(X, s=split_point, gamma=gamma)
        
        threshold = 0.10 if n_samples < l1 + l2 else 0.15
        if mmd_val > threshold:
            pattern_score = min(mmd_val / 0.25, 1.0)
        else:
            pattern_score = 0.0
        
        return pattern_score, mmd_val
    
    # Sliding window OW-MMD computation
    mmd_sequence = []
    step_size = 25
    
    for ref_start in range(0, n_samples - l1 - l2, step_size):
        ref_end = ref_start + l1
        test_end = ref_end + l2
        
        if test_end > n_samples:
            break
        
        ref_window = X[ref_start:ref_end]
        test_window = X[ref_end:test_end]
        
        window_combined = np.vstack([ref_window, test_window])
        mmd_val, _ = mmd_ow(window_combined, s=l1, gamma=gamma)
        mmd_sequence.append(mmd_val)
    
    if len(mmd_sequence) < 3:
        if mmd_sequence:
            max_mmd = max(mmd_sequence)
            pattern_score = min(max_mmd / 0.3, 1.0) if max_mmd > 0.15 else 0.0
            return pattern_score, max_mmd
        return 0.0, 0.0
    
    mmd_array = np.array(mmd_sequence)
    mmd_min, mmd_max_val = mmd_array.min(), mmd_array.max()
    
    if mmd_max_val - mmd_min < 1e-10:
        pattern_score = min(mmd_max_val / 0.3, 1.0) if mmd_max_val > 0.15 else 0.0
        return pattern_score, mmd_max_val
    
    mmd_norm = (mmd_array - mmd_min) / (mmd_max_val - mmd_min)
    
    if mode == 'simple':
        pattern_score = _simple_pattern_detection(mmd_norm)
    elif mode == 'enhanced':
        pattern_score = _enhanced_pattern_detection(mmd_array, mmd_norm)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return pattern_score, mmd_max_val


def shapedd_ow_mmd_enhanced(X, l1=50, l2=150, gamma='auto'):
    """
    LEGACY: Enhanced ShapeDD-OW-MMD with sophisticated pattern detection.
    
    Same as shapedd_ow_mmd() with mode='enhanced'.
    """
    return shapedd_ow_mmd(X, l1=l1, l2=l2, gamma=gamma, mode='enhanced')

def _simple_pattern_detection(mmd_norm):
    """Simple 3-check triangle pattern detection."""
    peak_idx = np.argmax(mmd_norm)
    pattern_score = 0.0
    
    # Check 1: Peak in middle region (20%-80%)
    if 0.2 < peak_idx / len(mmd_norm) < 0.8:
        pattern_score += 0.3
    
    # Check 2: Rising before peak
    if peak_idx > 1:
        rise = mmd_norm[:peak_idx]
        if len(rise) > 0 and np.sum(np.diff(rise) > 0) / len(rise) > 0.5:
            pattern_score += 0.35
    
    # Check 3: Falling after peak
    if peak_idx < len(mmd_norm) - 2:
        fall = mmd_norm[peak_idx:]
        if len(fall) > 0 and np.sum(np.diff(fall) < 0) / len(fall) > 0.5:
            pattern_score += 0.35
    
    return pattern_score


def _enhanced_pattern_detection(mmd_array, mmd_norm):
    """Enhanced pattern detection with statistical validation."""
    pattern_score = 0.0
    
    # Check 1: Triangle shape
    if _check_triangle_shape(mmd_norm, tolerance=0.6):
        pattern_score += 0.4
    
    # Check 2: Zero-crossing in derivative
    if _check_zero_crossing(mmd_array):
        pattern_score += 0.2
    
    # Check 3: Statistically significant peak
    if _check_significant_peak(mmd_array, sigma=2.0):
        pattern_score += 0.4
    
    return min(pattern_score, 1.0)


def _check_triangle_shape(sequence, tolerance=0.5):
    """Check if sequence exhibits triangle-like shape."""
    n = len(sequence)
    if n < 5:
        return False
    
    peak_idx = np.argmax(sequence)
    
    # Peak should be in middle region
    if peak_idx < n * 0.2 or peak_idx > n * 0.8:
        return False
    
    # Check rising phase
    if peak_idx > 1:
        rise = sequence[:peak_idx]
        is_rising = np.sum(np.diff(rise) > 0) / len(rise) >= tolerance
    else:
        is_rising = True
    
    # Check falling phase
    if peak_idx < n - 2:
        fall = sequence[peak_idx:]
        is_falling = np.sum(np.diff(fall) < 0) / len(fall) >= tolerance
    else:
        is_falling = True
    
    return is_rising and is_falling


def _check_zero_crossing(sequence):
    """Check for zero-crossings in first derivative."""
    if len(sequence) < 3:
        return False
    
    derivative = np.diff(sequence)
    sign_changes = np.sum(np.diff(np.sign(derivative)) != 0)
    return sign_changes >= 1


def _check_significant_peak(sequence, sigma=2.0):
    """Check if sequence has statistically significant peak."""
    mean_val = np.mean(sequence)
    std_val = np.std(sequence)
    
    if std_val < 1e-10:
        return False
    
    return np.max(sequence) > mean_val + sigma * std_val

import numpy as np
from scipy.spatial.distance import pdist, cdist

# =============================================================================
# ENHANCED SHAPEDD++ FOR SUDDEN DRIFT DETECTION
# =============================================================================

# def rbf_kernel(X, Y, gamma):
#     """RBF kernel with given gamma."""
#     sq_dist = cdist(X, Y, 'sqeuclidean')
#     return np.exp(-gamma * sq_dist)


def compute_gamma_median(X):
    """Median heuristic for kernel bandwidth."""
    distances = pdist(X, 'euclidean')
    if len(distances) == 0 or np.all(distances == 0):
        return 1.0
    return 1.0 / (2 * np.median(distances[distances > 0])**2)


def studentized_mmd_test(X, Y, gamma=None, n_perm=200):
    """
    Studentized MMD with permutation test.
    
    Optimized for sudden drift detection with:
    - Variance normalization for stable thresholds
    - Efficient permutation testing
    """
    m, n = len(X), len(Y)
    combined = np.vstack([X, Y])
    
    if gamma is None:
        gamma = compute_gamma_median(combined)
    
    def compute_stat(X_s, Y_s):
        """Compute studentized MMD²."""
        m_s, n_s = len(X_s), len(Y_s)
        
        K_XX = rbf_kernel(X_s, X_s, gamma)
        K_YY = rbf_kernel(Y_s, Y_s, gamma)
        K_XY = rbf_kernel(X_s, Y_s, gamma)
        
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)
        
        # MMD²
        mmd_sq = (np.sum(K_XX) / (m_s * (m_s - 1)) +
                  np.sum(K_YY) / (n_s * (n_s - 1)) -
                  2 * np.sum(K_XY) / (m_s * n_s))
        
        # Variance estimate
        h_X = np.sum(K_XX, axis=1) / (m_s - 1) - np.sum(K_XY, axis=1) / n_s
        h_Y = np.sum(K_YY, axis=1) / (n_s - 1) - np.sum(K_XY, axis=0) / m_s
        
        var_est = 4 * (np.var(h_X) / m_s + np.var(h_Y) / n_s)
        
        return mmd_sq / np.sqrt(max(var_est, 1e-10))
    
    # Observed statistic
    stat_obs = compute_stat(X, Y)
    
    # Permutation test
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(m + n)
        stat_perm = compute_stat(combined[perm[:m]], combined[perm[m:]])
        if stat_perm >= stat_obs:
            count += 1
    
    p_value = (count + 1) / (n_perm + 1)
    mmd_value = np.sqrt(max(0, stat_obs))
    
    return mmd_value, p_value


def shape_dd_plus_plus(X, l1, l2=None, n_perm=300, 
                        use_studentized=True,
                        early_validation=True):
    """
    ShapeDD++ : Enhanced ShapeDD for Sudden Drift Detection.
    
    Improvements over original ShapeDD:
    1. Studentized MMD for variance reduction
    2. Efficient sliding window computation
    3. Early validation option for strong drifts
    
    Parameters:

    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int
        Window size for shape statistic (half-window)
    l2 : int, optional
        Window size for validation (default: 3*l1)
    n_perm : int
        Number of permutations for p-value
    use_studentized : bool
        Use studentized MMD (recommended for sudden drift)
    early_validation : bool
        Skip validation for very strong shape signals
    
    Returns:

    results : list of dict
        Each dict contains: position, mmd, p_value, shape_strength
    """
    n = X.shape[0]
    if l2 is None:
        l2 = 3 * l1
    
    # Validate inputs
    if n < 4 * l1:
        raise ValueError(f"Stream too short: {n} < 4*l1 = {4*l1}")
    
    # Step 1: Compute drift magnitude on sliding windows
    gamma = compute_gamma_median(X)
    sigma = np.zeros(n)
    
    for t in range(2*l1, n):
        ref = X[t-2*l1:t-l1]
        test = X[t-l1:t]
        
        if len(ref) >= l1 and len(test) >= l1:
            if use_studentized:
                mmd_val, _ = studentized_mmd_test(ref, test, gamma, n_perm=50)
            else:
                # Standard MMD (faster, less accurate)
                K_XX = rbf_kernel(ref, ref, gamma)
                K_YY = rbf_kernel(test, test, gamma)
                K_XY = rbf_kernel(ref, test, gamma)
                np.fill_diagonal(K_XX, 0)
                np.fill_diagonal(K_YY, 0)
                mmd_sq = (np.sum(K_XX)/(l1*(l1-1)) + 
                          np.sum(K_YY)/(l1*(l1-1)) - 
                          2*np.sum(K_XY)/(l1*l1))
                mmd_val = np.sqrt(max(0, mmd_sq))
            
            sigma[t] = mmd_val
    
    # Step 2: Shape matching via matched filter
    w = np.concatenate([np.ones(l1)/l1, -np.ones(l1)/l1])
    shape_stat = np.convolve(sigma, w, mode='same')
    
    # Step 3: Find zero-crossings (positive to negative)
    candidates = []
    for t in range(1, n):
        if shape_stat[t-1] > 0 and shape_stat[t] <= 0:
            # Drift position is at t - l1 (center of triangle)
            drift_pos = t - l1
            if drift_pos > l1 and drift_pos < n - l1:
                candidates.append((drift_pos, shape_stat[t-1]))
    
    # Step 4: Validate candidates with permutation test
    results = []
    
    for pos, strength in candidates:
        # Extract validation window
        a = max(0, pos - l2//2)
        b = min(n, pos + l2//2)
        window = X[a:b]
        split = pos - a
        
        if split < 5 or len(window) - split < 5:
            continue
        
        ref_window = window[:split]
        test_window = window[split:]
        
        # Early validation: skip full test for very strong signals
        if early_validation and strength > 0.5:
            # Quick check with fewer permutations
            mmd_val, p_val = studentized_mmd_test(
                ref_window, test_window, gamma, n_perm=100
            )
        else:
            # Full validation
            mmd_val, p_val = studentized_mmd_test(
                ref_window, test_window, gamma, n_perm=n_perm
            )
        
        results.append({
            'position': pos,
            'mmd': mmd_val,
            'p_value': p_val,
            'shape_strength': strength,
            'is_drift': p_val < 0.05
        })
    
    return results


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def detect_sudden_drift(X, window_size=50, significance=0.05, n_perm=300):
    """
    Simple interface for sudden drift detection.
    
    Parameters:

    X : array-like
        Data stream
    window_size : int
        Size of detection window
    significance : float
        Significance level for drift detection
    n_perm : int
        Number of permutations
    
    Returns:

    drift_points : list of int
        Detected drift positions
    """
    results = shape_dd_plus_plus(
        X, 
        l1=window_size, 
        n_perm=n_perm,
        use_studentized=True
    )
    
    drift_points = [r['position'] for r in results if r['p_value'] < significance]
    
    return drift_points
