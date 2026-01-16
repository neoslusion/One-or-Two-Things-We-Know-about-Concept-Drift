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

def mmd_fast(X, K, s=None, n_perm=500):
    """
    Original fast MMD using precomputed kernel and einsum.
    """
    if s is None:
        s = X.shape[0] // 2
    
    W = gen_window_matrix(s, K.shape[0] - s, n_perm)
    stats = np.einsum('ij,ij->i', np.dot(W, K), W)
    p = (stats[0] < stats).sum() / n_perm
    
    return stats[0], p


def mmd_studentized_fast(K, s, n_perm=500):
    """
    Studentized MMD using precomputed kernel matrix.
    
    Maintains O(n²) complexity while adding variance normalization.
    
    Key insight: We can estimate variance from the kernel matrix
    without recomputing kernels.
    """
    n = K.shape[0]
    m = s  # Reference window size
    
    # Extract sub-matrices
    K_XX = K[:m, :m]
    K_YY = K[m:, m:]
    K_XY = K[:m, m:]
    
    n_y = n - m
    
    # MMD² unbiased
    K_XX_sum = np.sum(K_XX) - np.trace(K_XX)  # Exclude diagonal
    K_YY_sum = np.sum(K_YY) - np.trace(K_YY)
    K_XY_sum = np.sum(K_XY)
    
    if m > 1 and n_y > 1:
        mmd_sq = (K_XX_sum / (m * (m - 1)) + 
                  K_YY_sum / (n_y * (n_y - 1)) - 
                  2 * K_XY_sum / (m * n_y))
    else:
        return 0.0, 1.0
    
    # Variance estimate using h-statistics
    # h_X[i] = mean(K_XX[i,:]) - mean(K_XY[i,:])
    K_XX_rowsum = (np.sum(K_XX, axis=1) - np.diag(K_XX)) / (m - 1)
    K_YY_rowsum = (np.sum(K_YY, axis=1) - np.diag(K_YY)) / (n_y - 1)
    K_XY_rowsum = np.sum(K_XY, axis=1) / n_y
    K_YX_colsum = np.sum(K_XY, axis=0) / m
    
    h_X = K_XX_rowsum - K_XY_rowsum
    h_Y = K_YY_rowsum - K_YX_colsum
    
    var_est = 4 * (np.var(h_X) / m + np.var(h_Y) / n_y)
    std_est = np.sqrt(max(var_est, 1e-10))
    
    # Studentized statistic
    stat_obs = mmd_sq / std_est
    
    # Permutation test
    count = 0
    indices = np.arange(n)
    
    for _ in range(n_perm):
        perm = np.random.permutation(indices)
        K_perm = K[np.ix_(perm, perm)]
        
        # Fast MMD² on permuted kernel
        K_XX_p = K_perm[:m, :m]
        K_YY_p = K_perm[m:, m:]
        K_XY_p = K_perm[:m, m:]
        
        K_XX_sum_p = np.sum(K_XX_p) - np.trace(K_XX_p)
        K_YY_sum_p = np.sum(K_YY_p) - np.trace(K_YY_p)
        K_XY_sum_p = np.sum(K_XY_p)
        
        mmd_sq_p = (K_XX_sum_p / (m * (m - 1)) + 
                    K_YY_sum_p / (n_y * (n_y - 1)) - 
                    2 * K_XY_sum_p / (m * n_y))
        
        # Variance for permuted
        K_XX_rowsum_p = (np.sum(K_XX_p, axis=1) - np.diag(K_XX_p)) / (m - 1)
        K_YY_rowsum_p = (np.sum(K_YY_p, axis=1) - np.diag(K_YY_p)) / (n_y - 1)
        K_XY_rowsum_p = np.sum(K_XY_p, axis=1) / n_y
        K_YX_colsum_p = np.sum(K_XY_p, axis=0) / m
        
        h_X_p = K_XX_rowsum_p - K_XY_rowsum_p
        h_Y_p = K_YY_rowsum_p - K_YX_colsum_p
        
        var_p = 4 * (np.var(h_X_p) / m + np.var(h_Y_p) / n_y)
        std_p = np.sqrt(max(var_p, 1e-10))
        
        stat_perm = mmd_sq_p / std_p
        
        if stat_perm >= stat_obs:
            count += 1
    
    p_value = (count + 1) / (n_perm + 1)
    mmd_value = np.sqrt(max(0, mmd_sq))
    
    return mmd_value, p_value


def mmd_studentized_fast_v2(K, s, n_perm=500):
    """
    Even faster studentized MMD - uses einsum like original.
    
    Trade-off: Slightly less accurate variance estimate but much faster.
    """
    n = K.shape[0]
    m = s
    n_y = n - m
    
    if m < 2 or n_y < 2:
        return 0.0, 1.0
    
    # Use weight matrix approach like original
    W = gen_window_matrix(m, n_y, n_perm)
    
    # Compute all MMD values at once
    stats = np.einsum('ij,ij->i', np.dot(W, K), W)
    
    # For studentization, estimate variance from observed statistic
    # Simple approach: use bootstrap variance from permutation stats
    mmd_obs = stats[0]
    mmd_perms = stats[1:]
    
    # Studentize using permutation std
    std_null = np.std(mmd_perms)
    if std_null < 1e-10:
        std_null = 1e-10
    
    stat_obs = mmd_obs / std_null
    stat_perms = mmd_perms / std_null
    
    p_value = (np.sum(stat_perms >= stat_obs) + 1) / (n_perm + 1)
    mmd_value = np.sqrt(max(0, mmd_obs))
    
    return mmd_value, p_value

def shape_plus_plus(X, l1, l2, n_perm, mode='fast'):
    """
    ShapeDD++ Optimized: Same speed as original + studentized MMD.
    
    Parameters:

    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int
        Half-window size for shape statistic
    l2 : int
        Window size for validation
    n_perm : int
        Number of permutations
    mode : str
        'fast' - Fastest, uses permutation std (recommended)
        'accurate' - More accurate variance estimate, slower
        'original' - Same as original ShapeDD (no studentization)
    
    Returns:

    res : array, shape (n, 3)
        [:, 0] - Shape statistic
        [:, 1] - MMD value
        [:, 2] - p-value
    """
    w = np.array(l1*[1.] + l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    
    # Single kernel computation for entire stream
    K_full = apply_kernel(X, metric="rbf")
    
    # Build weight matrix (vectorized)
    W = np.zeros((n - 2*l1, n))
    for i in range(n - 2*l1):
        W[i, i:i+2*l1] = w
    
    # Compute shape statistic (same as original)
    stat = np.einsum('ij,ij->i', np.dot(W, K_full), W)
    shape_stat = np.convolve(stat, w)
    shape_prime = shape_stat[1:] * shape_stat[:-1]
    
    # Initialize results
    res = np.zeros((n, 3))
    res[:, 2] = 1  # Default p-value
    
    # Validate at zero-crossings
    for pos in np.where(shape_prime < 0)[0]:
        if shape_stat[pos] > 0:
            res[pos, 0] = shape_stat[pos]
            
            # Extract window
            a = max(0, pos - int(l2/2))
            b = min(n, pos + int(l2/2))
            s = pos - a  # Split point
            
            # Skip if window too small
            if s < 2 or (b - a - s) < 2:
                continue
            
            # Get kernel submatrix (no recomputation!)
            K_window = K_full[a:b, a:b]
            
            # Choose MMD variant
            if mode == 'original':
                mmd_val, p_val = mmd_fast(X[a:b], K_window, s, n_perm)
            elif mode == 'fast':
                mmd_val, p_val = mmd_studentized_fast_v2(K_window, s, n_perm)
            elif mode == 'accurate':
                mmd_val, p_val = mmd_studentized_fast(K_window, s, n_perm)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            res[pos, 1] = mmd_val
            res[pos, 2] = p_val
    
    return res


# =============================================================================
# SECTION 4: CONVENIENCE WRAPPERS
# =============================================================================

def detect_drift(X, l1=50, l2=150, n_perm=500, significance=0.05, mode='fast'):
    """
    Simple interface for drift detection.
    
    Returns:

    drift_points : list of int
        Positions where drift was detected
    """
    res = shape_plus_plus(X, l1, l2, n_perm, mode)
    return np.where(res[:, 2] < significance)[0].tolist()
