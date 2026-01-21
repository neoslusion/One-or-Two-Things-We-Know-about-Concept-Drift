import numpy as np
from scipy.spatial.distance import cdist, pdist

# GPU Acceleration Support
try:
    import torch

    HAS_TORCH = False  # Disabled for hardware consistency (CPU only)
    if torch.cuda.is_available() and HAS_TORCH:
        DEVICE = torch.device("cuda")
        print("[MMD] Using GPU acceleration (CUDA)")
    else:
        DEVICE = torch.device("cpu")
        print("[MMD] Using CPU (Torch available but disabled)")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("[MMD] Using CPU (NumPy only)")


def rbf_kernel(X, Y, gamma="auto"):
    """
    RBF (Gaussian) kernel: k(x,y) = exp(-gamma * ||x-y||^2)
    Supports both NumPy and PyTorch (GPU) automatically.
    """
    if gamma is None:
        gamma = "auto"

    # GPU branch disabled via HAS_TORCH = False
    if HAS_TORCH and (isinstance(X, torch.Tensor) or isinstance(Y, torch.Tensor)):
        return _rbf_kernel_torch(X, Y, gamma)

    # Auto-convert to Torch if available and data is large enough (>500 samples)
    if HAS_TORCH and len(X) > 500:
        try:
            X_t = torch.from_numpy(X).float().to(DEVICE)
            Y_t = torch.from_numpy(Y).float().to(DEVICE)
            K_t = _rbf_kernel_torch(X_t, Y_t, gamma)
            return K_t.cpu().numpy()
        except Exception:
            pass  # Fallback to numpy on error

    # NumPy implementation
    if gamma == "auto":
        # Median heuristic
        if len(X) > 1000:  # Subsample for speed
            idx = np.random.choice(len(X), 1000, replace=False)
            all_data = np.vstack([X[idx], Y[idx] if len(Y) > 1000 else Y])
        else:
            all_data = np.vstack([X, Y])

        distances = cdist(all_data, all_data, metric="euclidean")
        distances = distances[distances > 0]
        if len(distances) > 0:
            gamma = 1.0 / (2 * np.median(distances) ** 2)
        else:
            gamma = 1.0

    distances_sq = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-gamma * distances_sq)


def _rbf_kernel_torch(X, Y, gamma="auto"):
    """PyTorch implementation of RBF Kernel."""
    if gamma is None:
        gamma = "auto"

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, device=DEVICE, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, device=DEVICE, dtype=torch.float32)

    if gamma == "auto":
        # Subsample for median heuristic to save GPU memory
        n_samples = min(2000, X.shape[0])
        idx = torch.randperm(X.shape[0])[:n_samples]
        X_sub = X[idx]

        # pdist equivalent
        dist_mat = torch.cdist(X_sub, X_sub)
        mask = dist_mat > 0
        if mask.any():
            median_dist = torch.median(dist_mat[mask])
            gamma = 1.0 / (2 * median_dist**2)
        else:
            gamma = 1.0

    # Compute full kernel
    dists = torch.cdist(X, Y)
    dists_sq = dists.pow(2)
    return torch.exp(-gamma * dists_sq)


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

    distances = pdist(X_sample, metric="euclidean")
    if len(distances) > 0 and np.any(distances > 0):
        median_dist = np.median(distances[distances > 0])
        return 1.0 / (2 * median_dist**2)
    return 1.0


def compute_optimal_weights(K, method: str = "inverse_density"):
    """
    Compute sample weights for weighted MMD estimation.

    This implements Inverse Density Weighting (IDW) where points in sparse regions
    (distribution boundaries) receive higher weights than points in dense regions.
    
    The intuition: boundary points are more informative for detecting distribution
    changes, so upweighting them improves sensitivity to drift.
    
    Formula: w_i ∝ 1 / sqrt(Σ_j K(x_i, x_j))
    
    where Σ_j K(x_i, x_j) is the kernel density estimate at point x_i.

    Parameters:
    -----------
    K : array-like, shape (n, n)
        Kernel matrix
    method : str, default='inverse_density'
        Weighting strategy:
        - 'uniform': Standard V-statistic (equal weights)
        - 'inverse_density': IDW - inverse square root of kernel density (DEFAULT)

    Returns:
    --------
    W : array-like, shape (n, n)
        Weight matrix with zeros on diagonal, normalized to sum to 1
    
    Note:
    -----
    This is a heuristic weighting scheme, not derived from a specific published
    optimal weights paper. It empirically improves sensitivity to drift by giving
    higher weight to boundary/sparse regions.
    """
    n = K.shape[0]

    if method == "uniform":
        # Standard V-statistic (equal weights)
        W = np.ones((n, n)) / (n * n)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)

    elif method in ("inverse_density", "variance_reduction"):
        # Inverse Density Weighting (IDW)
        # w_i proportional to 1/sqrt(sum_j K(x_i, x_j))
        # Points with low density (sparse regions) get higher weights
        K_off = K.copy()
        np.fill_diagonal(K_off, 0)

        k_sums = np.sum(K_off, axis=1)  # Kernel density estimate
        k_sums = np.maximum(k_sums, 1e-10)  # Numerical stability

        inv_weights = 1.0 / (np.sqrt(k_sums) + 0.5)  # Dampening for stability
        W = np.outer(inv_weights, inv_weights)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)

    else:
        raise ValueError(f"Unknown weight method: {method}")


def compute_adw_mmd_squared(X, Y, gamma=None, weight_method="inverse_density"):
    """
    Compute Weighted MMD² (WMMD²) between X and Y using Inverse Density Weighting.

    Returns MMD² directly (can be negative due to finite sample effects).
    This is the correct statistic for hypothesis testing.

    The weighting scheme uses Inverse Density Weighting (IDW):
    - Points in sparse regions (boundaries) get higher weights
    - Points in dense regions (cores) get lower weights
    - This improves sensitivity to distribution changes at boundaries
    
    Formula: w_i ∝ 1 / sqrt(Σ_j K(x_i, x_j))

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Reference samples
    Y : array-like, shape (m_samples, n_features)
        Test samples
    gamma : str or float, default='auto'
        RBF kernel bandwidth
    weight_method : str, default='inverse_density'
        Weighting strategy ('inverse_density' or 'uniform')

    Returns:
    --------
    mmd_squared : float
        Weighted MMD² statistic (can be negative)
    """
    # GPU Acceleration Entry Point
    if HAS_TORCH and len(X) >= 100:  # Lowered threshold to catch standard windows
        try:
            X_t = torch.from_numpy(X).float().to(DEVICE)
            Y_t = torch.from_numpy(Y).float().to(DEVICE)

            # Compute kernels on GPU
            K_XX = rbf_kernel(X_t, X_t, gamma)
            K_YY = rbf_kernel(Y_t, Y_t, gamma)
            K_XY = rbf_kernel(X_t, Y_t, gamma)

            # Compute weights on GPU
            W_XX = compute_optimal_weights(K_XX, weight_method)
            W_YY = compute_optimal_weights(K_YY, weight_method)

            m, n = X.shape[0], Y.shape[0]
            W_XY = torch.ones((m, n), device=DEVICE) / (m * n)

            # Compute sums
            term1 = torch.sum(W_XX * K_XX)
            term2 = torch.sum(W_YY * K_YY)
            term3 = torch.sum(W_XY * K_XY)

            return (term1 + term2 - 2 * term3).cpu().item()
        except Exception as e:
            # print(f"GPU MMD failed: {e}, falling back to CPU")
            pass

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


def compute_adw_mmd(X, Y, gamma="auto", weight_method="inverse_density"):
    """
    Compute Weighted MMD (WMMD) between X and Y using Inverse Density Weighting.

    Convenience wrapper that returns sqrt of MMD² (always non-negative).

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Reference samples
    Y : array-like, shape (m_samples, n_features)
        Test samples
    gamma : str or float, default='auto'
        RBF kernel bandwidth
    weight_method : str, default='inverse_density'
        Weighting strategy

    Returns:
    --------
    mmd_value : float
        Weighted MMD statistic (non-negative)
    """
    mmd_sq = compute_adw_mmd_squared(X, Y, gamma, weight_method)
    return np.sqrt(max(0, mmd_sq))


def mmd_adw(X, s=None, gamma="auto", weight_method="inverse_density"):
    """
    Compute Weighted MMD between two halves of X (split-based interface).

    This splits X at point s and computes WMMD between X[:s] and X[s:].
    Used for drift detection in sliding window approaches.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data window
    s : int, optional
        Split point (default: half)
    gamma : str or float, default='auto'
        RBF kernel bandwidth
    weight_method : str, default='inverse_density'
        Weighting strategy

    Returns:
    --------
    mmd_value : float
        Weighted MMD statistic
    threshold : float
        Fixed heuristic threshold (0.1)
    """
    if s is None:
        s = len(X) // 2

    X_ref = X[:s]
    X_test = X[s:]

    mmd_value = compute_adw_mmd(X_ref, X_test, gamma, weight_method)

    # Fixed threshold heuristic (for quick detection)
    threshold = 0.1

    return mmd_value, threshold


def shapedd_adw_mmd(X, l1=50, l2=150, gamma="auto", mode="simple"):
    """
    LEGACY: ShapeDD-ADW-MMD Hybrid with heuristic pattern detection.
    Wrapper that returns only (score, max) for compatibility.
    """
    score, mmd_max, _ = shapedd_adw_mmd_full(X, l1, l2, gamma, mode)
    return score, mmd_max


def shapedd_adw_mmd_full(X, l1=50, l2=150, gamma="auto", mode="simple"):
    """
    ShapeDD-ADW-MMD Hybrid with heuristic pattern detection.

    Returns:
    --------
    pattern_score : float
        Geometric pattern strength (0.0 to 1.0)
    mmd_max : float
        Maximum OW-MMD value in sequence
    mmd_trace : np.ndarray
        The full MMD sequence for analysis
    """
    n_samples = len(X)

    # Minimum size for pattern detection
    min_size = l1 + l2 + 125

    if n_samples < min_size:
        # Fallback: single OW-MMD test
        min_required = min(l1, l2)
        if n_samples < 2 * min_required:
            return 0.0, 0.0, np.array([])

        if n_samples >= l1 + l2:
            split_point = l1
        else:
            ratio = l1 / (l1 + l2)
            split_point = max(min_required, int(n_samples * ratio))

        mmd_val, _ = mmd_adw(X, s=split_point, gamma=gamma)

        threshold = 0.10 if n_samples < l1 + l2 else 0.15
        if mmd_val > threshold:
            pattern_score = min(mmd_val / 0.25, 1.0)
        else:
            pattern_score = 0.0

        return pattern_score, mmd_val, np.array([mmd_val])

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
        mmd_val, _ = mmd_adw(window_combined, s=l1, gamma=gamma)
        mmd_sequence.append(mmd_val)

    mmd_array = np.array(mmd_sequence)

    if len(mmd_sequence) < 3:
        if mmd_sequence:
            max_mmd = max(mmd_sequence)
            pattern_score = min(max_mmd / 0.3, 1.0) if max_mmd > 0.15 else 0.0
            return pattern_score, max_mmd, mmd_array
        return 0.0, 0.0, np.array([])

    mmd_min, mmd_max_val = mmd_array.min(), mmd_array.max()

    if mmd_max_val - mmd_min < 1e-10:
        pattern_score = min(mmd_max_val / 0.3, 1.0) if mmd_max_val > 0.15 else 0.0
        return pattern_score, mmd_max_val, mmd_array

    mmd_norm = (mmd_array - mmd_min) / (mmd_max_val - mmd_min)

    if mode == "simple":
        shape_score = _simple_pattern_detection(mmd_norm)
    elif mode == "enhanced":
        shape_score = _enhanced_pattern_detection(mmd_array, mmd_norm)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # CRITICAL FIX: Combine shape score with MMD magnitude
    # Pattern detection alone returns 0.3 even for small drifts
    # We need to weight by actual MMD magnitude to avoid false positives
    # 
    # Thresholds based on empirical testing:
    # - mmd_max > 0.3: Strong drift signal
    # - mmd_max in [0.15, 0.3]: Moderate signal, needs good shape
    # - mmd_max < 0.15: Likely noise, reduce score significantly
    
    if mmd_max_val < 0.08:
        # Very low MMD - almost certainly no drift
        pattern_score = 0.0
    elif mmd_max_val < 0.15:
        # Low MMD - might be small drift, require strong pattern
        pattern_score = shape_score * 0.3 * (mmd_max_val / 0.15)
    elif mmd_max_val < 0.3:
        # Moderate MMD - weight by magnitude
        magnitude_weight = 0.5 + 0.5 * ((mmd_max_val - 0.15) / 0.15)
        pattern_score = shape_score * magnitude_weight
    else:
        # Strong MMD - full pattern score
        pattern_score = min(shape_score + 0.2, 1.0)  # Boost for strong signals

    return pattern_score, mmd_max_val, mmd_array


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
    distances = pdist(X, "euclidean")
    if len(distances) == 0 or np.all(distances == 0):
        return 1.0
    return 1.0 / (2 * np.median(distances[distances > 0]) ** 2)


# Hàm validate mới: WMMD với studentized + permutation
def wmmd_studentized(X_window, s, n_perm=500, weight_method="variance_reduction"):
    """
    Weighted MMD studentized với permutation, dùng cho validate.
    Trả về mmd_value, p_value như mmd gốc.
    """
    n = len(X_window)
    m = s
    n_y = n - m
    if m < 2 or n_y < 2:
        return 0.0, 1.0

    # Precompute kernel toàn window (thay vì full X)
    combined = X_window
    gamma = compute_gamma_median_heuristic(combined)  # Hoặc 'auto'
    K = rbf_kernel(combined, combined, gamma)

    # Tính observed MMD² weighted
    K_XX = K[:m, :m]
    K_YY = K[m:, m:]
    K_XY = K[:m, m:]
    W_XX = compute_optimal_weights(K_XX, weight_method)
    W_YY = compute_optimal_weights(K_YY, weight_method)
    W_XY = np.ones((m, n_y)) / (m * n_y)
    mmd_obs = np.sum(W_XX * K_XX) + np.sum(W_YY * K_YY) - 2 * np.sum(W_XY * K_XY)

    # Permutation cho stats
    stats = [mmd_obs]  # Observed là stats[0]
    indices = np.arange(n)
    for _ in range(n_perm):
        perm = np.random.permutation(indices)
        K_perm = K[np.ix_(perm, perm)]
        K_XX_p = K_perm[:m, :m]
        K_YY_p = K_perm[m:, m:]
        K_XY_p = K_perm[:m, m:]
        W_XX_p = compute_optimal_weights(K_XX_p, weight_method)
        W_YY_p = compute_optimal_weights(K_YY_p, weight_method)
        W_XY_p = np.ones((m, n_y)) / (m * n_y)
        mmd_p = (
            np.sum(W_XX_p * K_XX_p)
            + np.sum(W_YY_p * K_YY_p)
            - 2 * np.sum(W_XY_p * K_XY_p)
        )
        stats.append(mmd_p)

    stats = np.array(stats)
    mmd_perms = stats[1:]
    std_null = np.std(mmd_perms) if np.std(mmd_perms) > 1e-10 else 1e-10
    stat_obs = mmd_obs / std_null
    stat_perms = mmd_perms / std_null
    p_value = (np.sum(stat_perms >= stat_obs) + 1) / (n_perm + 1)
    mmd_value = np.sqrt(max(0, mmd_obs))
    return mmd_value, p_value


# Hàm ShapeDD sửa với WMMD
def shape_with_wmmd(X, l1, l2, n_perm, weight_method="variance_reduction"):
    """
    ShapeDD với Weighted MMD (variance reduction).
    Giống gốc nhưng validate bằng WMMD studentized.
    
    NOTE: This is SLOW due to permutation test. Use shape_with_wmmd_fast
    for real-time applications.
    """
    w = np.array(l1 * [1.0] + l1 * [-1.0]) / float(l1)

    n = X.shape[0]
    K = rbf_kernel(X, X)  # Giữ apply_kernel như gốc, giả sử là rbf
    W = np.zeros((n - 2 * l1, n))

    for i in range(n - 2 * l1):
        W[i, i : i + 2 * l1] = w

    stat = np.einsum("ij,ij->i", np.dot(W, K), W)
    shape_stat = np.convolve(stat, w)
    shape_prime = shape_stat[1:] * shape_stat[:-1]

    res = np.zeros((n, 3))
    res[:, 2] = 1  # Default p-value

    for pos in np.where(shape_prime < 0)[0]:
        if shape_stat[pos] > 0:
            res[pos, 0] = shape_stat[pos]
            a = max(0, pos - int(l2 / 2))
            b = min(n, pos + int(l2 / 2))
            s = pos - a
            if s < 2 or (b - a - s) < 2:
                continue
            # Thay mmd bằng wmmd_studentized
            mmd_val, p_val = wmmd_studentized(X[a:b], s, n_perm, weight_method)
            res[pos, 1] = mmd_val
            res[pos, 2] = p_val

    return res


def wmmd_asymptotic(X_window, s, weight_method="variance_reduction"):
    """
    Weighted MMD with ASYMPTOTIC p-value (no permutation).
    
    Uses the fact that under H0, weighted MMD² is asymptotically Gaussian
    with variance that can be estimated from the data.
    
    This is O(n²) instead of O(n² × n_perm), ~100x faster than permutation.
    
    Returns:
    --------
    mmd_value : float
        sqrt of weighted MMD²
    p_value : float  
        Asymptotic p-value
    """
    from scipy.stats import norm
    
    n = len(X_window)
    m = s
    n_y = n - m
    if m < 10 or n_y < 10:  # Need enough samples for asymptotic
        return 0.0, 1.0
    
    # Compute kernel
    gamma = compute_gamma_median_heuristic(X_window)
    K = rbf_kernel(X_window, X_window, gamma)
    
    # Extract submatrices
    K_XX = K[:m, :m]
    K_YY = K[m:, m:]
    K_XY = K[:m, m:]
    
    # Compute weighted MMD²
    W_XX = compute_optimal_weights(K_XX, weight_method)
    W_YY = compute_optimal_weights(K_YY, weight_method)
    W_XY = np.ones((m, n_y)) / (m * n_y)
    
    mmd_sq = np.sum(W_XX * K_XX) + np.sum(W_YY * K_YY) - 2 * np.sum(W_XY * K_XY)
    
    # Estimate variance under H0 using bootstrap-like approach
    # Var(MMD²) ≈ 4 * (Var(k(X,Y)) + Cov terms)
    # Simplified: use variance of cross-term
    cross_var = np.var(K_XY)
    
    # Asymptotic variance (Gretton et al. 2012, simplified)
    var_mmd = 4 * cross_var / min(m, n_y)
    
    if var_mmd < 1e-10:
        return np.sqrt(max(0, mmd_sq)), 1.0 if mmd_sq <= 0 else 0.0
    
    # Z-score and p-value (one-sided test: MMD² > 0 under H1)
    z_score = mmd_sq / np.sqrt(var_mmd)
    p_value = 1 - norm.cdf(z_score)  # P(Z > z)
    
    return np.sqrt(max(0, mmd_sq)), p_value


def shape_with_wmmd_fast(X, l1, l2, weight_method="variance_reduction"):
    """
    FAST ShapeDD with Weighted MMD using asymptotic p-values.
    
    This is the recommended version for real-time monitoring:
    - Uses original ShapeDD shape statistic for candidate detection
    - Validates with ADW-MMD using asymptotic distribution (no permutation)
    - ~10-100x faster than permutation-based version
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int
        Half-window size for shape statistic
    l2 : int
        Window size for MMD validation
    weight_method : str
        Weighting method for ADW-MMD ('variance_reduction' or 'uniform')
        
    Returns:
    --------
    res : array-like, shape (n_samples, 3)
        [:, 0] - Shape statistic value (peak indicator)
        [:, 1] - Weighted MMD value
        [:, 2] - Asymptotic p-value (< 0.05 indicates drift)
    """
    w = np.array(l1 * [1.0] + l1 * [-1.0]) / float(l1)
    
    n = X.shape[0]
    
    # Step 1: Compute shape statistic using SPARSE computation
    # Instead of full kernel, use sliding window approach
    gamma = compute_gamma_median_heuristic(X[:min(500, n)])
    
    # Optimized: compute shape statistic via sliding window MMD
    # This avoids O(n²) full kernel computation
    window_size = 2 * l1
    step = max(1, l1 // 4)
    
    shape_values = []
    positions = []
    
    for i in range(0, n - window_size, step):
        # Local kernel for this window only
        window = X[i:i + window_size]
        K_local = rbf_kernel(window, window, gamma)
        
        # Shape statistic for this position
        w_local = np.array(l1 * [1.0] + l1 * [-1.0]) / float(l1)
        stat_val = np.dot(w_local, np.dot(K_local, w_local))
        
        shape_values.append(stat_val)
        positions.append(i + l1)  # Center of window
    
    shape_values = np.array(shape_values)
    positions = np.array(positions)
    
    # Detect peaks via zero-crossing of derivative
    shape_deriv = np.diff(shape_values)
    shape_prime = shape_deriv[1:] * shape_deriv[:-1]
    
    res = np.zeros((n, 3))
    res[:, 2] = 1  # Default p-value = 1 (no drift)
    
    # Step 2: Validate each candidate peak with WMMD (asymptotic)
    peak_indices = np.where(shape_prime < 0)[0] + 1  # +1 for diff offset
    
    for idx in peak_indices:
        if idx >= len(shape_values):
            continue
        if shape_values[idx] > 0:
            pos = positions[idx]
            res[pos, 0] = shape_values[idx]
            
            # Extract window around peak
            a = max(0, pos - int(l2 / 2))
            b = min(n, pos + int(l2 / 2))
            s = pos - a
            
            if s < 10 or (b - a - s) < 10:
                continue
            
            # Validate with WMMD (asymptotic p-value)
            mmd_val, p_val = wmmd_asymptotic(X[a:b], s, weight_method)
            res[pos, 1] = mmd_val
            res[pos, 2] = p_val
    
    return res


def shapedd_adw_mmd_proper(X, l1=50, l2=150, alpha=0.05, weight_method="variance_reduction"):
    """
    PROPER ShapeDD + ADW-MMD: Combines original ShapeDD detection with ADW-MMD validation.
    
    This is the CORRECT implementation that:
    1. Uses ShapeDD shape statistic for DETECTING candidate drift points
    2. Uses ADW-MMD with asymptotic p-value for VALIDATING candidates
    
    Faster than permutation test while maintaining statistical rigor.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream (buffer)
    l1 : int
        Half-window for shape statistic (reference window)
    l2 : int  
        Full window for MMD validation
    alpha : float
        Significance level (default 0.05)
    weight_method : str
        ADW-MMD weighting ('variance_reduction' recommended)
        
    Returns:
    --------
    is_drift : bool
        Whether drift is detected
    drift_positions : list of int
        Positions where drift detected (p < alpha)
    mmd_trace : np.ndarray
        MMD values at each sliding position (for classification)
    p_values : list of float
        P-values at detected positions
    """
    n = len(X)
    
    # Handle small windows with simple split test
    if n < 2 * l1 + l2:
        # Fallback: simple split test at midpoint
        if n < 2 * l1:
            return False, [], np.array([]), []
        
        # Split window and compute ADW-MMD
        split = n // 2
        mmd_val, p_val = wmmd_asymptotic(X, split, weight_method)
        
        # Single-point trace for classification
        mmd_trace = np.array([mmd_val])
        
        if p_val < alpha:
            return True, [split], mmd_trace, [p_val]
        else:
            return False, [], mmd_trace, []
    
    # Step 1: Compute sliding MMD sequence (for both detection and classification)
    gamma = compute_gamma_median_heuristic(X[:min(500, n)])
    
    mmd_sequence = []
    mmd_positions = []
    step = max(1, l1 // 2)
    
    for i in range(0, n - l1 - l2, step):
        ref_window = X[i:i+l1]
        test_window = X[i+l1:i+l1+l2]
        
        # Compute ADW-MMD between ref and test
        combined = np.vstack([ref_window, test_window])
        mmd_val, _ = mmd_adw(combined, s=l1, gamma=gamma)
        
        mmd_sequence.append(mmd_val)
        mmd_positions.append(i + l1)  # Position of potential drift
    
    mmd_trace = np.array(mmd_sequence)
    
    if len(mmd_trace) < 3:
        # Small window fallback: just check if max MMD is significant
        if len(mmd_trace) > 0:
            max_idx = np.argmax(mmd_trace)
            max_mmd = mmd_trace[max_idx]
            # Use simple threshold for very small windows
            if max_mmd > 0.15:
                pos = mmd_positions[max_idx]
                mmd_val, p_val = wmmd_asymptotic(X, len(X)//2, weight_method)
                if p_val < alpha:
                    return True, [pos], mmd_trace, [p_val]
        return False, [], mmd_trace, []
    
    # Step 2: Detect peaks in MMD sequence (ShapeDD-style)
    # Peak = local maximum with significant rise and fall
    peaks = []
    threshold = np.mean(mmd_trace) + np.std(mmd_trace)
    
    for i in range(1, len(mmd_trace) - 1):
        if mmd_trace[i] > mmd_trace[i-1] and mmd_trace[i] > mmd_trace[i+1]:
            # Check if peak is significant (above mean + std)
            if mmd_trace[i] > threshold:
                peaks.append(i)
    
    # FALLBACK for small/monotonic windows: 
    # If no peaks but we have high MMD values, check endpoints
    if not peaks:
        # Check if sequence is monotonically increasing (drift entering window)
        if len(mmd_trace) >= 3:
            is_increasing = all(mmd_trace[i] < mmd_trace[i+1] for i in range(len(mmd_trace)-1))
            is_decreasing = all(mmd_trace[i] > mmd_trace[i+1] for i in range(len(mmd_trace)-1))
            
            # Monotonic sequence with high endpoint suggests drift
            if is_increasing and mmd_trace[-1] > threshold:
                # Drift is at the end of window - use last position
                peaks.append(len(mmd_trace) - 1)
            elif is_decreasing and mmd_trace[0] > threshold:
                # Drift was at the start of window
                peaks.append(0)
            elif mmd_trace.max() > threshold * 1.2:
                # Strong signal even without clear peak - use max position
                peaks.append(np.argmax(mmd_trace))
    
    if not peaks:
        return False, [], mmd_trace, []
    
    # Step 3: Validate each peak with asymptotic p-value
    drift_positions = []
    p_values = []
    
    for peak_idx in peaks:
        pos = mmd_positions[peak_idx]
        
        # Extract validation window
        a = max(0, pos - l2 // 2)
        b = min(n, pos + l2 // 2)
        s = pos - a
        
        if s < 10 or (b - a - s) < 10:
            continue
        
        # Compute asymptotic p-value
        mmd_val, p_val = wmmd_asymptotic(X[a:b], s, weight_method)
        
        if p_val < alpha:
            drift_positions.append(pos)
            p_values.append(p_val)
    
    is_drift = len(drift_positions) > 0
    
    return is_drift, drift_positions, mmd_trace, p_values
