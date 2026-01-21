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


def compute_optimal_weights(K, method: str = "variance_reduction"):
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
        - 'adaptive': Density-based weighting (synonym for variance_reduction in this context)

    Returns:
    --------
    W : array-like, shape (n, n)
        Weight matrix with zeros on diagonal, normalized to sum to 1
    """
    n = K.shape[0]

    if method == "uniform":
        # Standard V-statistic (equal weights)
        W = np.ones((n, n)) / (n * n)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)

    elif method == "variance_reduction":
        # Variance-optimal weights (Bharti et al., 2023)
        # w_i proportional to 1/sqrt(sum_j K(x_i, x_j))
        K_off = K.copy()
        np.fill_diagonal(K_off, 0)

        k_sums = np.sum(K_off, axis=1)
        k_sums = np.maximum(k_sums, 1e-10)  # Numerical stability

        inv_weights = 1.0 / (np.sqrt(k_sums) + 0.5)  # Dampening alpha=0.5
        W = np.outer(inv_weights, inv_weights)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)

    elif method == "adaptive":
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


def compute_adw_mmd_squared(X, Y, gamma=None, weight_method="variance_reduction"):
    """
    Compute Adaptive Density-Weighted MMD² (ADW-MMD²) between X and Y.

    Returns MMD² directly (can be negative due to variance reduction).
    This is the correct statistic for hypothesis testing.

    The weighting scheme corresponds to the "Optimally-Weighted" estimator from
    Bharti et al. (2023), which uses weights inversely proportional to the
    square root of the kernel density to minimize estimation variance.
    We rename it to ADW-MMD to avoid confusion with "Optimal MMD" (test power optimization)
    and "Importance Weighted MMD" (covariate shift).

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
        ADW-MMD² statistic (can be negative)
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


def compute_adw_mmd(X, Y, gamma="auto", weight_method="variance_reduction"):
    """
    Compute Adaptive Density-Weighted MMD (ADW-MMD) between X and Y.

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
        ADW-MMD statistic (non-negative)
    """
    mmd_sq = compute_adw_mmd_squared(X, Y, gamma, weight_method)
    return np.sqrt(max(0, mmd_sq))


def mmd_adw(X, s=None, gamma="auto", weight_method="variance_reduction"):
    """
    Compute ADW-MMD between two halves of X (split-based interface).

    This is the basic ADW-MMD computation with a fixed threshold.

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
        ADW-MMD statistic
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
        pattern_score = _simple_pattern_detection(mmd_norm)
    elif mode == "enhanced":
        pattern_score = _enhanced_pattern_detection(mmd_array, mmd_norm)
    else:
        raise ValueError(f"Unknown mode: {mode}")

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
