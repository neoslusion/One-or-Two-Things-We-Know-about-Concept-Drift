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
        # print("[MMD] Using CPU (Torch available but disabled)")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    # print("[MMD] Using CPU (NumPy only)")


# =============================================================================
# CORE KERNEL & UTILS
# =============================================================================

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
    """Compute gamma using median heuristic."""
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


# =============================================================================
# ADAPTIVE DENSITY-WEIGHTED MMD (ADW-MMD)
# =============================================================================

def compute_optimal_weights(K, method: str = "inverse_density"):
    """
    Compute sample weights using Inverse Density Weighting (IDW).
    
    Contribution: Upweights boundary points to improve drift sensitivity.
    w_i ∝ 1 / sqrt(Σ_j K(x_i, x_j))
    """
    n = K.shape[0]

    if method == "uniform":
        W = np.ones((n, n)) / (n * n)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)

    elif method in ("inverse_density", "variance_reduction"):
        K_off = K.copy()
        np.fill_diagonal(K_off, 0)

        k_sums = np.sum(K_off, axis=1)  # Kernel density estimate
        k_sums = np.maximum(k_sums, 1e-10)

        inv_weights = 1.0 / (np.sqrt(k_sums) + 0.5)
        W = np.outer(inv_weights, inv_weights)
        np.fill_diagonal(W, 0)
        return W / np.sum(W)

    else:
        raise ValueError(f"Unknown weight method: {method}")


def compute_adw_mmd_squared(X, Y, gamma=None, weight_method="inverse_density"):
    """
    Compute Weighted MMD² between X and Y using IDW.
    """
    # GPU Acceleration Entry Point
    if HAS_TORCH and len(X) >= 100:
        try:
            X_t = torch.from_numpy(X).float().to(DEVICE)
            Y_t = torch.from_numpy(Y).float().to(DEVICE)

            K_XX = rbf_kernel(X_t, X_t, gamma)
            K_YY = rbf_kernel(Y_t, Y_t, gamma)
            K_XY = rbf_kernel(X_t, Y_t, gamma)

            W_XX = compute_optimal_weights(K_XX, weight_method)
            W_YY = compute_optimal_weights(K_YY, weight_method)

            m, n = X.shape[0], Y.shape[0]
            W_XY = torch.ones((m, n), device=DEVICE) / (m * n)

            term1 = torch.sum(W_XX * K_XX)
            term2 = torch.sum(W_YY * K_YY)
            term3 = torch.sum(W_XY * K_XY)

            return (term1 + term2 - 2 * term3).cpu().item()
        except Exception:
            pass

    m, n = X.shape[0], Y.shape[0]

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    W_XX = compute_optimal_weights(K_XX, weight_method)
    W_YY = compute_optimal_weights(K_YY, weight_method)
    W_XY = np.ones((m, n)) / (m * n)

    term1 = np.sum(W_XX * K_XX)
    term2 = np.sum(W_YY * K_YY)
    term3 = np.sum(W_XY * K_XY)

    return term1 + term2 - 2 * term3


def compute_adw_mmd(X, Y, gamma="auto", weight_method="inverse_density"):
    """Wrapper for sqrt(MMD²)."""
    mmd_sq = compute_adw_mmd_squared(X, Y, gamma, weight_method)
    return np.sqrt(max(0, mmd_sq))


def mmd_adw(X, s=None, gamma="auto", weight_method="inverse_density"):
    """Split-window WMMD computation for drift signals."""
    if s is None:
        s = len(X) // 2

    X_ref = X[:s]
    X_test = X[s:]

    mmd_value = compute_adw_mmd(X_ref, X_test, gamma, weight_method)
    return mmd_value, 0.1  # Threshold placeholder


def wmmd_asymptotic(X_window, s, weight_method="variance_reduction"):
    """
    Weighted MMD with ASYMPTOTIC p-value (Fast Validation).
    
    Uses asymptotic Gaussian distribution of MMD² under H0.
    Replaces slow permutation tests.
    """
    from scipy.stats import norm
    
    n = len(X_window)
    m = s
    n_y = n - m
    if m < 10 or n_y < 10:
        return 0.0, 1.0
    
    gamma = compute_gamma_median_heuristic(X_window)
    K = rbf_kernel(X_window, X_window, gamma)
    
    K_XX = K[:m, :m]
    K_YY = K[m:, m:]
    K_XY = K[:m, m:]
    
    W_XX = compute_optimal_weights(K_XX, weight_method)
    W_YY = compute_optimal_weights(K_YY, weight_method)
    W_XY = np.ones((m, n_y)) / (m * n_y)
    
    mmd_sq = np.sum(W_XX * K_XX) + np.sum(W_YY * K_YY) - 2 * np.sum(W_XY * K_XY)
    
    # Simplified variance estimation for asymptotic test
    cross_var = np.var(K_XY)
    var_mmd = 4 * cross_var / min(m, n_y)
    
    if var_mmd < 1e-10:
        return np.sqrt(max(0, mmd_sq)), 1.0 if mmd_sq <= 0 else 0.0
    
    z_score = mmd_sq / np.sqrt(var_mmd)
    p_value = 1 - norm.cdf(z_score)
    
    return np.sqrt(max(0, mmd_sq)), p_value


# =============================================================================
# SE-CDT MAIN DETECTION LOGIC (PROPER)
# =============================================================================

def shapedd_adw_mmd_proper(X, l1=50, l2=150, alpha=0.05, weight_method="variance_reduction"):
    """
    PROPER ShapeDD + ADW-MMD.
    
    The MAIN detection algorithm for SE-CDT.
    1. Detection: Sliding window ADW-MMD (ShapeDD style)
    2. Validation: Asymptotic p-value (Fast WMMD)
    
    Returns: is_drift, positions, mmd_trace, p_values
    """
    n = len(X)
    
    # Handle small windows
    if n < 2 * l1 + l2:
        if n < 2 * l1:
            return False, [], np.array([]), []
        
        split = n // 2
        mmd_val, p_val = wmmd_asymptotic(X, split, weight_method)
        mmd_trace = np.array([mmd_val])
        if p_val < alpha:
            return True, [split], mmd_trace, [p_val]
        return False, [], mmd_trace, []
    
    # 1. Compute sliding MMD signal
    gamma = compute_gamma_median_heuristic(X[:min(500, n)])
    
    mmd_sequence = []
    mmd_positions = []
    step = max(1, l1 // 2)
    
    for i in range(0, n - l1 - l2, step):
        ref_window = X[i:i+l1]
        test_window = X[i+l1:i+l1+l2]
        
        combined = np.vstack([ref_window, test_window])
        mmd_val, _ = mmd_adw(combined, s=l1, gamma=gamma)
        
        mmd_sequence.append(mmd_val)
        mmd_positions.append(i + l1)
    
    mmd_trace = np.array(mmd_sequence)
    
    if len(mmd_trace) < 3:
        if len(mmd_trace) > 0 and np.max(mmd_trace) > 0.15:
            pos = mmd_positions[np.argmax(mmd_trace)]
            _, p_val = wmmd_asymptotic(X, len(X)//2, weight_method)
            if p_val < alpha:
                return True, [pos], mmd_trace, [p_val]
        return False, [], mmd_trace, []
    
    # 2. Peak Detection
    peaks = []
    threshold = np.mean(mmd_trace) + np.std(mmd_trace)
    
    for i in range(1, len(mmd_trace) - 1):
        if mmd_trace[i] > mmd_trace[i-1] and mmd_trace[i] > mmd_trace[i+1]:
            if mmd_trace[i] > threshold:
                peaks.append(i)
    
    # Fallback for boundary/monotonic peaks
    if not peaks and len(mmd_trace) >= 3:
        if mmd_trace.max() > threshold * 1.2:
            peaks.append(np.argmax(mmd_trace))
    
    if not peaks:
        return False, [], mmd_trace, []
    
    # 3. Asymptotic Validation
    drift_positions = []
    p_values = []
    
    for peak_idx in peaks:
        pos = mmd_positions[peak_idx]
        
        a = max(0, pos - l2 // 2)
        b = min(n, pos + l2 // 2)
        s = pos - a
        
        if s < 10 or (b - a - s) < 10:
            continue
        
        _, p_val = wmmd_asymptotic(X[a:b], s, weight_method)
        
        if p_val < alpha:
            drift_positions.append(pos)
            p_values.append(p_val)
    
    return len(drift_positions) > 0, drift_positions, mmd_trace, p_values
