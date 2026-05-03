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
# IDW-MMD (Inverse Density-Weighted MMD)
# =============================================================================
# The contribution of this thesis: a weighted MMD variant where each sample
# is weighted inversely proportional to the square root of its local kernel
# density estimate:
#       w_i  ∝  1 / sqrt( Σ_j  k(x_i, x_j) )
# This up-weights boundary / low-density points, increasing sensitivity to
# distributional shifts at the support boundary while keeping false-positive
# control tight.
# =============================================================================

def compute_optimal_weights(K, method: str = "inverse_density"):
    """
    Compute sample weights using Inverse Density Weighting (IDW-MMD).

    Contribution: up-weights boundary / low-density points to improve
    drift sensitivity (see thesis §3.2 for the derivation).

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


def compute_idw_mmd_squared(X, Y, gamma=None, weight_method="inverse_density"):
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


def compute_idw_mmd(X, Y, gamma="auto", weight_method="inverse_density"):
    """Wrapper for sqrt(MMD²)."""
    mmd_sq = compute_idw_mmd_squared(X, Y, gamma, weight_method)
    return np.sqrt(max(0, mmd_sq))


def mmd_idw(X, s=None, gamma="auto", weight_method="inverse_density"):
    """Split-window IDW-MMD computation for drift signals."""
    if s is None:
        s = len(X) // 2

    X_ref = X[:s]
    X_test = X[s:]

    mmd_value = compute_idw_mmd(X_ref, X_test, gamma, weight_method)
    return mmd_value, 0.1  # Threshold placeholder


def _wmmd_from_K(K, m, n_y, indices_x, indices_y, weight_method):
    """
    Compute the (possibly weighted) MMD² statistic from a *pre-computed* kernel
    matrix ``K`` using the given index sets ``indices_x``, ``indices_y``.

    This helper is used by ``wmmd_gamma`` to produce a fast bootstrap null
    distribution: the kernel matrix is computed *once* on the pooled sample,
    and each "permutation" only re-indexes into ``K`` rather than recomputing
    O(n²) kernel evaluations.

    The weighting scheme matches the IDW-MMD definition used in the thesis:
    XX and YY use IDW weights, the cross-term XY uses uniform 1/(mn_y).
    """
    indices_x = np.asarray(indices_x, dtype=int)
    indices_y = np.asarray(indices_y, dtype=int)
    K_xx = K[np.ix_(indices_x, indices_x)]
    K_yy = K[np.ix_(indices_y, indices_y)]
    K_xy = K[np.ix_(indices_x, indices_y)]
    W_xx = compute_optimal_weights(K_xx, weight_method)
    W_yy = compute_optimal_weights(K_yy, weight_method)
    W_xy = np.ones((m, n_y)) / (m * n_y)
    return float(np.sum(W_xx * K_xx) + np.sum(W_yy * K_yy) - 2.0 * np.sum(W_xy * K_xy))


def wmmd_gamma(
    X_window,
    s,
    weight_method="variance_reduction",
    n_null_samples: int = 20,
    seed: int = 0,
):
    """
    (Weighted) MMD with **moment-matched Gamma null** p-value.

    This replaces the previous ``wmmd_asymptotic`` which applied the
    *Gaussian H₁ asymptotic* of standard MMD as if it were the H₀ null
    distribution -- an incorrect derivation that produced a systematically
    over-conservative test (empirical Type-I error ≈ 0 on stationary
    streams instead of the nominal α). See thesis §3.2.4 and the audit
    note in this module for context.

    Mathematical justification
    --------------------------
    Under H₀ (P = Q), the limiting distribution of the (biased) MMD²
    statistic is a sum of weighted χ² variables (Gretton et al. 2012,
    Thm. 12), *not* a Gaussian centered at zero.  A simple, correct
    fast approximation is the **two-moment Gamma fit** proposed by
    Gretton et al. NIPS 2009, which fits a Gamma(k, θ) distribution
    matching the empirical first two moments of the null statistic:

        k = (E[ξ])² / Var[ξ],      θ = Var[ξ] / E[ξ],
        p-value = 1 - F_Γ(observed; k, θ).

    To estimate (E[ξ], Var[ξ]) under H₀ for the *weighted* IDW-MMD
    (whose null moments do not have a clean closed form because of the
    asymmetric IDW vs. uniform cross-term), we use a **fast index
    permutation** of the *single* kernel matrix already computed on
    the pooled window: the kernel is O(n²) once, then each null
    sample only indexes into ``K`` (no kernel re-computation).

    With ``n_null_samples = 20`` this costs ~20× a single MMD
    evaluation -- still ~125× cheaper than the original 2500-permutation
    test, while remaining properly calibrated at α = 0.05.

    Parameters
    ----------
    X_window : (n, d) ndarray
        Concatenated [reference; test] window.
    s : int
        Split point: X = X_window[:s], Y = X_window[s:].
    weight_method : str
        Passed to ``compute_optimal_weights`` (e.g. "variance_reduction"
        for IDW, "uniform" for unweighted).
    n_null_samples : int, default 20
        Number of label permutations used to estimate the null
        distribution moments. The variance of the moment estimate scales
        as 1/n_null_samples; B=20 is enough for a stable Gamma fit while
        keeping latency low.
    seed : int, default 0
        Seed for the permutation RNG.  Fixed by default so each call to
        the detector is deterministic given the input window (which is
        what callers in ``shapedd_idw_mmd_proper`` rely on).

    Returns
    -------
    mmd_sqrt : float
        sqrt(max(0, observed MMD²)).
    p_value : float
        Right-tail Gamma p-value.  Falls back to the empirical bootstrap
        p-value when the fitted Gamma is degenerate (zero/near-zero
        mean or variance under H₀).
    """
    from scipy.stats import gamma as gamma_dist

    n = len(X_window)
    m = s
    n_y = n - m
    if m < 10 or n_y < 10:
        return 0.0, 1.0

    g = compute_gamma_median_heuristic(X_window)
    K = rbf_kernel(X_window, X_window, g)  # one-shot kernel computation

    obs_mmd_sq = _wmmd_from_K(
        K, m, n_y,
        np.arange(m),
        np.arange(m, n),
        weight_method,
    )

    # --- Build empirical null distribution by index permutation ---
    rng = np.random.RandomState(seed)
    null_samples = np.empty(n_null_samples, dtype=float)
    for b in range(n_null_samples):
        perm = rng.permutation(n)
        null_samples[b] = _wmmd_from_K(
            K, m, n_y, perm[:m], perm[m:m + n_y], weight_method
        )

    null_mean = float(np.mean(null_samples))
    null_var = float(np.var(null_samples, ddof=1)) if n_null_samples > 1 else 0.0

    # --- Gamma moment-matching fit ---
    # If the null moments are degenerate (e.g. all permutations gave
    # numerically identical statistics, which can happen for tiny windows
    # or pathological inputs), fall back to a direct empirical p-value.
    if (
        not np.isfinite(null_mean)
        or not np.isfinite(null_var)
        or null_mean <= 1e-12
        or null_var <= 1e-12
    ):
        # Empirical right-tail bootstrap p-value with +1/+1 smoothing.
        rejections = int(np.sum(null_samples >= obs_mmd_sq))
        p_value = float((rejections + 1) / (n_null_samples + 1))
    else:
        k_param = (null_mean ** 2) / null_var          # shape
        theta_param = null_var / null_mean             # scale
        # P(ξ ≥ obs | H0) under Gamma(k, θ).
        p_value = float(1.0 - gamma_dist.cdf(obs_mmd_sq, a=k_param, scale=theta_param))
        p_value = float(np.clip(p_value, 0.0, 1.0))

    return float(np.sqrt(max(0.0, obs_mmd_sq))), p_value


def wmmd_asymptotic(X_window, s, weight_method="variance_reduction"):
    """DEPRECATED: kept as an alias for backward compatibility.

    The original Gaussian-asymptotic implementation applied the H₁
    leading-order variance ``4·Var(K_XY)/min(n,m)`` from Gretton 2012
    (Thm. 8) to compute a one-sided z-test under H₀.  Under H₀ the true
    MMD² limit is a (degenerate) sum of weighted χ², not a Gaussian
    centered at zero, so that derivation was incorrect; on stationary
    streams it produced a systematically over-conservative test
    (empirical Type-I error ≈ 0).

    We now route this entry point to :func:`wmmd_gamma`, which uses a
    moment-matched Gamma fit on a fast (B=20) index-permutation null.
    See ``wmmd_gamma`` for the mathematical justification.
    """
    return wmmd_gamma(X_window, s, weight_method=weight_method)


# =============================================================================
# SE-CDT MAIN DETECTION LOGIC (PROPER)
# =============================================================================

def shapedd_idw_mmd_proper(X, l1=50, l2=150, alpha=0.05, weight_method="variance_reduction"):
    """
    ShapeDD + IDW-MMD — Hybrid two-role detector.

    Architecture (matches original ShapeDD evaluation paradigm):
    -----------------------------------------------------------------------
    Role 1 — MMD TRACE (compact 2×l1 windows, same as original ShapeDD):
        Uses *Standard* (unweighted) MMD computed via sliding contrast
        weight vectors over the full n×n kernel — produces a dense trace
        with step=1. Fits any window >= 2*l1 samples.

    Role 2 — VALIDATION p-value (IDW-MMD + Gamma null):
        At each trace peak, extracts a validation window of size l2,
        runs wmmd_gamma (IDW-MMD statistic, Gamma-approximation p-value).

    Pipeline:
    1. Compute n×n kernel once
    2. Trace: sliding 2×l1 contrast vectors → MMD² signal
    3. Peaks: local maxima above mean + std
    4. Validate: wmmd_gamma at each peak → confirm/reject

    Returns: is_drift, positions, mmd_trace, p_values
    """
    n = len(X)

    # Fallback for tiny windows
    if n < 2 * l1:
        return False, [], np.array([]), []

    # -----------------------------------------------------------------------
    # 1. TRACE: Compact 2×l1 sliding contrast vectors (same as original
    #    ShapeDD).  This avoids the l1+l2 per-trace-point cost and
    #    yields step=1 trace density for any window >= 2*l1.
    # -----------------------------------------------------------------------
    gamma = compute_gamma_median_heuristic(X[:min(500, n)])
    K = rbf_kernel(X, X, gamma)       # n×n kernel, computed once

    # Contrast weight vector: [+1/l1 repeated l1 times, -1/l1 repeated l1 times]
    w = np.empty(2 * l1, dtype=float)
    w[:l1] = 1.0 / l1
    w[l1:] = -1.0 / l1

    trace_len = n - 2 * l1 + 1
    mmd_trace = np.empty(trace_len, dtype=float)
    mmd_positions = np.arange(l1, n - l1 + 1)  # centre of each window

    # Precompute the weight matrix W (trace_len × n) and compute all
    # MMD² trace values via efficient matrix multiply, matching the
    # original ShapeDD formulation.
    W = np.zeros((trace_len, n), dtype=float)
    for i in range(trace_len):
        W[i, i : i + 2 * l1] = w

    # stat[i] = w^T K_sub w = MMD²(ref[i:i+l1], test[i+l1:i+2*l1])
    WK = np.dot(W, K)                                    # (trace_len, n)
    mmd_sq_trace = np.einsum("ij,ij->i", WK, W)         # (trace_len,)
    mmd_trace = np.sqrt(np.maximum(0.0, mmd_sq_trace))

    # -----------------------------------------------------------------------
    # 2. PEAK DETECTION on the standard-MMD trace
    # -----------------------------------------------------------------------
    if len(mmd_trace) < 3:
        return False, [], mmd_trace, []

    threshold = np.mean(mmd_trace) + np.std(mmd_trace)
    peaks = []
    for i in range(1, len(mmd_trace) - 1):
        if mmd_trace[i] > mmd_trace[i - 1] and mmd_trace[i] > mmd_trace[i + 1]:
            if mmd_trace[i] > threshold:
                peaks.append(i)

    # Fallback for boundary/monotonic peaks
    if not peaks and len(mmd_trace) >= 3:
        if mmd_trace.max() > threshold * 1.2:
            peaks.append(np.argmax(mmd_trace))

    if not peaks:
        return False, [], mmd_trace, []

    # -----------------------------------------------------------------------
    # 3. VALIDATION: IDW-MMD + Gamma-approximation p-value at each peak
    # -----------------------------------------------------------------------
    drift_positions = []
    p_values = []

    for peak_idx in peaks:
        pos = int(mmd_positions[peak_idx])

        a = max(0, pos - l2 // 2)
        b = min(n, pos + l2 // 2)
        s = pos - a

        if s < 10 or (b - a - s) < 10:
            continue

        _, p_val = wmmd_gamma(X[a:b], s, weight_method)

        if p_val < alpha:
            drift_positions.append(pos)
            p_values.append(p_val)

    return len(drift_positions) > 0, drift_positions, mmd_trace, p_values
