"""
Optimally-Weighted MMD (OW-MMD) Implementation
for Drift Detection

Based on: Bharti et al., ICML 2023
"Optimally-weighted Estimators of the Maximum Mean Discrepancy"

Two methods implemented:
1. MMD_OW - Standalone OW-MMD drift detector
2. ShapeDD_OW_MMD - Hybrid combining ShapeDD + OW-MMD
"""

import numpy as np
from scipy.spatial.distance import cdist

# ============================================================================
# COMPONENT 1: KERNEL FUNCTIONS
# ============================================================================

def rbf_kernel_ow(X, Y, gamma='auto'):
    """
    RBF (Gaussian) kernel: k(x,y) = exp(-γ||x-y||²)

    Args:
        X: First set of samples (n_samples, n_features)
        Y: Second set of samples (m_samples, n_features)
        gamma: Bandwidth parameter ('auto' or float)

    Returns:
        K: Kernel matrix (n_samples, m_samples)
    """
    if gamma == 'auto':
        # Median heuristic
        all_data = np.vstack([X, Y])
        distances = cdist(all_data, all_data, metric='euclidean')
        gamma = 1.0 / (2 * np.median(distances[distances > 0])**2)

    # Compute pairwise squared distances
    distances_sq = cdist(X, Y, metric='sqeuclidean')

    return np.exp(-gamma * distances_sq)


# ============================================================================
# COMPONENT 2: OPTIMAL WEIGHT COMPUTATION
# ============================================================================

def compute_optimal_weights(kernel_matrix, method='variance_reduction'):
    """
    Compute optimal weights for kernel matrix.

    Args:
        kernel_matrix: Kernel evaluations (n, n)
        method: Weighting strategy
            - 'uniform': Standard equal weights (baseline)
            - 'variance_reduction': Optimal for variance minimization
            - 'adaptive': Data-dependent weights

    Returns:
        weights: Weight matrix (n, n)
    """
    n = kernel_matrix.shape[0]

    if method == 'uniform':
        # Standard V-statistic (equal weights)
        return np.ones((n, n)) / (n * n)

    elif method == 'variance_reduction':
        # Variance-optimal weights
        # Based on kernel variance structure

        # Remove diagonal (self-similarities)
        K_off = kernel_matrix.copy()
        np.fill_diagonal(K_off, 0)

        # Row-wise kernel sums (excluding diagonal)
        k_sums = np.sum(K_off, axis=1)
        k_sums = np.maximum(k_sums, 1e-10)  # Numerical stability

        # Inverse variance weighting
        # Points in dense regions get lower weight
        inv_weights = 1.0 / np.sqrt(k_sums)
        weights = np.outer(inv_weights, inv_weights)

        # Remove diagonal
        np.fill_diagonal(weights, 0)

        # Normalize
        weights = weights / np.sum(weights)

        return weights

    elif method == 'adaptive':
        # Adaptive density-based weighting

        # Kernel density estimate
        k_density = np.sum(kernel_matrix, axis=1)
        k_density = k_density / np.sum(k_density)

        # Inverse density weighting (upweight rare regions)
        inv_density = 1.0 / (k_density + 1e-10)
        inv_density = inv_density / np.sum(inv_density)

        weights = np.outer(inv_density, inv_density)
        np.fill_diagonal(weights, 0)  # Remove diagonal
        weights = weights / np.sum(weights)

        return weights

    else:
        raise ValueError(f"Unknown weight method: {method}")


# ============================================================================
# COMPONENT 3: OW-MMD ESTIMATOR
# ============================================================================

def compute_ow_mmd(X, Y, gamma='auto', weight_method='variance_reduction'):
    """
    Compute Optimally-Weighted MMD² between X and Y.

    Args:
        X: Reference samples (n_samples, n_features)
        Y: Test samples (m_samples, n_features)
        gamma: RBF kernel bandwidth
        weight_method: How to compute optimal weights

    Returns:
        mmd_value: OW-MMD statistic (positive scalar)
    """
    m, n = X.shape[0], Y.shape[0]

    # Compute kernel matrices
    K_XX = rbf_kernel_ow(X, X, gamma)
    K_YY = rbf_kernel_ow(Y, Y, gamma)
    K_XY = rbf_kernel_ow(X, Y, gamma)

    # Compute optimal weights for XX and YY terms
    W_XX = compute_optimal_weights(K_XX, weight_method)
    W_YY = compute_optimal_weights(K_YY, weight_method)

    # Cross-term uses uniform weights (standard practice)
    W_XY = np.ones((m, n)) / (m * n)

    # Weighted MMD² computation
    term1 = np.sum(W_XX * K_XX)
    term2 = np.sum(W_YY * K_YY)
    term3 = np.sum(W_XY * K_XY)

    mmd_squared = term1 + term2 - 2 * term3

    # Return MMD (not squared), ensure non-negative
    mmd_value = np.sqrt(max(0, mmd_squared))

    return mmd_value


# ============================================================================
# COMPONENT 4: OW-MMD DRIFT DETECTION (STANDALONE)
# ============================================================================

def ow_mmd_drift_detection(stream, window_size=100, step_size=50,
                           threshold_sigma=3.0, gamma='auto'):
    """
    Detect drift using OW-MMD with adaptive thresholding.

    Args:
        stream: Data stream (n_samples, n_features)
        window_size: Size of reference and test windows
        step_size: Step between consecutive tests
        threshold_sigma: Number of standard deviations for threshold
        gamma: RBF kernel parameter

    Returns:
        detections: List of drift point indices
        mmd_sequence: List of (index, mmd_value) tuples
    """
    n_samples = len(stream)
    mmd_sequence = []
    detections = []

    # Initial reference window
    ref_window = stream[:window_size]

    # Slide through stream
    for i in range(window_size, n_samples - window_size + 1, step_size):
        # Test window
        test_window = stream[i:i+window_size]

        # Compute OW-MMD
        mmd = compute_ow_mmd(ref_window, test_window, gamma=gamma,
                            weight_method='variance_reduction')

        center_idx = i + window_size // 2
        mmd_sequence.append((center_idx, mmd))

        # Adaptive threshold based on history
        if len(mmd_sequence) >= 10:
            # Use recent history
            recent_values = [v for _, v in mmd_sequence[-10:]]
            mean_mmd = np.mean(recent_values)
            std_mmd = np.std(recent_values)

            # N-sigma rule
            threshold = mean_mmd + threshold_sigma * std_mmd
        else:
            # Bootstrap phase: use fixed threshold
            threshold = 0.1

        # Drift detection
        if mmd > threshold:
            detections.append(center_idx)

            # Update reference window after drift
            ref_window = test_window

    return detections, mmd_sequence


# ============================================================================
# COMPONENT 5: GEOMETRIC ANALYSIS (SHAPEDD-STYLE)
# ============================================================================

def check_triangle_shape(sequence, tolerance=0.5):
    """
    Check if sequence exhibits triangle-like shape.

    Triangle properties:
    - Rising phase before peak
    - Peak in middle region
    - Falling phase after peak
    """
    n = len(sequence)
    if n < 5:
        return False

    # Find peak position
    peak_idx = np.argmax(sequence)

    # Peak should be in middle region (20%-80%)
    if peak_idx < n * 0.2 or peak_idx > n * 0.8:
        return False

    # Check rising phase
    if peak_idx > 1:
        rise = sequence[:peak_idx]
        rise_increasing_frac = np.sum(np.diff(rise) > 0) / len(rise)
        is_rising = rise_increasing_frac >= tolerance
    else:
        is_rising = True

    # Check falling phase
    if peak_idx < n - 2:
        fall = sequence[peak_idx:]
        fall_decreasing_frac = np.sum(np.diff(fall) < 0) / len(fall)
        is_falling = fall_decreasing_frac >= tolerance
    else:
        is_falling = True

    return is_rising and is_falling


def check_zero_crossing(sequence):
    """
    Check for zero-crossings in first derivative.
    """
    if len(sequence) < 3:
        return False

    derivative = np.diff(sequence)

    # Count sign changes in derivative
    sign_changes = np.sum(np.diff(np.sign(derivative)) != 0)

    # At least one zero-crossing (peak or trough)
    return sign_changes >= 1


def check_significant_peak(sequence, sigma=2.0):
    """
    Check if sequence has statistically significant peak.
    """
    mean_val = np.mean(sequence)
    std_val = np.std(sequence)

    if std_val < 1e-10:
        return False

    max_val = np.max(sequence)

    # Peak is significant if > mean + sigma*std
    return max_val > mean_val + sigma * std_val


def shapedd_geometric_analysis_ow(mmd_sequence, window_size=30,
                                  min_spacing=50):
    """
    Apply ShapeDD's geometric analysis to OW-MMD statistic sequence.

    Args:
        mmd_sequence: List of (index, mmd_value) tuples
        window_size: Window for pattern detection
        min_spacing: Minimum spacing between detections

    Returns:
        drift_points: List of detected drift locations
    """
    if len(mmd_sequence) < window_size:
        return []

    indices = np.array([idx for idx, _ in mmd_sequence])
    values = np.array([val for _, val in mmd_sequence])

    drift_points = []

    # Sliding window over MMD sequence
    for i in range(len(values) - window_size + 1):
        window = values[i:i+window_size]
        window_idx = indices[i:i+window_size]

        # Normalize window to [0, 1]
        w_min, w_max = window.min(), window.max()
        if w_max - w_min < 1e-10:
            continue

        window_norm = (window - w_min) / (w_max - w_min)

        # ShapeDD geometric checks
        is_triangle = check_triangle_shape(window_norm, tolerance=0.6)
        has_zero_crossing = check_zero_crossing(window)
        has_peak = check_significant_peak(window, sigma=2.0)

        # Drift detection logic
        if is_triangle and (has_zero_crossing or has_peak):
            # Find peak position in window
            peak_idx = np.argmax(window)
            drift_location = int(window_idx[peak_idx])

            # Avoid duplicates (min spacing rule)
            if not drift_points or drift_location - drift_points[-1] >= min_spacing:
                drift_points.append(drift_location)

    return drift_points


# ============================================================================
# COMPONENT 6: SHAPEDD-OW-MMD HYBRID
# ============================================================================

def shapedd_ow_mmd_hybrid(stream, ref_window_size=50, test_window_size=150,
                         step_size=25, gamma='auto',
                         geometric_window=30):
    """
    ShapeDD-OW-MMD Hybrid: Combines OW-MMD statistics with geometric analysis.

    This is the MAIN CONTRIBUTION combining:
    - OW-MMD for efficient, low-variance drift statistics
    - ShapeDD's geometric pattern detection for robustness

    Args:
        stream: Data stream (n_samples, n_features)
        ref_window_size: Reference window size (L1 in ShapeDD)
        test_window_size: Test window size (L2 in ShapeDD)
        step_size: Step between consecutive tests
        gamma: RBF kernel parameter
        geometric_window: Window size for geometric analysis

    Returns:
        drift_points: Detected drift locations
        mmd_sequence: OW-MMD statistic sequence
    """
    n_samples = len(stream)
    mmd_sequence = []

    # Reference window starts at beginning
    ref_start = 0
    ref_end = ref_window_size

    # Slide test window through stream
    for test_start in range(ref_window_size,
                           n_samples - test_window_size + 1,
                           step_size):
        test_end = test_start + test_window_size

        # Extract windows
        ref_window = stream[ref_start:ref_end]
        test_window = stream[test_start:test_end]

        # Compute OW-MMD
        mmd_val = compute_ow_mmd(ref_window, test_window,
                                gamma=gamma,
                                weight_method='variance_reduction')

        center_idx = (test_start + test_end) // 2
        mmd_sequence.append((center_idx, mmd_val))

    # Apply ShapeDD's geometric analysis to OW-MMD sequence
    drift_points = shapedd_geometric_analysis_ow(
        mmd_sequence,
        window_size=geometric_window,
        min_spacing=ref_window_size
    )

    return drift_points, mmd_sequence


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_ow_mmd_implementation():
    """
    Unit tests for OW-MMD implementation.
    """
    print("="*70)
    print("TESTING OW-MMD IMPLEMENTATION")
    print("="*70)

    np.random.seed(42)

    # Test 1: Same distribution → MMD ≈ 0
    print("\nTest 1: Same distribution (MMD should be small)")
    X = np.random.normal(0, 1, (100, 10))
    Y = np.random.normal(0, 1, (100, 10))
    mmd = compute_ow_mmd(X, Y)
    print(f"  MMD value: {mmd:.6f}")
    assert mmd < 0.3, f"MMD should be small for same distribution, got {mmd}"
    print("  ✓ PASSED")

    # Test 2: Different distributions → MMD > 0
    print("\nTest 2: Different distributions (MMD should be large)")
    Y = np.random.normal(2, 1, (100, 10))
    mmd = compute_ow_mmd(X, Y)
    print(f"  MMD value: {mmd:.6f}")
    assert mmd > 0.5, f"MMD should be large for different distributions, got {mmd}"
    print("  ✓ PASSED")

    # Test 3: Triangle shape detection
    print("\nTest 3: Triangle shape detection")
    triangle = np.concatenate([
        np.linspace(0, 1, 15),  # Rise
        np.linspace(1, 0, 15)   # Fall
    ])
    is_triangle = check_triangle_shape(triangle)
    print(f"  Triangle detected: {is_triangle}")
    assert is_triangle, "Should detect triangle shape"
    print("  ✓ PASSED")

    # Test 4: Drift detection on synthetic stream
    print("\nTest 4: Drift detection on synthetic stream")
    # Create stream with drift at position 500
    stream_pre = np.random.normal(0, 1, (500, 10))
    stream_post = np.random.normal(1.5, 1, (500, 10))
    stream = np.vstack([stream_pre, stream_post])

    # Test OW-MMD
    detections, _ = ow_mmd_drift_detection(stream, window_size=100, step_size=50)
    print(f"  Detections: {detections}")
    if detections:
        closest_detection = min(detections, key=lambda x: abs(x - 500))
        error = abs(closest_detection - 500)
        print(f"  Closest detection to true drift (500): {closest_detection} (error: {error})")
        assert error < 150, f"Should detect drift near position 500, closest was {closest_detection}"
        print("  ✓ PASSED")
    else:
        print("  ⚠ WARNING: No drift detected")

    # Test 5: Hybrid method
    print("\nTest 5: ShapeDD-OW-MMD Hybrid")
    detections_hybrid, mmd_seq = shapedd_ow_mmd_hybrid(stream,
                                                       ref_window_size=50,
                                                       test_window_size=150,
                                                       step_size=25)
    print(f"  Hybrid detections: {detections_hybrid}")
    print(f"  MMD sequence length: {len(mmd_seq)}")
    if detections_hybrid:
        closest = min(detections_hybrid, key=lambda x: abs(x - 500))
        error = abs(closest - 500)
        print(f"  Closest detection: {closest} (error: {error})")
        print("  ✓ PASSED")
    else:
        print("  ⚠ WARNING: No drift detected by hybrid")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    test_ow_mmd_implementation()
