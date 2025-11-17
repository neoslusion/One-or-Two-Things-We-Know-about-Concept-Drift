import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SpectraDrift:
    def __init__(self, k=None, n_eigenvalues=10, alpha=0.05):
        """
        SPECTRA-DRIFT: Spectral Graph-based Drift Detection

        Parameters:
        -----------
        k : int or None
            Number of neighbors in k-NN graph. If None, uses sqrt(n)
        n_eigenvalues : int
            Number of eigenvalues to extract (default: 10)
        alpha : float
            False positive rate for adaptive threshold (default: 0.05)
        """
        self.k = k
        self.n_eigenvalues = n_eigenvalues
        self.alpha = alpha
        self.reference_features = None
        self.feature_scaler = StandardScaler()
        self.threshold = None
        self.calibration_scores = []

    def _build_graph(self, X):
        """Build k-NN graph with Gaussian weights"""
        n = len(X)
        k = self.k if self.k is not None else max(5, int(np.sqrt(n)))
        k = min(k, n - 1)  # Ensure k < n

        # k-NN search
        knn = NearestNeighbors(n_neighbors=k + 1)  # +1 because includes self
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        # Remove self-loops (first column is self with distance 0)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Self-tuning bandwidth (Zelnik-Manor & Perona, 2004)
        # Use median of k-th nearest neighbor distances
        sigma = np.median(distances[:, -1]) / np.sqrt(2)
        if sigma == 0:
            sigma = 1e-6  # Avoid division by zero

        # Gaussian weights
        weights = np.exp(-distances**2 / (2 * sigma**2))

        # Build sparse adjacency matrix
        row_ind = np.repeat(np.arange(n), k)
        col_ind = indices.flatten()
        data = weights.flatten()

        W = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

        # Symmetrize
        W = (W + W.T) / 2

        return W

    def _compute_laplacian(self, W):
        """Compute normalized symmetric Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}"""
        # Degree matrix
        D = np.array(W.sum(axis=1)).flatten()

        # Handle isolated nodes (degree = 0)
        D_inv_sqrt = np.zeros_like(D)
        nonzero = D > 0
        D_inv_sqrt[nonzero] = np.power(D[nonzero], -0.5)

        # Create diagonal sparse matrix (CORRECT way)
        D_inv_sqrt_mat = diags(D_inv_sqrt)

        # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
        n = W.shape[0]
        I = eye(n)
        L_sym = I - D_inv_sqrt_mat @ W @ D_inv_sqrt_mat

        return L_sym

    def _extract_features(self, eigenvalues):
        """
        Extract multi-scale spectral features

        Features (11-dimensional):
        1. lambda_2 (Fiedler value) - global connectivity
        2. spectral gap (lambda_2 - lambda_1)
        3. mean(lambda_2:lambda_5) - low eigenvalues (mid-scale)
        4. std(lambda_2:lambda_5)
        5. mean(high eigenvalues) - local structure
        6. spectral entropy H(lambda)
        7. sum(eigenvalues) - trace
        8. sum(eigenvalues^2) - 2nd moment
        9. max eigenvalue
        10. eigenvalue range (max - min)
        11. median eigenvalue
        """
        n_eig = len(eigenvalues)

        # 1-2: Fiedler value and spectral gap
        lambda_1 = eigenvalues[0] if n_eig > 0 else 0
        lambda_2 = eigenvalues[1] if n_eig > 1 else 0
        spectral_gap = lambda_2 - lambda_1

        # 3-4: Low eigenvalues (mid-scale structure)
        low_eigs = eigenvalues[1:min(5, n_eig)]
        mean_low = np.mean(low_eigs) if len(low_eigs) > 0 else 0
        std_low = np.std(low_eigs) if len(low_eigs) > 0 else 0

        # 5: High eigenvalues (local structure)
        high_eigs = eigenvalues[-5:] if n_eig >= 5 else eigenvalues
        mean_high = np.mean(high_eigs) if len(high_eigs) > 0 else 0

        # 6: Spectral entropy
        probs = eigenvalues / (eigenvalues.sum() + 1e-10)
        probs = probs[probs > 1e-10]
        entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0

        # 7-8: Moments
        trace = eigenvalues.sum()
        second_moment = np.sum(eigenvalues**2)

        # 9-11: Range statistics
        eig_max = eigenvalues.max() if n_eig > 0 else 0
        eig_range = eigenvalues.max() - eigenvalues.min() if n_eig > 0 else 0
        eig_median = np.median(eigenvalues) if n_eig > 0 else 0

        features = np.array([
            lambda_2,
            spectral_gap,
            mean_low,
            std_low,
            mean_high,
            entropy,
            trace,
            second_moment,
            eig_max,
            eig_range,
            eig_median
        ])

        return features

    def fit(self, X):
        """
        Fit detector on reference data (no drift)

        This calibrates the threshold via bootstrap
        """
        X = np.asarray(X)

        # Build graph and compute eigenvalues
        W = self._build_graph(X)
        L = self._compute_laplacian(W)

        k = min(self.n_eigenvalues, L.shape[0] - 2)
        if k < 2:
            raise ValueError(f"Not enough samples ({L.shape[0]}) for eigenvalue computation")

        eigenvalues, _ = eigsh(L, k=k, which='SA')  # Smallest Algebraic
        eigenvalues = np.sort(eigenvalues)

        # Extract and normalize features
        features = self._extract_features(eigenvalues)

        # Fit scaler on reference features
        self.feature_scaler.fit(features.reshape(1, -1))
        self.reference_features = self.feature_scaler.transform(features.reshape(1, -1)).flatten()

        # Calibrate threshold via bootstrap
        self._calibrate_threshold(X)

        return self

    def _calibrate_threshold(self, X, n_bootstrap=50):
        """
        Calibrate threshold via bootstrap on reference data

        Split X randomly into two halves, compute drift score (null distribution)
        Threshold = (1-alpha) quantile of null scores
        """
        n = len(X)
        if n < 100:
            # Not enough data, use default threshold
            self.threshold = 2.0  # Conservative default
            return

        null_scores = []

        for _ in range(n_bootstrap):
            # Random split
            indices = np.random.permutation(n)
            X1 = X[indices[:n//2]]
            X2 = X[indices[n//2:]]

            try:
                # Compute features for both halves
                W1 = self._build_graph(X1)
                L1 = self._compute_laplacian(W1)
                k1 = min(self.n_eigenvalues, L1.shape[0] - 2)
                eigs1, _ = eigsh(L1, k=k1, which='SA')
                eigs1 = np.sort(eigs1)
                feat1 = self._extract_features(eigs1)
                feat1 = self.feature_scaler.transform(feat1.reshape(1, -1)).flatten()

                W2 = self._build_graph(X2)
                L2 = self._compute_laplacian(W2)
                k2 = min(self.n_eigenvalues, L2.shape[0] - 2)
                eigs2, _ = eigsh(L2, k=k2, which='SA')
                eigs2 = np.sort(eigs2)
                feat2 = self._extract_features(eigs2)
                feat2 = self.feature_scaler.transform(feat2.reshape(1, -1)).flatten()

                # Null score (no drift)
                score = np.linalg.norm(feat1 - feat2)
                null_scores.append(score)
            except:
                continue

        if len(null_scores) > 0:
            # Threshold = (1-alpha) quantile
            self.threshold = np.quantile(null_scores, 1 - self.alpha)
            self.calibration_scores = null_scores
        else:
            self.threshold = 2.0  # Fallback

    def detect(self, X):
        """
        Detect drift in current window

        Returns:
        --------
        dict with keys:
            - drift_detected: bool
            - score: float (normalized distance)
            - drift_type: str (sudden/gradual/incremental/stable)
            - eigenvalues: array
            - features: array (normalized)
        """
        if self.reference_features is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        X = np.asarray(X)

        # Build graph and compute eigenvalues
        W = self._build_graph(X)
        L = self._compute_laplacian(W)

        k = min(self.n_eigenvalues, L.shape[0] - 2)
        eigenvalues, _ = eigsh(L, k=k, which='SA')
        eigenvalues = np.sort(eigenvalues)

        # Extract and normalize features
        features = self._extract_features(eigenvalues)
        features_normalized = self.feature_scaler.transform(features.reshape(1, -1)).flatten()

        # Compute drift score (L2 distance in normalized space)
        score = np.linalg.norm(features_normalized - self.reference_features)

        # Drift detection
        drift_detected = score > self.threshold

        # Classify drift type (if drift detected)
        drift_type = self._classify_drift(features_normalized, eigenvalues) if drift_detected else 'stable'

        return {
            'drift_detected': drift_detected,
            'score': score,
            'threshold': self.threshold,
            'drift_type': drift_type,
            'eigenvalues': eigenvalues,
            'features': features_normalized,
            'raw_features': features
        }

    def _classify_drift(self, features_norm, eigenvalues):
        """
        Classify drift type based on spectral features

        - Sudden: Large change in lambda_2 (spectral gap)
        - Gradual: Moderate change in mid-spectrum
        - Incremental: Change in high eigenvalues
        - Stable: No significant change
        """
        # Feature indices
        lambda_2_idx = 0
        spectral_gap_idx = 1
        mean_low_idx = 2
        mean_high_idx = 4

        # Changes (normalized space)
        lambda_2_change = abs(features_norm[lambda_2_idx] - self.reference_features[lambda_2_idx])
        gap_change = abs(features_norm[spectral_gap_idx] - self.reference_features[spectral_gap_idx])
        low_change = abs(features_norm[mean_low_idx] - self.reference_features[mean_low_idx])
        high_change = abs(features_norm[mean_high_idx] - self.reference_features[mean_high_idx])

        # Classification logic (thresholds in normalized space)
        if lambda_2_change > 1.5 or gap_change > 1.5:
            # Large drop in connectivity → sudden drift
            return 'sudden'
        elif low_change > high_change and low_change > 1.0:
            # Mid-spectrum change dominant → gradual drift
            return 'gradual'
        elif high_change > low_change and high_change > 1.0:
            # High eigenvalue change → incremental drift
            return 'incremental'
        else:
            return 'stable'

# ============================================================================
# Utility Functions (API compatible with shape_dd.py)
# ============================================================================

def spectra(X_ref, X_current, k=None, n_eigenvalues=10, alpha=0.05):
    """
    SPECTRA-DRIFT: Spectral Graph Theory-based Drift Detection

    Parameters:
    -----------
    X_ref : array (n_ref, d)
        Reference window (stable data, no drift)
    X_current : array (n_curr, d)
        Current window (test for drift)
    k : int or None
        Number of neighbors in k-NN graph (default: sqrt(n))
    n_eigenvalues : int
        Number of eigenvalues to extract (default: 10)
    alpha : float
        False positive rate (default: 0.05 = 5%)

    Returns:
    --------
    drift_detected : bool
        True if drift detected
    score : float
        Drift score (L2 distance in normalized feature space)
    drift_type : str
        Type of drift: 'sudden', 'gradual', 'incremental', or 'stable'

    Example:
    --------
    >>> X_ref = stream[:500]  # First 500 samples (no drift)
    >>> X_current = stream[1500:2000]  # Window after drift
    >>> detected, score, dtype = spectra(X_ref, X_current)
    >>> print(f"Drift: {detected}, Score: {score:.3f}, Type: {dtype}")
    """
    detector = SpectraDrift(k=k, n_eigenvalues=n_eigenvalues, alpha=alpha)
    detector.fit(X_ref)
    result = detector.detect(X_current)

    return result['drift_detected'], result['score'], result['drift_type']


def spectra_streaming(X, window_size=500, k=None, n_eigenvalues=10, alpha=0.05,
                      update_on_drift=True):
    """
    Streaming version of SPECTRA-DRIFT (compatible with shape() API)

    Parameters:
    -----------
    X : array (n_samples, n_features)
        Full data stream
    window_size : int
        Size of sliding window (default: 500)
    k : int or None
        Number of neighbors (default: sqrt(window_size))
    n_eigenvalues : int
        Number of eigenvalues to extract (default: 10)
    alpha : float
        False positive rate (default: 0.05)
    update_on_drift : bool
        If True, update reference window when drift detected (default: True)

    Returns:
    --------
    res : array (n_samples, 3)
        [:, 0] = spectral drift score (normalized)
        [:, 1] = drift_type_encoded (0=stable, 1=sudden, 2=gradual, 3=incremental)
        [:, 2] = binary detection (1=drift, 0=no_drift)

    Example:
    --------
    >>> X = generate_stream_with_drift(10000)
    >>> results = spectra_streaming(X, window_size=500)
    >>> plt.plot(results[:, 0])  # Plot drift scores
    >>> plt.plot(results[:, 2] * 5)  # Plot detections
    """
    n = X.shape[0]
    res = np.zeros((n, 3))

    # Need at least 2 windows
    if n < window_size * 2:
        print(f"Warning: Stream too short ({n} < {window_size*2}). Returning zeros.")
        return res

    # Initialize detector on first window
    detector = SpectraDrift(k=k, n_eigenvalues=n_eigenvalues, alpha=alpha)
    X_ref = X[:window_size]

    try:
        detector.fit(X_ref)
    except Exception as e:
        print(f"Error fitting detector: {e}")
        return res

    # Drift type encoding
    drift_type_map = {'stable': 0, 'sudden': 1, 'gradual': 2, 'incremental': 3}

    # Sliding window detection
    for i in range(window_size, n - window_size + 1):
        X_curr = X[i:i+window_size]

        try:
            result = detector.detect(X_curr)

            res[i, 0] = result['score']
            res[i, 1] = drift_type_map.get(result['drift_type'], 0)
            res[i, 2] = 1 if result['drift_detected'] else 0

            # Update reference if drift detected
            if update_on_drift and result['drift_detected']:
                detector.fit(X_curr)

        except Exception as e:
            # If detection fails, keep previous values
            if i > window_size:
                res[i, :] = res[i-1, :]

    return res


# ============================================================================
# Quick Test Function
# ============================================================================

def test_spectra():
    """Quick test on synthetic data"""
    from sklearn.datasets import make_blobs

    print("=" * 60)
    print("SPECTRA-DRIFT Quick Test")
    print("=" * 60)

    # Generate reference data (no drift)
    X_ref, _ = make_blobs(n_samples=300, centers=2, n_features=5,
                          random_state=42, cluster_std=1.0)

    # Generate current data WITH drift (different centers)
    X_drift, _ = make_blobs(n_samples=300, centers=2, n_features=5,
                            random_state=99, cluster_std=1.5,
                            center_box=(-5, 5))

    # Generate current data NO drift (same distribution)
    X_no_drift, _ = make_blobs(n_samples=300, centers=2, n_features=5,
                               random_state=123, cluster_std=1.0)

    # Test
    print("\n1. Testing on NO DRIFT data:")
    detected, score, dtype = spectra(X_ref, X_no_drift, alpha=0.05)
    print(f"   Drift Detected: {detected}")
    print(f"   Score: {score:.3f}")
    print(f"   Type: {dtype}")
    print(f"   Expected: False (no drift)")

    print("\n2. Testing on DRIFT data:")
    detected, score, dtype = spectra(X_ref, X_drift, alpha=0.05)
    print(f"   Drift Detected: {detected}")
    print(f"   Score: {score:.3f}")
    print(f"   Type: {dtype}")
    print(f"   Expected: True (drift detected)")

    print("\n" + "=" * 60)
    print("Test completed! If both results match expectations, code is working.")
    print("=" * 60)


if __name__ == "__main__":
    test_spectra()
