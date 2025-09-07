from __future__ import annotations

import collections
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from river.base import DriftDetector


class ShapeDD(DriftDetector):
    r"""Shape Drift Detection method for concept drift detection.

    ShapeDD is a multivariate concept drift detection method that identifies changes in the 
    data distribution by analyzing the shape of kernel-based statistics over a sliding window.
    It uses kernel methods to detect structural changes in the data and performs statistical
    tests to confirm drift occurrence.

    Parameters
    ----------
    window_size
        Size of the sliding window for storing recent data points.
    l1
        Half-width of the shape detection window.
    l2
        Half-width of the MMD test window around detected change points.
    n_perm
        Number of permutations for the statistical test.
    alpha
        Significance level for drift detection (p-value threshold).
    kernel
        Kernel function to use for similarity computation.

    Examples
    --------
    >>> import numpy as np
    >>> from river import drift

    >>> # Create ShapeDD detector
    >>> shapedd = drift.ShapeDD(window_size=100, l1=10, l2=15, alpha=0.05)

    >>> # Simulate a data stream with concept drift
    >>> np.random.seed(42)
    >>> data_stream = []
    >>> # First distribution
    >>> for i in range(150):
    ...     x = {"feature_" + str(j): np.random.normal(0, 1) for j in range(3)}
    ...     data_stream.append(x)
    >>> # Second distribution (shifted)
    >>> for i in range(150):
    ...     x = {"feature_" + str(j): np.random.normal(3, 1) for j in range(3)}
    ...     data_stream.append(x)

    >>> # Update drift detector
    >>> for i, x in enumerate(data_stream):
    ...     shapedd.update(x)
    ...     if shapedd.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         break
    Change detected at index 165

    References
    ----------
    [^1]: Based on kernel-based shape analysis for multivariate concept drift detection.

    """

    def __init__(
        self,
        window_size: int = 200,
        l1: int = 20,
        l2: int = 30,
        n_perm: int = 1000,
        alpha: float = 0.05,
        kernel: str = "rbf",
    ):
        super().__init__()
        self.window_size = window_size
        self.l1 = l1
        self.l2 = l2
        self.n_perm = n_perm
        self.alpha = alpha
        self.kernel = kernel
        self._mmd_cache = {}
        self._reset()

    def _reset(self):
        """Reset the detector's state."""
        super()._reset()
        self.window = collections.deque(maxlen=self.window_size)
        self._n_detections = 0
        self.change_points = []
        self.last_detection_index = -1

    def _gen_window_matrix(self, l1: int, l2: int, n_perm: int):
        """Generate window matrix for MMD test with caching."""
        cache_key = (l1, l2, n_perm)
        if cache_key not in self._mmd_cache:
            w = np.array(l1 * [1./l1] + l2 * [-1./l2])
            W = np.array([w] + [np.random.permutation(w) for _ in range(n_perm)])
            self._mmd_cache[cache_key] = W
        return self._mmd_cache[cache_key]

    def _mmd_test(self, X: np.ndarray, s: int = None, n_perm: int = 1000):
        """Perform Maximum Mean Discrepancy test."""
        try:
            K = apply_kernel(X, metric=self.kernel)
            if s is None:
                s = int(X.shape[0] / 2)
            
            # Ensure we have enough data for the test
            if s <= 0 or s >= K.shape[0]:
                return 0.0, 1.0
            
            W = self._gen_window_matrix(s, K.shape[0] - s, n_perm)
            test_stats = np.einsum('ij,ij->i', np.dot(W, K), W)
            p_value = (test_stats[0] < test_stats[1:]).sum() / n_perm
            
            return test_stats[0], p_value
        except Exception:
            return 0.0, 1.0

    def _convert_to_array(self, sample):
        """Convert dict sample to numpy array."""
        if isinstance(sample, dict):
            # Sort keys for consistent ordering
            keys = sorted(sample.keys())
            return np.array([sample[k] for k in keys])
        elif isinstance(sample, (list, tuple)):
            return np.array(sample)
        else:
            return np.array([sample])

    def _shape_analysis(self, X: np.ndarray):
        """Perform shape analysis to detect potential change points."""
        n = X.shape[0]
        
        # Need minimum samples for analysis
        if n < 2 * self.l1:
            return []
        
        try:
            # Create shape detection window
            w = np.array(self.l1 * [1.] + self.l1 * [-1.]) / float(self.l1)
            
            K = apply_kernel(X, metric=self.kernel)
            W = np.zeros((n - 2 * self.l1, n))
            
            for i in range(n - 2 * self.l1):
                W[i, i:i + 2 * self.l1] = w
            
            stat = np.einsum('ij,ij->i', np.dot(W, K), W)
            
            # Detect shape changes
            shape = np.convolve(stat, w, mode='full')
            shape_prime = shape[1:] * shape[:-1]
            
            # Find potential change points
            change_candidates = []
            for pos in np.where(shape_prime < 0)[0]:
                if pos < len(shape) and shape[pos] > 0:
                    # Perform MMD test around this position
                    a, b = max(0, pos - int(self.l2/2)), min(n, pos + int(self.l2/2))
                    if b - a > self.l1:  # Ensure we have enough data for MMD test
                        mmd_stat, p_value = self._mmd_test(X[a:b], pos - a, self.n_perm)
                        change_candidates.append({
                            'position': pos,
                            'shape_stat': shape[pos],
                            'mmd_stat': mmd_stat,
                            'p_value': p_value
                        })
            
            return change_candidates
        except Exception:
            return []

    def update(self, x):
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            Input sample. Can be a dict, list, tuple, or scalar.

        Returns
        -------
        self

        """
        # Reset detection flag first (but don't reset counters)
        was_detected = self.drift_detected
        if was_detected:
            # Keep track of detections before reset
            detections_before = self._n_detections
            self._reset()
            self._n_detections = detections_before  # Restore detection count

        # Convert input to numpy array
        x_array = self._convert_to_array(x)
        
        # Add to window
        self.window.append(x_array)
        
        # Perform drift analysis when window is full
        if len(self.window) >= self.window_size:
            X = np.array(list(self.window))
            change_candidates = self._shape_analysis(X)
            
            # Check for significant changes
            drift_detected = False
            for candidate in change_candidates:
                if candidate['p_value'] <= self.alpha:
                    drift_detected = True
                    self.change_points.append({
                        'global_position': len(self.window) - self.window_size + candidate['position'],
                        'local_position': candidate['position'],
                        'p_value': candidate['p_value'],
                        'mmd_stat': candidate['mmd_stat'],
                        'shape_stat': candidate['shape_stat']
                    })
                    break
            
            if drift_detected:
                self._drift_detected = True
                self._n_detections += 1
                self.last_detection_index = len(self.window) - 1
            else:
                self._drift_detected = False
        else:
            self._drift_detected = False

        return self

    @property
    def n_detections(self) -> int:
        """The total number of detected changes."""
        return self._n_detections

    @property
    def estimation(self) -> float:
        """P-value from the most recent detection."""
        if self.change_points:
            return self.change_points[-1]['p_value']
        return 1.0

    @property
    def most_recent_change_point(self) -> dict | None:
        """Information about the most recent change point detected."""
        if self.change_points:
            return self.change_points[-1]
        return None

    @classmethod
    def _unit_test_params(cls):
        """Parameters for unit testing."""
        yield {"window_size": 50, "l1": 5, "l2": 10, "n_perm": 100}
