from __future__ import annotations

import collections
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from river.base import DriftDetector


class DAWIDD(DriftDetector):
    r"""Drift Detection using Adapted Windowing and Independence Distance (DAWIDD).

    DAWIDD is a multivariate concept drift detection method that uses kernel-based statistical tests
    to detect changes in the data distribution. It maintains a sliding window of recent data and
    uses kernel methods to measure the distance between temporal patterns in the data.

    Parameters
    ----------
    window_size
        Size of the sliding window for storing recent data points.
    t_kernel
        Kernel function for temporal patterns. Options include 'rbf', 'linear', 'polynomial', 'min'.
    x_kernel
        Kernel function for feature space. Options include 'rbf', 'linear', 'polynomial'.
    n_perm
        Number of permutations for the statistical test.
    alpha
        Significance level for drift detection (p-value threshold).

    Examples
    --------
    >>> import numpy as np
    >>> from river import drift

    >>> # Create DAWIDD detector
    >>> dawidd = drift.DAWIDD(window_size=50, alpha=0.05)

    >>> # Simulate a data stream with concept drift
    >>> np.random.seed(42)
    >>> data_stream = []
    >>> # First distribution
    >>> for i in range(100):
    ...     x = {"feature_" + str(j): np.random.normal(0, 1) for j in range(3)}
    ...     data_stream.append(x)
    >>> # Second distribution (shifted)
    >>> for i in range(100):
    ...     x = {"feature_" + str(j): np.random.normal(2, 1) for j in range(3)}
    ...     data_stream.append(x)

    >>> # Update drift detector
    >>> for i, x in enumerate(data_stream):
    ...     dawidd.update(x)
    ...     if dawidd.drift_detected:
    ...         print(f"Change detected at index {i}")
    ...         break
    Change detected at index 120

    References
    ----------
    [^1]: Based on kernel-based statistical tests for multivariate concept drift detection.

    """

    def __init__(
        self,
        window_size: int = 100,
        t_kernel: str = "rbf",
        x_kernel: str = "rbf", 
        n_perm: int = 1000,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.window_size = window_size
        self.t_kernel = t_kernel
        self.x_kernel = x_kernel
        self.n_perm = n_perm
        self.alpha = alpha
        self._kernel_cache = {}
        self._reset()

    def _reset(self):
        """Reset the detector's state."""
        super()._reset()
        self.window = collections.deque(maxlen=self.window_size)
        self._n_detections = 0
        self.p_value = 1.0
        self.test_statistic = 0.0

    def _get_time_kernel(self, n_size: int, n_perm: int, kernel: str):
        """Compute time kernel matrix with permutations."""
        cache_key = (n_size, n_perm, kernel)
        if cache_key not in self._kernel_cache:
            T = np.linspace(-1, 1, n_size).reshape(-1, 1)
            H = np.eye(n_size) - (1/n_size) * np.ones((n_size, n_size))
            
            if kernel == "min":
                K = [H @ np.minimum(T[None, :], T[:, None])[:, :, 0] @ H]
                for _ in range(n_perm):
                    T_perm = np.random.permutation(T)
                    K.append(H @ np.minimum(T_perm[None, :], T_perm[:, None])[:, :, 0] @ H)
            else:
                K = [H @ apply_kernel(T, metric=kernel) @ H]
                for _ in range(n_perm):
                    T_perm = np.random.permutation(T)
                    K.append(H @ apply_kernel(T_perm, metric=kernel) @ H)
            
            K = np.array(K)
            K = K.reshape(n_perm + 1, n_size * n_size)
            self._kernel_cache[cache_key] = K
        
        return self._kernel_cache[cache_key]

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

    def _dawidd_test(self, X: np.ndarray):
        """Perform DAWIDD statistical test."""
        n_size = X.shape[0]
        
        if n_size < 3:  # Need minimum samples for meaningful test
            return 0.0, 1.0
        
        try:
            # Compute feature kernel
            K_X = apply_kernel(X, metric=self.x_kernel)
            
            # Get time kernel with permutations
            K_time = self._get_time_kernel(n_size, self.n_perm, self.t_kernel)
            
            # Compute test statistics
            s = K_time @ K_X.ravel()
            
            # P-value calculation
            p_value = (s[0] < s[1:]).sum() / self.n_perm
            
            # Test statistic (normalized)
            test_stat = (1/n_size)**2 * s[0]
            
            return test_stat, p_value
            
        except Exception:
            # Return safe values if computation fails
            return 0.0, 1.0

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
        
        # Perform drift test when window is full
        if len(self.window) >= self.window_size:
            X = np.array(list(self.window))
            self.test_statistic, self.p_value = self._dawidd_test(X)
            
            # Detect drift if p-value is below threshold
            if self.p_value <= self.alpha:
                self._drift_detected = True
                self._n_detections += 1
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
        """Current p-value from the statistical test."""
        return self.p_value

    @classmethod
    def _unit_test_params(cls):
        """Parameters for unit testing."""
        yield {"window_size": 20, "n_perm": 100}

