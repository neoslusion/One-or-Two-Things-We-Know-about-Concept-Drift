"""
U-CDT_MSW: Unsupervised Concept Drift Type Identification 
           based on Multi-Sliding Windows with MMD and Shape-based Denoising

Integrates:
- Original ShapeDD (Hinder et al., IEEE SSCI 2021)
- CDT_MSW framework (Guo et al., Information Sciences 2022)

Author: [Your Name]
Date: 2024
"""

import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: KERNEL AND MMD UTILITIES
# =============================================================================

def apply_kernel(X, metric="rbf", gamma=None):
    """
    Compute kernel matrix with median heuristic for gamma.
    
    Args:
        X: np.array of shape (n_samples, n_features)
        metric: kernel type ('rbf', 'linear', etc.)
        gamma: kernel bandwidth (None for median heuristic)
    
    Returns:
        K: kernel matrix of shape (n_samples, n_samples)
    """
    if gamma is None and metric == "rbf":
        # Median heuristic
        dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist + 1e-10)
    
    return pairwise_kernels(X, metric=metric, gamma=gamma)


def mmd(X, split, n_perm=1000):
    """
    Compute MMD statistic and p-value using permutation test.
    
    Args:
        X: np.array of shape (n_samples, n_features)
        split: index to split X into two windows
        n_perm: number of permutations
    
    Returns:
        tuple: (mmd_statistic, p_value)
    """
    n = X.shape[0]
    if split <= 0 or split >= n:
        return 0.0, 1.0
    
    K = apply_kernel(X)
    
    # Compute MMD statistic
    def compute_mmd_from_kernel(K, n1, n2):
        """Compute biased MMD from kernel matrix."""
        K_XX = K[:n1, :n1]
        K_YY = K[n1:, n1:]
        K_XY = K[:n1, n1:]
        
        # Biased estimator
        mmd_sq = (np.sum(K_XX) / (n1 * n1) + 
                  np.sum(K_YY) / (n2 * n2) - 
                  2 * np.sum(K_XY) / (n1 * n2))
        return max(0, mmd_sq)
    
    n1, n2 = split, n - split
    observed_mmd = compute_mmd_from_kernel(K, n1, n2)
    
    # Permutation test
    count = 0
    indices = np.arange(n)
    for _ in range(n_perm):
        np.random.shuffle(indices)
        K_perm = K[np.ix_(indices, indices)]
        perm_mmd = compute_mmd_from_kernel(K_perm, n1, n2)
        if perm_mmd >= observed_mmd:
            count += 1
    
    p_value = (count + 1) / (n_perm + 1)
    return np.sqrt(observed_mmd), p_value


def compute_mmd_unbiased(X, Y, gamma=None):
    """
    Compute unbiased MMD between two samples.
    
    Args:
        X: np.array of shape (n, d) - samples from P
        Y: np.array of shape (m, d) - samples from Q
        gamma: kernel bandwidth (None for median heuristic)
    
    Returns:
        float: MMD value
    """
    if len(X) < 2 or len(Y) < 2:
        return 0.0
    
    n, m = len(X), len(Y)
    
    if gamma is None:
        XY = np.vstack([X, Y])
        dists = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=-1)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist + 1e-10)
    
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    
    # Unbiased estimator
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    term1 = np.sum(K_XX) / (n * (n - 1))
    term2 = np.sum(K_YY) / (m * (m - 1))
    term3 = np.sum(K_XY) / (n * m)
    
    mmd_squared = term1 + term2 - 2 * term3
    return np.sqrt(max(0, mmd_squared))


# =============================================================================
# PART 2: ORIGINAL SHAPEDD (FROM YOUR CODE)
# =============================================================================

def shape_detect(X, l1, l2, n_perm=500):
    """
    Original ShapeDD drift detection algorithm.
    
    Detects drift by finding "triangle peaks" in the MMD statistic curve.
    When a drift occurs, the MMD between sliding windows forms a characteristic
    triangular shape with a peak at the drift location.
    
    Parameters:

    X : array-like, shape (n_samples, n_features)
        Data stream
    l1 : int
        Half-window size for shape statistic computation
    l2 : int
        Window size for MMD statistical test
    n_perm : int
        Number of permutations for MMD p-value estimation
    
    Returns:

    res : array-like, shape (n_samples, 3)
        [:, 0] - Shape statistic value
        [:, 1] - MMD statistic  
        [:, 2] - p-value (< 0.05 indicates significant drift)
    """
    n = X.shape[0]
    
    if n <= 2 * l1:
        res = np.zeros((n, 3))
        res[:, 2] = 1
        return res
    
    # Weight vector for shape detection: [1/l1, ..., 1/l1, -1/l1, ..., -1/l1]
    w = np.array(l1 * [1.] + l1 * [-1.]) / float(l1)
    
    # Compute kernel matrix
    K = apply_kernel(X, metric="rbf")
    
    # Compute MMD statistics using sliding windows
    W = np.zeros((n - 2 * l1, n))
    for i in range(n - 2 * l1):
        W[i, i:i + 2 * l1] = w
    
    # stat[i] = MMD^2 between windows [i, i+l1) and [i+l1, i+2*l1)
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)
    
    # Convolve with weight vector to find shape pattern
    shape_stat = np.convolve(stat, w, mode='full')
    
    # Find zero crossings (sign changes)
    shape_prime = shape_stat[1:] * shape_stat[:-1]
    
    # Initialize results
    res = np.zeros((n, 3))
    res[:, 2] = 1  # Default p-value = 1 (no drift)
    
    # Check each zero crossing
    for pos in np.where(shape_prime < 0)[0]:
        if pos < n and shape_stat[pos] > 0:
            res[pos, 0] = shape_stat[pos]
            # Validate with MMD test
            a, b = max(0, pos - int(l2 / 2)), min(n, pos + int(l2 / 2))
            if b - a > 4:  # Need enough samples
                res[pos, 1:] = mmd(X[a:b], pos - a, n_perm)
    
    return res


class ShapeDD:
    """
    Shape-based Drift Detector wrapper class.
    
    Provides a clean interface to the original ShapeDD algorithm.
    """
    
    def __init__(self, l1=50, l2=100, n_perm=500, p_threshold=0.05):
        """
        Args:
            l1: Half-window size for shape statistic
            l2: Window size for MMD test validation
            n_perm: Number of permutations for p-value
            p_threshold: Significance level for drift detection
        """
        self.l1 = l1
        self.l2 = l2
        self.n_perm = n_perm
        self.p_threshold = p_threshold
    
    def detect(self, X):
        """
        Detect drift in data stream X.
        
        Args:
            X: np.array of shape (n_samples, n_features)
        
        Returns:
            dict: Detection results
        """
        res = shape_detect(X, self.l1, self.l2, self.n_perm)
        
        # Find significant drift points
        drift_mask = res[:, 2] < self.p_threshold
        drift_positions = np.where(drift_mask)[0]
        
        if len(drift_positions) == 0:
            return {
                'detected': False,
                'positions': [],
                'strengths': [],
                'p_values': [],
                'raw_results': res
            }
        
        # Get the most significant drift
        best_idx = drift_positions[np.argmin(res[drift_positions, 2])]
        
        return {
            'detected': True,
            'position': best_idx,
            'positions': drift_positions.tolist(),
            'strength': res[best_idx, 0],
            'strengths': res[drift_positions, 0].tolist(),
            'p_value': res[best_idx, 2],
            'p_values': res[drift_positions, 2].tolist(),
            'mmd_stat': res[best_idx, 1],
            'raw_results': res
        }


# =============================================================================
# PART 3: U-CDT_MSW - MAIN ALGORITHM
# =============================================================================

class U_CDT_MSW:
    """
    Unsupervised Concept Drift Type Identification based on Multi-Sliding Windows.
    
    Combines CDT_MSW's drift type identification framework with ShapeDD's 
    unsupervised MMD-based detection.
    
    Three phases:
    1. Detection: Find drift position using ShapeDD
    2. Growth: Determine drift length and category (TCD/PCD) using MMD stability
    3. Tracking: Identify subcategory using MMD Evolution Curve (MEC)
    """
    
    def __init__(self,
                 # ShapeDD parameters
                 l1=50,
                 l2=100,
                 n_perm=500,
                 p_threshold=0.05,
                 # Growth process parameters
                 stability_threshold=0.01,
                 n_adjoint_windows=4,
                 max_drift_length=20,
                 # Tracking parameters
                 tracking_steps=15,
                 block_size=50):
        """
        Args:
            l1: ShapeDD half-window size
            l2: ShapeDD validation window size
            n_perm: Permutations for p-value
            p_threshold: Significance level
            stability_threshold: δ_MMD for stability detection
            n_adjoint_windows: n in adjoint window W_R
            max_drift_length: Maximum drift length to search
            tracking_steps: k steps for tracking process
            block_size: Samples per block for Growth/Tracking
        """
        # ShapeDD
        self.shape_detector = ShapeDD(l1, l2, n_perm, p_threshold)
        self.l1 = l1
        
        # Growth
        self.stability_threshold = stability_threshold
        self.n = n_adjoint_windows
        self.max_drift_length = max_drift_length
        
        # Tracking
        self.k = tracking_steps
        self.block_size = block_size
    
    def _create_blocks(self, X):
        """Split data stream into blocks."""
        n_samples = len(X)
        n_blocks = n_samples // self.block_size
        blocks = []
        for i in range(n_blocks):
            start = i * self.block_size
            end = start + self.block_size
            blocks.append(X[start:end])
        return blocks
    
    def _sample_to_block_index(self, sample_idx):
        """Convert sample index to block index."""
        return sample_idx // self.block_size
    
    # =========================================================================
    # PHASE 1: DETECTION (using ShapeDD)
    # =========================================================================
    
    def _detection_process(self, X):
        """
        Phase 1: Detect drift position using ShapeDD.
        
        Returns sample-level position, which will be converted to block index.
        """
        result = self.shape_detector.detect(X)
        
        if not result['detected']:
            return {
                'drift_detected': False,
                'drift_position_sample': None,
                'drift_position_block': None,
                'drift_strength': 0,
                'p_value': 1.0,
                'all_candidates': []
            }
        
        return {
            'drift_detected': True,
            'drift_position_sample': result['position'],
            'drift_position_block': self._sample_to_block_index(result['position']),
            'drift_strength': result['strength'],
            'p_value': result['p_value'],
            'mmd_stat': result['mmd_stat'],
            'all_candidates': list(zip(result['positions'], 
                                       result['p_values'],
                                       result['strengths']))
        }
    
    # =========================================================================
    # PHASE 2: GROWTH PROCESS
    # =========================================================================
    
    def _growth_process(self, blocks, drift_block_idx):
        """
        Phase 2: Determine drift length and category using MMD stability.
        
        Replaces CDT_MSW's accuracy-based variance:
            Original: σ²_R = Var(accuracies in W_R) ≤ δ
            Proposed: σ²_MMD = Var(consecutive MMDs in W_R) ≤ δ_MMD
        
        Logic:
        - If distribution stabilizes immediately (m=1): TCD
        - If distribution takes time to stabilize (m>1): PCD
        
        Args:
            blocks: List of data blocks
            drift_block_idx: Block index where drift was detected
        
        Returns:
            dict: Growth results with drift_length and category
        """
        n_blocks = len(blocks)
        t = drift_block_idx
        
        # Check bounds
        if t + 1 + self.n >= n_blocks:
            return {
                'drift_length': 1,
                'category': 'TCD',
                'mmd_variances': [],
                'mmd_means': []
            }
        
        def compute_window_stability(start_pos):
            """
            Compute stability of distribution in adjoint window.
            
            Stability is measured by variance of consecutive MMDs.
            Low variance = stable distribution.
            """
            mmds = []
            for j in range(self.n - 1):
                idx1 = start_pos + j
                idx2 = start_pos + j + 1
                if idx2 >= n_blocks:
                    break
                mmd_val = compute_mmd_unbiased(blocks[idx1], blocks[idx2])
                mmds.append(mmd_val)
            
            if len(mmds) < 2:
                return 0.0, 0.0
            
            return np.var(mmds), np.mean(mmds)
        
        mmd_variances = []
        mmd_means = []
        
        # Start with W_R at position t+1
        m = 1
        var_R, mean_R = compute_window_stability(t + 1)
        mmd_variances.append(var_R)
        mmd_means.append(mean_R)
        
        # Check if immediately stable (TCD, m=1)
        if var_R <= self.stability_threshold:
            return {
                'drift_length': 1,
                'category': 'TCD',
                'mmd_variances': mmd_variances,
                'mmd_means': mmd_means
            }
        
        # Slide W_R forward until stable (PCD)
        max_m = min(self.max_drift_length, n_blocks - t - self.n)
        
        while var_R > self.stability_threshold and m < max_m:
            m += 1
            var_R, mean_R = compute_window_stability(t + m)
            mmd_variances.append(var_R)
            mmd_means.append(mean_R)
        
        category = 'TCD' if m == 1 else 'PCD'
        
        return {
            'drift_length': m,
            'category': category,
            'mmd_variances': mmd_variances,
            'mmd_means': mmd_means
        }
    
    # =========================================================================
    # PHASE 3: TRACKING PROCESS - MMD EVOLUTION CURVE (MEC)
    # =========================================================================
    
    def _tracking_process(self, blocks, drift_block_idx, drift_length):
        """
        Phase 3: Generate MMD Evolution Curve (MEC) and classify subcategory.
        
        Replaces CDT_MSW's TFR curve:
            Original: TFR(i) = α_B^i / α_A^i (accuracy ratio)
            Proposed: MEC(i) = MMD(W'_A^i, W'_B) (distribution distance)
        
        Key insight from CDT_MSW paper:
        - W'_B is STATIC at drift position
        - W'_A SLIDES toward W'_B
        - The curve pattern reveals drift subcategory
        
        Args:
            blocks: List of data blocks
            drift_block_idx: Block index where drift was detected
            drift_length: Detected drift length (m)
        
        Returns:
            dict: Tracking results with MEC curve and subcategory
        """
        n_blocks = len(blocks)
        t = drift_block_idx
        m = max(1, drift_length)
        
        # W'_B: STATIC composite window at drift position
        # Contains m blocks starting from drift position
        end_B = min(t + m, n_blocks)
        if end_B <= t or t >= n_blocks:
            return {
                'mec_curve': np.array([]),
                'subcategory': 'Unknown',
                'mec_features': {}
            }
        
        W_B = np.vstack(blocks[t:end_B])
        
        # Generate MEC: W'_A slides from before drift toward W'_B
        # Start from far before drift position
        mec_curve = []
        positions = []
        
        start_tracking = max(0, t - self.k)
        
        for i in range(self.k):
            pos_A = start_tracking + i
            end_A = pos_A + m
            
            # Stop if W'_A overlaps with or passes W'_B
            if pos_A >= t or end_A > n_blocks:
                break
            
            W_A = np.vstack(blocks[pos_A:end_A])
            mmd_value = compute_mmd_unbiased(W_A, W_B)
            mec_curve.append(mmd_value)
            positions.append(pos_A)
        
        # Continue tracking past the drift point
        for i in range(self.k):
            pos_A = t + i + 1
            end_A = pos_A + m
            
            if end_A > n_blocks:
                break
            
            W_A = np.vstack(blocks[pos_A:end_A])
            mmd_value = compute_mmd_unbiased(W_A, W_B)
            mec_curve.append(mmd_value)
            positions.append(pos_A)
        
        mec_curve = np.array(mec_curve)
        
        # Extract features and classify
        mec_features = self._extract_mec_features(mec_curve)
        category = 'TCD' if drift_length == 1 else 'PCD'
        subcategory = self._classify_subcategory(mec_curve, mec_features, category)
        
        return {
            'mec_curve': mec_curve,
            'mec_positions': positions,
            'subcategory': subcategory,
            'mec_features': mec_features
        }
    
    def _extract_mec_features(self, mec_curve):
        """
        Extract features from MMD Evolution Curve.
        
        Features designed to distinguish:
        - TCD: Sudden, Blip, Recurrent
        - PCD: Incremental, Gradual
        """
        if len(mec_curve) < 4:
            return {
                'n_peaks': 0,
                'peak_prominence': 0,
                'smoothness': 0,
                'oscillation': 0,
                'decay_rate': 0,
                'tail_ratio': 0,
                'max_value': 0,
                'mean_value': 0
            }
        
        features = {}
        n = len(mec_curve)
        
        # Normalize
        max_val = np.max(mec_curve)
        mec_norm = mec_curve / (max_val + 1e-10)
        
        # 1. Peak analysis
        peaks, properties = find_peaks(mec_norm, height=0.1, prominence=0.1)
        features['n_peaks'] = len(peaks)
        features['peak_prominence'] = np.mean(properties.get('prominences', [0])) if len(peaks) > 0 else 0
        
        # 2. Smoothness (inverse of second derivative variance)
        if n > 2:
            second_deriv = np.diff(np.diff(mec_norm))
            features['smoothness'] = 1.0 / (1.0 + np.var(second_deriv) * 100)
        else:
            features['smoothness'] = 1.0
        
        # 3. Oscillation (sign changes in first derivative)
        if n > 1:
            first_deriv = np.diff(mec_norm)
            sign_changes = np.sum(np.abs(np.diff(np.sign(first_deriv))) > 0)
            features['oscillation'] = sign_changes / n
        else:
            features['oscillation'] = 0
        
        # 4. Decay rate
        if mec_curve[0] > mec_curve[-1] and mec_curve[0] > 0:
            ratio = mec_curve[-1] / (mec_curve[0] + 1e-10)
            features['decay_rate'] = -np.log(max(ratio, 1e-10)) / n
        else:
            features['decay_rate'] = 0
        
        # 5. Tail behavior (last 30%)
        tail_start = int(0.7 * n)
        tail = mec_norm[tail_start:]
        head = mec_norm[:int(0.3 * n)]
        features['tail_ratio'] = np.mean(tail) / (np.mean(head) + 1e-10) if len(head) > 0 else 0
        
        # 6. Global statistics
        features['max_value'] = max_val
        features['mean_value'] = np.mean(mec_curve)
        
        return features
    
    def _classify_subcategory(self, mec_curve, features, category):
        """
        Classify drift subcategory based on MEC features.
        
        TCD Subcategories (drift_length = 1):
        - Sudden: MEC shows single peak, then drops to near zero
        - Blip: MEC shows peak, drops, then rises again
        - Recurrent: MEC shows multiple oscillating peaks
        
        PCD Subcategories (drift_length > 1):
        - Incremental: MEC shows smooth, monotonic decay
        - Gradual: MEC shows oscillating decay
        """
        if len(mec_curve) < 4:
            return 'Unknown'
        
        n_peaks = features['n_peaks']
        oscillation = features['oscillation']
        smoothness = features['smoothness']
        tail_ratio = features['tail_ratio']
        
        if category == 'TCD':
            # === TCD Subcategories ===
            
            # Recurrent: Multiple peaks with high oscillation
            if n_peaks >= 3 or oscillation > 0.4:
                return 'Recurrent'
            
            # Blip: Peak then rise in tail (tail_ratio > threshold)
            if tail_ratio > 0.5 and n_peaks <= 2:
                return 'Blip'
            
            # Sudden: Single peak, low tail
            if n_peaks <= 1 and tail_ratio < 0.3:
                return 'Sudden'
            
            # Default for TCD
            if tail_ratio > 0.4:
                return 'Blip'
            else:
                return 'Sudden'
        
        else:  # PCD
            # === PCD Subcategories ===
            
            # Incremental: Smooth transition (high smoothness, low oscillation)
            if smoothness > 0.5 and oscillation < 0.2:
                return 'Incremental'
            
            # Gradual: Oscillating transition
            if oscillation > 0.2 or smoothness < 0.5:
                return 'Gradual'
            
            # Default for PCD
            return 'Incremental'
    
    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================
    
    def fit(self, X):
        """
        Run full U-CDT_MSW pipeline on data stream.
        
        Args:
            X: np.array of shape (n_samples, n_features) - data stream
        
        Returns:
            dict: Complete analysis results
        """
        n_samples = len(X)
        
        if n_samples < 4 * self.l1:
            return {
                'error': 'Not enough samples for detection',
                'n_samples': n_samples,
                'min_required': 4 * self.l1
            }
        
        # Create blocks for Growth and Tracking
        blocks = self._create_blocks(X)
        n_blocks = len(blocks)
        
        if n_blocks < 10:
            return {
                'error': 'Not enough blocks',
                'n_blocks': n_blocks
            }
        
        # =====================================================================
        # PHASE 1: DETECTION
        # =====================================================================
        detection_result = self._detection_process(X)
        
        if not detection_result['drift_detected']:
            return {
                'drift_detected': False,
                'detection': detection_result,
                'growth': None,
                'tracking': None
            }
        
        drift_block_idx = detection_result['drift_position_block']
        
        # =====================================================================
        # PHASE 2: GROWTH
        # =====================================================================
        growth_result = self._growth_process(blocks, drift_block_idx)
        
        # =====================================================================
        # PHASE 3: TRACKING
        # =====================================================================
        tracking_result = self._tracking_process(
            blocks,
            drift_block_idx,
            growth_result['drift_length']
        )
        
        return {
            'drift_detected': True,
            'drift_position_sample': detection_result['drift_position_sample'],
            'drift_position_block': drift_block_idx,
            'drift_length': growth_result['drift_length'],
            'category': growth_result['category'],
            'subcategory': tracking_result['subcategory'],
            'p_value': detection_result['p_value'],
            'drift_strength': detection_result['drift_strength'],
            'detection': detection_result,
            'growth': growth_result,
            'tracking': tracking_result
        }
    
    def fit_streaming(self, X, window_size=500, step_size=100):
        """
        Run U-CDT_MSW in streaming mode with sliding window.
        
        Args:
            X: Full data stream
            window_size: Size of analysis window
            step_size: Step size for sliding
        
        Returns:
            list: Results for each window
        """
        results = []
        n_samples = len(X)
        
        for start in range(0, n_samples - window_size, step_size):
            end = start + window_size
            window_data = X[start:end]
            
            result = self.fit(window_data)
            
            if result.get('drift_detected', False):
                # Adjust position to global coordinates
                result['global_position'] = start + result['drift_position_sample']
            
            result['window_start'] = start
            result['window_end'] = end
            results.append(result)
        
        return results


# =============================================================================
# PART 4: DATA GENERATORS
# =============================================================================

class DriftDataGenerator:
    """Generate synthetic data streams with various drift types."""
    
    @staticmethod
    def generate_sudden_drift(n_samples=2000, n_features=10, drift_point=1000, 
                              shift=2.0, seed=42):
        """Generate data with sudden (abrupt) drift."""
        np.random.seed(seed)
        X_before = np.random.randn(drift_point, n_features)
        X_after = np.random.randn(n_samples - drift_point, n_features) + shift
        X = np.vstack([X_before, X_after])
        return X, {'type': 'Sudden', 'position': drift_point, 'shift': shift}
    
    @staticmethod
    def generate_gradual_drift(n_samples=2000, n_features=10,
                               drift_start=800, drift_end=1200, 
                               shift=2.0, seed=42):
        """Generate data with gradual drift (oscillating transition)."""
        np.random.seed(seed)
        X = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            if i < drift_start:
                X[i] = np.random.randn(n_features)
            elif i >= drift_end:
                X[i] = np.random.randn(n_features) + shift
            else:
                # Oscillating: randomly sample from old or new distribution
                p = (i - drift_start) / (drift_end - drift_start)
                if np.random.rand() < p:
                    X[i] = np.random.randn(n_features) + shift
                else:
                    X[i] = np.random.randn(n_features)
        
        return X, {'type': 'Gradual', 'start': drift_start, 'end': drift_end}
    
    @staticmethod
    def generate_incremental_drift(n_samples=2000, n_features=10,
                                   drift_start=800, drift_end=1200,
                                   shift=2.0, seed=42):
        """Generate data with incremental drift (smooth transition)."""
        np.random.seed(seed)
        X = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            if i < drift_start:
                mean = 0
            elif i >= drift_end:
                mean = shift
            else:
                # Linear interpolation
                mean = shift * (i - drift_start) / (drift_end - drift_start)
            X[i] = np.random.randn(n_features) + mean
        
        return X, {'type': 'Incremental', 'start': drift_start, 'end': drift_end}
    
    @staticmethod
    def generate_recurrent_drift(n_samples=2000, n_features=10, 
                                 period=500, shift=2.0, seed=42):
        """Generate data with recurrent drift."""
        np.random.seed(seed)
        X = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            if (i // period) % 2 == 0:
                X[i] = np.random.randn(n_features)
            else:
                X[i] = np.random.randn(n_features) + shift
        
        return X, {'type': 'Recurrent', 'period': period}
    
    @staticmethod
    def generate_blip_drift(n_samples=2000, n_features=10,
                            blip_start=900, blip_end=1100,
                            shift=2.0, seed=42):
        """Generate data with blip (temporary) drift."""
        np.random.seed(seed)
        X = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            if blip_start <= i < blip_end:
                X[i] = np.random.randn(n_features) + shift
            else:
                X[i] = np.random.randn(n_features)
        
        return X, {'type': 'Blip', 'start': blip_start, 'end': blip_end}


# =============================================================================
# PART 5: EXPERIMENTS
# =============================================================================

def run_experiments():
    """Run experiments on synthetic datasets."""
    
    print("=" * 70)
    print("U-CDT_MSW: Unsupervised Concept Drift Type Identification")
    print("with Original ShapeDD Integration")
    print("=" * 70)
    
    # Initialize detector with parameters tuned for synthetic data
    detector = U_CDT_MSW(
        # ShapeDD parameters
        l1=50,              # Half-window for shape detection
        l2=100,             # Validation window
        n_perm=200,         # Permutations (reduce for speed)
        p_threshold=0.05,   # Significance level
        # Growth parameters
        stability_threshold=0.005,
        n_adjoint_windows=4,
        max_drift_length=15,
        # Tracking parameters
        tracking_steps=20,
        block_size=50
    )
    
    # Test configurations
    experiments = [
        ('Sudden', DriftDataGenerator.generate_sudden_drift, 
         {'n_samples': 2000, 'drift_point': 1000, 'shift': 2.5}),
        ('Gradual', DriftDataGenerator.generate_gradual_drift,
         {'n_samples': 2000, 'drift_start': 800, 'drift_end': 1200, 'shift': 2.5}),
        ('Incremental', DriftDataGenerator.generate_incremental_drift,
         {'n_samples': 2000, 'drift_start': 800, 'drift_end': 1200, 'shift': 2.5}),
        ('Recurrent', DriftDataGenerator.generate_recurrent_drift,
         {'n_samples': 2000, 'period': 500, 'shift': 2.5}),
        ('Blip', DriftDataGenerator.generate_blip_drift,
         {'n_samples': 2000, 'blip_start': 900, 'blip_end': 1100, 'shift': 2.5}),
    ]
    
    results = []
    
    for name, generator, params in experiments:
        print(f"\n{'='*70}")
        print(f"Testing: {name} Drift")
        print('='*70)
        
        # Generate data
        X, ground_truth = generator(n_features=10, **params)
        print(f"Ground truth: {ground_truth}")
        
        # Run detection
        result = detector.fit(X)
        
        if result.get('drift_detected', False):
            print(f"\n✓ Drift Detected!")
            print(f"  - Position (sample): {result['drift_position_sample']}")
            print(f"  - Position (block):  {result['drift_position_block']}")
            print(f"  - p-value:           {result['p_value']:.4f}")
            print(f"  - Drift Strength:    {result['drift_strength']:.4f}")
            print(f"  - Drift Length:      {result['drift_length']}")
            print(f"  - Category:          {result['category']}")
            print(f"  - Subcategory:       {result['subcategory']}")
            
            # Show MEC features
            if result['tracking']:
                features = result['tracking']['mec_features']
                print(f"\n  MEC Features:")
                print(f"    - Peaks:       {features.get('n_peaks', 0)}")
                print(f"    - Oscillation: {features.get('oscillation', 0):.4f}")
                print(f"    - Smoothness:  {features.get('smoothness', 0):.4f}")
                print(f"    - Tail Ratio:  {features.get('tail_ratio', 0):.4f}")
        else:
            print(f"\n✗ No Drift Detected")
            if 'error' in result:
                print(f"  Error: {result['error']}")
        
        results.append({
            'experiment': name,
            'ground_truth': ground_truth,
            'result': result
        })
    
    return results


def evaluate_accuracy(results):
    """Evaluate detection and classification accuracy."""
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    # Category mapping
    ground_truth_category = {
        'Sudden': 'TCD',
        'Blip': 'TCD',
        'Recurrent': 'TCD',
        'Incremental': 'PCD',
        'Gradual': 'PCD'
    }
    
    detection_correct = 0
    category_correct = 0
    subcategory_correct = 0
    total = len(results)
    
    print(f"\n{'Experiment':<15} {'Detected':<10} {'Category':<12} {'Subcategory':<15}")
    print("-" * 52)
    
    for r in results:
        exp_name = r['experiment']
        result = r['result']
        
        det_ok = '✓' if result.get('drift_detected', False) else '✗'
        cat_ok = '-'
        sub_ok = '-'
        
        if result.get('drift_detected', False):
            detection_correct += 1
            
            expected_cat = ground_truth_category[exp_name]
            actual_cat = result.get('category', '')
            cat_ok = '✓' if actual_cat == expected_cat else '✗'
            if actual_cat == expected_cat:
                category_correct += 1
            
            actual_sub = result.get('subcategory', '')
            sub_ok = '✓' if actual_sub == exp_name else '✗'
            if actual_sub == exp_name:
                subcategory_correct += 1
        
        print(f"{exp_name:<15} {det_ok:<10} {cat_ok:<12} {sub_ok:<15}")
    
    print("-" * 52)
    print(f"\nDetection Accuracy:    {detection_correct}/{total} "
          f"({100*detection_correct/total:.1f}%)")
    print(f"Category Accuracy:     {category_correct}/{total} "
          f"({100*category_correct/total:.1f}%)")
    print(f"Subcategory Accuracy:  {subcategory_correct}/{total} "
          f"({100*subcategory_correct/total:.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run experiments
    results = run_experiments()
    
    # Evaluate
    evaluate_accuracy(results)
    
    print("\n" + "=" * 70)
    print("Experiments completed!")
    print("=" * 70)
