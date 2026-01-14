"""
ShapeDD_OW_CDT: ShapeDD with Optimally-Weighted MMD for Drift Type Classification
===================================================================================

Combines:
- OW-MMD (Bharti et al., ICML 2023) for faster, more accurate drift detection
- CDT classification logic from ShapedCDT_V5

Key improvements over standard ShapeDD+CDT:
- Variance-optimal weighting reduces noise
- Better bandwidth selection via median heuristic
- More robust pattern detection
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Tuple, List, Optional
import sys
from pathlib import Path

# Add backup to path for ow_mmd
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent
backup_path = project_root / "experiments" / "backup"
if str(backup_path) not in sys.path:
    sys.path.insert(0, str(backup_path))

# Import OW-MMD functions
try:
    from ow_mmd import compute_ow_mmd, compute_ow_mmd_squared, shapedd_ow_mmd
    HAS_OW_MMD = True
except ImportError:
    HAS_OW_MMD = False
    print("[WARNING] ow_mmd not available, using fallback standard MMD")


@dataclass
class DriftResult:
    detected: bool
    positions: List[int]
    subcategory: Optional[str]
    category: Optional[str]
    confidence: float
    mmd_signal: np.ndarray
    times: np.ndarray
    debug_info: dict = None


class ShapeDD_OW_CDT:
    """
    ShapeDD with Optimally-Weighted MMD for Drift Type Classification.
    
    Uses OW-MMD (Bharti et al., 2023) which:
    - Has lower variance than standard MMD
    - Is more robust to outliers
    - Provides better detection performance
    
    Note: OW-MMD values are typically smaller than standard MMD,
    so thresholds are adjusted accordingly.
    
    Parameters
    ----------
    window_size : int
        Size of sliding window for MMD computation
    stride : int
        Step size for sliding window
    smoothing_sigma : float
        Gaussian smoothing parameter for signal
    use_ow : bool
        Whether to use OW-MMD (default) or fallback to standard MMD
    ow_threshold_factor : float
        Adjustment factor for OW-MMD thresholds (default 0.3 vs 1.0 for standard)
    """
    
    def __init__(self, 
                 window_size: int = 200,
                 stride: int = 50,
                 smoothing_sigma: float = 1.5,
                 use_ow: bool = True,
                 ow_threshold_factor: float = 0.3):
        self.window_size = window_size
        self.stride = stride
        self.sigma = smoothing_sigma
        self.use_ow = use_ow and HAS_OW_MMD
        self.ow_threshold_factor = ow_threshold_factor
        
    def _compute_mmd_ow(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute OW-MMD between two windows."""
        if len(X) < 2 or len(Y) < 2:
            return 0.0
        
        if self.use_ow:
            try:
                return compute_ow_mmd(X, Y, gamma='auto', weight_method='variance_reduction')
            except Exception:
                pass
        
        # Fallback to standard MMD
        return self._compute_mmd_standard(X, Y)
    
    def _compute_mmd_standard(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Fallback standard MMD computation."""
        if len(X) < 2 or len(Y) < 2:
            return 0.0
        
        combined = np.vstack([X, Y])
        dists = pdist(combined, 'sqeuclidean')
        if len(dists) == 0 or np.median(dists) == 0:
            return 0.0
        
        gamma = 1.0 / (2 * np.median(dists) + 1e-10)
        m, n = len(X), len(Y)
        
        K_XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
        K_YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
        K_XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
        
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)
        
        mmd_sq = (np.sum(K_XX) / (m * (m - 1)) + 
                  np.sum(K_YY) / (n * (n - 1)) - 
                  2 * np.mean(K_XY))
        
        return np.sqrt(max(0, mmd_sq))

    
    def _compute_drift_magnitude(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute drift magnitude signal using OW-MMD."""
        n = len(X)
        l = self.window_size
        times, sigma = [], []
        
        for t in range(2 * l, n, self.stride):
            mmd_val = self._compute_mmd_ow(X[t - 2*l : t - l], X[t - l : t])
            sigma.append(mmd_val)
            times.append(t - l)
        
        return np.array(times), np.array(sigma)
    
    def _detect_peaks(self, sigma: np.ndarray) -> np.ndarray:
        """Detect peaks in MMD signal."""
        if len(sigma) < 3:
            return np.array([])
        
        mean_s, std_s = np.mean(sigma), np.std(sigma)
        
        # OW-MMD produces lower values, use lower threshold
        if self.use_ow:
            threshold = mean_s + 0.1 * std_s  # Lower threshold for OW-MMD
            min_prominence = std_s * 0.05
        else:
            threshold = mean_s + 0.3 * std_s
            min_prominence = std_s * 0.2
        
        peaks, _ = find_peaks(
            sigma,
            height=threshold,
            distance=max(2, len(sigma) // 25),
            prominence=min_prominence
        )
        return peaks
    
    def _analyze_peak_shape(self, sigma: np.ndarray, peak_idx: int) -> dict:
        """Analyze shape of a peak."""
        n = len(sigma)
        left_bound = max(0, peak_idx - 10)
        right_bound = min(n, peak_idx + 10)
        
        left_segment = sigma[left_bound:peak_idx]
        right_segment = sigma[peak_idx:right_bound]
        peak_val = sigma[peak_idx]
        
        if len(left_segment) > 1:
            rise_rate = (peak_val - np.min(left_segment)) / len(left_segment)
        else:
            rise_rate = 0
            
        if len(right_segment) > 1:
            fall_rate = (peak_val - np.min(right_segment)) / len(right_segment)
        else:
            fall_rate = 0
        
        symmetry = 1 - abs(rise_rate - fall_rate) / (max(rise_rate, fall_rate) + 1e-10)
        
        if len(left_segment) >= 2 and len(right_segment) >= 2:
            left_drop = peak_val - sigma[left_bound]
            right_drop = peak_val - sigma[min(right_bound-1, n-1)]
            sharpness = (left_drop + right_drop) / (2 * peak_val + 1e-10)
        else:
            sharpness = 0.5
        
        return {
            'symmetry': symmetry,
            'sharpness': sharpness,
            'peak_value': peak_val
        }
    
    def _has_blip_pattern(self, sigma: np.ndarray, peaks: np.ndarray) -> bool:
        """Check for blip pattern (temporary change with return)."""
        if len(peaks) == 0 or len(sigma) < 10:
            return False
        
        n = len(sigma)
        begin_mean = np.mean(sigma[:n//5])
        end_mean = np.mean(sigma[-n//5:])
        peak_max = np.max([sigma[p] for p in peaks])
        
        both_low = begin_mean < 0.4 * peak_max and end_mean < 0.4 * peak_max
        similar = abs(begin_mean - end_mean) < 0.2 * peak_max
        
        if len(peaks) >= 2:
            return both_low and similar
        elif len(peaks) == 1:
            peak_pos = peaks[0]
            is_middle = 0.2 * n < peak_pos < 0.8 * n
            
            if peak_pos < n - 5:
                after_peak_min = np.min(sigma[peak_pos + 2:])
                returns = after_peak_min < 0.3 * peak_max
                return is_middle and returns and both_low and similar
        
        return False
    
    def _check_monotonic_trend(self, sigma: np.ndarray) -> Tuple[bool, float]:
        """Check for monotonic increasing trend (incremental drift)."""
        if len(sigma) < 5:
            return False, 0.0
        
        x = np.arange(len(sigma))
        slope, _ = np.polyfit(x, sigma, 1)
        
        predicted = slope * x + np.mean(sigma)
        ss_res = np.sum((sigma - predicted) ** 2)
        ss_tot = np.sum((sigma - np.mean(sigma)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        is_monotonic = slope > 0.002 and r_squared > 0.3
        return is_monotonic, r_squared
    
    def _classify(self, sigma: np.ndarray, peaks: np.ndarray) -> Tuple[str, str, float, dict]:
        """Classify drift type based on signal analysis."""
        debug = {}
        n_peaks = len(peaks)
        
        if len(sigma) == 0:
            return "no_drift", "none", 0.0, debug
        
        debug['n_peaks'] = n_peaks
        
        # Check monotonic trend
        is_monotonic, trend_r2 = self._check_monotonic_trend(sigma)
        debug['is_monotonic'] = is_monotonic
        debug['trend_r2'] = trend_r2
        
        # No peaks case
        if n_peaks == 0:
            if is_monotonic:
                return "incremental", "PCD", 0.75, debug
            return "no_drift", "none", 0.5, debug
        
        # Analyze peaks
        peak_shapes = [self._analyze_peak_shape(sigma, p) for p in peaks]
        avg_symmetry = np.mean([ps['symmetry'] for ps in peak_shapes])
        avg_sharpness = np.mean([ps['sharpness'] for ps in peak_shapes])
        
        debug['avg_symmetry'] = avg_symmetry
        debug['avg_sharpness'] = avg_sharpness
        
        # Check patterns
        has_blip = self._has_blip_pattern(sigma, peaks)
        debug['has_blip'] = has_blip
        
        # Periodicity check
        if n_peaks >= 3:
            intervals = np.diff(peaks)
            period_cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        else:
            period_cv = 1.0
        debug['period_cv'] = period_cv
        
        # Classification Logic
        # 1. INCREMENTAL
        if is_monotonic and avg_sharpness < 0.4:
            return "incremental", "PCD", 0.80, debug
        
        # 2. RECURRENT
        if n_peaks >= 4 and period_cv < 0.35 and avg_sharpness > 0.6:
            return "recurrent", "TCD", 0.85, debug
        
        # 3. BLIP
        if has_blip and n_peaks == 2:
            return "blip", "TCD", 0.85, debug
        
        # 4. SUDDEN
        if n_peaks == 1:
            if avg_sharpness > 0.7 and avg_symmetry > 0.7:
                return "sudden", "TCD", 0.85, debug
            elif avg_sharpness < 0.3:
                return "gradual", "PCD", 0.80, debug
            else:
                if avg_symmetry > 0.8:
                    return "sudden", "TCD", 0.70, debug
                else:
                    return "gradual", "PCD", 0.70, debug
        
        # 5. Multiple peaks (2-3)
        if n_peaks in [2, 3]:
            if avg_sharpness > 0.6:
                return "sudden", "TCD", 0.75, debug
            else:
                return "gradual", "PCD", 0.75, debug
        
        # 6. Many peaks (4+)
        if n_peaks >= 4:
            if avg_sharpness > 0.5:
                return "sudden", "TCD", 0.70, debug
            else:
                return "gradual", "PCD", 0.70, debug
        
        return "unknown", "unknown", 0.5, debug
    
    def detect(self, X: np.ndarray, y: np.ndarray = None) -> DriftResult:
        """Main detection method (compatible with benchmark interface)."""
        return self.detect_and_classify(X, y)
    
    def detect_and_classify(self, X: np.ndarray, y: np.ndarray = None) -> DriftResult:
        """Detect and classify drift type."""
        times, sigma_raw = self._compute_drift_magnitude(X)
        
        if len(sigma_raw) < 3:
            return DriftResult(False, [], None, None, 0.0, sigma_raw, times, {})
        
        sigma = gaussian_filter1d(sigma_raw, sigma=self.sigma)
        peaks = self._detect_peaks(sigma)
        
        subcategory, category, confidence, debug = self._classify(sigma, peaks)
        
        positions = times[peaks].tolist() if len(peaks) > 0 else []
        
        debug['using_ow_mmd'] = self.use_ow
        
        return DriftResult(
            detected=len(peaks) > 0 or subcategory in ["incremental"],
            positions=positions,
            subcategory=subcategory,
            category=category,
            confidence=confidence,
            mmd_signal=sigma,
            times=times,
            debug_info=debug
        )


# ================================================================
# STANDALONE TEST
# ================================================================

def generate_test_data():
    """Generate test datasets."""
    np.random.seed(42)
    n, d = 5000, 10
    datasets = {}
    
    # SUDDEN
    X = np.random.randn(n, d)
    X[n//2:] += 3
    datasets['sudden'] = (X, 'TCD', 'sudden')
    
    # GRADUAL
    X = np.random.randn(n, d)
    for i in range(n):
        if i > n//4:
            progress = min(1.0, (i - n//4) / (n//2))
            X[i] += progress * 3
    datasets['gradual'] = (X, 'PCD', 'gradual')
    
    # BLIP
    X = np.random.randn(n, d)
    X[n//3:n//2] += 4
    datasets['blip'] = (X, 'TCD', 'blip')
    
    # RECURRENT
    X = np.random.randn(n, d)
    period = n // 5
    for i in range(1, 5):
        X[i*period:] += 2.0
    datasets['recurrent'] = (X, 'TCD', 'recurrent')
    
    # INCREMENTAL
    X = np.random.randn(n, d)
    for i in range(n):
        X[i] += (i / n) * 3
    datasets['incremental'] = (X, 'PCD', 'incremental')
    
    return datasets


def run_tests():
    """Run standalone tests."""
    print("=" * 70)
    print("ShapeDD_OW_CDT: OW-MMD based Drift Type Classification")
    print(f"OW-MMD Available: {HAS_OW_MMD}")
    print("=" * 70)
    
    detector = ShapeDD_OW_CDT(window_size=200, stride=50)
    datasets = generate_test_data()
    
    correct_cat = 0
    correct_sub = 0
    
    for name, (X, expected_cat, expected_sub) in datasets.items():
        result = detector.detect_and_classify(X)
        
        cat_ok = result.category == expected_cat
        sub_ok = result.subcategory == expected_sub
        
        if cat_ok:
            correct_cat += 1
        if sub_ok:
            correct_sub += 1
        
        cat_mark = "✓" if cat_ok else "✗"
        sub_mark = "✓" if sub_ok else "✗"
        
        print(f"\n{cat_mark} {name.upper()}")
        print(f"   Expected: {expected_cat} ({expected_sub})")
        print(f"   Detected: {result.category} ({result.subcategory}) {sub_mark}")
        print(f"   Confidence: {result.confidence:.2f}")
        
        if result.debug_info:
            d = result.debug_info
            print(f"   Debug: peaks={d.get('n_peaks', 0)}, "
                  f"sharp={d.get('avg_sharpness', 0):.2f}, "
                  f"ow_mmd={d.get('using_ow_mmd', False)}")
    
    print("\n" + "=" * 70)
    print(f"Category accuracy:    {correct_cat}/{len(datasets)} ({100*correct_cat/len(datasets):.0f}%)")
    print(f"Subcategory accuracy: {correct_sub}/{len(datasets)} ({100*correct_sub/len(datasets):.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
