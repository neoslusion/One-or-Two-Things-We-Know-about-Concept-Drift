import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Tuple, List, Optional


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


class ShapedCDT_V5:
    """
    SHAPED-CDT V5: Final version with proper priority logic
    
    Key fixes:
    1. Sudden vs Blip: Sudden has 1 peak, Blip has 2+ peaks (or clear return pattern)
    2. Gradual vs Recurrent: Use sharpness, not just periodicity
    3. Incremental: Check monotonic trend even with peaks
    """
    
    def __init__(self, 
                 window_size: int = 200,
                 stride: int = 50,
                 smoothing_sigma: float = 1.5):
        self.window_size = window_size
        self.stride = stride
        self.sigma = smoothing_sigma
        
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        if len(X) < 2 or len(Y) < 2:
            return 0.0
        
        combined = np.vstack([X, Y])
        dists = pdist(combined, 'sqeuclidean')
        if len(dists) == 0 or np.median(dists) == 0:
            return 0.0
        
        gamma = 1.0 / (2 * np.median(dists) + 1e-10)
        m, n = len(X), len(Y)
        
        K_XX = np.exp(-gamma * squareform(pdist(X, 'sqeuclidean')))
        K_YY = np.exp(-gamma * squareform(pdist(Y, 'sqeuclidean')))
        K_XY = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2))
        
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)
        
        mmd_sq = (np.sum(K_XX) / (m * (m - 1)) + 
                  np.sum(K_YY) / (n * (n - 1)) - 
                  2 * np.mean(K_XY))
        
        return np.sqrt(max(0, mmd_sq))
    
    def _compute_drift_magnitude(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        l = self.window_size
        times, sigma = [], []
        
        for t in range(2 * l, n, self.stride):
            mmd_val = self._compute_mmd(X[t - 2*l : t - l], X[t - l : t])
            sigma.append(mmd_val)
            times.append(t - l)
        
        return np.array(times), np.array(sigma)
    
    def _detect_peaks(self, sigma: np.ndarray) -> np.ndarray:
        if len(sigma) < 3:
            return np.array([])
        
        mean_s, std_s = np.mean(sigma), np.std(sigma)
        threshold = mean_s + 0.3 * std_s
        
        peaks, _ = find_peaks(
            sigma,
            height=threshold,
            distance=max(2, len(sigma) // 25),
            prominence=std_s * 0.2
        )
        return peaks
    
    def _analyze_peak_shape(self, sigma: np.ndarray, peak_idx: int) -> dict:
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
        """
        Blip = temporary change with CLEAR return to baseline
        
        Requirements:
        1. At least 2 peaks (up transition + down transition)
        2. OR: 1 peak but signal clearly returns to start level
        """
        if len(peaks) == 0 or len(sigma) < 10:
            return False
        
        n = len(sigma)
        begin_mean = np.mean(sigma[:n//5])
        end_mean = np.mean(sigma[-n//5:])
        peak_max = np.max([sigma[p] for p in peaks])
        
        # Blip: begin and end are BOTH low and SIMILAR
        both_low = begin_mean < 0.4 * peak_max and end_mean < 0.4 * peak_max
        similar = abs(begin_mean - end_mean) < 0.2 * peak_max
        
        # For blip, we need 2+ peaks (one for each transition)
        # OR clear evidence of return (end much lower than peak region)
        if len(peaks) >= 2:
            return both_low and similar
        elif len(peaks) == 1:
            # Single peak - check if it's in the middle and signal returns
            peak_pos = peaks[0]
            is_middle = 0.2 * n < peak_pos < 0.8 * n
            
            # Check if signal after peak returns to baseline
            if peak_pos < n - 5:
                after_peak_min = np.min(sigma[peak_pos + 2:])
                returns = after_peak_min < 0.3 * peak_max
                return is_middle and returns and both_low and similar
        
        return False
    
    def _check_monotonic_trend(self, sigma: np.ndarray) -> Tuple[bool, float]:
        """Check for monotonic increasing trend (incremental drift)"""
        if len(sigma) < 5:
            return False, 0.0
        
        x = np.arange(len(sigma))
        slope, _ = np.polyfit(x, sigma, 1)
        
        # Compute R²
        predicted = slope * x + np.mean(sigma)
        ss_res = np.sum((sigma - predicted) ** 2)
        ss_tot = np.sum((sigma - np.mean(sigma)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Incremental: positive slope with reasonable fit
        is_monotonic = slope > 0.002 and r_squared > 0.3
        return is_monotonic, r_squared
    
    def _classify(self, sigma: np.ndarray, peaks: np.ndarray) -> Tuple[str, str, float, dict]:
        debug = {}
        n_peaks = len(peaks)
        
        if len(sigma) == 0:
            return "no_drift", "none", 0.0, debug
        
        debug['n_peaks'] = n_peaks
        
        # === Check monotonic trend (for incremental) ===
        is_monotonic, trend_r2 = self._check_monotonic_trend(sigma)
        debug['is_monotonic'] = is_monotonic
        debug['trend_r2'] = trend_r2
        
        # === No peaks case ===
        if n_peaks == 0:
            if is_monotonic:
                return "incremental", "PCD", 0.75, debug
            return "no_drift", "none", 0.5, debug
        
        # === Analyze peaks ===
        peak_shapes = [self._analyze_peak_shape(sigma, p) for p in peaks]
        avg_symmetry = np.mean([ps['symmetry'] for ps in peak_shapes])
        avg_sharpness = np.mean([ps['sharpness'] for ps in peak_shapes])
        
        debug['avg_symmetry'] = avg_symmetry
        debug['avg_sharpness'] = avg_sharpness
        
        # === Check patterns ===
        has_blip = self._has_blip_pattern(sigma, peaks)
        debug['has_blip'] = has_blip
        
        # Periodicity check
        if n_peaks >= 3:
            intervals = np.diff(peaks)
            period_cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
        else:
            period_cv = 1.0
        debug['period_cv'] = period_cv
        
        # === CLASSIFICATION LOGIC ===
        # Priority: Incremental → Recurrent → Blip → Sudden → Gradual
        
        # 1. INCREMENTAL: Monotonic trend with many small peaks
        if is_monotonic and avg_sharpness < 0.4:
            return "incremental", "PCD", 0.80, debug
        
        # 2. RECURRENT: 4+ peaks with regular intervals AND sharp peaks
        if n_peaks >= 4 and period_cv < 0.35 and avg_sharpness > 0.6:
            return "recurrent", "TCD", 0.85, debug
        
        # 3. BLIP: Clear return-to-baseline pattern with 2 peaks
        if has_blip and n_peaks == 2:
            return "blip", "TCD", 0.85, debug
        
        # 4. SUDDEN: Single sharp symmetric peak
        if n_peaks == 1:
            if avg_sharpness > 0.7 and avg_symmetry > 0.7:
                return "sudden", "TCD", 0.85, debug
            elif avg_sharpness < 0.3:
                return "gradual", "PCD", 0.80, debug
            else:
                # Borderline - default to sudden if symmetric
                if avg_symmetry > 0.8:
                    return "sudden", "TCD", 0.70, debug
                else:
                    return "gradual", "PCD", 0.70, debug
        
        # 5. Multiple peaks (2-3) - not blip, not recurrent
        if n_peaks in [2, 3]:
            if avg_sharpness > 0.6:
                return "sudden", "TCD", 0.75, debug  # Multiple sudden drifts
            else:
                return "gradual", "PCD", 0.75, debug
        
        # 6. Many peaks (4+) - not recurrent (failed periodicity check)
        if n_peaks >= 4:
            if avg_sharpness > 0.5:
                return "sudden", "TCD", 0.70, debug
            else:
                return "gradual", "PCD", 0.70, debug
        
        return "unknown", "unknown", 0.5, debug
    
    def detect(self, X: np.ndarray, y: np.ndarray = None) -> DriftResult:
        """Alias for benchmark compatibility"""
        return self.detect_and_classify(X, y)

    def detect_and_classify(self, X: np.ndarray, y: np.ndarray = None) -> DriftResult:
        times, sigma_raw = self._compute_drift_magnitude(X)
        
        if len(sigma_raw) < 3:
            return DriftResult(False, [], None, None, 0.0, sigma_raw, times, {})
        
        sigma = gaussian_filter1d(sigma_raw, sigma=self.sigma)
        peaks = self._detect_peaks(sigma)
        
        subcategory, category, confidence, debug = self._classify(sigma, peaks)
        
        positions = times[peaks].tolist() if len(peaks) > 0 else []
        
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


def generate_test_data():
    np.random.seed(42)
    n, d = 5000, 10
    datasets = {}
    
    # 1. SUDDEN: Single abrupt permanent shift
    X = np.random.randn(n, d)
    X[n//2:] += 3
    datasets['sudden'] = (X, 'TCD', 'sudden')
    
    # 2. GRADUAL: Slow transition (low sharpness expected)
    X = np.random.randn(n, d)
    for i in range(n):
        if i > n//4:
            progress = min(1.0, (i - n//4) / (n//2))
            X[i] += progress * 3
    datasets['gradual'] = (X, 'PCD', 'gradual')
    
    # 3. BLIP: Temporary change with 2 clear transitions
    X = np.random.randn(n, d)
    X[n//3:n//2] += 4
    datasets['blip'] = (X, 'TCD', 'blip')
    
    # 4. RECURRENT: 4+ regular periodic sharp shifts
    X = np.random.randn(n, d)
    period = n // 5
    for i in range(1, 5):
        X[i*period:] += 2.0  # Stronger shift for sharper peaks
    datasets['recurrent'] = (X, 'TCD', 'recurrent')
    
    # 5. INCREMENTAL: Continuous slow drift (monotonic)
    X = np.random.randn(n, d)
    for i in range(n):
        X[i] += (i / n) * 3  # Stronger trend
    datasets['incremental'] = (X, 'PCD', 'incremental')
    
    return datasets


def run_tests():
    print("="*70)
    print("SHAPED-CDT V5: Final Version")
    print("="*70)
    
    detector = ShapedCDT_V5(window_size=200, stride=50)
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
                  f"sym={d.get('avg_symmetry', 0):.2f}, "
                  f"blip={d.get('has_blip', False)}, "
                  f"mono={d.get('is_monotonic', False)}, "
                  f"cv={d.get('period_cv', 1):.2f}")
    
    print("\n" + "="*70)
    print(f"Category accuracy:    {correct_cat}/{len(datasets)} ({100*correct_cat/len(datasets):.0f}%)")
    print(f"Subcategory accuracy: {correct_sub}/{len(datasets)} ({100*correct_sub/len(datasets):.0f}%)")
    print("="*70)


if __name__ == "__main__":
    run_tests()
