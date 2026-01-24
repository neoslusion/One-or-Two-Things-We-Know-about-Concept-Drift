import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import time

# Import PROPER ShapeDD + ADW-MMD implementation
from .mmd_variants import shapedd_adw_mmd_proper, shapedd_adw_mmd_full

@dataclass
class SECDTResult:
    """Result from SE-CDT unified detection and classification."""
    is_drift: bool = False
    drift_type: str = "Unknown"  # TCD (Temporary), PCD (Permanent)
    subcategory: str = "Unknown" # Sudden, Blip, Gradual, Incremental, Recurrent
    features: Dict[str, float] = None
    score: float = 0.0
    p_value: float = 1.0  # NEW: Statistical p-value for drift detection
    mmd_trace: np.ndarray = None
    drift_positions: List[int] = field(default_factory=list)  # NEW: Detected positions
    classification_time: float = 0.0

class SE_CDT:
    """
    SE-CDT (ShapeDD-Enhanced Concept Drift Type identification).
    Unified Detector-Classifier System.
    
    This implementation PROPERLY combines:
    1. ShapeDD shape statistic for drift DETECTION (candidate identification)
    2. ADW-MMD with asymptotic p-value for VALIDATION
    3. Signal shape analysis for drift TYPE CLASSIFICATION
    
    Attributes:
    -----------
    l1 : int
        Reference window size (half-window for shape statistic)
    l2 : int
        Test window size for MMD validation
    alpha : float
        Significance level for drift detection (default 0.05)
    use_proper : bool
        If True, use PROPER ShapeDD+ADW-MMD (fast, with p-value)
        If False, use heuristic version (for backward compatibility)
    """
    
    def __init__(self, window_size: int = 50, l2: int = 150, 
                 threshold: float = 0.15, alpha: float = 0.05,
                 use_proper: bool = True):
        """
        Initialize SE-CDT detector.
        
        Parameters:
        -----------
        window_size : int
            Reference window size (l1). Default 50.
        l2 : int
            Test window size for MMD. Default 150.
        threshold : float
            Score threshold for heuristic mode. Default 0.15.
            (Ignored when use_proper=True)
        alpha : float
            Significance level for proper mode. Default 0.05.
        use_proper : bool
            Use PROPER ShapeDD+ADW-MMD implementation. Default True.
        """
        self.l1 = window_size
        self.l2 = l2
        self.threshold = threshold
        self.alpha = alpha
        self.use_proper = use_proper

    def monitor(self, window: np.ndarray) -> SECDTResult:
        """
        Monitor the stream window for drift.
        If drift is detected, automatically classify it.
        
        Uses PROPER ShapeDD + ADW-MMD design:
        1. ShapeDD shape statistic detects candidate drift points
        2. ADW-MMD with asymptotic p-value validates candidates
        3. MMD signal shape classifies drift type
        
        Parameters:
        -----------
        window : np.ndarray
            Current data window to check (buffer).
            
        Returns:
        --------
        result : SECDTResult
            Unified result containing:
            - is_drift: Whether drift was detected
            - p_value: Statistical significance (only in proper mode)
            - drift_type: TCD or PCD
            - subcategory: Sudden, Blip, Gradual, Incremental, Recurrent
            - mmd_trace: MMD signal for analysis
            - features: Extracted geometric features
        """
        if self.use_proper:
            return self._monitor_proper(window)
        else:
            return self._monitor_heuristic(window)
    
    def _monitor_proper(self, window: np.ndarray) -> SECDTResult:
        """
        PROPER implementation using ShapeDD + ADW-MMD with p-value.
        """
        # 1. Detection Step (PROPER ShapeDD + ADW-MMD)
        is_drift, drift_positions, mmd_trace, p_values = shapedd_adw_mmd_proper(
            window, l1=self.l1, l2=self.l2, alpha=self.alpha
        )
        
        # Compute aggregate score from p-values
        if p_values:
            min_p = min(p_values)
            # Convert p-value to score for backward compatibility
            # score = 1 - p_value (high score = more confident drift)
            score = 1.0 - min_p
        else:
            min_p = 1.0
            score = 0.0
        
        result = SECDTResult(
            is_drift=is_drift,
            score=score,
            p_value=min_p,
            mmd_trace=mmd_trace,
            drift_positions=drift_positions
        )
        
        # 2. Classification Step (if drift detected)
        if is_drift and len(mmd_trace) > 0:
            t0 = time.time()
            classification_res = self.classify(mmd_trace)
            t1 = time.time()
            
            result.drift_type = classification_res.drift_type
            result.subcategory = classification_res.subcategory
            result.features = classification_res.features
            result.classification_time = t1 - t0
            
        return result
    
    def _monitor_heuristic(self, window: np.ndarray) -> SECDTResult:
        """
        Heuristic implementation (backward compatible).
        Uses pattern score threshold instead of p-value.
        """
        # 1. Detection Step (Heuristic ShapeDD-ADW)
        pattern_score, mmd_max, mmd_trace = shapedd_adw_mmd_full(
            window, l1=self.l1, l2=self.l2, gamma='auto'
        )
        
        result = SECDTResult(
            is_drift=False,
            score=pattern_score,
            p_value=1.0,  # No p-value in heuristic mode
            mmd_trace=mmd_trace
        )
        
        # 2. Trigger Check (threshold-based)
        if pattern_score > self.threshold:
            result.is_drift = True
            
            # 3. Classification Step (Algorithm 3.4)
            t0 = time.time()
            classification_res = self.classify(mmd_trace)
            t1 = time.time()
            
            result.drift_type = classification_res.drift_type
            result.subcategory = classification_res.subcategory
            result.features = classification_res.features
            result.classification_time = t1 - t0
            
        return result

    def extract_features(self, sigma_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract geometric features from the MMD signal.
        Expected input: A window of MMD values centered around the detected drift.
        """
        if len(sigma_signal) == 0:
            return {}
            
        # 1. Smoothing (Moderate)
        sigma_s = gaussian_filter1d(sigma_signal, sigma=4) # Reverted to 4
        
        # 2. Peak Detection
        threshold = np.mean(sigma_s) + 0.3 * np.std(sigma_s)
        peaks, properties = find_peaks(sigma_s, height=threshold)
        
        n_p = len(peaks)
        
        # 3. Calculate Features
        features = {
            'n_p': n_p,
            'WR': 0.0,    # Width Ratio
            'SNR': 0.0,   # Signal-to-Noise Ratio
            'CV': 0.0,    # Coefficient of Variation (Periodicity)
            'Mean': np.mean(sigma_s),
            'peak_positions': peaks.tolist() if n_p > 0 else [],  # For Blip detection
            'PPR': 0.0,   # Peak Proximity Ratio (for Blip)
            'DPAR': 0.0,  # Dual-Peak Amplitude Ratio (for Blip)
            # Temporal features (for Incremental vs Gradual)
            'LTS': 0.0,   # Linear Trend Strength
            'SDS': 0.0,   # Step Detection Score
            'MS': 0.0     # Monotonicity Score
        }
        
        # SNR
        median_val = np.median(sigma_s)
        max_val = np.max(sigma_s) if len(sigma_s) > 0 else 0
        features['SNR'] = max_val / (median_val + 1e-10)
        
        if n_p > 0:
            best_peak_idx = np.argmax(properties['peak_heights'])
            
            # calculate widths at half height
            widths, width_heights, left_ips, right_ips = peak_widths(
                sigma_s, [peaks[best_peak_idx]], rel_height=0.5
            )
            fwhm = widths[0]
            features['WR'] = fwhm / (2 * self.l1)
            
            # Periodicity (CV) if multiple peaks
            if n_p >= 2:
                peak_distances = np.diff(peaks)
                if len(peak_distances) > 0:
                    features['CV'] = np.std(peak_distances) / (np.mean(peak_distances) + 1e-10)
                
                # PPR (Peak Proximity Ratio) - for Blip detection
                # Ratio of closest peak distance to signal length
                min_peak_distance = np.min(peak_distances)
                features['PPR'] = min_peak_distance / len(sigma_s)
                
                # DPAR (Dual-Peak Amplitude Ratio) - for Blip detection
                # If 2 peaks close together with similar heights = Blip
                if n_p == 2:
                    peak_heights = properties['peak_heights']
                    h1, h2 = peak_heights[0], peak_heights[1]
                    features['DPAR'] = min(h1, h2) / (max(h1, h2) + 1e-10)
        
        # 4. Extract Temporal Features (for Incremental vs Gradual)
        temporal_features = self.extract_temporal_features(sigma_s)
        features.update(temporal_features)
        
        return features

    def extract_temporal_features(self, sigma_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features to distinguish Incremental from Gradual drift.
        
        Incremental: Stepwise changes with monotonic trend
        Gradual: Smooth curve with oscillations
        
        Returns:
        --------
        temporal_features : dict
            - LTS (Linear Trend Strength): R² of linear fit
            - SDS (Step Detection Score): Number of significant steps
            - MS (Monotonicity Score): Ratio of monotonic segments
        """
        if len(sigma_signal) < 10:
            return {'LTS': 0.0, 'SDS': 0.0, 'MS': 0.0}
        
        signal = sigma_signal.copy()
        n = len(signal)
        
        # 1. LTS (Linear Trend Strength) - R² coefficient
        # High LTS = strong linear trend (Incremental)
        # Low LTS = curved/oscillating (Gradual)
        x = np.arange(n)
        if np.std(signal) > 1e-10:
            # Linear regression
            A = np.vstack([x, np.ones(n)]).T
            m, c = np.linalg.lstsq(A, signal, rcond=None)[0]
            y_pred = m * x + c
            
            # R² calculation
            ss_tot = np.sum((signal - np.mean(signal))**2)
            ss_res = np.sum((signal - y_pred)**2)
            lts = 1 - (ss_res / (ss_tot + 1e-10))
            lts = max(0.0, min(1.0, lts))  # Clamp to [0, 1]
        else:
            lts = 0.0
        
        # 2. SDS (Step Detection Score) - Count significant jumps
        # High SDS = many steps (Incremental)
        # Low SDS = smooth changes (Gradual)
        diffs = np.diff(signal)
        if len(diffs) > 0:
            diff_threshold = np.std(diffs) * 1.5  # 1.5 std = significant step
            significant_steps = np.sum(np.abs(diffs) > diff_threshold)
            sds = significant_steps / len(diffs)  # Normalized
        else:
            sds = 0.0
        
        # 3. MS (Monotonicity Score) - Ratio of monotonic direction
        # High MS = mostly increasing or decreasing (Incremental)
        # Low MS = back-and-forth oscillations (Gradual)
        if len(diffs) > 0:
            positive_changes = np.sum(diffs > 0)
            negative_changes = np.sum(diffs < 0)
            total_changes = positive_changes + negative_changes
            
            if total_changes > 0:
                # Monotonicity = how much one direction dominates
                ms = abs(positive_changes - negative_changes) / total_changes
            else:
                ms = 0.0
        else:
            ms = 0.0
        
        return {
            'LTS': float(lts),
            'SDS': float(sds),
            'MS': float(ms)
        }

    def classify(self, sigma_signal: np.ndarray) -> SECDTResult:
        """
        Classify drift type based on signal shape (Algorithm 3.4).
        Decision order: Sudden → Blip → Recurrent → Gradual → Incremental
        """
        features = self.extract_features(sigma_signal)
        if not features:
            return SECDTResult()
            
        n_p = features['n_p']
        wr = features['WR']
        snr = features['SNR']
        cv = features['CV']
        mean_val = features['Mean']
        peak_positions = features.get('peak_positions', [])
        ppr = features.get('PPR', 0.0)  # Peak Proximity Ratio
        dpar = features.get('DPAR', 0.0)  # Dual-Peak Amplitude Ratio
        
        result = SECDTResult(features=features)
        
        # Decision Logic (Enhanced with PPR/DPAR for Blip)
        
        # 1. Blip Drift (TCD) - CHECK FIRST!
        # Pattern: drift → revert quickly (2 peaks close together, similar height)
        # Enhanced with PPR and DPAR features
        if n_p == 2 and len(peak_positions) >= 2:
            peak_distance = abs(peak_positions[1] - peak_positions[0])
            
            # Multi-condition Blip detection (relaxed thresholds):
            # - PPR < 0.20: Peaks are close (< 20% of signal length, was 0.15)
            # - DPAR > 0.65: Similar peak heights (height ratio > 0.65, was 0.7)
            # - peak_distance < 35: Close in smoothed signal units (was 30)
            # - wr < 0.30: Not too wide (was 0.25)
            is_blip = (
                (ppr > 0 and ppr < 0.20 and dpar > 0.65) or
                (peak_distance < 35 and wr < 0.30 and dpar > 0.60)
            )
            
            if is_blip:
                result.drift_type = "TCD"
                result.subcategory = "Blip"
                return result
        
        # 2. Sudden Drift (TCD)
        # Sharp single peak, high SNR, narrow width
        # THRESHOLD TUNING: Relaxed wr (0.12→0.15) and lowered snr (2.5→2.0)
        # Rationale: Recapture TCD events previously misclassified as Recurrent/PCD
        # Expected: TCD accuracy 25% → 60-70%, improves CAT accuracy
        if n_p <= 3 and wr < 0.15 and snr > 2.0:
            result.drift_type = "TCD"
            result.subcategory = "Sudden"
            return result
        
        # 3. Extract temporal features EARLY for disambiguation
        lts = features.get('LTS', 0.0)
        sds = features.get('SDS', 0.0)
        ms = features.get('MS', 0.0)
        
        # 4. Recurrent Drift (PCD)
        # Multiple evenly-spaced peaks BUT NOT with strong upward trend
        # If strong temporal trend (LTS > 0.5), it's Incremental, not Recurrent
        if n_p >= 4 and cv < 0.3 and lts < 0.5:
            result.drift_type = "PCD"
            result.subcategory = "Recurrent"
            return result
        
        # 5. Gradual vs Incremental (PCD) - Use TEMPORAL features
        # DEFAULT: All remaining cases are PCD, use temporal analysis
        result.drift_type = "PCD"
        
        # Decision logic for Incremental vs Gradual:
        # THRESHOLD TUNING: Raised LTS from 0.3 → 0.5, added lts > 0.3 for compound conditions
        # Rationale: Previous 0.3 threshold was too permissive, causing TCD/PCD boundary blur
        #            Requires clear monotonic trend (R² > 0.5) for Incremental classification
        # Trade-off: Incremental accuracy 40% → 25-30%, CAT accuracy 60% → 75-80%
        # Expected behavior:
        #   - Incremental: Clear upward trend with R² > 0.5 (e.g., linear parameter shift)
        #   - Gradual: Oscillating/smooth curves with R² < 0.5 (e.g., slow RBF drift)
        # Incremental can be EITHER:
        #   A) Strong trend: LTS > 0.5 (STRENGTHENED - requires clear upward trend)
        #   B) Weak trend + monotonic: MS > 0.6 AND LTS > 0.3 (STRENGTHENED both conditions)
        #   C) Weak trend + steps: SDS > 0.12 AND LTS > 0.3 (STRENGTHENED lts requirement)
        #   D) Many peaks: n_p >= 7 (plateau pattern, unchanged)
        # Gradual: Low temporal scores (oscillating, smooth curve without trend)
        
        is_incremental = (
            (lts > 0.5) or                    # Strong upward trend (STRENGTHENED from 0.3)
            (ms > 0.6 and lts > 0.3) or       # Monotonic with moderate trend (STRENGTHENED)
            (sds > 0.12 and lts > 0.3) or     # Steps with moderate trend (STRENGTHENED)
            (n_p >= 7) or                      # Many peaks = plateau-like
            (n_p == 0 and mean_val > 0.0001 and lts > 0.5)  # Plateau + strong trend (STRENGTHENED)
        )
        
        if is_incremental:
            result.subcategory = "Incremental"
        else:
            result.subcategory = "Gradual"
        
        return result

