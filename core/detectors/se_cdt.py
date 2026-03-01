import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import time

# Import PROPER ShapeDD + ADW-MMD implementation
from .mmd_variants import shapedd_adw_mmd_proper

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
            Legacy threshold parameter (kept for API compatibility).
        alpha : float
            Significance level for proper mode. Default 0.05.
        use_proper : bool
            Legacy parameter (ignored, always uses proper mode).
        """
        self.l1 = window_size
        self.l2 = l2
        self.threshold = threshold
        self.alpha = alpha

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
        return self._monitor_proper(window)
    
    def _monitor_proper(self, window: np.ndarray) -> SECDTResult:
        """
        PROPER implementation using ShapeDD + ADW-MMD with p-value.
        
        Pipeline:
        1. Detection: ShapeDD + ADW-MMD finds drift candidates
        2. Growth: Width Ratio analysis measures drift length → TCD vs PCD
        3. Classification: Shape + temporal features → subcategory
        """
        # 1. Detection Step (PROPER ShapeDD + ADW-MMD)
        is_drift, drift_positions, mmd_trace, p_values = shapedd_adw_mmd_proper(
            window, l1=self.l1, l2=self.l2, alpha=self.alpha
        )
        
        # Compute aggregate score from p-values
        if p_values:
            min_p = min(p_values)
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
        
        # 2. Growth + Classification Step (if drift detected)
        if is_drift and len(mmd_trace) > 0:
            t0 = time.time()
            
            # Growth process: measure drift length from MMD signal shape
            drift_length = self._growth_process(window, mmd_trace=mmd_trace)
            
            # Classification: use drift_length as primary TCD/PCD discriminator
            classification_res = self.classify(mmd_trace, drift_length=drift_length)
            t1 = time.time()
            
            result.drift_type = classification_res.drift_type
            result.subcategory = classification_res.subcategory
            result.features = classification_res.features
            result.classification_time = t1 - t0
            
        return result
    
    def _growth_process(self, data_window: np.ndarray, mmd_trace: np.ndarray = None) -> int:
        """
        Growth process (CDT-MSW Algorithm 2, adapted for unsupervised MMD).
        
        Measures the Width Ratio (WR) of the dominant MMD peak to distinguish:
        - TCD (WR < 0.12): Sharp, narrow peak → sudden change
        - PCD (WR >= 0.12): Wide peak → gradual change
        
        Parameters:
        -----------
        data_window : np.ndarray
            Raw data buffer.
        mmd_trace : np.ndarray, optional
            Pre-computed MMD trace from detection step.
            
        Returns:
        --------
        drift_length : int
            1 = TCD (sharp change), >1 = PCD (gradual change).
        """
        from scipy.signal import find_peaks, peak_widths
        
        if mmd_trace is None or len(mmd_trace) < 10:
            return 1
        
        sigma_s = gaussian_filter1d(mmd_trace, sigma=4)
        threshold = np.mean(sigma_s) + 0.3 * np.std(sigma_s)
        peaks, properties = find_peaks(sigma_s, height=threshold)
        
        if len(peaks) == 0:
            return 1
        
        best_peak_idx = np.argmax(properties['peak_heights'])
        widths, _, _, _ = peak_widths(sigma_s, [peaks[best_peak_idx]], rel_height=0.5)
        fwhm = widths[0]
        wr = fwhm / (2 * self.l1)
        
        if wr < 0.12:
            return 1  # TCD
        else:
            return max(2, int(fwhm / 3))  # PCD

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

    def classify(self, sigma_signal: np.ndarray, drift_length: int = None) -> SECDTResult:
        """
        Classify drift type based on Growth process + signal shape.
        
        Enhanced Algorithm (CDT-MSW inspired):
        1. If drift_length > 1 (Growth process says PCD): use temporal features
        2. Otherwise: use original shape-based decision tree
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
        ppr = features.get('PPR', 0.0)
        dpar = features.get('DPAR', 0.0)
        lts = features.get('LTS', 0.0)
        sds = features.get('SDS', 0.0)
        ms = features.get('MS', 0.0)
        
        result = SECDTResult(features=features)
        
        # =====================================================================
        # PRIMARY: Use drift_length from Growth process (CDT-MSW Algorithm 2)
        # drift_length > 1 → PCD (distribution changed gradually)
        # =====================================================================
        
        if drift_length is not None and drift_length > 1:
            result.drift_type = "PCD"
            is_incremental = (
                (lts > 0.5) or
                (ms > 0.6 and lts > 0.3) or
                (sds > 0.12 and lts > 0.3) or
                (n_p >= 7) or
                (n_p == 0 and mean_val > 0.0001 and lts > 0.5)
            )
            result.subcategory = "Incremental" if is_incremental else "Gradual"
            return result
        
        # =====================================================================
        # FALLBACK: Original shape-based decision tree
        # Used when Growth process returns drift_length == 1
        # =====================================================================
        
        # 1. Blip Drift (TCD)
        if n_p == 2 and len(peak_positions) >= 2:
            peak_distance = abs(peak_positions[1] - peak_positions[0])
            is_blip = (
                (ppr > 0 and ppr < 0.20 and dpar > 0.65) or
                (peak_distance < 35 and wr < 0.30 and dpar > 0.60)
            )
            if is_blip:
                result.drift_type = "TCD"
                result.subcategory = "Blip"
                return result
        
        # 2. Sudden Drift (TCD)
        if n_p <= 3 and wr < 0.15 and snr > 2.0:
            result.drift_type = "TCD"
            result.subcategory = "Sudden"
            return result
        
        # 3. Recurrent Drift (TCD)
        if n_p >= 4 and cv < 0.3 and lts < 0.5:
            result.drift_type = "TCD"
            result.subcategory = "Recurrent"
            return result
        
        # 4. Gradual vs Incremental (PCD) — fallback
        result.drift_type = "PCD"
        is_incremental = (
            (lts > 0.5) or
            (ms > 0.6 and lts > 0.3) or
            (sds > 0.12 and lts > 0.3) or
            (n_p >= 7) or
            (n_p == 0 and mean_val > 0.0001 and lts > 0.5)
        )
        result.subcategory = "Incremental" if is_incremental else "Gradual"
        return result

