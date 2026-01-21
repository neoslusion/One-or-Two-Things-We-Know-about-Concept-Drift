import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import time

# Import shapedd_adw_mmd_full for internal detection
from .mmd_variants import shapedd_adw_mmd_full

@dataclass
class SECDTResult:
    is_drift: bool = False
    drift_type: str = "Unknown"  # TCD, PCD
    subcategory: str = "Unknown" # Sudden, Blip, Gradual, Incremental, Recurrent
    features: Dict[str, float] = None
    score: float = 0.0
    mmd_trace: np.ndarray = None
    classification_time: float = 0.0

class SE_CDT:
    """
    SE-CDT (ShapeDD-Enhanced Concept Drift Type identification).
    Unified Detector-Classifier System.
    """
    
    def __init__(self, window_size: int = 50, l2: int = 150, threshold: float = 0.5):
        self.l1 = window_size
        self.l2 = l2
        self.threshold = threshold

    def monitor(self, window: np.ndarray) -> SECDTResult:
        """
        Monitor the stream window for drift.
        If drift is detected, automatically classify it.
        
        Parameters:
        -----------
        window : np.ndarray
            Current data window to check.
            
        Returns:
        --------
        result : SECDTResult
            Unified result containing detection and classification info.
        """
        # 1. Detection Step (ShapeDD-ADW)
        pattern_score, mmd_max, mmd_trace = shapedd_adw_mmd_full(
            window, l1=self.l1, l2=self.l2, gamma='auto'
        )
        
        result = SECDTResult(
            is_drift=False,
            score=pattern_score,
            mmd_trace=mmd_trace
        )
        
        # 2. Trigger Check
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
            'peak_positions': peaks.tolist() if n_p > 0 else []  # For Blip detection
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
        
        return features

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
        
        result = SECDTResult(features=features)
        
        # Decision Logic (Algorithm 3.4 - Fixed Order)
        
        # 1. Sudden Drift (TCD)
        # Sharp single peak, high SNR, narrow width
        if n_p <= 3 and wr < 0.12 and snr > 2.5:
            result.drift_type = "TCD"
            result.subcategory = "Sudden"
            return result
        
        # 2. Blip Drift (TCD) - NEW!
        # Two peaks close together (bump up then down in signal)
        if n_p == 2 and len(peak_positions) >= 2:
            peak_distance = abs(peak_positions[1] - peak_positions[0])
            # Close peaks (< 30 units in smoothed signal) = Blip
            if peak_distance < 30 and wr < 0.2:
                result.drift_type = "TCD"
                result.subcategory = "Blip"
                return result
        
        # 3. Recurrent Drift (PCD)
        # Multiple evenly-spaced peaks
        if n_p >= 4 and cv < 0.3:
            result.drift_type = "PCD"
            result.subcategory = "Recurrent"
            return result
            
        # 4. Gradual Drift (PCD)
        # Wide peak (WR >= 0.12) with moderate peak count
        if wr >= 0.12 and n_p <= 6:
            result.drift_type = "PCD"
            result.subcategory = "Gradual"
            return result
        
        # 5. Incremental Drift (PCD) - Checked LAST as per thesis
        # Many peaks or plateau-like signal (high mean, low SNR)
        if n_p >= 7 or (n_p == 0 and mean_val > 0.0001) or (snr < 2.0 and mean_val > 0.0005):
            result.drift_type = "PCD"
            result.subcategory = "Incremental"
            return result
        
        # Fallback: Gradual (most common PCD)
        result.drift_type = "PCD"
        result.subcategory = "Gradual"
        
        return result

