import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

@dataclass
class SECDTResult:
    drift_type: str = "Unknown"  # TCD, PCD
    subcategory: str = "Unknown" # Sudden, Blip, Gradual, Incremental, Recurrent
    features: Dict[str, float] = None
    
class SE_CDT:
    """
    SE-CDT (ShapeDD-Enhanced Concept Drift Type identification).
    Algorithm 3.4 from Thesis: Unsupervised drift classification 
    using drift magnitude signal shape.
    """
    
    def __init__(self, window_size: int = 50):
        self.l1 = window_size

    def extract_features(self, sigma_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract geometric features from the MMD signal.
        Expected input: A window of MMD values centered around the detected drift.
        """
        if len(sigma_signal) == 0:
            return {}
            
        # 1. Smoothing
        sigma_s = gaussian_filter1d(sigma_signal, sigma=2)
        
        # 2. Peak Detection
        # Threshold from paper: mean + 0.3 * std
        threshold = np.mean(sigma_s) + 0.3 * np.std(sigma_s)
        peaks, properties = find_peaks(sigma_s, height=threshold)
        
        n_p = len(peaks)
        
        # 3. Calculate Features
        features = {
            'n_p': n_p,
            'WR': 0.0,    # Width Ratio
            'SNR': 0.0,   # Signal-to-Noise Ratio
            'CV': 0.0     # Coefficient of Variation (Periodicity)
        }
        
        # SNR
        median_val = np.median(sigma_s)
        max_val = np.max(sigma_s) if len(sigma_s) > 0 else 0
        features['SNR'] = max_val / (median_val + 1e-10)
        
        if n_p > 0:
            # Width Ratio (WR) of the most prominent peak
            # Use the highest peak for width calculation
            best_peak_idx = np.argmax(properties['peak_heights'])
            
            # calculate widths at half height
            widths, width_heights, left_ips, right_ips = peak_widths(
                sigma_s, [peaks[best_peak_idx]], rel_height=0.5
            )
            fwhm = widths[0]
            
            # WR = FWHM / (2 * l1)
            # Note: signal usually covers a range, but WR is relative to window size
            features['WR'] = fwhm / (2 * self.l1)
            
            # Periodicity (CV) if multiple peaks
            if n_p >= 2:
                peak_distances = np.diff(peaks)
                if len(peak_distances) > 0:
                    mean_dist = np.mean(peak_distances)
                    std_dist = np.std(peak_distances)
                    features['CV'] = std_dist / (mean_dist + 1e-10)
        
        return features

    def classify(self, sigma_signal: np.ndarray) -> SECDTResult:
        """
        Classify drift type based on signal shape (Algorithm 3.4).
        """
        features = self.extract_features(sigma_signal)
        if not features:
            return SECDTResult()
            
        n_p = features['n_p']
        wr = features['WR']
        snr = features['SNR']
        cv = features['CV']
        
        result = SECDTResult(features=features)
        
        # Decision Logic (Algorithm 3.4)
        
        # 1. Sudden Drift (TCD)
        # Sharp single peak, high SNR
        if n_p <= 4 and wr < 1.2 and snr > 4:
            result.drift_type = "TCD"
            result.subcategory = "Sudden"
            return result
            
        # 2. Recurrent Drift (PCD)
        # Multiple regular peaks
        if n_p >= 4 and cv < 0.2:
            result.drift_type = "PCD"
            result.subcategory = "Recurrent"
            return result
            
        # 3. Gradual Drift (PCD)
        # Wide peak
        if wr > 2.0:
            result.drift_type = "PCD"
            result.subcategory = "Gradual"
            return result
            
        # 4. Incremental Drift (PCD)
        # Very wide or many peaks (stepped)
        if n_p >= 10 and wr > 5:
            result.drift_type = "PCD"
            result.subcategory = "Incremental"
            return result
        
        # Default Fallback (usually Gradual for undefined wide signals)
        result.drift_type = "PCD"
        result.subcategory = "Gradual"
        
        return result
