"""
ShapeDD_OW_CDT v2: Using Original shapedd_ow_mmd() for Classification
======================================================================

Uses the original shapedd_ow_mmd() function directly which already has:
- Proper pattern_score calculation
- Calibrated thresholds for OW-MMD
- Both simple and enhanced pattern detection modes

Classification is based on pattern_score and mmd_max from original function.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import sys
from pathlib import Path

# Add backup to path for ow_mmd
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent
backup_path = project_root / "experiments" / "backup"
if str(backup_path) not in sys.path:
    sys.path.insert(0, str(backup_path))

# Import original OW-MMD functions
try:
    from ow_mmd import shapedd_ow_mmd, shapedd_ow_mmd_hybrid
    HAS_OW_MMD = True
except ImportError:
    HAS_OW_MMD = False
    print("[WARNING] ow_mmd not available")


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


class ShapeDD_OW_CDT_V2:
    """
    ShapeDD_OW_CDT using original shapedd_ow_mmd() function.
    
    This version uses the well-calibrated original function instead of
    re-implementing peak detection.
    
    Parameters
    ----------
    l1 : int
        Reference window size (default: 50)
    l2 : int  
        Test window size (default: 150)
    stride : int
        Step size for sliding windows
    detection_threshold : float
        Pattern score threshold for drift detection (default: 0.5)
    """
    
    def __init__(self, 
                 l1: int = 50,
                 l2: int = 150,
                 stride: int = 50,
                 detection_threshold: float = 0.5,
                 window_size: int = 200):
        self.l1 = l1
        self.l2 = l2
        self.stride = stride
        self.detection_threshold = detection_threshold
        self.window_size = window_size  # For compatibility
        
    def _compute_signal(self, X: np.ndarray):
        """Compute pattern scores over sliding windows."""
        n = len(X)
        window_size = self.l1 + self.l2
        
        pattern_scores = []
        mmd_values = []
        times = []
        
        for t in range(window_size, n, self.stride):
            window = X[t - window_size:t]
            
            if HAS_OW_MMD and len(window) >= self.l1 + self.l2:
                pattern_score, mmd_max = shapedd_ow_mmd(
                    window, l1=self.l1, l2=self.l2, gamma='auto', mode='enhanced'
                )
            else:
                pattern_score, mmd_max = 0.0, 0.0
            
            pattern_scores.append(pattern_score)
            mmd_values.append(mmd_max)
            times.append(t - window_size // 2)
        
        return np.array(times), np.array(pattern_scores), np.array(mmd_values)
    
    def _classify_from_signal(self, pattern_scores: np.ndarray, mmd_values: np.ndarray):
        """Classify drift type based on pattern scores and MMD values."""
        if len(pattern_scores) == 0:
            return "no_drift", "none", 0.0, {}
        
        debug = {}
        
        # Find drift points (where pattern_score > threshold)
        drift_mask = pattern_scores > self.detection_threshold
        n_drift_points = np.sum(drift_mask)
        
        debug['n_drift_points'] = n_drift_points
        debug['max_pattern_score'] = float(np.max(pattern_scores)) if len(pattern_scores) > 0 else 0.0
        debug['max_mmd'] = float(np.max(mmd_values)) if len(mmd_values) > 0 else 0.0
        
        if n_drift_points == 0:
            # Check for gradual/incremental based on trend
            if len(mmd_values) >= 5:
                slope = np.polyfit(range(len(mmd_values)), mmd_values, 1)[0]
                if slope > 0.001:
                    return "incremental", "PCD", 0.6, debug
            return "no_drift", "none", 0.3, debug
        
        # Analyze pattern of drift points
        max_score = np.max(pattern_scores)
        mean_score = np.mean(pattern_scores[drift_mask]) if n_drift_points > 0 else 0.0
        confidence = max_score
        
        debug['mean_score'] = mean_score
        
        # Count clusters of drift points
        drift_indices = np.where(drift_mask)[0]
        if len(drift_indices) > 1:
            gaps = np.diff(drift_indices)
            n_clusters = 1 + np.sum(gaps > 3)  # Gap > 3 means new cluster
        else:
            n_clusters = n_drift_points
        
        debug['n_clusters'] = n_clusters
        
        # Classification logic based on pattern
        n_total = len(pattern_scores)
        drift_ratio = n_drift_points / n_total
        
        # RECURRENT: Many separate clusters
        if n_clusters >= 4:
            return "recurrent", "TCD", confidence, debug
        
        # BLIP: 2 clusters (start and end of temporary change)
        if n_clusters == 2 and drift_ratio < 0.4:
            return "blip", "TCD", confidence, debug
        
        # SUDDEN: 1 concentrated cluster with high score
        if n_clusters == 1 and max_score > 0.7 and drift_ratio < 0.3:
            return "sudden", "TCD", confidence, debug
        
        # GRADUAL: Spread out drift points with lower scores
        if drift_ratio > 0.3 and mean_score < 0.6:
            return "gradual", "PCD", confidence, debug
        
        # INCREMENTAL: Continuous increase pattern
        if drift_ratio > 0.5:
            return "incremental", "PCD", confidence, debug
        
        # Default based on concentration
        if n_clusters <= 2:
            return "sudden", "TCD", confidence, debug
        else:
            return "gradual", "PCD", confidence, debug
    
    def detect(self, X: np.ndarray, y: np.ndarray = None) -> DriftResult:
        """Main detection interface."""
        times, pattern_scores, mmd_values = self._compute_signal(X)
        
        subcategory, category, confidence, debug = self._classify_from_signal(
            pattern_scores, mmd_values
        )
        
        # Find positions of significant drift points
        if len(pattern_scores) > 0:
            drift_mask = pattern_scores > self.detection_threshold
            positions = times[drift_mask].tolist() if np.any(drift_mask) else []
        else:
            positions = []
        
        debug['using_ow_mmd'] = HAS_OW_MMD
        
        return DriftResult(
            detected=len(positions) > 0 or subcategory in ["incremental", "gradual"],
            positions=positions,
            subcategory=subcategory,
            category=category,
            confidence=confidence,
            mmd_signal=pattern_scores,
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
    print("ShapeDD_OW_CDT V2: Using Original shapedd_ow_mmd()")
    print(f"OW-MMD Available: {HAS_OW_MMD}")
    print("=" * 70)
    
    detector = ShapeDD_OW_CDT_V2(l1=50, l2=150, stride=50)
    datasets = generate_test_data()
    
    correct_cat = 0
    correct_sub = 0
    
    for name, (X, expected_cat, expected_sub) in datasets.items():
        result = detector.detect(X)
        
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
            print(f"   Debug: drift_pts={d.get('n_drift_points', 0)}, "
                  f"clusters={d.get('n_clusters', 0)}, "
                  f"max_score={d.get('max_pattern_score', 0):.2f}")
    
    print("\n" + "=" * 70)
    print(f"Category accuracy:    {correct_cat}/{len(datasets)} ({100*correct_cat/len(datasets):.0f}%)")
    print(f"Subcategory accuracy: {correct_sub}/{len(datasets)} ({100*correct_sub/len(datasets):.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
