"""
CDT_MSW: Concept Drift Type Identification Based on Multi-Sliding Windows
=========================================================================

Implementation following Guo et al. (2022), Information Sciences.

Three main processes:
1. Detection Process: Find drift position using accuracy ratio
2. Growth Process: Determine drift length and category (TCD/PCD)
3. Tracking Process: Identify subcategory using TFR curves

Reference:
    Guo, H., Li, H., Ren, Q., & Wang, W. (2022). Concept drift type 
    identification based on multi-sliding windows. Information Sciences, 
    585, 1-23.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CDT_MSW_Config:
    """Configuration for CDT_MSW algorithm."""
    # Window parameters
    window_size: int = 200          # Basic window size s
    adjoint_windows: int = 5        # Number of basic windows in adjoint window n
    step_size: int = 50             # Sliding step
    
    # Detection thresholds
    detection_threshold: float = 0.85   # ξ for P̃ᵗdet
    stability_threshold: float = 0.01   # δ for variance stability
    stability_patience: int = 3         # Consecutive stable checks
    
    # Tracking parameters
    tracking_steps: int = 10        # k for tracking process
    
    # Learner type: 'perceptron', 'logistic', 'mlp', 'tree'
    learner: str = 'logistic'


@dataclass
class CDT_MSW_Result:
    """Result of CDT_MSW classification."""
    drift_detected: bool
    drift_position: int
    drift_length: int
    category: str  # 'TCD' or 'PCD'
    subcategory: str  # 'sudden', 'gradual', 'incremental', 'recurrent', 'blip'
    detection_ratio: float
    tfr_curve: List[float]
    note: str = ""


class CDT_MSW:
    """
    CDT_MSW: Multi-Sliding Window Concept Drift Type Identification.
    
    Uses accuracy-based detection (supervised approach).
    """
    
    def __init__(self, config: Optional[CDT_MSW_Config] = None):
        self.config = config or CDT_MSW_Config()
        self.learner = self._create_learner()
    
    def _create_learner(self):
        """Create the base learner for accuracy computation."""
        if self.config.learner == 'perceptron':
            return Perceptron(max_iter=100, random_state=42)
        elif self.config.learner == 'logistic':
            return LogisticRegression(max_iter=100, solver='lbfgs', random_state=42)
        elif self.config.learner == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, random_state=42)
        elif self.config.learner == 'tree':
            return DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            return LogisticRegression(max_iter=100, solver='lbfgs', random_state=42)
    
    def _compute_accuracy(self, X_train, y_train, X_test, y_test) -> float:
        """Train model on (X_train, y_train), test on (X_test, y_test)."""
        if len(np.unique(y_train)) < 2:
            return 0.5  # Cannot train with single class
        
        try:
            model = self._create_learner()
            model.fit(X_train, y_train)
            return model.score(X_test, y_test)
        except Exception:
            return 0.5
    
    # =========================================================================
    # PHASE 1: DETECTION PROCESS
    # =========================================================================
    
    def detection_process(self, X: np.ndarray, y: np.ndarray) -> Tuple[bool, int, float]:
        """
        Detection Process: Find drift position.
        
        Algorithm:
            1. Set W_A as reference window (static)
            2. Slide W_B forward
            3. Compute P̃ᵗdet = aᵗB / a*A
            4. Drift detected when P̃ᵗdet < ξ
        
        Returns:
            (drift_detected, drift_position, detection_ratio)
        """
        n = len(X)
        s = self.config.window_size
        step = self.config.step_size
        threshold = self.config.detection_threshold
        
        if n < 2 * s:
            return False, 0, 1.0
        
        # W_A: Reference window (first s samples)
        X_A = X[:s]
        y_A = y[:s]
        
        # Compute a*A: accuracy of model trained on W_A
        a_star_A = self._compute_accuracy(X_A, y_A, X_A, y_A)
        
        if a_star_A < 0.1:
            return False, 0, 1.0  # Model is not useful
        
        # Slide W_B forward
        for t in range(s, n - s + 1, step):
            X_B = X[t:t + s]
            y_B = y[t:t + s]
            
            # Compute aᵗB: accuracy of model trained on W_B, tested on W_A
            a_t_B = self._compute_accuracy(X_B, y_B, X_A, y_A)
            
            # Detection flow ratio
            P_det = a_t_B / a_star_A
            
            if P_det < threshold:
                # Drift detected at position t
                return True, t, P_det
        
        return False, n, 1.0
    
    # =========================================================================
    # PHASE 2: GROWTH PROCESS
    # =========================================================================
    
    def growth_process(self, X: np.ndarray, y: np.ndarray, 
                       drift_position: int) -> Tuple[int, str]:
        """
        Growth Process: Determine drift length and category.
        
        Algorithm:
            1. Fill adjoint window W_R with data after drift
            2. Compute variance of accuracy in W_R subwindows
            3. When variance < δ → distribution stable → drift ends
            4. Drift length m determines category:
               - m = 1 → TCD (Transient Concept Drift)
               - m > 1 → PCD (Progressive Concept Drift)
        
        Returns:
            (drift_length, category)
        """
        n = len(X)
        s = self.config.window_size
        n_adj = self.config.adjoint_windows
        delta = self.config.stability_threshold
        patience = self.config.stability_patience
        
        # Reference window before drift
        ref_start = max(0, drift_position - s)
        X_ref = X[ref_start:drift_position]
        y_ref = y[ref_start:drift_position]
        
        if len(X_ref) < s // 2:
            return 1, "TCD"
        
        # Start growing from drift position
        m = 0
        stable_count = 0
        t_end = drift_position + s
        
        while t_end <= n and stable_count < patience:
            m += 1
            
            # Compute accuracy in current window
            if t_end - s < drift_position:
                continue
                
            X_cur = X[t_end - s:t_end]
            y_cur = y[t_end - s:t_end]
            
            # Compute accuracy relative to reference
            acc = self._compute_accuracy(X_ref, y_ref, X_cur, y_cur)
            
            # Simple stability check: does accuracy stabilize?
            if m > 1:
                # Check if accuracy has stabilized (close to previous)
                if hasattr(self, '_prev_acc'):
                    if abs(acc - self._prev_acc) < delta:
                        stable_count += 1
                    else:
                        stable_count = 0
                self._prev_acc = acc
            else:
                self._prev_acc = acc
            
            t_end += self.config.step_size
        
        # Determine category
        if m <= 1:
            category = "TCD"
        else:
            category = "PCD"
        
        return m, category
    
    # =========================================================================
    # PHASE 3: TRACKING PROCESS
    # =========================================================================
    
    def tracking_process(self, X: np.ndarray, y: np.ndarray,
                         drift_position: int, category: str) -> Tuple[str, List[float]]:
        """
        Tracking Process: Identify drift subcategory.
        
        Algorithm:
            1. Build Tracking Flow Ratio (TFR) curve
            2. Analyze TFR curve pattern
            3. Classify subcategory based on pattern
        
        Returns:
            (subcategory, tfr_curve)
        """
        n = len(X)
        s = self.config.window_size
        k = self.config.tracking_steps
        step = self.config.step_size
        
        # Composite windows
        ref_start = max(0, drift_position - s)
        X_A_composite = X[ref_start:drift_position]
        y_A_composite = y[ref_start:drift_position]
        
        if len(X_A_composite) < s // 2:
            return "undetermined", []
        
        # Build TFR curve
        tfr_curve = []
        t = drift_position
        
        for i in range(k):
            if t + s > n:
                break
                
            X_B = X[t:t + s]
            y_B = y[t:t + s]
            
            # Compute tracking flow ratio
            # P̃ᵢtra = acc(model_A, on B) / acc(model_A, on A)
            try:
                model = self._create_learner()
                model.fit(X_A_composite, y_A_composite)
                
                acc_on_A = model.score(X_A_composite, y_A_composite)
                acc_on_B = model.score(X_B, y_B)
                
                if acc_on_A > 0.1:
                    P_tra = acc_on_B / acc_on_A
                else:
                    P_tra = 1.0
            except Exception:
                P_tra = 1.0
            
            tfr_curve.append(P_tra)
            
            # Slide composite window forward
            slide_end = min(t + step, n - s)
            if slide_end > ref_start:
                X_A_composite = X[ref_start:slide_end]
                y_A_composite = y[ref_start:slide_end]
            
            t += step
        
        # Classify subcategory based on TFR curve
        subcategory = self._classify_tfr_curve(tfr_curve, category)
        
        return subcategory, tfr_curve
    
    def _classify_tfr_curve(self, tfr_curve: List[float], category: str) -> str:
        """
        Classify subcategory based on TFR curve pattern.
        
        TCD patterns:
            - Sudden: Sharp single drop, stays low
            - Blip: Brief drop, returns to high
            - Recurrent: Periodic pattern
        
        PCD patterns:
            - Gradual: Oscillating, gradual decrease
            - Incremental: Monotonic gradual decrease
        """
        if len(tfr_curve) < 2:
            return "undetermined"
        
        arr = np.array(tfr_curve)
        
        # Statistics
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # Trend analysis
        diffs = np.diff(arr)
        sign_changes = np.sum((diffs[1:] * diffs[:-1]) < 0) if len(diffs) > 1 else 0
        
        # Monotonicity check
        is_decreasing = np.all(diffs <= 0.1)
        
        if category == "TCD":
            # Check for blip: returns to high after drop
            if min_val < 0.5 and arr[-1] > 0.7:
                return "blip"
            
            # Check for recurrent: periodic pattern
            if sign_changes >= 3:
                return "recurrent"
            
            # Default TCD: sudden
            return "sudden"
        
        else:  # PCD
            # Check for incremental: gradual monotonic
            if is_decreasing and std_val < 0.2:
                return "incremental"
            
            # Default PCD: gradual
            return "gradual"
    
    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================
    
    def classify(self, X: np.ndarray, y: np.ndarray) -> CDT_MSW_Result:
        """
        Main entry point: Classify drift type in data stream.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        
        Returns:
            CDT_MSW_Result with classification details
        """
        # Phase 1: Detection
        detected, position, det_ratio = self.detection_process(X, y)
        
        if not detected:
            return CDT_MSW_Result(
                drift_detected=False,
                drift_position=len(X),
                drift_length=0,
                category="none",
                subcategory="none",
                detection_ratio=det_ratio,
                tfr_curve=[],
                note="No drift detected"
            )
        
        # Phase 2: Growth
        length, category = self.growth_process(X, y, position)
        
        # Phase 3: Tracking
        subcategory, tfr_curve = self.tracking_process(X, y, position, category)
        
        return CDT_MSW_Result(
            drift_detected=True,
            drift_position=position,
            drift_length=length,
            category=category,
            subcategory=subcategory,
            detection_ratio=det_ratio,
            tfr_curve=tfr_curve,
            note=f"Detected at t={position}, length={length}"
        )


# =============================================================================
# SYNTHETIC DATA GENERATORS (Following paper's experimental setup)
# =============================================================================

def generate_sudden_drift(n_samples: int = 10000, n_features: int = 10,
                          n_drifts: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[int], str, str]:
    """
    Generate sudden drift dataset.
    Distribution changes abruptly at drift points.
    """
    np.random.seed(seed)
    
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    segment_len = n_samples // (n_drifts + 1)
    drift_points = []
    
    for i in range(n_drifts + 1):
        start = i * segment_len
        end = min((i + 1) * segment_len, n_samples)
        
        # Different concept for each segment
        if i % 2 == 0:
            # Concept A: class 0 = negative features, class 1 = positive
            X[start:end] = np.random.randn(end - start, n_features)
            y[start:end] = (X[start:end, 0] > 0).astype(int)
        else:
            # Concept B: inverted decision boundary
            X[start:end] = np.random.randn(end - start, n_features)
            y[start:end] = (X[start:end, 0] < 0).astype(int)
        
        if i > 0:
            drift_points.append(start)
    
    return X, y, drift_points, "TCD", "sudden"


def generate_gradual_drift(n_samples: int = 10000, n_features: int = 10,
                           n_drifts: int = 3, transition_width: int = 1000,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[int], str, str]:
    """
    Generate gradual drift dataset.
    Concepts mix probabilistically during transition.
    """
    np.random.seed(seed)
    
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    segment_len = n_samples // (n_drifts + 1)
    drift_points = []
    
    current_concept = 0
    
    for t in range(n_samples):
        segment_idx = t // segment_len
        pos_in_seg = t % segment_len
        
        # Generate sample
        X[t] = np.random.randn(n_features)
        
        # Determine concept with probabilistic mixing during transition
        if pos_in_seg < transition_width and segment_idx > 0:
            # During transition - probabilistic mixing
            alpha = pos_in_seg / transition_width
            if np.random.random() < alpha:
                concept = segment_idx % 2
            else:
                concept = (segment_idx - 1) % 2
        else:
            concept = segment_idx % 2
        
        # Generate label based on concept
        if concept == 0:
            y[t] = int(X[t, 0] > 0)
        else:
            y[t] = int(X[t, 0] < 0)
        
        if t > 0 and t == segment_idx * segment_len:
            drift_points.append(t)
    
    return X, y, drift_points, "PCD", "gradual"


def generate_incremental_drift(n_samples: int = 10000, n_features: int = 10,
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[int], str, str]:
    """
    Generate incremental drift dataset.
    Decision boundary shifts linearly over time.
    """
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    
    # Linear shift in decision boundary
    boundary_shift = np.linspace(-1, 1, n_samples)
    y = (X[:, 0] > boundary_shift).astype(int)
    
    # No discrete drift points
    return X, y, [], "PCD", "incremental"


def generate_recurrent_drift(n_samples: int = 10000, n_features: int = 10,
                             period: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[int], str, str]:
    """
    Generate recurrent drift dataset.
    Concepts alternate cyclically.
    """
    np.random.seed(seed)
    
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    drift_points = []
    
    for t in range(n_samples):
        X[t] = np.random.randn(n_features)
        
        concept = (t // period) % 2
        
        if concept == 0:
            y[t] = int(X[t, 0] > 0)
        else:
            y[t] = int(X[t, 0] < 0)
        
        if t > 0 and t % period == 0:
            drift_points.append(t)
    
    return X, y, drift_points, "PCD", "recurrent"


def generate_blip_drift(n_samples: int = 10000, n_features: int = 10,
                        n_blips: int = 5, blip_duration: int = 100,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[int], str, str]:
    """
    Generate blip drift dataset.
    Temporary outlier concept returns to original.
    """
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] > 0).astype(int)  # Base concept
    
    drift_points = []
    
    for i in range(n_blips):
        blip_start = (i + 1) * n_samples // (n_blips + 1)
        blip_end = min(blip_start + blip_duration, n_samples)
        
        # Inverted concept during blip
        y[blip_start:blip_end] = (X[blip_start:blip_end, 0] < 0).astype(int)
        drift_points.append(blip_start)
    
    return X, y, drift_points, "TCD", "blip"


# =============================================================================
# BENCHMARK
# =============================================================================

def run_cdt_msw_benchmark():
    """Run comprehensive CDT_MSW benchmark."""
    print("=" * 80)
    print("CDT_MSW BENCHMARK (Paper Reproduction)")
    print("Guo et al. (2022): Concept drift type identification based on multi-sliding windows")
    print("=" * 80)
    
    generators = [
        ("sudden", generate_sudden_drift),
        ("gradual", generate_gradual_drift),
        ("incremental", generate_incremental_drift),
        ("recurrent", generate_recurrent_drift),
        ("blip", generate_blip_drift),
    ]
    
    learners = ['logistic', 'perceptron']
    n_runs = 3
    
    results = {}
    
    for learner in learners:
        print(f"\n{'='*60}")
        print(f"Learner: {learner.upper()}")
        print(f"{'='*60}")
        
        config = CDT_MSW_Config(
            window_size=200,
            step_size=50,
            detection_threshold=0.85,
            learner=learner
        )
        
        cdt_msw = CDT_MSW(config)
        
        for dtype, gen_fn in generators:
            print(f"\n{dtype.upper()}:", end=" ")
            
            type_correct = 0
            cat_correct = 0
            total = 0
            
            for run in range(n_runs):
                X, y, drift_points, true_cat, true_type = gen_fn(seed=run * 100)
                
                result = cdt_msw.classify(X, y)
                
                if result.drift_detected:
                    if result.subcategory == true_type:
                        type_correct += 1
                    if result.category == true_cat:
                        cat_correct += 1
                    total += 1
                
                print(".", end="", flush=True)
            
            if total > 0:
                type_acc = type_correct / total * 100
                cat_acc = cat_correct / total * 100
            else:
                type_acc = 0
                cat_acc = 0
            
            results[(learner, dtype)] = {
                'type_acc': type_acc,
                'cat_acc': cat_acc,
                'total': total
            }
            
            print(f" TypeAcc={type_acc:.1f}%, CatAcc={cat_acc:.1f}%")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"\n{'Learner':<15} {'Type Accuracy':<15} {'Category Accuracy':<20}")
    print("-" * 50)
    
    for learner in learners:
        type_accs = [results[(learner, d)]['type_acc'] for d, _ in generators]
        cat_accs = [results[(learner, d)]['cat_acc'] for d, _ in generators]
        
        avg_type = np.mean(type_accs)
        avg_cat = np.mean(cat_accs)
        
        print(f"{learner:<15} {avg_type:>12.1f}%    {avg_cat:>17.1f}%")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_cdt_msw_benchmark()
