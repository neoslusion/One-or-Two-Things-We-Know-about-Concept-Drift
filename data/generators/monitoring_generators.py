import numpy as np
from typing import List, Tuple, Optional


def generate_joint_drift_stream(
    n_samples: int = 5000,
    n_drifts: int = 5,
    n_features: int = 5,
    drift_type: str = "sudden",
    drift_magnitude: float = 3.0,
    noise: float = 0.05,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Generate data stream with JOINT drift: P(X) shift + P(Y|X) rotation.

    Academic Approach (Approach B):
        - P(X) drift: Features drawn from N(μ_k, I) where μ_k shifts at each
          drift point. This creates covariate shift detectable by unsupervised
          methods (MMD, SE-CDT).
        - P(Y|X) drift: Each concept k has a different hyperplane w_k.
          Labels are y = sign(w_k · (X - μ_k)). Since (X - μ_k) is centered,
          class balance is always ~50/50 regardless of drift magnitude.

    This satisfies BOTH requirements for the SE-CDT adaptation evaluation:
        1. SE-CDT detects drift via P(X) change (MMD signal)
        2. Model accuracy drops due to P(Y|X) change (boundary rotation)
        3. Retraining helps because the new model learns the new boundary

    Drift types supported:
        - "sudden": Abrupt μ shift (alternating ±magnitude)
        - "stepping": Cumulative μ shift (staircase)
        - "gradual": Smooth μ interpolation over transition window
        - "incremental": Continuous linear μ shift
        - "recurrent": Alternating between concepts (A→B→A→B)

    References:
        - Lu et al. (2018): Joint drift = P(X) + P(Y|X) change
        - Gama et al. (2014): Concept drift taxonomy
        - Hinder et al. (2024): One or Two Things We Know about Concept Drift

    Args:
        n_samples: Total stream length
        n_drifts: Number of drift events
        n_features: Feature dimensionality
        drift_type: Type of drift ("sudden", "stepping", "gradual",
                    "incremental", "recurrent")
        drift_magnitude: Mean shift magnitude (controls P(X) drift intensity)
        noise: Label noise probability (default 5%)
        random_seed: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels
        drift_points: List of drift positions
        drift_types: List of drift type strings
    """
    rng = np.random.RandomState(random_seed)

    segment_len = n_samples // (n_drifts + 1)
    drift_points = [(i + 1) * segment_len for i in range(n_drifts)]

    # --- Step 1: Generate concept means (controls P(X) drift) ---
    # Each concept k has mean μ_k. The shift pattern depends on drift_type.
    n_shift = max(2, n_features // 2)  # Shift first half of features

    # Pre-compute means for each segment
    segment_means = [np.zeros(n_features)]  # Segment 0: origin

    for k in range(1, n_drifts + 1):
        prev_mean = segment_means[-1].copy()

        if drift_type == "sudden":
            # Alternating: +magnitude, -magnitude, +magnitude, ...
            new_mean = np.zeros(n_features)
            if k % 2 == 1:
                new_mean[:n_shift] = drift_magnitude
            # else: back to origin
            segment_means.append(new_mean)

        elif drift_type == "stepping":
            # Cumulative: each drift adds +magnitude
            new_mean = prev_mean.copy()
            new_mean[:n_shift] += drift_magnitude
            segment_means.append(new_mean)

        elif drift_type in ("gradual", "incremental"):
            # Same as sudden for final state, transition handled below
            new_mean = np.zeros(n_features)
            if k % 2 == 1:
                new_mean[:n_shift] = drift_magnitude
            segment_means.append(new_mean)

        elif drift_type == "recurrent":
            # Alternating: same as sudden
            new_mean = np.zeros(n_features)
            if k % 2 == 1:
                new_mean[:n_shift] = drift_magnitude
            segment_means.append(new_mean)

        else:
            # Default: sudden behavior
            new_mean = np.zeros(n_features)
            if k % 2 == 1:
                new_mean[:n_shift] = drift_magnitude
            segment_means.append(new_mean)

    # --- Step 2: Generate hyperplane weights (controls P(Y|X) drift) ---
    # Each concept has a different decision boundary direction.
    # Use orthogonal-ish rotations for maximum class accuracy drop.
    concept_weights = []
    for k in range(n_drifts + 1):
        # Generate a deterministic but distinct weight vector per concept
        w_rng = np.random.RandomState(random_seed + k * 137)
        w = w_rng.randn(n_features)
        w = w / np.linalg.norm(w)  # Normalize
        concept_weights.append(w)

    # --- Step 3: Generate feature stream X ---
    X = np.empty((n_samples, n_features))
    concept_id = np.zeros(n_samples, dtype=int)

    all_boundaries = list(zip([0] + drift_points, drift_points + [n_samples]))

    for seg_idx, (seg_start, seg_end) in enumerate(all_boundaries):
        seg_len = seg_end - seg_start
        mean_k = segment_means[seg_idx]

        if drift_type == "gradual" and seg_idx > 0:
            # Gradual: smooth transition over first 1/3 of segment
            transition_width = min(500, seg_len // 3)
            prev_mean = segment_means[seg_idx - 1]

            for t in range(seg_len):
                idx = seg_start + t
                if t < transition_width:
                    alpha = t / transition_width
                    # Probabilistic mixture (true gradual drift definition)
                    if rng.random() < alpha:
                        X[idx] = rng.randn(n_features) + mean_k
                        concept_id[idx] = seg_idx
                    else:
                        X[idx] = rng.randn(n_features) + prev_mean
                        concept_id[idx] = seg_idx - 1
                else:
                    X[idx] = rng.randn(n_features) + mean_k
                    concept_id[idx] = seg_idx

        elif drift_type == "incremental" and seg_idx > 0:
            # Incremental: continuous linear interpolation of mean
            transition_width = min(seg_len // 2, 500)
            prev_mean = segment_means[seg_idx - 1]

            for t in range(seg_len):
                idx = seg_start + t
                if t < transition_width:
                    progress = t / transition_width
                    current_mean = (1 - progress) * prev_mean + progress * mean_k
                    X[idx] = rng.randn(n_features) + current_mean
                    concept_id[idx] = seg_idx
                else:
                    X[idx] = rng.randn(n_features) + mean_k
                    concept_id[idx] = seg_idx
        else:
            # Sudden / Stepping / Recurrent: instant switch
            X[seg_start:seg_end] = rng.randn(seg_len, n_features) + mean_k
            concept_id[seg_start:seg_end] = seg_idx

    # --- Step 4: Generate labels using concept-relative hyperplane ---
    # y = sign(w_k · (X - μ_k))
    # This ensures ~50/50 balance because (X - μ_k) ~ N(0, I)
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        k = concept_id[i]
        x_centered = X[i] - segment_means[k]
        decision = np.dot(concept_weights[k], x_centered)
        y[i] = 1 if decision > 0 else 0

    # --- Step 5: Add label noise ---
    noise_mask = rng.random(n_samples) < noise
    y[noise_mask] = 1 - y[noise_mask]

    # Determine drift type labels for output
    if drift_type == "stepping":
        type_labels = ["sudden"] * n_drifts  # Stepping = cumulative sudden
    elif drift_type == "recurrent":
        type_labels = ["recurrent"] * n_drifts
    else:
        type_labels = [drift_type] * n_drifts

    return X, y, drift_points, type_labels

def generate_sea_concepts(n_samples=5000, n_drifts=3, random_seed=42, noise=0.1):
    """
    Generate SEA Concepts stream (Street and Kim, 2001).
    Standard benchmark for abrupt drift.
    
    Features: 3 (only first 2 relevant). Range [0, 10].
    Concept: x1 + x2 <= theta
    Drift: theta changes (e.g., 8 -> 9 -> 7).
    """
    np.random.seed(random_seed)
    
    X = np.random.rand(n_samples, 3) * 10
    y = np.zeros(n_samples, dtype=int)
    
    # 4 concepts (thresholds)
    thresholds = [8.0, 9.0, 7.0, 9.5]
    
    # Noise injection
    noise_mask = np.random.rand(n_samples) < noise
    
    drift_points = []
    segment_len = n_samples // (n_drifts + 1)
    
    for i in range(n_samples):
        segment = min(i // segment_len, len(thresholds) - 1)
        theta = thresholds[segment]
        
        # SEA logic: x1 + x2 <= theta
        label = 1 if (X[i, 0] + X[i, 1] <= theta) else 0
        
        if noise_mask[i]:
            label = 1 - label
            
        y[i] = label
        
        # Track drift points
        if i > 0 and i % segment_len == 0 and i // segment_len <= n_drifts:
            drift_points.append(i)
            
    return X, y, drift_points, ['sudden'] * len(drift_points)


def generate_rotating_hyperplane(n_samples=5000, n_drifts=3, n_features=5, 
                                 mag_change=0.1, drift_type='gradual', 
                                 random_seed=42):
    """
    Generate Rotating Hyperplane stream (Hulten et al., 2001).
    Standard benchmark for gradual/incremental drift.
    
    Concept: Hyperplane in d-dimensions rotates/moves continuously.
    """
    np.random.seed(random_seed)
    
    X = np.random.rand(n_samples, n_features)
    y = np.zeros(n_samples, dtype=int)
    
    # Initial weights
    weights = np.random.rand(n_features) - 0.5
    
    drift_points = []
    segment_len = n_samples // (n_drifts + 1)
    
    # Velocity for rotation
    velocity = np.zeros(n_features)
    
    # State tracking
    is_drifting = False
    drift_start = 0
    transition_window = 500  # Samples for gradual transition
    
    for i in range(n_samples):
        # Determine segment
        segment = i // segment_len
        
        # Drift logic
        if i > 0 and i % segment_len == 0:
            if segment <= n_drifts:
                drift_points.append(i)
                # Change rotation direction/velocity at drift points
                velocity = (np.random.rand(n_features) - 0.5) * mag_change
                is_drifting = True
                drift_start = i
        
        # Apply drift based on type
        if is_drifting:
            if drift_type == 'sudden':
                # Apply change immediately (once per segment)
                if i == drift_start:
                    weights += velocity * 10 # Big jump
                    is_drifting = False
            
            elif drift_type == 'incremental':
                # Constant continuous change
                weights += velocity
                # No stopping, continuous drift
                
            elif drift_type == 'gradual':
                # Weights change, but we mix old and new concept?
                # Actually hyperplane rotation IS incremental by definition.
                # Standard "Gradual" usually means mixing two distributions.
                # Here we'll simulate 'Gradual' as a fast rotation over a window
                if i < drift_start + transition_window:
                    weights += velocity
                else:
                    is_drifting = False
        
        # Generate label
        # Normalize weights to keep decision boundary meaningful
        # weights = weights / np.linalg.norm(weights) 
        
        decision = np.dot(X[i], weights)
        y[i] = 1 if decision > 0 else 0
        
        # Noise
        if np.random.rand() < 0.05:
            y[i] = 1 - y[i]
            
    return X, y, drift_points, [drift_type] * len(drift_points)

def generate_mixed_drift_dataset(n_samples=6000, random_seed=42, n_features=5, drift_magnitude=2.0):
    """
    Generate a dataset with mixed drift types for comprehensive evaluation.
    
    CRITICAL: For unsupervised drift detection (SE-CDT), we must change
    FEATURE DISTRIBUTIONS (X), not just decision boundaries (y).
    
    Drift types:
    1. Sudden: Abrupt mean shift in X
    2. Gradual: Smooth transition in X over a window
    3. Recurrent: Return to a previous X distribution
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples (will be slightly adjusted for segments)
    random_seed : int
        Random seed for reproducibility
    n_features : int
        Number of features in X
    drift_magnitude : float
        Base magnitude of drift (mean shift). Higher = easier to detect.
        
    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    drift_points : list of int
    drift_types : list of str
    """
    np.random.seed(random_seed)
    
    segment_len = n_samples // 4
    total_samples = segment_len * 4
    
    X = np.zeros((total_samples, n_features))
    y = np.zeros(total_samples, dtype=int)
    drift_points = []
    drift_types = []
    
    # Define base distributions (means for each concept)
    mean_A = np.zeros(n_features)  # Concept A: centered at origin
    mean_B = np.ones(n_features) * drift_magnitude  # Concept B: shifted
    mean_C = np.array([drift_magnitude if i % 2 == 0 else -drift_magnitude 
                       for i in range(n_features)])  # Concept C: alternating
    
    # Decision boundaries for labels (separate from X distribution)
    def label_concept_A(x):
        return int(np.sum(x[:2]) > 0)
    
    def label_concept_B(x):
        return int(x[0] > drift_magnitude / 2)
    
    def label_concept_C(x):
        return int(np.dot(x, np.ones(n_features)) > 0)
    
    # ============================================
    # Segment 1: Base Concept A (0 to segment_len)
    # ============================================
    X[:segment_len] = np.random.randn(segment_len, n_features) + mean_A
    for i in range(segment_len):
        y[i] = label_concept_A(X[i])
    
    # ============================================
    # Drift 1: SUDDEN (at segment_len)
    # Abrupt change from Concept A to Concept B
    # ============================================
    drift_points.append(segment_len)
    drift_types.append('sudden')
    
    # Segment 2: Concept B
    start_2 = segment_len
    end_2 = segment_len * 2
    X[start_2:end_2] = np.random.randn(segment_len, n_features) + mean_B
    for i in range(start_2, end_2):
        y[i] = label_concept_B(X[i])
    
    # ============================================
    # Drift 2: GRADUAL (at segment_len * 2)
    # Smooth transition from B to C over 500 samples
    # ============================================
    drift_points.append(segment_len * 2)
    drift_types.append('gradual')
    
    # Segment 3: Gradual transition B -> C
    start_3 = segment_len * 2
    end_3 = segment_len * 3
    transition_window = min(500, segment_len // 3)
    
    for i in range(segment_len):
        idx = start_3 + i
        
        if i < transition_window:
            # Gradual transition: interpolate between B and C
            alpha = i / transition_window  # 0 -> 1
            current_mean = (1 - alpha) * mean_B + alpha * mean_C
            X[idx] = np.random.randn(n_features) + current_mean
            
            # Labels: probabilistic mix
            if np.random.rand() < alpha:
                y[idx] = label_concept_C(X[idx])
            else:
                y[idx] = label_concept_B(X[idx])
        else:
            # Post-transition: pure Concept C
            X[idx] = np.random.randn(n_features) + mean_C
            y[idx] = label_concept_C(X[idx])
    
    # ============================================
    # Drift 3: RECURRENT (at segment_len * 3)
    # Return to Concept A (same distribution as start)
    # ============================================
    drift_points.append(segment_len * 3)
    drift_types.append('recurrent')
    
    # Segment 4: Back to Concept A
    start_4 = segment_len * 3
    end_4 = segment_len * 4
    X[start_4:end_4] = np.random.randn(segment_len, n_features) + mean_A
    for i in range(start_4, end_4):
        y[i] = label_concept_A(X[i])
    
    # Add small noise to avoid perfectly separable data
    noise_mask = np.random.rand(total_samples) < 0.05
    y[noise_mask] = 1 - y[noise_mask]
    
    return X, y, drift_points, drift_types
