import numpy as np

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

def generate_mixed_drift_dataset(n_samples=6000, random_seed=42):
    """
    Generate a dataset with mixed drift types for comprehensive evaluation.
    1. Sudden (SEA)
    2. Gradual (Rot. Hyperplane)
    3. Recurrent (ABBA)
    """
    np.random.seed(random_seed)
    X = []
    y = []
    drift_points = []
    drift_types = []
    
    segment_len = 1500
    
    # Segment 1: Base Concept (SEA theta=8)
    X1, y1, _, _ = generate_sea_concepts(segment_len, n_drifts=0, random_seed=random_seed)
    X.append(X1); y.append(y1)
    
    # Drift 1: Sudden (SEA theta=8 -> 5)
    drift_points.append(segment_len)
    drift_types.append('sudden')
    
    # Segment 2: New Concept (SEA theta=5)
    X2 = np.random.rand(segment_len, 3) * 10
    y2 = np.array([1 if (x[0]+x[1] <= 5.0) else 0 for x in X2])
    X.append(X2); y.append(y2)
    
    # Drift 2: Gradual (Transition to Hyperplane)
    drift_points.append(segment_len * 2)
    drift_types.append('gradual')
    
    # Segment 3: Gradual Transition
    # Use Interleaved chunks to simulate gradual
    X3 = np.random.rand(segment_len, 3) * 10
    y3 = np.zeros(segment_len, dtype=int)
    
    # Concept A (SEA 5) vs Concept B (Hyperplane)
    w_hyper = np.array([0.5, 0.5, -0.5])
    
    for i in range(segment_len):
        # Probability of B increases from 0 to 1 (Sigmoid-like)
        prob_b = 1 / (1 + np.exp(-10 * (i/segment_len - 0.5)))
        
        if np.random.rand() < prob_b:
            # Concept B
            y3[i] = 1 if np.dot(X3[i], w_hyper) > 0 else 0
        else:
            # Concept A
            y3[i] = 1 if (X3[i,0]+X3[i,1] <= 5.0) else 0
            
    X.append(X3); y.append(y3)
    
    # Drift 3: Recurrent (Back to SEA theta=8)
    drift_points.append(segment_len * 3)
    drift_types.append('recurrent')
    
    # Segment 4: Back to Start
    X4, y4, _, _ = generate_sea_concepts(segment_len, n_drifts=0, random_seed=random_seed+1)
    # Ensure exact same concept as Seg 1 (theta=8)
    # X4 is new data, but logic is same
    y4 = np.array([1 if (x[0]+x[1] <= 8.0) else 0 for x in X4])
    
    X.append(X4); y.append(y4)
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    return X, y, drift_points, drift_types
