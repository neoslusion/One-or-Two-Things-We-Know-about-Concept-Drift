"""
Dataset generation functions for drift detection benchmarks.

Contains generators for various synthetic and semi-real datasets including:
- Sudden drift: SEA, STAGGER, Hyperplane, gen_random
- Gradual drift: SEA gradual, Hyperplane gradual, Agrawal gradual, Circles gradual
- Incremental drift: RBF with moving centroids
- Real-world: Electricity, Covertype (sorted versions)
- Noise robustness: Sine family with irrelevant features
"""

import numpy as np
from river.datasets import synth

# Import gen_random from backup module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backup')))
from gen_data import gen_random


# ============================================================================
# SUDDEN DRIFT DATASET GENERATORS
# ============================================================================

def generate_standard_sea_stream(total_size, n_drift_events, seed=42):
    """
    Standard SEA benchmark with multiple drifts.
    Creates sudden drifts by switching between SEA variants.
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    variants = [0, 1, 2, 3]  # SEA has 4 variants
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    segments = [0] + drift_positions + [total_size]

    for seg_idx in range(len(segments) - 1):
        start, end = segments[seg_idx], segments[seg_idx + 1]
        size = end - start
        variant = variants[seg_idx % len(variants)]

        stream = synth.SEA(seed=seed + seg_idx * 100, variant=variant)
        for i, (x, y) in enumerate(stream.take(size)):
            X_list.append(list(x.values()))
            y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, drift_positions


def generate_enhanced_sea_stream(total_size, n_drift_events, seed=42,
                                  scale_factors=(1.8, 1.5, 2.0),
                                  shift_amounts=(5.0, 4.0, 8.0)):
    """Enhanced SEA with multiple drifts and transformations."""
    np.random.seed(seed)

    X_list, y_list = [], []

    variants = [0, 1, 2, 3]
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    segments = [0] + drift_positions + [total_size]

    for seg_idx in range(len(segments) - 1):
        start, end = segments[seg_idx], segments[seg_idx + 1]
        size = end - start
        variant = variants[seg_idx % len(variants)]

        stream = synth.SEA(seed=seed + seg_idx * 100, variant=variant)

        for i, (x, y) in enumerate(stream.take(size)):
            x_vals = list(x.values())

            # Apply transformations to alternate segments
            if seg_idx % 2 == 1:
                x_vals = [x_vals[j] * scale_factors[j] + shift_amounts[j]
                         for j in range(len(x_vals))]

            X_list.append(x_vals)
            y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, drift_positions


def generate_stagger_stream(total_size, n_drift_events, seed=42):
    """STAGGER concepts with multiple sudden drifts."""
    np.random.seed(seed)

    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    segments = [0] + drift_positions + [total_size]

    X_segments, y_segments = [], []

    for seg_idx in range(len(segments) - 1):
        start, end = segments[seg_idx], segments[seg_idx + 1]
        size = end - start

        X_seg = np.random.randn(size, 5)

        # Different concepts for each segment
        if seg_idx % 3 == 0:
            X_seg[:, 0] += 2.0
            y_seg = (X_seg[:, 0] + X_seg[:, 1] > 1.5).astype(int)
        elif seg_idx % 3 == 1:
            X_seg[:, 0] -= 2.0
            y_seg = (X_seg[:, 0] * X_seg[:, 1] > 0).astype(int)
        else:
            X_seg[:, 1] += 1.5
            y_seg = (X_seg[:, 1] + X_seg[:, 2] > 0.5).astype(int)

        X_segments.append(X_seg)
        y_segments.append(y_seg)

    X = np.vstack(X_segments)
    y = np.hstack(y_segments)

    return X, y, drift_positions

def generate_hyperplane_stream(
    total_size: int,
    n_drift_events: int,
    seed: int = 42,
    n_features: int = 10,
    noise_percentage: float = 0.05,
    mag_change: float = 0.001
):
    """
    Generate Rotating Hyperplane stream (STANDARD LITERATURE BENCHMARK).
    
    This is the standard benchmark used in D3, DAWIDD, MMD, and other
    unsupervised drift detection papers. The hyperplane continuously
    rotates, creating gradual/incremental drift that changes P(X).
    
    References:
        - Hulten et al. (2001) "Mining time-changing data streams"
        - MOA framework standard configuration
        - Gözüaçık et al. (2019) D3 paper
        - Hinder et al. (2020) DAWIDD paper
    
    Parameters
    ----------
    total_size : int
        Total number of samples to generate.
    n_drift_events : int
        Number of drift markers (for evaluation, evenly spaced).
    seed : int, default=42
        Random seed for reproducibility.
    n_features : int, default=10
        Number of features (MOA standard: 10).
    noise_percentage : float, default=0.05
        Label noise probability.
    mag_change : float, default=0.001
        Magnitude of rotation per sample (MOA standard: 0.001).
        Higher = faster drift. Range: 0.0001 (slow) to 0.01 (fast).
    
    Returns
    -------
    X : np.ndarray of shape (total_size, n_features)
        Feature matrix.
    y : np.ndarray of shape (total_size,)
        Target labels.
    drift_positions : list of int
        Evenly spaced drift markers for evaluation.
    """
    X_list, y_list = [], []
    
    # Standard rotating hyperplane configuration (matches MOA/literature)
    stream = synth.Hyperplane(
        seed=seed,
        n_features=n_features,
        n_drift_features=n_features // 2,  # Half features drift (standard)
        mag_change=mag_change,              # Continuous rotation!
        sigma=0.1,                          # Direction change probability
        noise_percentage=noise_percentage
    )
    
    # Generate continuous stream
    for x, y in stream.take(total_size):
        X_list.append(list(x.values()))
        y_list.append(y)
    
    # Drift positions are evenly spaced (for evaluation purposes)
    # In rotating hyperplane, drift is continuous, but we mark evaluation points
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    
    return np.array(X_list), np.array(y_list), drift_positions

def generate_genrandom_stream(total_size, n_drift_events, seed=42,
                               dims=5, intens=0.125, dist="unif", alt=False):
    """Custom synthetic data using gen_random with multiple drifts."""
    np.random.seed(seed)

    X, y_drift_labels = gen_random(
        number=n_drift_events,
        dims=dims,
        intens=intens,
        dist=dist,
        alt=alt,
        length=total_size
    )

    # Find actual drift positions
    drift_indices = np.where(np.diff(y_drift_labels) != 0)[0] + 1
    drift_positions = drift_indices.tolist()

    # Generate synthetic binary classification labels
    # Use simple threshold on first feature
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)

    return X, y, drift_positions


# ============================================================================
# GRADUAL DRIFT DATASET GENERATORS
# ============================================================================

def generate_sea_gradual_stream(total_size, n_drift_events, seed=42, transition_width=1000):
    """
    SEA benchmark with GRADUAL drifts (smooth transitions between variants).

    Instead of instant variant switches, blends samples from old→new concept
    over transition_width samples.

    Args:
        transition_width: Number of samples for gradual transition (default: 1000)
                         During transition, samples are blended:
                         Start: 100% old, 0% new → End: 0% old, 100% new
    """
    np.random.seed(seed)

    X_list, y_list = [], []
    variants = [0, 1, 2, 3]

    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]

    # Generate segments with gradual transitions
    for seg_idx in range(n_drift_events + 1):
        old_variant = variants[seg_idx % len(variants)]
        new_variant = variants[(seg_idx + 1) % len(variants)]

        if seg_idx == 0:
            # First segment - no transition at start
            stream = synth.SEA(seed=seed + seg_idx * 100, variant=old_variant)
            for i, (x, y) in enumerate(stream.take(segment_size)):
                X_list.append(list(x.values()))
                y_list.append(y)
        else:
            # Gradual transition segment
            transition_start = len(X_list)

            # Generate samples from both concepts
            stream_old = synth.SEA(seed=seed + seg_idx * 100, variant=old_variant)
            stream_new = synth.SEA(seed=seed + (seg_idx + 1) * 100, variant=new_variant)

            samples_old = list(stream_old.take(segment_size))
            samples_new = list(stream_new.take(segment_size))

            for i in range(segment_size):
                # Calculate blend ratio (linear interpolation)
                if i < transition_width:
                    # Gradual transition: old → new
                    alpha = i / transition_width  # 0 → 1

                    x_old = np.array(list(samples_old[i][0].values()))
                    x_new = np.array(list(samples_new[i][0].values()))
                    y_old = samples_old[i][1]
                    y_new = samples_new[i][1]

                    # Blend features
                    x_blended = (1 - alpha) * x_old + alpha * x_new

                    # Blend labels probabilistically
                    if np.random.rand() < alpha:
                        y_blended = y_new
                    else:
                        y_blended = y_old

                    X_list.append(x_blended.tolist())
                    y_list.append(y_blended)
                else:
                    # After transition - pure new concept
                    X_list.append(list(samples_new[i][0].values()))
                    y_list.append(samples_new[i][1])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, drift_positions


def generate_hyperplane_gradual_stream(total_size, n_drift_events, seed=42, n_features=10):
    """
    Rotating Hyperplane with CONTINUOUS gradual drift.

    Uses very small mag_change to create smooth, continuous rotation
    instead of sudden jumps between segments.
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    # Use VERY SMALL mag_change for gradual rotation
    # This creates continuous drift throughout the stream
    stream = synth.Hyperplane(
        seed=seed,
        n_features=n_features,
        n_drift_features=5,  # More features drifting for observable change
        mag_change=0.0001,   # VERY small = gradual
        sigma=0.1,           # Small noise
        noise_percentage=0.05
    )

    # Generate full stream (drift happens continuously)
    for i, (x, y) in enumerate(stream.take(total_size)):
        X_list.append(list(x.values()))
        y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    # Estimate drift positions (evenly spaced)
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]

    return X, y, drift_positions


def generate_agrawal_gradual_stream(total_size, n_drift_events, seed=42, transition_width=1000):
    """
    Agrawal generator with GRADUAL transitions between classification functions.

    Agrawal has 10 different classification functions. We gradually blend
    between them over transition_width samples.
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]

    # Cycle through classification functions (0-9)
    for seg_idx in range(n_drift_events + 1):
        old_func = seg_idx % 10
        new_func = (seg_idx + 1) % 10

        if seg_idx == 0:
            # First segment - no transition
            stream = synth.Agrawal(seed=seed + seg_idx * 100, classification_function=old_func)
            for i, (x, y) in enumerate(stream.take(segment_size)):
                X_list.append(list(x.values()))
                y_list.append(y)
        else:
            # Gradual transition segment
            stream_old = synth.Agrawal(seed=seed + seg_idx * 100, classification_function=old_func)
            stream_new = synth.Agrawal(seed=seed + (seg_idx + 1) * 100, classification_function=new_func)

            samples_old = list(stream_old.take(segment_size))
            samples_new = list(stream_new.take(segment_size))

            for i in range(segment_size):
                if i < transition_width:
                    # Gradual transition via probabilistic label selection
                    alpha = i / transition_width

                    # Use old sample's features but blend labels probabilistically
                    x_features = list(samples_old[i][0].values())
                    y_old = samples_old[i][1]
                    y_new = samples_new[i][1]

                    # Blend labels probabilistically
                    if np.random.rand() < alpha:
                        y_blended = y_new
                    else:
                        y_blended = y_old

                    X_list.append(x_features)
                    y_list.append(y_blended)
                else:
                    # Pure new concept
                    X_list.append(list(samples_new[i][0].values()))
                    y_list.append(samples_new[i][1])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, drift_positions


def generate_circles_gradual_stream(total_size, n_drift_events, seed=42, transition_width=500):
    """
    Circles dataset with GRADUAL drifts (circles move/resize smoothly).

    Classic synthetic benchmark: 2D data with circular decision boundaries
    that gradually move and resize over transition windows.
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]

    # Define circle configurations (center_x, center_y, radius)
    circles = [
        (0.25, 0.25, 0.15),  # Circle 1
        (0.75, 0.25, 0.15),  # Circle 2
        (0.25, 0.75, 0.15),  # Circle 3
        (0.75, 0.75, 0.15),  # Circle 4
    ]

    for seg_idx in range(n_drift_events + 1):
        # Get current and next circle configuration
        old_circle = circles[seg_idx % len(circles)]
        new_circle = circles[(seg_idx + 1) % len(circles)]

        for i in range(segment_size):
            # Generate random point in [0, 1] × [0, 1]
            x = np.random.rand()
            y = np.random.rand()

            if seg_idx == 0 or i >= transition_width:
                # Use current circle (before transition or after transition complete)
                if seg_idx == 0:
                    cx, cy, r = old_circle
                else:
                    cx, cy, r = new_circle
            else:
                # Gradual transition - interpolate circle parameters
                alpha = i / transition_width

                cx = (1 - alpha) * old_circle[0] + alpha * new_circle[0]
                cy = (1 - alpha) * old_circle[1] + alpha * new_circle[1]
                r = (1 - alpha) * old_circle[2] + alpha * new_circle[2]

            # Classification: inside circle = class 1, outside = class 0
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            label = 1 if distance <= r else 0

            X_list.append([x, y])
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, drift_positions


# ============================================================================
# INCREMENTAL DRIFT DATASET GENERATORS (MOA/River Standards)
# ============================================================================

def generate_rbf_stream(total_size, n_drift_events, seed=42, n_centroids=50, speed=0.0001):
    """
    Random RBF with moving centroids (INCREMENTAL/CONTINUOUS drift).

    STANDARD CONFIGURATION (following MOA papers):
    - 50 centroids (standard benchmark setting)
    - 10 features
    - 2 classes
    - Speed: 0.0001 (slow), 0.001 (moderate), 0.01 (fast)
    - All centroids drift continuously

    This simulates INCREMENTAL drift where cluster boundaries move
    continuously over time (different from sudden concept switches).

    Reference:
        "MOA: Massive Online Analysis" (Bifet et al., 2010)
        Standard RBF generator configuration for drift benchmarking

    Args:
        speed: Drift speed (0.0001=slow, 0.001=moderate, 0.01=fast)
        n_centroids: Number of RBF centroids (default: 50, MOA standard)
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    # STANDARD MOA CONFIGURATION
    # 50 centroids, 10 features, all centroids drift
    stream = synth.RandomRBFDrift(
        seed_model=seed,
        seed_sample=seed + 1000,
        n_classes=2,                    # Binary classification (standard)
        n_features=10,                  # 10 features (MOA standard)
        n_centroids=n_centroids,        # 50 centroids (standard)
        change_speed=speed,             # Drift speed parameter
        n_drift_centroids=n_centroids   # ALL centroids drift (maximum drift)
    )

    # Generate stream
    for i, (x, y) in enumerate(stream.take(total_size)):
        X_list.append(list(x.values()))
        y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    # Estimate drift positions (continuous drift, so evenly spaced markers)
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]

    return X, y, drift_positions


# ============================================================================
# REAL-WORLD DATASET GENERATORS
# ============================================================================

def generate_electricity_stream(total_size, n_drift_events, seed=42):
    """
    Electricity (Elec2) - Real-world electricity price prediction dataset.

    REAL-WORLD DRIFT BENCHMARK (most cited in literature):
    - 45,312 total instances (Australian NSW Electricity Market, 1996-1998)
    - Binary classification: Price UP or DOWN
    - 7 features: day, time, demand, supply indicators
    - Natural concept drift from market expansion
    - 30-minute intervals (48 instances per day)

    We don't know exactly when/where drifts occur in this real-world data.

    Use for:
    - Qualitative validation (detection patterns, stability, false positives)
    - Real-world robustness testing

    Do NOT use for:
    - Quantitative F1/MTTD metrics (no ground truth)

    Reference:
        "How good is the Electricity benchmark for evaluating concept drift adaptation"
        (Harries, 1999; used in 500+ papers)

    Args:
        total_size: Number of samples to extract (default: 10000)
        n_drift_events: Estimated number of drift events (for compatibility)
    """
    from river.datasets import Elec2

    X_list, y_list = [], []

    # Load Elec2 from River
    stream = Elec2()

    # Extract first total_size samples
    for i, (x, y) in enumerate(stream):
        if i >= total_size:
            break
        X_list.append(list(x.values()))
        y_list.append(1 if y == 'UP' else 0)  # Convert UP/DOWN to 1/0

    X = np.array(X_list)
    y = np.array(y_list)

    # ⚠️ NO GROUND TRUTH - Estimate drift positions (heuristic only)
    # Literature suggests drift from market expansion, but exact locations unknown
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]

    return X, y, drift_positions


def generate_electricity_sorted_stream(total_size, n_drift_events, seed=42, sort_feature="nswdemand"):
    """
    Electricity (Elec2) - SEMI-REAL dataset with CONTROLLED drift.

    Strategy: Sort by demand/price to create natural concept drift as
    market behavior changes with load levels (real economic phenomenon).

    DATASET INFO:
    - 45,312 total instances (Australian NSW Electricity Market)
    - 7 features: day, period, nswprice, nswdemand, vicprice, vicdemand, transfer
    - Binary classification: Price UP or DOWN
    - Sorting by demand creates natural drift (different pricing at different loads)

    Reference:
        Harries (1999), used in 500+ concept drift papers

    Args:
        total_size: Number of samples to extract
        n_drift_events: Number of drift events (controllable!)
        seed: Random seed
        sort_feature: Feature to sort by (default: nswdemand)
    """
    from river.datasets import Elec2

    np.random.seed(seed)

    # Feature name mapping for Elec2
    feature_names = ['day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']

    # Load Elec2
    data = []
    for i, (x, y) in enumerate(Elec2()):
        if i >= min(total_size * 2, 45312):  # Buffer for sampling, max is full dataset
            break
        row = list(x.values())
        # Get sort feature value
        sort_idx = feature_names.index(sort_feature) if sort_feature in feature_names else 3  # default to nswdemand
        sort_val = row[sort_idx]
        label = 1 if y == 'UP' else 0
        data.append((sort_val, row, label))

    # Sort by the chosen feature to create controlled drift
    data.sort(key=lambda t: t[0])

    # Sample evenly to get exact total_size
    if len(data) > total_size:
        step = len(data) // total_size
        sampled = [data[i * step] for i in range(total_size)]
    else:
        sampled = data[:total_size]

    X = np.array([s[1] for s in sampled])
    y = np.array([s[2] for s in sampled])

    # Calculate drift positions (concept changes at feature boundaries)
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]

    return X, y, drift_positions


def generate_covertype_sorted_stream(total_size, n_drift_events, seed=42, sort_feature="Elevation"):
    """
    Forest Covertype - SEMI-REAL dataset with CONTROLLED drift.

    Strategy: Sort by elevation to create natural concept drift as
    forest type changes with altitude (real ecological phenomenon).

    This is the standard "semi-real" benchmark approach - real data
    reordered to create known drift positions.

    DATASET INFO:
    - 581,012 total instances (Covertype from UCI)
    - 54 features (10 numerical + 44 binary wilderness/soil type)
    - 7 forest cover types → converted to binary
    - Sorting by elevation creates natural concept drift

    Reference:
        Blackard & Dean (1999), UCI ML Repository
        Used in 200+ concept drift papers

    Args:
        total_size: Number of samples to extract
        n_drift_events: Number of drift events (controllable!)
        seed: Random seed
        sort_feature: Feature to sort by (default: Elevation)
    """
    from river.datasets import Covtype

    np.random.seed(seed)

    # Load Covertype dataset (not pre-normalized, we'll normalize later)
    data = []
    for i, (x, y) in enumerate(Covtype()):
        if i >= min(total_size * 3, 100000):  # Buffer for sampling, cap at 100k
            break
        row = list(x.values())
        # Get elevation (first numerical feature in Covertype)
        # Feature order: Elevation, Aspect, Slope, etc.
        elevation = row[0] if sort_feature == "Elevation" else row[1]
        data.append((elevation, row, y))

    # Sort by elevation to create controlled drift
    data.sort(key=lambda t: t[0])

    # Sample evenly to get exact total_size
    if len(data) > total_size:
        step = len(data) // total_size
        sampled = [data[i * step] for i in range(total_size)]
    else:
        sampled = data[:total_size]

    X = np.array([s[1] for s in sampled])
    # Convert multi-class to binary (cover type 1-2 vs 3-7)
    y = np.array([1 if s[2] <= 2 else 0 for s in sampled])

    # Normalize features (Covtype is not pre-normalized)
    # Use StandardScaler approach: (x - mean) / std
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero for constant features
    X = (X - X_mean) / X_std

    # Calculate drift positions (concept changes at elevation boundaries)
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]

    return X, y, drift_positions


# ============================================================================
# SINE FAMILY: Classification Reversal + Noise Robustness Tests
# ============================================================================

def generate_sine1_stream(total_size, n_drift_events, seed=42):
    """
    Sine1 Dataset: y < sin(x) classification that REVERSES at drift points.

    Tests: Classification reversal drift (different from SEA's threshold shift)
    - Before drift: y = 1 if point is BELOW sin(x) curve
    - After drift:  y = 1 if point is ABOVE sin(x) curve

    This is a more dramatic P(Y|X) change than SEA's threshold shift.

    Args:
        total_size: Total number of samples
        n_drift_events: Number of drift events
        seed: Random seed

    Returns:
        X: Features (2D: x and y coordinates)
        y: Labels (binary classification)
        drift_positions: List of drift positions
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    segments = [0] + drift_positions + [total_size]

    for seg_idx in range(len(segments) - 1):
        start, end = segments[seg_idx], segments[seg_idx + 1]
        size = end - start

        # Generate random points in [0, 2π] × [0, 1]
        x_vals = np.random.uniform(0, 2 * np.pi, size)
        y_vals = np.random.uniform(0, 1, size)

        # Compute sin curve
        sin_vals = np.sin(x_vals)

        # Classification rule: REVERSES at each drift
        if seg_idx % 2 == 0:
            # Concept 0: Points BELOW sine curve are positive
            labels = (y_vals < (sin_vals + 1) / 2).astype(int)  # Normalize sin to [0,1]
        else:
            # Concept 1: Points ABOVE sine curve are positive (REVERSED!)
            labels = (y_vals >= (sin_vals + 1) / 2).astype(int)

        # Stack features: [x, y]
        X_seg = np.column_stack([x_vals, y_vals])
        X_list.append(X_seg)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y, drift_positions


def generate_sine2_stream(total_size, n_drift_events, seed=42):
    """
    Sine2 Dataset: y < 0.5 + 0.3*sin(3πx) classification that REVERSES.

    Similar to Sine1 but with:
    - Higher frequency (3π instead of 1)
    - Offset boundary (0.5)
    - Amplitude scaling (0.3)

    Args:
        total_size: Total number of samples
        n_drift_events: Number of drift events
        seed: Random seed

    Returns:
        X: Features (2D: x and y coordinates)
        y: Labels (binary classification)
        drift_positions: List of drift positions
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    segments = [0] + drift_positions + [total_size]

    for seg_idx in range(len(segments) - 1):
        start, end = segments[seg_idx], segments[seg_idx + 1]
        size = end - start

        # Generate random points in [0, 2π] × [0, 1]
        x_vals = np.random.uniform(0, 2 * np.pi, size)
        y_vals = np.random.uniform(0, 1, size)

        # Compute boundary: 0.5 + 0.3*sin(3πx)
        boundary = 0.5 + 0.3 * np.sin(3 * np.pi * x_vals)

        # Classification rule: REVERSES at each drift
        if seg_idx % 2 == 0:
            # Concept 0: Points BELOW boundary are positive
            labels = (y_vals < boundary).astype(int)
        else:
            # Concept 1: Points ABOVE boundary are positive (REVERSED!)
            labels = (y_vals >= boundary).astype(int)

        # Stack features: [x, y]
        X_seg = np.column_stack([x_vals, y_vals])
        X_list.append(X_seg)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y, drift_positions


def generate_sinirrel1_stream(total_size, n_drift_events, seed=42):
    """
    SINIRREL1: Sine1 + 2 IRRELEVANT random features (noise robustness test).

    Tests: How well methods handle irrelevant features
    - Features 0-1: Same as Sine1 (relevant)
    - Features 2-3: Uniformly random noise (irrelevant)

    This tests if methods waste capacity on noise dimensions.

    Args:
        total_size: Total number of samples
        n_drift_events: Number of drift events
        seed: Random seed

    Returns:
        X: Features (4D: 2 relevant + 2 irrelevant)
        y: Labels (binary classification)
        drift_positions: List of drift positions
    """
    # Generate Sine1 first
    X_sine1, y, drift_positions = generate_sine1_stream(total_size, n_drift_events, seed)

    # Add 2 irrelevant random features
    np.random.seed(seed + 999)
    noise_features = np.random.uniform(0, 1, (total_size, 2))

    X = np.hstack([X_sine1, noise_features])

    print(f"  SINIRREL1 (SUDDEN + NOISE): {X.shape[0]} samples, {X.shape[1]} features (2 relevant + 2 irrelevant)")
    print(f"    {len(drift_positions)} drifts, Noise: 50% of features")
    return X, y, drift_positions


def generate_sinirrel2_stream(total_size, n_drift_events, seed=42):
    """
    SINIRREL2: Sine2 + 2 IRRELEVANT random features (noise robustness test).

    Tests: Noise robustness with complex sine boundary
    - Features 0-1: Same as Sine2 (relevant)
    - Features 2-3: Uniformly random noise (irrelevant)

    Args:
        total_size: Total number of samples
        n_drift_events: Number of drift events
        seed: Random seed

    Returns:
        X: Features (4D: 2 relevant + 2 irrelevant)
        y: Labels (binary classification)
        drift_positions: List of drift positions
    """
    # Generate Sine2 first
    X_sine2, y, drift_positions = generate_sine2_stream(total_size, n_drift_events, seed)

    # Add 2 irrelevant random features
    np.random.seed(seed + 999)
    noise_features = np.random.uniform(0, 1, (total_size, 2))

    X = np.hstack([X_sine2, noise_features])

    return X, y, drift_positions


# ============================================================================
# RBF AND LED: Complex Distributions
# ============================================================================

def generate_rbfblips_stream(total_size, n_drift_events, seed=42, n_centroids=50, n_features=10):
    """
    RBFblips: Random RBF clusters with SUDDEN drift (blips).

    Creates STRONG P(X) changes using:
    1. Different RBF centroid configurations per segment
    2. Feature scaling/shifting at alternate segments to ensure detectable change

    Tests: Non-linear, high-dimensional distributions with abrupt cluster changes
    - Uses Radial Basis Functions (RBF) to create complex decision boundaries
    - Each drift event: centroids jump to new positions + feature transformation
    - Different from incremental RBF (which has continuous centroid movement)

    Args:
        total_size: Total number of samples
        n_drift_events: Number of drift events
        seed: Random seed
        n_centroids: Number of RBF centroids
        n_features: Feature dimensionality

    Returns:
        X: Features (n_features dimensions)
        y: Labels (binary classification)
        drift_positions: List of drift positions
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    segments = [0] + drift_positions + [total_size]

    for seg_idx in range(len(segments) - 1):
        start, end = segments[seg_idx], segments[seg_idx + 1]
        size = end - start

        # Use River's RandomRBF generator with DIFFERENT seed per segment
        stream = synth.RandomRBF(
            seed_model=seed + seg_idx * 1000,  # Different centroids per segment
            seed_sample=seed + seg_idx * 100,
            n_classes=2,
            n_features=n_features,
            n_centroids=n_centroids
        )

        # Generate samples from this RBF configuration
        for i, (x, y) in enumerate(stream.take(size)):
            x_vals = list(x.values())
            
            # Add feature transformation at alternating segments
            # This creates STRONG P(X) changes that detectors can detect
            if seg_idx % 2 == 1:
                # Scale and shift features to create clear distribution change
                x_vals = [v * 1.5 + 0.3 for v in x_vals]
            
            X_list.append(x_vals)
            y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, drift_positions


def generate_led_abrupt_stream(total_size, n_drift_events, seed=42, has_noise=False):
    """
    LED_abrupt: 7-segment LED digit prediction with SUDDEN function changes.

    Tests: Discrete binary features with abrupt concept drift
    - 7 binary features (LED segments: a-g)
    - Predicts digit displayed on LED
    - Each drift: LED encoding function changes

    Different from STAGGER:
    - LED: 7 binary features, digit classification (10 classes → binary)
    - STAGGER: 3 categorical features, logical rule changes

    Args:
        total_size: Total number of samples
        n_drift_events: Number of drift events
        seed: Random seed
        has_noise: Whether to add noise to LED (default: False for clean signal)

    Returns:
        X: Features (7 binary features)
        y: Labels (binary: even vs odd digit)
        drift_positions: List of drift positions
    """
    np.random.seed(seed)

    X_list, y_list = [], []

    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    segments = [0] + drift_positions + [total_size]

    for seg_idx in range(len(segments) - 1):
        start, end = segments[seg_idx], segments[seg_idx + 1]
        size = end - start

        # Use River's LED generator
        stream = synth.LED(
            seed=seed + seg_idx * 100,
            noise_percentage=0.1 if has_noise else 0.0,
            irrelevant_features=has_noise  # Add irrelevant features if noise=True
        )

        # Generate samples
        for i, (x, y_digit) in enumerate(stream.take(size)):
            X_list.append(list(x.values())[:7])  # First 7 features are LED segments

            # Convert 10-class digit to binary (even vs odd)
            # This changes the classification function at each drift
            if seg_idx % 2 == 0:
                # Concept 0: Even digits are positive
                y_binary = 1 if y_digit % 2 == 0 else 0
            else:
                # Concept 1: Odd digits are positive (REVERSED!)
                y_binary = 1 if y_digit % 2 == 1 else 0

            y_list.append(y_binary)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, drift_positions


# ============================================================================
# UNIFIED DATASET GENERATOR
# ============================================================================

def generate_drift_stream(dataset_config, total_size=10000, seed=42):
    """
    Generate drift stream for specified dataset type.

    Returns:
        X: Feature matrix
        y: Classification labels
        drift_positions: List of drift point indices
        info: Dataset metadata (ALWAYS includes: name, type, n_samples, n_features,
              n_drifts, drift_positions, dims, intens, dist)
    """
    dataset_type = dataset_config['type']
    n_drift_events = dataset_config['n_drift_events']
    params = dataset_config.get('params', {})

    print(f"  Generating {dataset_type} with {n_drift_events} drift events...")

    if dataset_type == "standard_sea":
        X, y, drift_positions = generate_standard_sea_stream(total_size, n_drift_events, seed)
        info = {
            'name': 'Standard SEA',
            'features': 3,
            'dims': 3,
            'intens': 'N/A',
            'dist': 'N/A',
            'drift_type': 'sudden'
        }

    elif dataset_type == "enhanced_sea":
        scale_factors = params.get('scale_factors', (1.8, 1.5, 2.0))
        shift_amounts = params.get('shift_amounts', (5.0, 4.0, 8.0))
        X, y, drift_positions = generate_enhanced_sea_stream(total_size, n_drift_events, seed,
                                                              scale_factors, shift_amounts)
        info = {
            'name': 'Enhanced SEA',
            'features': 3,
            'dims': 3,
            'intens': 'N/A',
            'dist': 'N/A',
            'drift_type': 'sudden'
        }

    elif dataset_type == "stagger":
        X, y, drift_positions = generate_stagger_stream(total_size, n_drift_events, seed)
        info = {
            'name': 'STAGGER',
            'features': 5,
            'dims': 5,
            'intens': 'N/A',
            'dist': 'N/A',
            'drift_type': 'sudden'
        }

    elif dataset_type == "hyperplane":
        n_features = params.get('n_features', 10)
        X, y, drift_positions = generate_hyperplane_stream(total_size, n_drift_events, seed, n_features)
        info = {
            'name': 'Hyperplane',
            'features': n_features,
            'dims': n_features,
            'intens': 'N/A',
            'dist': 'N/A',
            'drift_type': 'sudden'
        }

    elif dataset_type == "gen_random":
        dims = params.get('dims', 5)
        intens = params.get('intens', 0.125)
        dist = params.get('dist', 'unif')
        alt = params.get('alt', False)
        X, y, drift_positions = generate_genrandom_stream(total_size, n_drift_events, seed,
                                                          dims, intens, dist, alt)
        info = {
            'name': 'gen_random',
            'features': dims,
            'dims': dims,
            'intens': intens,
            'dist': dist,
            'drift_type': 'sudden'
        }

    # ========================================================================
    # GRADUAL DRIFT DATASETS
    # ========================================================================

    elif dataset_type == "sea_gradual":
        transition_width = params.get('transition_width', 1000)
        X, y, drift_positions = generate_sea_gradual_stream(total_size, n_drift_events, seed, transition_width)
        info = {
            'name': 'SEA Gradual',
            'features': 3,
            'dims': 3,
            'intens': 'N/A',
            'dist': 'N/A',
            'drift_type': 'gradual',
            'transition_width': transition_width
        }

    elif dataset_type == "hyperplane_gradual":
        n_features = params.get('n_features', 10)
        X, y, drift_positions = generate_hyperplane_gradual_stream(total_size, n_drift_events, seed, n_features)
        info = {
            'name': 'Hyperplane Gradual',
            'features': n_features,
            'dims': n_features,
            'intens': 'N/A',
            'dist': 'N/A',
            'drift_type': 'gradual',
            'transition_width': 'continuous'
        }

    elif dataset_type == "agrawal_gradual":
        transition_width = params.get('transition_width', 1000)
        X, y, drift_positions = generate_agrawal_gradual_stream(total_size, n_drift_events, seed, transition_width)
        info = {
            'name': 'Agrawal Gradual',
            'features': 9,
            'dims': 9,
            'intens': 'N/A',
            'dist': 'N/A',
            'drift_type': 'gradual',
            'transition_width': transition_width
        }

    elif dataset_type == "circles_gradual":
        transition_width = params.get('transition_width', 500)
        X, y, drift_positions = generate_circles_gradual_stream(total_size, n_drift_events, seed, transition_width)
        info = {
            'name': 'Circles Gradual',
            'features': 2,
            'dims': 2,
            'intens': 'N/A',
            'dist': 'N/A',
            'drift_type': 'gradual',
            'transition_width': transition_width
        }

    # ========================================================================
    # INCREMENTAL DRIFT DATASETS
    # ========================================================================

    elif dataset_type == "rbf":
        n_centroids = params.get('n_centroids', 50)
        speed = params.get('speed', 0.0001)
        X, y, drift_positions = generate_rbf_stream(total_size, n_drift_events, seed, n_centroids, speed)
        info = {
            'name': f'RBF (speed={speed})',
            'features': 10,
            'dims': 10,
            'intens': f'speed={speed}',
            'dist': 'RBF',
            'drift_type': 'incremental',
            'speed': speed,
            'n_centroids': n_centroids
        }

    elif dataset_type == "electricity":
        X, y, drift_positions = generate_electricity_stream(total_size, n_drift_events, seed)
        info = {
            'name': 'Electricity (Elec2)',
            'features': 7,
            'dims': 7,
            'intens': 'N/A (real-world)',
            'dist': 'Real-world',
            'drift_type': 'real-world',
            'has_ground_truth': False
        }

    elif dataset_type == "electricity_sorted":
        sort_feature = params.get('sort_feature', 'nswdemand')
        X, y, drift_positions = generate_electricity_sorted_stream(
            total_size, n_drift_events, seed, sort_feature
        )
        info = {
            'name': 'Electricity (Sorted)',
            'features': 7,
            'dims': 7,
            'intens': 'N/A (semi-real)',
            'dist': 'Real-world sorted',
            'drift_type': 'semi-real',
            'has_ground_truth': True,
            'sort_feature': sort_feature
        }

    # ========================================================================
    # SEMI-REAL DATASETS: Covertype with Controlled Drift
    # ========================================================================

    elif dataset_type == "covertype_sorted":
        sort_feature = params.get('sort_feature', 'Elevation')
        X, y, drift_positions = generate_covertype_sorted_stream(
            total_size, n_drift_events, seed, sort_feature
        )
        info = {
            'name': 'Covertype (Sorted)',
            'features': 54,
            'dims': 54,
            'intens': 'N/A (semi-real)',
            'dist': 'Real-world sorted',
            'drift_type': 'semi-real',
            'has_ground_truth': True,
            'sort_feature': sort_feature
        }

    # ========================================================================
    # SINE FAMILY
    # ========================================================================

    elif dataset_type == "sine1":
        X, y, drift_positions = generate_sine1_stream(total_size, n_drift_events, seed)
        info = {
            'name': 'Sine1',
            'features': 2,
            'dims': 2,
            'intens': 'N/A',
            'dist': 'Uniform',
            'drift_type': 'sudden',
            'has_ground_truth': True
        }

    elif dataset_type == "sine2":
        X, y, drift_positions = generate_sine2_stream(total_size, n_drift_events, seed)
        info = {
            'name': 'Sine2',
            'features': 2,
            'dims': 2,
            'intens': 'N/A',
            'dist': 'Uniform',
            'drift_type': 'sudden',
            'has_ground_truth': True
        }

    elif dataset_type == "sinirrel1":
        X, y, drift_positions = generate_sinirrel1_stream(total_size, n_drift_events, seed)
        info = {
            'name': 'SINIRREL1',
            'features': 4,
            'dims': 4,
            'intens': 'N/A (50% noise)',
            'dist': 'Uniform',
            'drift_type': 'sudden',
            'has_ground_truth': True
        }

    elif dataset_type == "sinirrel2":
        X, y, drift_positions = generate_sinirrel2_stream(total_size, n_drift_events, seed)
        info = {
            'name': 'SINIRREL2',
            'features': 4,
            'dims': 4,
            'intens': 'N/A (50% noise)',
            'dist': 'Uniform',
            'drift_type': 'sudden',
            'has_ground_truth': True
        }

    # ========================================================================
    # RBF AND LED
    # ========================================================================

    elif dataset_type == "rbfblips":
        n_centroids = params.get('n_centroids', 50)
        n_features = params.get('n_features', 10)
        X, y, drift_positions = generate_rbfblips_stream(total_size, n_drift_events, seed, n_centroids, n_features)
        info = {
            'name': f'RBFblips (c={n_centroids})',
            'features': n_features,
            'dims': n_features,
            'intens': f'{n_centroids} centroids',
            'dist': 'RBF',
            'drift_type': 'sudden',
            'has_ground_truth': True
        }

    elif dataset_type == "led_abrupt":
        has_noise = params.get('has_noise', False)
        X, y, drift_positions = generate_led_abrupt_stream(total_size, n_drift_events, seed, has_noise)
        info = {
            'name': 'LED_abrupt' + (' (noisy)' if has_noise else ''),
            'features': 7,
            'dims': 7,
            'intens': 'N/A',
            'dist': 'Binary',
            'drift_type': 'sudden',
            'has_ground_truth': True
        }

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Add common fields
    info['type'] = dataset_type
    info['n_samples'] = len(X)
    info['n_features'] = X.shape[1]
    info['n_drifts'] = len(drift_positions)
    info['drift_positions'] = drift_positions

    return X, y, drift_positions, info

