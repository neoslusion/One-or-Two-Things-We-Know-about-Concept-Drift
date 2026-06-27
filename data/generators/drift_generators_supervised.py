"""
Supervised drift stream generators for CDT_MSW fair comparison.
Generates streams with P(Y|X) changes for supervised drift detection methods.
"""

import numpy as np
from typing import Dict, Tuple, List, Any
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler


def generate_concept_aware_labels(X: np.ndarray, 
                                   concept_type: str,
                                   noise_level: float = 0.1) -> np.ndarray:
    """
    Generate labels with different concept relationships.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        concept_type: 'linear', 'nonlinear', 'cluster_based'
        noise_level: Label noise probability
        
    Returns:
        y: Binary labels (0/1) with concept dependency
    """
    n_samples, n_features = X.shape
    
    if concept_type == 'linear':
        # Linear decision boundary: y = sign(w^T x + b)
        w = np.random.randn(n_features)
        w = w / np.linalg.norm(w)
        b = np.random.randn()
        scores = X @ w + b
        y = (scores > 0).astype(int)
        
    elif concept_type == 'nonlinear':
        # Nonlinear boundary: y = sign(x1^2 + x2^2 - threshold)
        if n_features >= 2:
            scores = X[:, 0]**2 + X[:, 1]**2
            threshold = np.median(scores)
            y = (scores > threshold).astype(int)
        else:
            # Fallback to linear
            w = np.random.randn(n_features)
            y = (X @ w > 0).astype(int)
            
    elif concept_type == 'cluster_based':
        # Cluster-based: assign labels based on cluster membership
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        y = kmeans.fit_predict(X)
        
    else:
        raise ValueError(f"Unknown concept_type: {concept_type}")
    
    # Add label noise
    if noise_level > 0:
        n_flip = int(n_samples * noise_level)
        flip_idx = np.random.choice(n_samples, n_flip, replace=False)
        y[flip_idx] = 1 - y[flip_idx]
    
    return y


def generate_supervised_stream(drift_scenario: Dict[str, Any],
                                total_size: int = 5000,
                                n_features: int = 20,
                                random_state: int = None) -> Tuple[np.ndarray, np.ndarray, List[int], Dict]:
    """
    Generate supervised drift stream with P(Y|X) changes.
    
    Args:
        drift_scenario: {
            'type': 'sudden'|'gradual'|'incremental'|'recurrent'|'blip',
            'n_drift_events': int,
            'drift_magnitude': float (0.3-0.7)
        }
        total_size: Total stream length
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        X: Feature matrix (total_size, n_features)
        y: Label vector (total_size,) with concept drift
        drift_positions: List of drift start positions
        info: Metadata dict
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    drift_type = drift_scenario['type']
    n_drifts = drift_scenario['n_drift_events']
    magnitude = drift_scenario.get('drift_magnitude', 0.5)
    
    # Generate base features (P(X) remains relatively stable)
    X = np.random.randn(total_size, n_features)
    
    # Initialize labels
    y = np.zeros(total_size, dtype=int)
    
    # Determine drift positions
    segment_size = total_size // (n_drifts + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drifts)]
    
    # Concept sequence: alternate between different concepts
    concept_types = ['linear', 'nonlinear', 'cluster_based']
    
    info = {
        'drift_type': drift_type,
        'n_drifts': n_drifts,
        'drift_positions': drift_positions,
        'magnitude': magnitude,
        'concepts': []
    }
    
    if drift_type == 'sudden':
        # Sudden P(Y|X) change: instant concept switch
        current_pos = 0
        for drift_idx in range(n_drifts + 1):
            end_pos = drift_positions[drift_idx] if drift_idx < n_drifts else total_size
            
            # Switch concept type
            concept = concept_types[drift_idx % len(concept_types)]
            segment_X = X[current_pos:end_pos]
            y[current_pos:end_pos] = generate_concept_aware_labels(
                segment_X, concept, noise_level=0.1
            )
            
            info['concepts'].append({
                'range': (current_pos, end_pos),
                'type': concept
            })
            
            current_pos = end_pos
            
    elif drift_type == 'gradual':
        # Gradual P(Y|X) change: smooth transition between concepts
        transition_width = int(segment_size * 0.3)  # 30% of segment
        
        current_pos = 0
        for drift_idx in range(n_drifts + 1):
            end_pos = drift_positions[drift_idx] if drift_idx < n_drifts else total_size
            
            concept_old = concept_types[drift_idx % len(concept_types)]
            concept_new = concept_types[(drift_idx + 1) % len(concept_types)]
            
            # Before transition: old concept
            trans_start = end_pos - transition_width if drift_idx < n_drifts else end_pos
            segment_X = X[current_pos:trans_start]
            y[current_pos:trans_start] = generate_concept_aware_labels(
                segment_X, concept_old, noise_level=0.1
            )
            
            # During transition: blend concepts
            if drift_idx < n_drifts:
                trans_X = X[trans_start:end_pos]
                y_old = generate_concept_aware_labels(trans_X, concept_old, noise_level=0.0)
                y_new = generate_concept_aware_labels(trans_X, concept_new, noise_level=0.0)
                
                # Gradual blend
                alpha = np.linspace(0, 1, transition_width)
                rand_vals = np.random.rand(transition_width)
                y[trans_start:end_pos] = np.where(rand_vals < alpha, y_new, y_old)
            
            info['concepts'].append({
                'range': (current_pos, end_pos),
                'type': f"{concept_old}->{concept_new}",
                'transition': (trans_start, end_pos) if drift_idx < n_drifts else None
            })
            
            current_pos = end_pos
            
    elif drift_type == 'incremental':
        # Incremental P(Y|X) change: stepwise concept drift
        n_steps = 5
        step_size = segment_size // n_steps
        
        current_pos = 0
        for drift_idx in range(n_drifts + 1):
            end_pos = drift_positions[drift_idx] if drift_idx < n_drifts else total_size
            
            concept_old = concept_types[drift_idx % len(concept_types)]
            concept_new = concept_types[(drift_idx + 1) % len(concept_types)]
            
            # Incremental steps
            for step in range(n_steps):
                step_start = current_pos + step * step_size
                step_end = min(step_start + step_size, end_pos)
                
                # Interpolate between concepts
                alpha = step / (n_steps - 1)
                step_X = X[step_start:step_end]
                y_old = generate_concept_aware_labels(step_X, concept_old, noise_level=0.0)
                y_new = generate_concept_aware_labels(step_X, concept_new, noise_level=0.0)
                
                rand_vals = np.random.rand(step_end - step_start)
                y[step_start:step_end] = np.where(rand_vals < alpha, y_new, y_old)
            
            info['concepts'].append({
                'range': (current_pos, end_pos),
                'type': f"{concept_old}->{concept_new}_incremental",
                'n_steps': n_steps
            })
            
            current_pos = end_pos
            
    elif drift_type == 'recurrent':
        # Recurrent P(Y|X): concepts repeat periodically
        concepts_pool = concept_types[:2]  # Only use 2 concepts for recurrence
        
        current_pos = 0
        for drift_idx in range(n_drifts + 1):
            end_pos = drift_positions[drift_idx] if drift_idx < n_drifts else total_size
            
            # Alternate between same 2 concepts
            concept = concepts_pool[drift_idx % len(concepts_pool)]
            segment_X = X[current_pos:end_pos]
            y[current_pos:end_pos] = generate_concept_aware_labels(
                segment_X, concept, noise_level=0.1
            )
            
            info['concepts'].append({
                'range': (current_pos, end_pos),
                'type': f"{concept}_recurrent"
            })
            
            current_pos = end_pos
            
    elif drift_type == 'blip':
        # Blip P(Y|X): temporary concept change, then revert
        blip_width = int(segment_size * 0.1)  # 10% of segment
        
        current_pos = 0
        base_concept = concept_types[0]
        blip_concept = concept_types[1]
        
        for drift_idx in range(n_drifts + 1):
            end_pos = drift_positions[drift_idx] if drift_idx < n_drifts else total_size
            
            if drift_idx < n_drifts:
                # Before blip
                blip_start = drift_positions[drift_idx]
                segment_X = X[current_pos:blip_start]
                y[current_pos:blip_start] = generate_concept_aware_labels(
                    segment_X, base_concept, noise_level=0.1
                )
                
                # During blip
                blip_end = min(blip_start + blip_width, total_size)
                blip_X = X[blip_start:blip_end]
                y[blip_start:blip_end] = generate_concept_aware_labels(
                    blip_X, blip_concept, noise_level=0.1
                )
                
                # After blip (revert to base)
                after_X = X[blip_end:end_pos]
                y[blip_end:end_pos] = generate_concept_aware_labels(
                    after_X, base_concept, noise_level=0.1
                )
                
                info['concepts'].append({
                    'range': (current_pos, end_pos),
                    'type': f"{base_concept}_with_blip",
                    'blip': (blip_start, blip_end)
                })
            else:
                # Final segment
                segment_X = X[current_pos:end_pos]
                y[current_pos:end_pos] = generate_concept_aware_labels(
                    segment_X, base_concept, noise_level=0.1
                )
                
                info['concepts'].append({
                    'range': (current_pos, end_pos),
                    'type': base_concept
                })
            
            current_pos = end_pos
    
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")
    
    return X, y, drift_positions, info


def generate_mixed_stream_supervised(total_size: int = 5000,
                                      n_features: int = 20,
                                      n_drift_events: int = 3,
                                      random_state: int = None) -> Tuple[np.ndarray, np.ndarray, List[int], Dict]:
    """
    Generate supervised stream with mixed drift types for CDT_MSW testing.
    
    Returns:
        X: Feature matrix
        y: Labels with concept drift
        drift_positions: Drift locations
        info: Metadata including true drift types
    """
    # Random drift type selection
    if random_state is not None:
        np.random.seed(random_state)
    
    drift_types = ['sudden', 'gradual', 'incremental', 'recurrent', 'blip']
    selected_type = np.random.choice(drift_types)
    
    scenario = {
        'type': selected_type,
        'n_drift_events': n_drift_events,
        'drift_magnitude': np.random.uniform(0.4, 0.6)
    }
    
    X, y, drifts, info = generate_supervised_stream(
        scenario, total_size, n_features, random_state
    )
    
    info['true_drift_type'] = selected_type
    
    return X, y, drifts, info
