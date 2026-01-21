"""
Mathematically Rigorous Drift Stream Generators.

This module provides drift stream generators with proper mathematical foundations
following the standard definitions from concept drift literature:

References:
    - Gama et al. (2014): "A survey on concept drift adaptation"
    - Webb et al. (2016): "Characterizing concept drift"
    - Lu et al. (2018): "Learning under Concept Drift: A Review"
    - Hinder et al. (2024): "One or two things we know about concept drift"

Drift Types (Mathematical Definitions):
---------------------------------------

1. SUDDEN (Abrupt) Drift:
   - At time t₀: P_t(X,Y) instantly changes from P₁ to P₂
   - Mathematical: P_t = P₁ if t < t₀, else P₂
   
2. GRADUAL Drift:
   - Transition period [t₀, t₀+w]: Mixture of old and new concept
   - Mathematical: P_t = (1 - α(t))·P₁ + α(t)·P₂
   - where α(t) = (t - t₀) / w for t ∈ [t₀, t₀+w]
   - Samples are drawn from P₁ with prob (1-α) or P₂ with prob α

3. INCREMENTAL Drift:
   - Continuous small changes in distribution parameters
   - Mathematical: θ_t = θ₀ + v·t (linear parameter evolution)
   - No mixture - single distribution with evolving parameters

4. RECURRENT Drift:
   - Concepts cycle back to previous states
   - Mathematical: P_t = P_{(t mod T) // period} where concepts repeat

5. BLIP (Outlier/Temporary) Drift:
   - Short temporary change that reverts
   - Mathematical: P_t = P₂ for t ∈ [t₀, t₀+w], else P₁
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Literal
from dataclasses import dataclass


@dataclass
class DriftEvent:
    """Describes a single drift event."""
    drift_type: Literal["Sudden", "Gradual", "Incremental", "Recurrent", "Blip"]
    position: int  # Start position of drift
    width: int = 0  # Transition width (0 for sudden)
    magnitude: float = 2.0  # Magnitude of distribution shift
    
    def __post_init__(self):
        if self.drift_type == "Sudden":
            self.width = 0  # Sudden has no transition width


class ConceptDriftStreamGenerator:
    """
    Mathematically rigorous concept drift stream generator.
    
    Generates streams with controlled drift following standard definitions
    from concept drift literature.
    
    The generator creates P(X) drift (covariate shift) which is appropriate
    for UNSUPERVISED drift detection methods like SE-CDT.
    
    For SUPERVISED methods like CDT_MSW, use supervised_mode=True to also
    change P(Y|X) at drift points.
    """
    
    def __init__(
        self,
        n_features: int = 5,
        base_mean: float = 0.0,
        base_std: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize the stream generator.
        
        Args:
            n_features: Number of features in the stream
            base_mean: Mean of base distribution
            base_std: Standard deviation of base distribution
            seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self.base_mean = base_mean
        self.base_std = base_std
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def _generate_base_concept(self, n_samples: int) -> np.ndarray:
        """Generate samples from base concept: X ~ N(μ₀, σ₀²I)"""
        return self.rng.randn(n_samples, self.n_features) * self.base_std + self.base_mean
    
    def _generate_shifted_concept(
        self, 
        n_samples: int, 
        shift_magnitude: float
    ) -> np.ndarray:
        """Generate samples from shifted concept: X ~ N(μ₀ + δ, σ₀²I)"""
        X = self.rng.randn(n_samples, self.n_features) * self.base_std + self.base_mean
        # Shift first half of features (standard approach)
        n_shift = max(1, self.n_features // 2)
        X[:, :n_shift] += shift_magnitude
        return X
    
    def generate_sudden_drift(
        self,
        length: int,
        drift_position: int,
        magnitude: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate stream with SUDDEN (abrupt) drift.
        
        Mathematical Definition:
            P_t = P₁ if t < t₀
            P_t = P₂ if t >= t₀
            
        where P₁ = N(0, I), P₂ = N(δ, I) with δ = [magnitude, ..., 0, ..., 0]
        
        Args:
            length: Total stream length
            drift_position: Position where drift occurs
            magnitude: Shift magnitude (higher = more detectable)
            
        Returns:
            X: Feature matrix (length, n_features)
            concept_id: Concept assignment per sample
            drift_position: Exact drift position
        """
        X = np.empty((length, self.n_features))
        concept_id = np.zeros(length, dtype=int)
        
        # Before drift: Concept 0
        X[:drift_position] = self._generate_base_concept(drift_position)
        concept_id[:drift_position] = 0
        
        # After drift: Concept 1 (shifted)
        X[drift_position:] = self._generate_shifted_concept(
            length - drift_position, magnitude
        )
        concept_id[drift_position:] = 1
        
        return X, concept_id, drift_position
    
    def generate_gradual_drift(
        self,
        length: int,
        drift_position: int,
        transition_width: int,
        magnitude: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate stream with GRADUAL drift.
        
        Mathematical Definition:
            For t ∈ [t₀, t₀ + w]:
                α(t) = (t - t₀) / w
                Sample from P₁ with probability (1 - α(t))
                Sample from P₂ with probability α(t)
            
        This creates a smooth probabilistic transition between concepts.
        
        Args:
            length: Total stream length
            drift_position: Start of transition
            transition_width: Length of transition period
            magnitude: Shift magnitude
            
        Returns:
            X: Feature matrix
            concept_id: Concept assignment per sample  
            drift_position: Start of drift
        """
        X = np.empty((length, self.n_features))
        concept_id = np.zeros(length, dtype=int)
        
        transition_end = min(drift_position + transition_width, length)
        
        # Before drift: Concept 0
        X[:drift_position] = self._generate_base_concept(drift_position)
        
        # During transition: Probabilistic mixture
        for t in range(drift_position, transition_end):
            # α increases linearly from 0 to 1
            alpha = (t - drift_position) / transition_width
            
            if self.rng.random() < alpha:
                # Sample from new concept
                X[t] = self._generate_shifted_concept(1, magnitude).flatten()
                concept_id[t] = 1
            else:
                # Sample from old concept
                X[t] = self._generate_base_concept(1).flatten()
                concept_id[t] = 0
        
        # After transition: Concept 1
        if transition_end < length:
            X[transition_end:] = self._generate_shifted_concept(
                length - transition_end, magnitude
            )
            concept_id[transition_end:] = 1
        
        return X, concept_id, drift_position
    
    def generate_incremental_drift(
        self,
        length: int,
        drift_position: int,
        transition_width: int,
        magnitude: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate stream with INCREMENTAL drift.
        
        Mathematical Definition:
            For t >= t₀:
                μ(t) = μ₀ + v · min((t - t₀), w) / w · δ
                X_t ~ N(μ(t), σ²I)
            
        The mean shifts continuously (not probabilistic mixture).
        Different from Gradual: single evolving distribution, not mixture.
        
        Args:
            length: Total stream length
            drift_position: Start of incremental change
            transition_width: Duration of change
            magnitude: Final shift magnitude
            
        Returns:
            X: Feature matrix
            concept_id: Concept assignment (0=base, 1=during, 2=final)
            drift_position: Start of drift
        """
        X = np.empty((length, self.n_features))
        concept_id = np.zeros(length, dtype=int)
        
        n_shift = max(1, self.n_features // 2)
        transition_end = min(drift_position + transition_width, length)
        
        # Before drift
        X[:drift_position] = self._generate_base_concept(drift_position)
        
        # During transition: Continuous shift
        for t in range(drift_position, transition_end):
            progress = (t - drift_position) / transition_width
            current_shift = magnitude * progress
            
            X[t] = self.rng.randn(self.n_features) * self.base_std + self.base_mean
            X[t, :n_shift] += current_shift
            
            concept_id[t] = 1  # During transition
        
        # After transition: Final shifted concept
        if transition_end < length:
            X[transition_end:] = self._generate_shifted_concept(
                length - transition_end, magnitude
            )
            concept_id[transition_end:] = 2
        
        return X, concept_id, drift_position
    
    def generate_recurrent_drift(
        self,
        length: int,
        drift_position: int,
        period: int,
        magnitude: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Generate stream with RECURRENT drift.
        
        Mathematical Definition:
            P_t = P₁ if ((t - t₀) // period) % 2 == 0
            P_t = P₂ if ((t - t₀) // period) % 2 == 1
            
        Concepts alternate between P₁ and P₂ with given period.
        
        Args:
            length: Total stream length
            drift_position: Start of recurrent pattern
            period: Length of each concept phase
            magnitude: Shift magnitude
            
        Returns:
            X: Feature matrix
            concept_id: Concept assignment
            drift_positions: List of all drift positions
        """
        X = np.empty((length, self.n_features))
        concept_id = np.zeros(length, dtype=int)
        drift_positions = []
        
        # Before pattern starts
        X[:drift_position] = self._generate_base_concept(drift_position)
        
        # Recurrent pattern
        current_pos = drift_position
        cycle = 0
        
        while current_pos < length:
            cycle_end = min(current_pos + period, length)
            cycle_length = cycle_end - current_pos
            
            if cycle % 2 == 0:
                # Concept 0 (base)
                X[current_pos:cycle_end] = self._generate_base_concept(cycle_length)
                concept_id[current_pos:cycle_end] = 0
            else:
                # Concept 1 (shifted)
                X[current_pos:cycle_end] = self._generate_shifted_concept(
                    cycle_length, magnitude
                )
                concept_id[current_pos:cycle_end] = 1
            
            # Track drift positions (at cycle boundaries)
            if current_pos > drift_position:
                drift_positions.append(current_pos)
            
            current_pos = cycle_end
            cycle += 1
        
        # First drift is at drift_position
        drift_positions = [drift_position] + drift_positions
        
        return X, concept_id, drift_positions
    
    def generate_blip_drift(
        self,
        length: int,
        drift_position: int,
        blip_width: int,
        magnitude: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate stream with BLIP (temporary) drift.
        
        Mathematical Definition:
            P_t = P₂ if t ∈ [t₀, t₀ + w]
            P_t = P₁ otherwise
            
        A temporary excursion to a different concept, then return to original.
        
        Args:
            length: Total stream length
            drift_position: Start of blip
            blip_width: Duration of blip
            magnitude: Shift magnitude
            
        Returns:
            X: Feature matrix
            concept_id: Concept assignment
            drift_position: Start of blip
        """
        X = np.empty((length, self.n_features))
        concept_id = np.zeros(length, dtype=int)
        
        blip_end = min(drift_position + blip_width, length)
        
        # Before blip
        X[:drift_position] = self._generate_base_concept(drift_position)
        
        # During blip
        blip_length = blip_end - drift_position
        X[drift_position:blip_end] = self._generate_shifted_concept(
            blip_length, magnitude
        )
        concept_id[drift_position:blip_end] = 1
        
        # After blip: Return to original concept
        if blip_end < length:
            X[blip_end:] = self._generate_base_concept(length - blip_end)
            concept_id[blip_end:] = 0
        
        return X, concept_id, drift_position


def generate_mixed_stream_rigorous(
    events: List[Dict],
    length: int = None,
    n_features: int = 5,
    seed: int = 42,
    supervised_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a stream with multiple drift events using rigorous mathematical definitions.
    
    This is the RECOMMENDED generator for SE-CDT benchmarks. It creates proper P(X) 
    drift that can be detected by unsupervised methods.
    
    Args:
        events: List of drift events, each with:
            - 'type': "Sudden", "Gradual", "Incremental", "Recurrent", "Blip"
            - 'pos': Position where drift starts
            - 'width': Transition width (for Gradual/Incremental/Blip)
            - 'magnitude': Optional shift magnitude (default 2.0)
        length: Total stream length (auto-computed if None)
        n_features: Number of features
        seed: Random seed
        supervised_mode: If True, also change P(Y|X) at drift points
                        (required for supervised methods like CDT_MSW)
        
    Returns:
        X: Feature matrix (length, n_features)
        y: Labels (binary classification)
        concept_id: Concept assignment per sample
        
    Example:
        >>> events = [
        ...     {"type": "Sudden", "pos": 1000},
        ...     {"type": "Gradual", "pos": 2000, "width": 500},
        ...     {"type": "Blip", "pos": 3000, "width": 100}
        ... ]
        >>> X, y, concepts = generate_mixed_stream_rigorous(events, length=5000)
    """
    # Auto-compute length
    if length is None:
        last_event = max([e['pos'] + e.get('width', 0) for e in events])
        length = last_event + 1000
    
    generator = ConceptDriftStreamGenerator(n_features=n_features, seed=seed)
    
    # Initialize with base concept
    X = generator._generate_base_concept(length)
    concept_id = np.zeros(length, dtype=int)
    
    # Sort events by position
    events = sorted(events, key=lambda x: x['pos'])
    
    current_concept = 0
    
    for evt in events:
        dtype = evt['type']
        pos = evt['pos']
        width = evt.get('width', 200)
        magnitude = evt.get('magnitude', 2.0)
        
        n_shift = max(1, n_features // 2)
        
        if dtype == "Sudden":
            # Instant shift: X[pos:] gets shifted
            current_concept += 1
            shift = magnitude if current_concept % 2 == 1 else -magnitude
            X[pos:, :n_shift] += shift
            concept_id[pos:] = current_concept
            
        elif dtype == "Gradual":
            # Probabilistic mixture during transition
            end_pos = min(pos + width, length)
            
            for t in range(pos, end_pos):
                alpha = (t - pos) / width
                
                if generator.rng.random() < alpha:
                    # Sample from new concept
                    shift = magnitude if (current_concept + 1) % 2 == 1 else 0
                    X[t, :n_shift] = generator.rng.randn(n_shift) + shift
                    concept_id[t] = current_concept + 1
                # else keep current concept values
            
            # After transition
            if end_pos < length:
                current_concept += 1
                shift = magnitude if current_concept % 2 == 1 else 0
                base = generator.rng.randn(length - end_pos, n_shift)
                X[end_pos:, :n_shift] = base + shift
                concept_id[end_pos:] = current_concept
                
        elif dtype == "Incremental":
            # Continuous shift during transition
            end_pos = min(pos + width, length)
            
            for t in range(pos, end_pos):
                progress = (t - pos) / width
                current_shift = magnitude * progress
                X[t, :n_shift] += current_shift
                concept_id[t] = current_concept + 1 if progress > 0.5 else current_concept
            
            # After transition
            if end_pos < length:
                current_concept += 1
                X[end_pos:, :n_shift] += magnitude
                concept_id[end_pos:] = current_concept
                
        elif dtype == "Recurrent":
            # Alternating pattern
            period = max(1, width // 2)
            
            for t in range(pos, length, period):
                cycle_end = min(t + period, length)
                cycle = ((t - pos) // period) % 2
                
                if cycle == 1:
                    X[t:cycle_end, :n_shift] += magnitude
                    concept_id[t:cycle_end] = current_concept + 1
                else:
                    concept_id[t:cycle_end] = current_concept
                    
        elif dtype == "Blip":
            # Temporary shift
            blip_width = max(1, width // 5)
            end_blip = min(pos + blip_width, length)
            
            X[pos:end_blip, :n_shift] += magnitude
            concept_id[pos:end_blip] = current_concept + 1
            # Returns to original after blip (no permanent change)
    
    # Generate labels
    if supervised_mode:
        # Concept-aware labels: different decision boundaries per concept
        # This is REQUIRED for CDT_MSW to work properly
        y = np.zeros(length, dtype=int)
        for i in range(length):
            if concept_id[i] % 2 == 0:
                # Even concepts: boundary is X[:,0] + X[:,1] > 0
                y[i] = 1 if (X[i, 0] + X[i, 1]) > 0 else 0
            else:
                # Odd concepts: ROTATED boundary X[:,0] - X[:,1] > 0
                y[i] = 1 if (X[i, 0] - X[i, 1]) > 0 else 0
    else:
        # Unsupervised mode: Fixed decision boundary
        # SE-CDT detects P(X) change, doesn't need P(Y|X) change
        y = (np.sum(X[:, :2], axis=1) > 0).astype(int)
    
    return X, y, concept_id


def validate_drift_properties(
    X: np.ndarray,
    concept_id: np.ndarray,
    expected_drift_positions: List[int],
    n_features: int = None
) -> Dict:
    """
    Validate that generated drift has expected statistical properties.
    
    This function verifies:
    1. Mean shift between concepts is significant
    2. Drift positions are correct
    3. Distribution change is detectable
    
    Args:
        X: Feature matrix
        concept_id: Concept assignments
        expected_drift_positions: Expected drift positions
        n_features: Number of features to check
        
    Returns:
        Dictionary with validation results
    """
    if n_features is None:
        n_features = min(2, X.shape[1])
    
    unique_concepts = np.unique(concept_id)
    results = {
        "valid": True,
        "n_concepts": len(unique_concepts),
        "concept_means": {},
        "mean_shifts": [],
        "detected_drift_positions": [],
        "errors": []
    }
    
    # Compute per-concept statistics
    for c in unique_concepts:
        mask = concept_id == c
        if np.sum(mask) > 10:  # Need enough samples
            results["concept_means"][int(c)] = X[mask, :n_features].mean(axis=0).tolist()
    
    # Check mean shifts between consecutive concepts
    for c in range(len(unique_concepts) - 1):
        c1, c2 = unique_concepts[c], unique_concepts[c + 1]
        if c1 in results["concept_means"] and c2 in results["concept_means"]:
            mean1 = np.array(results["concept_means"][c1])
            mean2 = np.array(results["concept_means"][c2])
            shift = np.linalg.norm(mean2 - mean1)
            results["mean_shifts"].append(float(shift))
            
            if shift < 0.5:
                results["errors"].append(f"Weak shift between concept {c1} and {c2}: {shift:.3f}")
                results["valid"] = False
    
    # Detect actual drift positions (where concept_id changes)
    changes = np.where(np.diff(concept_id) != 0)[0] + 1
    results["detected_drift_positions"] = changes.tolist()
    
    # Verify expected positions
    for expected_pos in expected_drift_positions:
        closest = min(changes, key=lambda x: abs(x - expected_pos)) if len(changes) > 0 else -1
        if abs(closest - expected_pos) > 50:  # Allow small tolerance
            results["errors"].append(
                f"Expected drift at {expected_pos}, closest detected at {closest}"
            )
    
    return results


# Convenience function for benchmark_proper.py compatibility
def generate_mixed_stream(
    events: List[Dict],
    length: int = None,
    seed: int = 42,
    supervised_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for benchmark_proper.generate_mixed_stream.
    
    Uses rigorous mathematical definitions for drift generation.
    """
    X, y, _ = generate_mixed_stream_rigorous(
        events, length, n_features=5, seed=seed, supervised_mode=supervised_mode
    )
    return X, y
