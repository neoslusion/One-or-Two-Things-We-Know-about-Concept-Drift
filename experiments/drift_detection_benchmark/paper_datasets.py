"""
Paper Datasets: Sine, Circle Synthetic Datasets
Following CDT_MSW Paper specifications exactly.

References:
- Guo et al. (2022), Information Sciences 585
- Standard drift detection benchmarks
"""

import numpy as np
from typing import Dict, List, Tuple


class PaperDataGenerator:
    """
    Generate synthetic datasets exactly as described in CDT_MSW paper.
    
    Datasets:
    - Sine1, Sine2: Decision boundary = sine wave
    - Circle: Decision boundary = circle with shifting center/radius
    - Gaussian: Moving mean Gaussian distribution
    """
    
    def __init__(self, n_samples: int = 100000, noise: float = 0.1, 
                 random_state: int = 42):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        np.random.seed(random_state)
    
    # ================================================================
    # SINE DATASET (Abrupt Drift - TCD)
    # ================================================================
    
    def generate_sine1(self, drift_positions: List[int] = None) -> Dict:
        """
        Sine1 dataset: y = sin(πx)
        
        Classification: y_attr < sin(πx) → positive
        Drift: Reverse class labels at drift points
        """
        if drift_positions is None:
            drift_positions = [self.n_samples // 2]
        
        np.random.seed(self.random_state)
        
        # Generate attributes
        x = np.random.uniform(0, 1, self.n_samples)
        y_attr = np.random.uniform(0, 1, self.n_samples)
        
        # Initial classification: below sine curve = positive
        boundary = np.sin(np.pi * x)
        labels = (y_attr < boundary).astype(int)
        
        # Apply drift (reverse labels after drift points)
        drift_positions = sorted(drift_positions)
        current_flip = False
        
        for i, dp in enumerate(drift_positions):
            if i < len(drift_positions) - 1:
                end = drift_positions[i + 1]
            else:
                end = self.n_samples
            
            current_flip = not current_flip
            if current_flip:
                labels[dp:end] = 1 - labels[dp:end]
        
        # Add noise
        noise_mask = np.random.random(self.n_samples) < self.noise
        labels[noise_mask] = 1 - labels[noise_mask]
        
        X = np.column_stack([x, y_attr])
        
        return {
            'X': X,
            'y': labels,
            'drift_positions': drift_positions,
            'drift_type': 'sudden',
            'category': 'TCD',
            'drift_lengths': [1] * len(drift_positions),
            'name': 'Sine1'
        }
    
    def generate_sine2(self, drift_positions: List[int] = None) -> Dict:
        """
        Sine2 dataset: y = 0.5 + 0.3 * sin(3πx)
        """
        if drift_positions is None:
            drift_positions = [self.n_samples // 2]
        
        np.random.seed(self.random_state)
        
        x = np.random.uniform(0, 1, self.n_samples)
        y_attr = np.random.uniform(0, 1, self.n_samples)
        
        boundary = 0.5 + 0.3 * np.sin(3 * np.pi * x)
        labels = (y_attr < boundary).astype(int)
        
        # Apply drift
        for i, dp in enumerate(sorted(drift_positions)):
            labels[dp:] = 1 - labels[dp:]
        
        # Add noise
        noise_mask = np.random.random(self.n_samples) < self.noise
        labels[noise_mask] = 1 - labels[noise_mask]
        
        X = np.column_stack([x, y_attr])
        
        return {
            'X': X,
            'y': labels,
            'drift_positions': drift_positions,
            'drift_type': 'sudden',
            'category': 'TCD',
            'drift_lengths': [1] * len(drift_positions),
            'name': 'Sine2'
        }
    
    # ================================================================
    # CIRCLE DATASET (Gradual Drift - PCD)
    # ================================================================
    
    def generate_circle(self, n_concepts: int = 4) -> Dict:
        """
        Circle dataset: (x - x_c)² + (y - y_c)² = r_c²
        
        Paper sequence:
        (0.2, 0.5) r=0.15 → (0.4, 0.5) r=0.2 → (0.6, 0.5) r=0.25 → (0.8, 0.5) r=0.3
        """
        np.random.seed(self.random_state)
        
        # Circle parameters from paper
        concepts = [
            (0.2, 0.5, 0.15),
            (0.4, 0.5, 0.2),
            (0.6, 0.5, 0.25),
            (0.8, 0.5, 0.3)
        ][:n_concepts]
        
        X = np.random.uniform(0, 1, (self.n_samples, 2))
        y = np.zeros(self.n_samples, dtype=int)
        
        segment = self.n_samples // len(concepts)
        drift_positions = []
        drift_lengths = []
        
        transition_width = segment // 4  # Gradual transition
        
        for i, (x_c, y_c, r_c) in enumerate(concepts):
            start = i * segment
            end = (i + 1) * segment if i < len(concepts) - 1 else self.n_samples
            
            if i > 0:
                drift_positions.append(start)
            
            for t in range(start, end):
                dist = np.sqrt((X[t, 0] - x_c)**2 + (X[t, 1] - y_c)**2)
                
                # Gradual transition: probabilistic mixing
                if i > 0 and t - start < transition_width:
                    # Mix with previous concept
                    prev_x_c, prev_y_c, prev_r_c = concepts[i-1]
                    prev_dist = np.sqrt((X[t, 0] - prev_x_c)**2 + (X[t, 1] - prev_y_c)**2)
                    alpha = (t - start) / transition_width
                    
                    if np.random.random() < alpha:
                        y[t] = int(dist <= r_c)
                    else:
                        y[t] = int(prev_dist <= prev_r_c)
                else:
                    y[t] = int(dist <= r_c)
        
        # Add noise
        noise_mask = np.random.random(self.n_samples) < self.noise
        y[noise_mask] = 1 - y[noise_mask]
        
        return {
            'X': X,
            'y': y,
            'drift_positions': drift_positions,
            'drift_type': 'gradual',
            'category': 'PCD',
            'drift_lengths': [transition_width // (self.n_samples // 100)] * len(drift_positions),
            'name': 'Circle'
        }
    
    # ================================================================
    # GAUSSIAN DATASET (Incremental Drift - PCD)
    # ================================================================
    
    def generate_gaussian(self, n_features: int = 10) -> Dict:
        """
        Gaussian dataset: Moving mean Gaussian distribution
        
        Mean shifts linearly from 0 to target over time.
        """
        np.random.seed(self.random_state)
        
        mean_start = np.zeros(n_features)
        mean_end = np.ones(n_features) * 2.0
        
        X = np.zeros((self.n_samples, n_features))
        
        drift_start = self.n_samples // 4
        drift_end = 3 * self.n_samples // 4
        
        for t in range(self.n_samples):
            if t < drift_start:
                mean = mean_start
            elif t >= drift_end:
                mean = mean_end
            else:
                alpha = (t - drift_start) / (drift_end - drift_start)
                mean = (1 - alpha) * mean_start + alpha * mean_end
            
            X[t] = np.random.randn(n_features) * 0.5 + mean
        
        # Generate labels based on first feature
        y = (X[:, 0] > np.linspace(0, 2, self.n_samples) / 2).astype(int)
        
        # Add noise
        noise_mask = np.random.random(self.n_samples) < self.noise
        y[noise_mask] = 1 - y[noise_mask]
        
        return {
            'X': X,
            'y': y,
            'drift_positions': [drift_start],
            'drift_type': 'incremental',
            'category': 'PCD',
            'drift_lengths': [(drift_end - drift_start) // (self.n_samples // 100)],
            'name': 'Gaussian'
        }
    
    # ================================================================
    # RECURRENT DATASET (TCD)
    # ================================================================
    
    def generate_recurrent_sine(self, n_recurrences: int = 4) -> Dict:
        """
        Recurrent Sine: Alternates between two sine concepts
        """
        np.random.seed(self.random_state)
        
        x = np.random.uniform(0, 1, self.n_samples)
        y_attr = np.random.uniform(0, 1, self.n_samples)
        
        period = self.n_samples // (n_recurrences + 1)
        drift_positions = [period * (i + 1) for i in range(n_recurrences)]
        
        labels = np.zeros(self.n_samples, dtype=int)
        
        for t in range(self.n_samples):
            concept = (t // period) % 2
            
            if concept == 0:
                boundary = np.sin(np.pi * x[t])
            else:
                boundary = 0.5 + 0.3 * np.sin(3 * np.pi * x[t])
            
            labels[t] = int(y_attr[t] < boundary)
        
        # Add noise
        noise_mask = np.random.random(self.n_samples) < self.noise
        labels[noise_mask] = 1 - labels[noise_mask]
        
        X = np.column_stack([x, y_attr])
        
        return {
            'X': X,
            'y': labels,
            'drift_positions': drift_positions,
            'drift_type': 'recurrent',
            'category': 'TCD',
            'drift_lengths': [1] * len(drift_positions),
            'name': 'Recurrent_Sine'
        }
    
    # ================================================================
    # BLIP DATASET (TCD)
    # ================================================================
    
    def generate_blip_sine(self, blip_duration: int = None) -> Dict:
        """
        Blip: Temporary change in sine boundary, then returns
        """
        if blip_duration is None:
            blip_duration = self.n_samples // 6
        
        np.random.seed(self.random_state)
        
        x = np.random.uniform(0, 1, self.n_samples)
        y_attr = np.random.uniform(0, 1, self.n_samples)
        
        blip_start = self.n_samples // 3
        blip_end = blip_start + blip_duration
        
        labels = np.zeros(self.n_samples, dtype=int)
        
        for t in range(self.n_samples):
            if blip_start <= t < blip_end:
                # Different boundary during blip
                boundary = 0.5 + 0.3 * np.sin(3 * np.pi * x[t])
            else:
                # Normal boundary
                boundary = np.sin(np.pi * x[t])
            
            labels[t] = int(y_attr[t] < boundary)
        
        # Add noise
        noise_mask = np.random.random(self.n_samples) < self.noise
        labels[noise_mask] = 1 - labels[noise_mask]
        
        X = np.column_stack([x, y_attr])
        
        return {
            'X': X,
            'y': labels,
            'drift_positions': [blip_start, blip_end],
            'drift_type': 'blip',
            'category': 'TCD',
            'drift_lengths': [1, 1],
            'name': 'Blip_Sine'
        }
    
    # ================================================================
    # GENERATE ALL PAPER DATASETS
    # ================================================================
    
    def generate_all_paper_datasets(self) -> Dict[str, Dict]:
        """Generate all paper-style datasets"""
        return {
            'sine1_sudden': self.generate_sine1([self.n_samples // 2]),
            'sine2_sudden': self.generate_sine2([self.n_samples // 2]),
            'circle_gradual': self.generate_circle(),
            'gaussian_incremental': self.generate_gaussian(),
            'recurrent_sine': self.generate_recurrent_sine(),
            'blip_sine': self.generate_blip_sine()
        }


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PAPER DATASET GENERATOR TEST")
    print("=" * 60)
    
    gen = PaperDataGenerator(n_samples=10000, noise=0.1, random_state=42)
    datasets = gen.generate_all_paper_datasets()
    
    for name, data in datasets.items():
        print(f"\n{name}:")
        print(f"  Shape: X={data['X'].shape}, y={data['y'].shape}")
        print(f"  Type: {data['category']} ({data['drift_type']})")
        print(f"  Drift positions: {data['drift_positions']}")
        print(f"  Class balance: {np.mean(data['y']):.2f}")
