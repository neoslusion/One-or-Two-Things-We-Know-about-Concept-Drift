"""
Benchmark Framework for Concept Drift Type Identification
Following CDT_MSW paper: Guo et al., Information Sciences 585 (2022) 1-23

Metrics:
- EDR: Error Detection Rate
- MDR: Missed Detection Rate  
- ADD: Average Detection Delay (for PCD only)
- ACC_cat: Category Accuracy (TCD vs PCD)
- ACC_subcat: Subcategory Accuracy (Sudden, Blip, Recurrent, Incremental, Gradual)

Author: [Your Name]
"""

import numpy as np
from sklearn.datasets import make_classification
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
import time
from collections import defaultdict
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import methods from component files
from experiments.drift_detection_benchmark.cdt_msw import CDT_MSW
from experiments.drift_detection_benchmark.shapedd_cdt import ShapedCDT_V5 as SHAPED_CDT

warnings.filterwarnings('ignore')


# ============================================================
# DATA GENERATION (Following CDT_MSW Paper Section 5.1)
# ============================================================

class DriftDataGenerator:
    """
    Generate synthetic datasets with controlled concept drift.
    
    Following CDT_MSW paper specifications:
    - TCD (Transient): Sudden, Blip, Recurrent
    - PCD (Progressive): Incremental, Gradual
    
    Drift positions inserted at early, middle, and late stages.
    """
    
    def __init__(self, n_samples: int = 10000, n_features: int = 10, 
                 n_classes: int = 2, noise: float = 0.1, random_state: int = 42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.noise = noise
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _generate_concept(self, n: int, concept_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data from a specific concept (distribution)"""
        np.random.seed(self.random_state + concept_id * 100)
        X, y = make_classification(
            n_samples=n,
            n_features=self.n_features,
            n_informative=self.n_features // 2,
            n_redundant=self.n_features // 4,
            n_classes=self.n_classes,
            flip_y=self.noise,
            random_state=self.random_state + concept_id
        )
        return X, y
    
    def generate_sudden(self, drift_positions: List[int] = None) -> Dict:
        """
        Generate Sudden drift (TCD).
        Abrupt change from one distribution to another.
        """
        if drift_positions is None:
            drift_positions = [self.n_samples // 2]
        
        all_positions = [0] + sorted(drift_positions) + [self.n_samples]
        X_list, y_list = [], []
        
        for i in range(len(all_positions) - 1):
            start, end = all_positions[i], all_positions[i + 1]
            X_seg, y_seg = self._generate_concept(end - start, concept_id=i)
            X_list.append(X_seg)
            y_list.append(y_seg)
        
        return {
            'X': np.vstack(X_list),
            'y': np.concatenate(y_list),
            'drift_positions': drift_positions,
            'drift_type': 'sudden',
            'category': 'TCD',
            'drift_lengths': [1] * len(drift_positions)
        }
    
    def generate_blip(self, drift_position: int = None, blip_length: int = None) -> Dict:
        """
        Generate Blip drift (TCD).
        Temporary change then return to original distribution.
        """
        if drift_position is None:
            drift_position = self.n_samples // 3
        if blip_length is None:
            blip_length = self.n_samples // 6
        
        end_blip = drift_position + blip_length
        
        # Before blip
        X1, y1 = self._generate_concept(drift_position, concept_id=0)
        # During blip (different distribution)
        X2, y2 = self._generate_concept(blip_length, concept_id=1)
        # After blip (back to original)
        X3, y3 = self._generate_concept(self.n_samples - end_blip, concept_id=0)
        
        return {
            'X': np.vstack([X1, X2, X3]),
            'y': np.concatenate([y1, y2, y3]),
            'drift_positions': [drift_position, end_blip],
            'drift_type': 'blip',
            'category': 'TCD',
            'drift_lengths': [1, 1]
        }
    
    def generate_recurrent(self, n_recurrences: int = 4) -> Dict:
        """
        Generate Recurrent drift (TCD).
        Periodic alternation between concepts.
        """
        period = self.n_samples // (n_recurrences + 1)
        drift_positions = [period * (i + 1) for i in range(n_recurrences)]
        
        X_list, y_list = [], []
        all_positions = [0] + drift_positions + [self.n_samples]
        
        for i in range(len(all_positions) - 1):
            start, end = all_positions[i], all_positions[i + 1]
            concept_id = i % 2  # Alternate between 2 concepts
            X_seg, y_seg = self._generate_concept(end - start, concept_id=concept_id)
            X_list.append(X_seg)
            y_list.append(y_seg)
        
        return {
            'X': np.vstack(X_list),
            'y': np.concatenate(y_list),
            'drift_positions': drift_positions,
            'drift_type': 'recurrent',
            'category': 'TCD',
            'drift_lengths': [1] * len(drift_positions)
        }
    
    def generate_incremental(self, drift_start: int = None, drift_length: int = None) -> Dict:
        """
        Generate Incremental drift (PCD).
        Very gradual change with 10% step size (as per paper).
        """
        if drift_start is None:
            drift_start = self.n_samples // 4
        if drift_length is None:
            drift_length = self.n_samples // 2
        
        X_old, y_old = self._generate_concept(self.n_samples, concept_id=0)
        X_new, y_new = self._generate_concept(self.n_samples, concept_id=1)
        
        X = np.zeros_like(X_old)
        y = np.zeros_like(y_old)
        
        drift_end = drift_start + drift_length
        n_steps = 10  # 10% increments as per paper
        step_size = drift_length // n_steps
        
        for i in range(self.n_samples):
            if i < drift_start:
                X[i], y[i] = X_old[i], y_old[i]
            elif i >= drift_end:
                X[i], y[i] = X_new[i], y_new[i]
            else:
                # Incremental: step-wise increase
                step = (i - drift_start) // step_size
                prob_new = min(1.0, (step + 1) * 0.1)
                if np.random.random() < prob_new:
                    X[i], y[i] = X_new[i], y_new[i]
                else:
                    X[i], y[i] = X_old[i], y_old[i]
        
        return {
            'X': X,
            'y': y,
            'drift_positions': [drift_start],
            'drift_type': 'incremental',
            'category': 'PCD',
            'drift_lengths': [drift_length // (self.n_samples // 100)]  # In blocks
        }
    
    def generate_gradual(self, drift_start: int = None, drift_length: int = None) -> Dict:
        """
        Generate Gradual drift (PCD).
        Smooth transition between distributions.
        """
        if drift_start is None:
            drift_start = self.n_samples // 4
        if drift_length is None:
            drift_length = self.n_samples // 2
        
        X_old, y_old = self._generate_concept(self.n_samples, concept_id=0)
        X_new, y_new = self._generate_concept(self.n_samples, concept_id=1)
        
        X = np.zeros_like(X_old)
        y = np.zeros_like(y_old)
        
        drift_end = drift_start + drift_length
        
        for i in range(self.n_samples):
            if i < drift_start:
                X[i], y[i] = X_old[i], y_old[i]
            elif i >= drift_end:
                X[i], y[i] = X_new[i], y_new[i]
            else:
                # Gradual: linear probability increase
                prob_new = (i - drift_start) / drift_length
                if np.random.random() < prob_new:
                    X[i], y[i] = X_new[i], y_new[i]
                else:
                    X[i], y[i] = X_old[i], y_old[i]
        
        return {
            'X': X,
            'y': y,
            'drift_positions': [drift_start],
            'drift_type': 'gradual',
            'category': 'PCD',
            'drift_lengths': [drift_length // (self.n_samples // 100)]
        }
    
    def generate_all_types(self) -> Dict[str, Dict]:
        """Generate all drift types for benchmarking"""
        return {
            'sudden': self.generate_sudden([self.n_samples // 2]),
            'blip': self.generate_blip(),
            'recurrent': self.generate_recurrent(n_recurrences=4),
            'incremental': self.generate_incremental(),
            'gradual': self.generate_gradual()
        }


# ============================================================
# EVALUATION METRICS (Following CDT_MSW Paper Section 5.3)
# ============================================================

class DriftMetrics:
    """
    Evaluation metrics from CDT_MSW paper:
    - EDR: Error Detection Rate
    - MDR: Missed Detection Rate
    - ADD: Average Detection Delay (PCD only)
    - ACC_cat: Category Accuracy
    - ACC_subcat: Subcategory Accuracy
    """
    
    def __init__(self, buffer_size: int = 3):
        """
        buffer_size: Number of blocks tolerance for position matching
                    (as per paper's Buf_B and Buf_E)
        """
        self.buffer_size = buffer_size
    
    def compute_edr(self, detected: List[int], true_positions: List[int], 
                    block_size: int) -> float:
        """
        Error Detection Rate: Proportion of false positives
        EDR = |{d ∈ D | ∀i, d ∉ Length_i}| / |D|
        """
        if len(detected) == 0:
            return 0.0
        
        buffer = self.buffer_size * block_size
        false_positives = 0
        
        for d in detected:
            matched = any(abs(d - t) <= buffer for t in true_positions)
            if not matched:
                false_positives += 1
        
        return false_positives / len(detected)
    
    def compute_mdr(self, detected: List[int], true_positions: List[int],
                    block_size: int) -> float:
        """
        Missed Detection Rate: Proportion of missed drifts
        MDR = |{t_i ∈ T | ∀d, d ∉ Length_i}| / |T|
        """
        if len(true_positions) == 0:
            return 0.0
        
        buffer = self.buffer_size * block_size
        missed = 0
        
        for t in true_positions:
            matched = any(abs(d - t) <= buffer for d in detected)
            if not matched:
                missed += 1
        
        return missed / len(true_positions)
    
    def compute_add(self, detected: List[int], true_positions: List[int],
                    block_size: int) -> float:
        """
        Average Detection Delay: Mean delay in detecting PCD
        ADD = Σ min|d - b| / |matched|
        """
        if len(detected) == 0 or len(true_positions) == 0:
            return float('inf')
        
        buffer = self.buffer_size * block_size
        delays = []
        
        for t in true_positions:
            matched_delays = [abs(d - t) for d in detected if abs(d - t) <= buffer]
            if matched_delays:
                delays.append(min(matched_delays))
        
        if len(delays) == 0:
            return float('inf')
        
        return np.mean(delays) / block_size  # In blocks
    
    def compute_category_accuracy(self, detected_cat: str, true_cat: str) -> float:
        """Category accuracy (TCD vs PCD)"""
        return 1.0 if detected_cat == true_cat else 0.0
    
    def compute_subcategory_accuracy(self, detected_sub: str, true_sub: str) -> float:
        """Subcategory accuracy"""
        return 1.0 if detected_sub == true_sub else 0.0


# ============================================================
# BENCHMARK RUNNER
# ============================================================

class Benchmark:
    """
    Comprehensive benchmark following CDT_MSW paper methodology
    """
    
    def __init__(self, n_runs: int = 10, block_sizes: List[int] = None):
        self.n_runs = n_runs
        self.block_sizes = block_sizes or [20, 40, 60]
        self.metrics = DriftMetrics()
        self.results = defaultdict(lambda: defaultdict(list))
    
    def run_single_experiment(self, detector, data: Dict, block_size: int) -> Dict:
        """Run single experiment and compute metrics"""
        X, y = data['X'], data['y']
        true_positions = data['drift_positions']
        true_category = data['category']
        true_subcategory = data['drift_type']
        
        # Run detector
        start_time = time.time()
        result = detector.detect(X, y)
        end_time = time.time()
        
        # Normalize result format
        if isinstance(result, dict):
            # CDT_MSW returns a dict
            detected_positions = result.get('drift_positions', [])
            drift_cats = result.get('drift_categories', [])
            drift_subcats = result.get('drift_subcategories', [])
            
            detected_category = drift_cats[0] if drift_cats else None
            detected_subcategory = drift_subcats[0] if drift_subcats else None
            processing_time = end_time - start_time
        else:
            # SHAPED_CDT returns an object
            if hasattr(result, 'detected_positions'):
                detected_positions = result.detected_positions
                detected_category = result.detected_category
                detected_subcategory = result.detected_subcategory
                processing_time = result.processing_time
            else:
                detected_positions = result.positions
                detected_category = result.category
                detected_subcategory = result.subcategory
                processing_time = end_time - start_time
        
        # Compute metrics
        edr = self.metrics.compute_edr(detected_positions, true_positions, block_size)
        mdr = self.metrics.compute_mdr(detected_positions, true_positions, block_size)
        
        if true_category == 'PCD':
            add = self.metrics.compute_add(detected_positions, true_positions, block_size)
        else:
            add = None
        
        cat_acc = self.metrics.compute_category_accuracy(detected_category, true_category)
        sub_acc = self.metrics.compute_subcategory_accuracy(detected_subcategory, true_subcategory)
        
        return {
            'edr': edr,
            'mdr': mdr,
            'add': add,
            'cat_acc': cat_acc,
            'sub_acc': sub_acc,
            'time': processing_time,
            'n_detected': len(detected_positions),
            'n_true': len(true_positions)
        }
    
    def run_benchmark(self, detectors: Dict, datasets: Dict[str, Dict]) -> Dict:
        """Run full benchmark"""
        all_results = {}
        
        for det_name, detector_class in detectors.items():
            print(f"\n{'='*60}")
            print(f"Running: {det_name}")
            print('='*60)
            
            det_results = {}
            
            for block_size in self.block_sizes:
                print(f"\n  Block size s = {block_size}")
                
                # Initialize detector with appropriate parameters
                if det_name == 'CDT_MSW':
                    # Tuned parameters for sensitivity
                    detector = detector_class(s=block_size, sigma=0.85, d=0.005, n=6)
                else:
                    detector = detector_class(window_size=block_size * 5, stride=block_size)
                
                block_results = {}
                
                for drift_type, data in datasets.items():
                    type_metrics = defaultdict(list)
                    
                    for run in range(self.n_runs):
                        # Regenerate data with different seed
                        gen = DriftDataGenerator(n_samples=10000, random_state=42 + run)
                        
                        if drift_type == 'sudden':
                            data = gen.generate_sudden([5000])
                        elif drift_type == 'blip':
                            data = gen.generate_blip()
                        elif drift_type == 'recurrent':
                            data = gen.generate_recurrent()
                        elif drift_type == 'incremental':
                            data = gen.generate_incremental()
                        elif drift_type == 'gradual':
                            data = gen.generate_gradual()
                        
                        metrics = self.run_single_experiment(detector, data, block_size)
                        
                        for k, v in metrics.items():
                            if v is not None:
                                type_metrics[k].append(v)
                    
                    # Aggregate metrics
                    block_results[drift_type] = {
                        k: (np.mean(v), np.std(v)) for k, v in type_metrics.items()
                    }
                    
                    # Print summary
                    m = block_results[drift_type]
                    print(f"    {drift_type:12s}: EDR={m['edr'][0]:.3f}±{m['edr'][1]:.3f}, "
                          f"MDR={m['mdr'][0]:.3f}±{m['mdr'][1]:.3f}, "
                          f"CAT={m['cat_acc'][0]:.3f}, SUB={m['sub_acc'][0]:.3f}")
                
                det_results[block_size] = block_results
            
            all_results[det_name] = det_results
        
        return all_results

    def plot_results(self, results: Dict, output_file: str = 'benchmark_results.png'):
        """
        Visualize benchmark results: Grouped bar charts for metrics by Drift Type.
        Aggregates results across block sizes.
        """
        print(f"\nGenerating visualization: {output_file}...")
        
        # Prepare data for plotting
        plot_data = []
        for det_name, det_results in results.items():
            for block_size, block_results in det_results.items():
                for drift_type, metrics in block_results.items():
                    plot_data.append({
                        'Method': det_name,
                        'Drift Type': drift_type,
                        'Block Size': block_size,
                        'Category Accuracy': metrics['cat_acc'][0],
                        'Subcategory Accuracy': metrics['sub_acc'][0],
                        'EDR (False Pos)': metrics['edr'][0],
                        'MDR (Missed)': metrics['mdr'][0]
                    })
        
        df = pd.DataFrame(plot_data)
        
        # Aggregate across block sizes for a cleaner high-level view
        df_agg = df.groupby(['Method', 'Drift Type']).mean().reset_index()
        
        # Set up the figure
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics_to_plot = [
            ('Category Accuracy', 'Category Accuracy (Higher is Better)', axes[0, 0]),
            ('Subcategory Accuracy', 'Subcategory Accuracy (Higher is Better)', axes[0, 1]),
            ('EDR (False Pos)', 'Error Detection Rate (Lower is Better)', axes[1, 0]),
            ('MDR (Missed)', 'Missed Detection Rate (Lower is Better)', axes[1, 1])
        ]
        
        for metric, title, ax in metrics_to_plot:
            sns.barplot(data=df_agg, x='Drift Type', y=metric, hue='Method', ax=ax, palette='viridis')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.set_xlabel('')
            ax.legend(loc='upper right')
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)

        plt.suptitle(f'Benchmark Results: CDT_MSW vs SHAPED_CDT\n(Averaged across window sizes)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")


    
    def print_summary_table(self, results: Dict):
        """Print summary table in paper format"""
        print("\n" + "="*80)
        print("SUMMARY TABLE (Following CDT_MSW Paper Format)")
        print("="*80)
        
        # Table header
        print(f"\n{'Method':<15} {'s':<5} {'Type':<12} {'EDR':<12} {'MDR':<12} {'ADD':<10} {'CAT_ACC':<10} {'SUB_ACC':<10}")
        print("-"*80)
        
        for det_name, det_results in results.items():
            for block_size, block_results in det_results.items():
                for drift_type, metrics in block_results.items():
                    edr = f"{metrics['edr'][0]:.3f}±{metrics['edr'][1]:.2f}"
                    mdr = f"{metrics['mdr'][0]:.3f}±{metrics['mdr'][1]:.2f}"
                    add = f"{metrics.get('add', (float('nan'), 0))[0]:.2f}" if 'add' in metrics else "N/A"
                    cat = f"{metrics['cat_acc'][0]:.3f}"
                    sub = f"{metrics['sub_acc'][0]:.3f}"
                    
                    print(f"{det_name:<15} {block_size:<5} {drift_type:<12} {edr:<12} {mdr:<12} {add:<10} {cat:<10} {sub:<10}")
        
        # Average by method
        print("\n" + "-"*80)
        print("AVERAGE BY METHOD:")
        print("-"*80)
        
        for det_name, det_results in results.items():
            all_edr, all_mdr, all_cat, all_sub = [], [], [], []
            
            for block_results in det_results.values():
                for metrics in block_results.values():
                    all_edr.append(metrics['edr'][0])
                    all_mdr.append(metrics['mdr'][0])
                    all_cat.append(metrics['cat_acc'][0])
                    all_sub.append(metrics['sub_acc'][0])
            
            print(f"{det_name:<15}: EDR={np.mean(all_edr):.3f}, MDR={np.mean(all_mdr):.3f}, "
                  f"CAT_ACC={np.mean(all_cat):.3f}, SUB_ACC={np.mean(all_sub):.3f}")


def main():
    detectors = {
        'CDT_MSW': CDT_MSW,
        'SHAPED_CDT': SHAPED_CDT
    }
    
    gen = DriftDataGenerator(n_samples=10000, random_state=42)
    datasets = gen.generate_all_types()

    benchmark = Benchmark(n_runs=5, block_sizes=[20, 40, 60])
    results = benchmark.run_benchmark(detectors, datasets)
    
    benchmark.print_summary_table(results)
    
    # Generate Plots
    benchmark.plot_results(results, output_file='experiments/drift_detection_benchmark/publication_figures/benchmark_comparison.png')

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = main()
