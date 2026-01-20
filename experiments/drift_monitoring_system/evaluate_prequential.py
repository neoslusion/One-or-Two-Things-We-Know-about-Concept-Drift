"""
Prequential Accuracy Evaluation for Drift Adaptation (Enhanced Version).

System Architecture:
--------------------
This script implements the evaluation of the Closed-Loop Adaptive Learning System defined in the thesis.
It integrates the following modules:

1.  **Stream Interface:** Generates synthetic data streams (Sudden, Gradual, Recurrent, etc.).
2.  **Inference Module:** Uses an incremental Learner (SGD or Retrainable Batch Classifier) to make predictions.
3.  **SE-CDT System (Monitor):** A Unified Detector-Classifier that:
    *   Detects drift using ShapeDD-ADW (Adaptive Density-Weighted MMD).
    *   Classifies drift type (Sudden/Gradual/etc.) using Signal Shape Analysis (Algorithm 3.4).
4.  **Adaptation Manager:** Selects the optimal adaptation strategy based on the classified drift type:
    *   Sudden -> Reset/Retrain
    *   Recurrent -> Retrieve from Cache
    *   Gradual -> Partial Update (or Retrain in this implementation)
    *   Blip -> Ignore

Usage:
------
    python evaluate_prequential.py --n_samples 5000 --drift_type sudden --w_ref 50 --sudden_thresh 0.5
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Callable, Optional, Dict, List

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SHAPE_DD_DIR = REPO_ROOT / "experiments" / "backup"
if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.insert(0, str(SHAPE_DD_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Import SE_CDT detector (Unified Detector-Classifier)
try:
    from se_cdt import SE_CDT
except ImportError:
    print("[WARNING] SE_CDT not found, using dummy detector")
    SE_CDT = None

# Import adaptation strategies
from adaptation_strategies import (
    adapt_sudden_drift,
    adapt_incremental_drift,
    adapt_gradual_drift,
    adapt_recurrent_drift,
    adapt_blip_drift
)

# Configuration
from config import (
    BUFFER_SIZE, CHUNK_SIZE, SHAPE_L1, SHAPE_L2, DRIFT_ALPHA, SHAPE_N_PERM,
    INITIAL_TRAINING_SIZE, PREQUENTIAL_WINDOW, ADAPTATION_WINDOW
)

# Strategy mapping
STRATEGY_MAP = {
    'sudden': adapt_sudden_drift,
    'incremental': adapt_incremental_drift,
    'gradual': adapt_gradual_drift,
    'recurrent': adapt_recurrent_drift,
    'blip': adapt_blip_drift,
}


def create_model():
    """Create sklearn Pipeline for classification."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])


def model_factory():
    """Factory function for creating new models (used by adapt_sudden_drift)."""
    return create_model()


# Import SOTA Generators
from data_generators import generate_sea_concepts, generate_mixed_drift_dataset

def generate_synthetic_stream(n_samples: int = 5000, n_drifts: int = 5, 
                               n_features: int = 5, drift_type: str = 'sudden',
                               random_seed: int = 42):
    """
    Wrapper for SOTA generators.
    """
    if drift_type == 'mixed':
        return generate_mixed_drift_dataset(n_samples, random_seed)
    else:
        # Default to SEA concepts for specific types (Sudden) or others
        # For simplicity in this script, we map types to SEA variations or Mixed
        if drift_type == 'sudden':
            return generate_sea_concepts(n_samples, n_drifts, random_seed)
        else:
            return generate_mixed_drift_dataset(n_samples, random_seed)

# ... (Configuration and imports preserved) ...

class AdaptationMode:
    """Enum-like class for adaptation modes."""
    NONE = 'no_adaptation'
    SIMPLE = 'simple_retrain'
    TYPE_SPECIFIC = 'type_specific'

def evaluate_with_adaptation(X, y, drift_points, mode: str = AdaptationMode.TYPE_SPECIFIC,
                              cache_dir: Optional[Path] = None, 
                              w_ref: int = 50, sudden_thresh: float = 0.5):
    """
    Evaluate adaptation strategies using SE-CDT Unified System.
    
    Modules:
    1. Stream Interface: Sliding window processing.
    2. Inference Module: Model prediction & accuracy tracking.
    3. SE-CDT System: Unified Detector-Classifier (Monitor).
    4. Adaptation Manager: Strategy selection and model update.
    """
    n_samples = len(X)
    
    # Initialize
    model = create_model()
    buffer = deque(maxlen=BUFFER_SIZE)
    recent_correct = deque(maxlen=PREQUENTIAL_WINDOW)
    
    accuracy_history = []
    drift_detections = []
    adaptations = []
    classification_times = []
    
    # Initialize SE-CDT Unified System
    # Note: window_size corresponds to l1 (reference window)
    se_cdt_system = SE_CDT(window_size=w_ref, l2=SHAPE_L2, threshold=sudden_thresh) if SE_CDT else None
    
    # Phase 1: Initial training
    train_end = min(INITIAL_TRAINING_SIZE, n_samples)
    model.fit(X[:train_end], y[:train_end])
    print(f"  [{mode}] Initial training on {train_end} samples")
    
    # Init buffer
    for i in range(train_end):
        buffer.append({'idx': i, 'x': X[i], 'y': y[i]})
        
    drift_detected = False
    samples_since_drift = 0
    adaptation_delay = 50
    current_drift_type = 'sudden' # Default
    
    # CRITICAL: For No Adaptation, we must NOT update this model object
    # We keep 'model' as is.
    
    for idx in range(train_end, n_samples):
        x = X[idx:idx+1]
        y_true = y[idx]
        
        # Predict
        y_pred = model.predict(x)[0]
        is_correct = (y_pred == y_true)
        recent_correct.append(is_correct)
        
        accuracy = np.mean(recent_correct)
        accuracy_history.append({'idx': idx, 'accuracy': accuracy})
        
        buffer.append({'idx': idx, 'x': X[idx], 'y': y_true})
        
        # Drift monitoring via SE-CDT System
        if len(buffer) >= BUFFER_SIZE and idx % CHUNK_SIZE == 0:
            if se_cdt_system is not None and not drift_detected:
                buffer_X = np.array([item['x'] for item in buffer])
                try:
                    # Call Unified Monitor
                    result = se_cdt_system.monitor(buffer_X)
                    
                    if result.is_drift:
                        drift_detected = True
                        drift_detections.append(idx)
                        classification_times.append(result.classification_time)
                        
                        # Use classified drift type immediately
                        detected_drift_type = result.subcategory
                        print(f"    [{mode}] Drift detected at sample {idx} (Score={result.score:.4f}, Type={detected_drift_type})")
                        
                        # Store result for adaptation step logic
                        current_drift_type = detected_drift_type
                        
                except Exception as e:
                    print(f"Error in SE-CDT monitor: {e}")
                    pass

        # Adaptation Logic
        if drift_detected:
            samples_since_drift += 1
            if samples_since_drift >= adaptation_delay:
                # 1. NO ADAPTATION
                if mode == AdaptationMode.NONE:
                    # Do nothing to the model! 
                    drift_detected = False
                    continue 

                # Prepare data
                adapt_start = max(0, len(buffer) - ADAPTATION_WINDOW)
                adapt_data = list(buffer)[adapt_start:]
                adapt_X = np.array([item['x'] for item in adapt_data])
                adapt_y = np.array([item['y'] for item in adapt_data])
                
                # 2. SIMPLE RETRAIN
                if mode == AdaptationMode.SIMPLE:
                    # Naively retrain on recent window
                    # Force a new model instance to be sure
                    model = create_model()
                    model.fit(adapt_X, adapt_y)
                    strategy_used = 'simple_retrain'
                    
                # 3. TYPE SPECIFIC
                elif mode == AdaptationMode.TYPE_SPECIFIC:
                    # Use the type detected by SE-CDT
                    drift_type = current_drift_type
                    print(f"    [{mode}] Adapting for: {drift_type}")
                    
                    if drift_type == 'Sudden' or drift_type == 'TCD': # Handle capitalization variations
                        model = adapt_sudden_drift(model_factory, adapt_X, adapt_y)
                        strategy_used = 'sudden'
                    elif drift_type == 'Recurrent':
                        # Try to load from cache
                        model = adapt_recurrent_drift(model_factory, model, adapt_X, adapt_y, cache_dir=cache_dir)
                        strategy_used = 'recurrent'
                    elif drift_type == 'Gradual':
                        # Use weighted approach or wait?
                        # For now, standard adapt
                        model = adapt_gradual_drift(model, adapt_X, adapt_y)
                        strategy_used = 'gradual'
                    elif drift_type == 'Blip':
                         # Blip: Do nothing or filter
                         print(f"    [{mode}] Blip detected - Ignoring adaptation")
                         strategy_used = 'blip_ignored'
                         # We don't update model for Blip
                    else:
                        model = adapt_incremental_drift(model, adapt_X, adapt_y)
                        strategy_used = 'incremental'
                
                adaptations.append({'idx': idx, 'strategy': strategy_used if mode == AdaptationMode.TYPE_SPECIFIC else 'simple'})
                drift_detected = False
                samples_since_drift = 0
                
    return accuracy_history, drift_detections, adaptations, classification_times


def plot_prequential_comparison(results: Dict[str, Dict], drift_points: List[int], output_path: Path):
    """
    Plot prequential accuracy comparison for all modes.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    colors = {
        AdaptationMode.TYPE_SPECIFIC: 'green',
        AdaptationMode.SIMPLE: 'blue',
        AdaptationMode.NONE: 'red'
    }
    labels = {
        AdaptationMode.TYPE_SPECIFIC: 'Type-Specific Adaptation (5 strategies)',
        AdaptationMode.SIMPLE: 'Simple Retrain (baseline)',
        AdaptationMode.NONE: 'No Adaptation'
    }
    
    for mode, data in results.items():
        idx_list = [r['idx'] for r in data['accuracy']]
        acc_list = [r['accuracy'] for r in data['accuracy']]
        ax.plot(idx_list, acc_list, color=colors.get(mode, 'gray'), 
                linewidth=1.5, label=labels.get(mode, mode), alpha=0.8)
        
        # Mark adaptations for type-specific
        if mode == AdaptationMode.TYPE_SPECIFIC:
            for adapt in data['adaptations']:
                ax.scatter(adapt['idx'], 0.95, marker='^', color='purple', s=100, zorder=5)
    
    # Mark true drift points
    for i, dp in enumerate(drift_points):
        ax.axvline(x=dp, color='gray', linestyle='--', alpha=0.5, 
                   label='True Drift' if i == 0 else '_nolegend_')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Prequential Accuracy', fontsize=12)
    ax.set_title('Prequential Accuracy: Type-Specific vs Simple vs No Adaptation', fontsize=14)
    ax.legend(loc='lower left')
    ax.set_ylim([0.3, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('↑ Adaptation events', xy=(0.02, 0.96), xycoords='axes fraction', 
                fontsize=10, color='purple')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def calculate_metrics(results: Dict[str, Dict], drift_points: List[int]) -> Dict:
    """
    Calculate comparison metrics for all modes.
    """
    metrics = {}
    
    for mode, data in results.items():
        acc_list = [r['accuracy'] for r in data['accuracy']]
        idx_list = [r['idx'] for r in data['accuracy']]
        
        # Overall accuracy
        mean_acc = np.mean(acc_list)
        
        # Post-drift accuracy (500 samples after each drift)
        post_drift_acc = []
        for dp in drift_points:
            for i, idx in enumerate(idx_list):
                if dp <= idx < dp + 500:
                    post_drift_acc.append(acc_list[i])
        
        post_drift_mean = np.mean(post_drift_acc) if post_drift_acc else 0
        
        # Classification time (only for Type Specific)
        avg_class_time = 0
        if 'classification_times' in data and len(data['classification_times']) > 0:
            avg_class_time = np.mean(data['classification_times']) * 1000 # Convert to ms
        
        metrics[mode] = {
            'overall_accuracy': mean_acc,
            'post_drift_accuracy': post_drift_mean,
            'n_detections': len(data['detections']),
            'n_adaptations': len(data['adaptations']),
            'avg_classification_time_ms': avg_class_time
        }
    
    # Calculate improvements
    if AdaptationMode.NONE in metrics and AdaptationMode.TYPE_SPECIFIC in metrics:
        baseline = metrics[AdaptationMode.NONE]['overall_accuracy']
        type_spec = metrics[AdaptationMode.TYPE_SPECIFIC]['overall_accuracy']
        simple = metrics.get(AdaptationMode.SIMPLE, {}).get('overall_accuracy', 0)
        
        metrics['improvement_vs_none'] = type_spec - baseline
        metrics['improvement_pct_vs_none'] = (type_spec - baseline) / baseline * 100 if baseline > 0 else 0
        metrics['improvement_vs_simple'] = type_spec - simple
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Prequential Accuracy Evaluation (Enhanced)')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--n_drifts', type=int, default=5, help='Number of drift points')
    parser.add_argument('--drift_type', type=str, default='sudden', 
                        choices=['sudden', 'gradual', 'incremental', 'recurrent'],
                        help='Type of drift to simulate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./prequential_results', help='Output directory')
    # Added Arguments
    parser.add_argument('--w_ref', type=int, default=50, help='Reference window size (l1)')
    parser.add_argument('--sudden_thresh', type=float, default=0.5, help='Threshold for sudden drift detection')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache directory for recurrent drift
    cache_dir = output_dir / 'model_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PREQUENTIAL ACCURACY EVALUATION (Enhanced with 5 Strategies)")
    print("="*80)
    print(f"Samples: {args.n_samples}, Drifts: {args.n_drifts}, Type: {args.drift_type}, Seed: {args.seed}")
    print(f"SE-CDT Config: w_ref={args.w_ref}, thresh={args.sudden_thresh}")
    print("="*80)
    
    # Generate data
    print("\n[1] Generating synthetic data stream...")
    X, y, drift_points, drift_types = generate_synthetic_stream(
        n_samples=args.n_samples,
        n_drifts=args.n_drifts,
        drift_type=args.drift_type,
        random_seed=args.seed
    )
    
    results = {}
    
    # Evaluate WITH type-specific adaptation
    print("\n[2] Evaluating WITH type-specific adaptation (5 strategies)...")
    acc_ts, det_ts, adapt_ts, times_ts = evaluate_with_adaptation(
        X, y, drift_points, mode=AdaptationMode.TYPE_SPECIFIC, cache_dir=cache_dir,
        w_ref=args.w_ref, sudden_thresh=args.sudden_thresh
    )
    results[AdaptationMode.TYPE_SPECIFIC] = {
        'accuracy': acc_ts, 'detections': det_ts, 'adaptations': adapt_ts, 'classification_times': times_ts
    }
    
    # Evaluate WITH simple adaptation
    print("\n[3] Evaluating WITH simple retrain adaptation...")
    acc_simple, det_simple, adapt_simple, _ = evaluate_with_adaptation(
        X, y, drift_points, mode=AdaptationMode.SIMPLE
    )
    results[AdaptationMode.SIMPLE] = {'accuracy': acc_simple, 'detections': det_simple, 'adaptations': adapt_simple}
    
    # Evaluate WITHOUT adaptation
    print("\n[4] Evaluating WITHOUT adaptation (baseline)...")
    acc_none, det_none, adapt_none, _ = evaluate_with_adaptation(
        X, y, drift_points, mode=AdaptationMode.NONE
    )
    results[AdaptationMode.NONE] = {'accuracy': acc_none, 'detections': det_none, 'adaptations': adapt_none}
    
    # Calculate metrics
    print("\n[5] Calculating metrics...")
    metrics = calculate_metrics(results, drift_points)
    
    # Calculate SOTA Adaptation Metrics (Recovery Speed, etc.)
    print("\n[5.1] Calculating SOTA Adaptation Metrics (Recovery Speed, Performance Loss)...")
    sota_metrics = {}
    for mode, data in results.items():
        sota_metrics[mode] = calculate_sota_adaptation_metrics(
            data['accuracy'], drift_points, n_samples=args.n_samples, adaptation_window=500
        )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nDrift Type: {args.drift_type.upper()}")
    print("-"*40)
    for mode in [AdaptationMode.TYPE_SPECIFIC, AdaptationMode.SIMPLE, AdaptationMode.NONE]:
        m = metrics.get(mode, {})
        s = sota_metrics.get(mode, {})
        time_str = f", ClassTime={m.get('avg_classification_time_ms', 0):.2f}ms" if mode == AdaptationMode.TYPE_SPECIFIC else ""
        
        print(f"{mode:20s}: Accuracy = {m.get('overall_accuracy', 0):.4f}, "
              f"Restoration Time = {s.get('avg_restoration_time', 0):.1f} samples, "
              f"Perf Loss = {s.get('avg_performance_loss', 0):.4f}" + time_str)
    
    print(f"\nImprovement (Type-Specific vs No-Adaptation): "
          f"+{metrics.get('improvement_vs_none', 0):.4f} "
          f"(+{metrics.get('improvement_pct_vs_none', 0):.1f}%)")
    print(f"Improvement (Type-Specific vs Simple):        "
          f"+{metrics.get('improvement_vs_simple', 0):.4f}")
    print("="*80)
    
    # Generate plot
    print("\n[6] Generating plot...")
    plot_path = output_dir / f"prequential_accuracy_{args.drift_type}.png"
    plot_prequential_comparison(results, drift_points, plot_path)
    
    # Save metrics to file
    metrics_path = output_dir / f"metrics_{args.drift_type}.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Prequential Accuracy Evaluation Results (SOTA Enhanced)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Samples: {args.n_samples}, Drifts: {args.n_drifts}, Type: {args.drift_type}, Seed: {args.seed}\n\n")
        f.write(f"Config: w_ref={args.w_ref}, thresh={args.sudden_thresh}\n")
        
        for mode in [AdaptationMode.TYPE_SPECIFIC, AdaptationMode.SIMPLE, AdaptationMode.NONE]:
            m = metrics.get(mode, {})
            s = sota_metrics.get(mode, {})
            f.write(f"\n{mode}:\n")
            f.write(f"  Overall Accuracy: {m.get('overall_accuracy', 0):.4f}\n")
            f.write(f"  Post-Drift Acc:   {m.get('post_drift_accuracy', 0):.4f}\n")
            f.write(f"  Adaptations:      {m.get('n_adaptations', 0)}\n")
            if mode == AdaptationMode.TYPE_SPECIFIC:
                f.write(f"  Avg Class. Time:  {m.get('avg_classification_time_ms', 0):.4f} ms\n")
            f.write(f"  Restoration Time: {s.get('avg_restoration_time', float('inf')):.1f} samples\n")
            f.write(f"  Performance Loss: {s.get('avg_performance_loss', 0):.4f}\n")
            f.write(f"  Convergence Rate: {s.get('convergence_rate', 0):.5f}\n")

    print(f"Metrics saved to: {metrics_path}")
    
    print("\n✓ Evaluation complete!")
    return metrics, sota_metrics

def calculate_sota_adaptation_metrics(accuracy_history, drift_points, n_samples, adaptation_window=500):
    """
    Calculate SOTA adaptation metrics:
    1. Restoration Time: Samples to reach 95% of pre-drift accuracy (or 90% absolute).
    2. Performance Loss: Average accuracy drop during drift period.
    3. Convergence Rate: Slope of recovery.
    """
    metrics = {
        'avg_restoration_time': 0.0,
        'avg_performance_loss': 0.0,
        'convergence_rate': 0.0
    }
    
    if not drift_points:
        return metrics
        
    restoration_times = []
    perf_losses = []
    slopes = []
    
    # Convert history to easier format
    acc_map = {item['idx']: item['accuracy'] for item in accuracy_history}
    
    for drift_idx in drift_points:
        # Pre-drift accuracy (baseline) - avg of 100 samples before
        pre_drift = [acc_map.get(i, 0) for i in range(drift_idx - 100, drift_idx)]
        baseline = np.mean(pre_drift) if pre_drift else 0.8  # Default fallback
        target_acc = max(0.85, baseline * 0.95) # Target to be considered "restored"
        
        # Analyze post-drift window
        window_end = min(n_samples, drift_idx + adaptation_window)
        post_drift_accs = [acc_map.get(i, 0) for i in range(drift_idx, window_end)]
        
        # 1. Restoration Time
        restored_at = float('inf')
        for i, acc in enumerate(post_drift_accs):
            # Check if stabilized above target for at least 30 samples
            if acc >= target_acc:
                # Look ahead 30 samples to ensure stability
                future_accs = post_drift_accs[i:i+30]
                if len(future_accs) > 0 and np.mean(future_accs) >= target_acc:
                    restored_at = i
                    break
        
        restoration_times.append(restored_at if restored_at != float('inf') else adaptation_window)
        
        # 2. Performance Loss (Integral of error)
        # Loss = Sum(Baseline - Current) for samples where Current < Baseline
        loss = np.sum([max(0, baseline - acc) for acc in post_drift_accs]) / len(post_drift_accs) if post_drift_accs else 0
        perf_losses.append(loss)
        
        # 3. Convergence Rate (Slope of linear fit to first 100 samples)
        if len(post_drift_accs) >= 10:
            fit_window = min(100, len(post_drift_accs))
            y_vals = post_drift_accs[:fit_window]
            x_vals = np.arange(len(y_vals))
            slope, _ = np.polyfit(x_vals, y_vals, 1)
            slopes.append(slope)
            
    metrics['avg_restoration_time'] = np.mean(restoration_times)
    metrics['avg_performance_loss'] = np.mean(perf_losses)
    metrics['convergence_rate'] = np.mean(slopes) if slopes else 0
    
    return metrics


if __name__ == "__main__":
    main()
