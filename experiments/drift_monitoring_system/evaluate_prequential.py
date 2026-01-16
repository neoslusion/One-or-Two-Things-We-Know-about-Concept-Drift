"""
Prequential Accuracy Evaluation for Drift Adaptation (Enhanced Version).

This script evaluates the 5 drift-type-specific adaptation strategies in an offline batch setting,
comparing:
1. WITH drift-type-specific adaptation (sudden, incremental, gradual, recurrent, blip)
2. WITH simple retrain (baseline adaptation - same as sudden for all)
3. WITHOUT adaptation (no adaptation baseline)

Generates accuracy-over-time plots to demonstrate the value of type-specific adaptation.

Usage:
    python evaluate_prequential.py [--n_samples 5000] [--n_drifts 5] [--drift_type sudden]
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

# Import ShapeDD detector
try:
    from mmd_variants import shapedd_ow_mmd_buffer as shapedd_detect
except ImportError:
    print("[WARNING] ow_mmd not found, using dummy detector")
    shapedd_detect = None

# Import adaptation strategies
from adaptation_strategies import (
    adapt_sudden_drift,
    adapt_incremental_drift,
    adapt_gradual_drift,
    adapt_recurrent_drift,
    adapt_blip_drift
)

# Import drift type classifier
from drift_type_classifier import classify_drift_at_detection, DriftTypeConfig

# Configuration
from config import (
    BUFFER_SIZE, CHUNK_SIZE, SHAPE_L1, SHAPE_L2, DRIFT_ALPHA,
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


def generate_synthetic_stream(n_samples: int = 5000, n_drifts: int = 5, 
                               n_features: int = 5, drift_type: str = 'sudden',
                               random_seed: int = 42):
    """
    Generate synthetic data stream with known drift points.
    
    Args:
        n_samples: Total number of samples
        n_drifts: Number of drift events
        n_features: Number of features
        drift_type: Type of drift ('sudden', 'gradual', 'incremental', 'recurrent')
        random_seed: Random seed for reproducibility
    
    Returns: X, y, drift_points, drift_types
    """
    np.random.seed(random_seed)
    
    X = []
    y = []
    drift_points = []
    drift_types = []
    
    samples_per_segment = n_samples // (n_drifts + 1)
    
    for segment in range(n_drifts + 1):
        start_idx = segment * samples_per_segment
        end_idx = start_idx + samples_per_segment if segment < n_drifts else n_samples
        n_segment = end_idx - start_idx
        
        if drift_type == 'sudden':
            # Abrupt mean shift
            mean_shift = segment * 2.0
            X_segment = np.random.randn(n_segment, n_features) + mean_shift
        elif drift_type == 'gradual':
            # Gradual transition within segment
            mean_start = (segment - 1) * 2.0 if segment > 0 else 0
            mean_end = segment * 2.0
            alphas = np.linspace(0, 1, n_segment).reshape(-1, 1)
            means = mean_start * (1 - alphas) + mean_end * alphas
            X_segment = np.random.randn(n_segment, n_features) + means
        elif drift_type == 'incremental':
            # Slow continuous drift
            mean_base = segment * 0.5
            increments = np.linspace(0, 1, n_segment).reshape(-1, 1)
            X_segment = np.random.randn(n_segment, n_features) + mean_base + increments * 0.5
        elif drift_type == 'recurrent':
            # Alternating between two concepts
            if segment % 2 == 0:
                mean_shift = 0
            else:
                mean_shift = 3.0
            X_segment = np.random.randn(n_segment, n_features) + mean_shift
        else:
            mean_shift = segment * 2.0
            X_segment = np.random.randn(n_segment, n_features) + mean_shift
        
        # Generate labels (decision boundary shifts with drift)
        weights = np.random.randn(n_features)
        if segment % 2 == 1:  # Change weights for odd segments
            weights = -weights
        
        logits = X_segment @ weights
        probs = 1 / (1 + np.exp(-logits))
        y_segment = (probs > 0.5).astype(int)
        
        X.append(X_segment)
        y.append(y_segment)
        
        if segment > 0:
            drift_points.append(start_idx)
            drift_types.append(drift_type)
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    print(f"Generated {n_samples} samples with {n_drifts} {drift_type} drift points at: {drift_points}")
    return X, y, drift_points, drift_types


class AdaptationMode:
    """Enum-like class for adaptation modes."""
    NONE = 'no_adaptation'
    SIMPLE = 'simple_retrain'
    TYPE_SPECIFIC = 'type_specific'


def evaluate_with_adaptation(X, y, drift_points, mode: str = AdaptationMode.TYPE_SPECIFIC,
                              cache_dir: Optional[Path] = None):
    """
    Evaluate model with different adaptation modes.
    
    Args:
        X: Feature matrix
        y: Labels
        drift_points: Known drift positions
        mode: AdaptationMode.NONE, SIMPLE, or TYPE_SPECIFIC
        cache_dir: Directory for model caching (for recurrent drift)
    
    Returns: accuracy_history, drift_detections, adaptations
    """
    n_samples = len(X)
    
    # Initialize
    model = create_model()
    buffer = deque(maxlen=BUFFER_SIZE)
    recent_correct = deque(maxlen=PREQUENTIAL_WINDOW)
    
    accuracy_history = []
    drift_detections = []
    adaptations = []
    
    # Drift type classifier config
    drift_type_cfg = DriftTypeConfig(
        w_ref=250,
        w_basic=100,
        sudden_len_thresh=250
    )
    
    # Phase 1: Initial training
    train_end = min(INITIAL_TRAINING_SIZE, n_samples)
    model.fit(X[:train_end], y[:train_end])
    print(f"  [{mode}] Initial training on {train_end} samples")
    
    # Fill buffer with training data
    for i in range(train_end):
        buffer.append({'idx': i, 'x': X[i], 'y': y[i]})
    
    # Phase 2: Deployment with evaluation
    drift_detected = False
    drift_detected_at = None
    samples_since_drift = 0
    adaptation_delay = 50  # Samples to wait after detection before adapting
    
    for idx in range(train_end, n_samples):
        x = X[idx:idx+1]
        y_true = y[idx]
        
        # Predict
        y_pred = model.predict(x)[0]
        is_correct = (y_pred == y_true)
        recent_correct.append(is_correct)
        
        # Calculate prequential accuracy
        accuracy = np.mean(recent_correct)
        accuracy_history.append({'idx': idx, 'accuracy': accuracy})
        
        # Add to buffer
        buffer.append({'idx': idx, 'x': X[idx], 'y': y_true})
        
        # Drift detection (every CHUNK_SIZE samples)
        if len(buffer) >= BUFFER_SIZE and idx % CHUNK_SIZE == 0:
            if shapedd_detect is not None and not drift_detected:
                buffer_X = np.array([item['x'] for item in buffer])
                
                try:
                    shp_results = shapedd_detect(buffer_X, l1=SHAPE_L1, l2=SHAPE_L2)
                    
                    # Check recent chunk for drift
                    chunk_start = max(0, len(buffer_X) - CHUNK_SIZE)
                    chunk_pvals = shp_results[chunk_start:, 2]
                    p_min = float(chunk_pvals.min())
                    
                    if p_min < DRIFT_ALPHA:
                        drift_detected = True
                        drift_detected_at = idx
                        drift_detections.append(idx)
                        samples_since_drift = 0
                        print(f"    [{mode}] Drift detected at sample {idx} (p={p_min:.4f})")
                except Exception as e:
                    pass  # Ignore detection errors
        
        # Adaptation logic
        if drift_detected:
            samples_since_drift += 1
            
            # Adapt after delay
            if samples_since_drift >= adaptation_delay:
                if mode == AdaptationMode.NONE:
                    # No adaptation - just reset detection flag
                    print(f"    [{mode}] Drift ignored at sample {idx}")
                    drift_detected = False
                    drift_detected_at = None
                    continue
                
                # Get adaptation window
                adapt_start = max(0, len(buffer) - ADAPTATION_WINDOW)
                adapt_data = list(buffer)[adapt_start:]
                adapt_X = np.array([item['x'] for item in adapt_data])
                adapt_y = np.array([item['y'] for item in adapt_data])
                
                if mode == AdaptationMode.SIMPLE:
                    # Simple retrain (same as sudden for all)
                    model.fit(adapt_X, adapt_y)
                    strategy_used = 'simple_retrain'
                    print(f"    [{mode}] Simple retrain at sample {idx} using {len(adapt_X)} samples")
                    
                elif mode == AdaptationMode.TYPE_SPECIFIC:
                    # Classify drift type
                    buffer_X = np.array([item['x'] for item in buffer])
                    drift_idx_in_buffer = len(buffer) - 1
                    
                    classification = classify_drift_at_detection(
                        buffer_X, 
                        drift_idx_in_buffer, 
                        drift_type_cfg
                    )
                    drift_type = classification.get('subcategory', 'sudden')
                    
                    print(f"    [{mode}] Classified drift as: {drift_type}")
                    
                    # Apply type-specific strategy
                    if drift_type == 'sudden':
                        model = adapt_sudden_drift(model_factory, adapt_X, adapt_y)
                        strategy_used = 'sudden'
                    elif drift_type == 'incremental':
                        model = adapt_incremental_drift(model, adapt_X, adapt_y)
                        strategy_used = 'incremental'
                    elif drift_type == 'gradual':
                        model = adapt_gradual_drift(model, adapt_X, adapt_y)
                        strategy_used = 'gradual'
                    elif drift_type == 'recurrent':
                        model = adapt_recurrent_drift(
                            model_factory, model, adapt_X, adapt_y,
                            cache_dir=cache_dir or Path('./model_cache'),
                            drift_idx=len(drift_detections)
                        )
                        strategy_used = 'recurrent'
                    elif drift_type == 'blip':
                        model = adapt_blip_drift(model, adapt_X, adapt_y)
                        strategy_used = 'blip'
                    else:
                        # Default to sudden
                        model = adapt_sudden_drift(model_factory, adapt_X, adapt_y)
                        strategy_used = 'sudden (default)'
                    
                    print(f"    [{mode}] Applied {strategy_used} strategy at sample {idx}")
                
                adaptations.append({
                    'idx': idx, 
                    'n_samples': len(adapt_X),
                    'strategy': strategy_used if mode == AdaptationMode.TYPE_SPECIFIC else 'simple'
                })
                
                drift_detected = False
                drift_detected_at = None
    
    return accuracy_history, drift_detections, adaptations


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
        
        metrics[mode] = {
            'overall_accuracy': mean_acc,
            'post_drift_accuracy': post_drift_mean,
            'n_detections': len(data['detections']),
            'n_adaptations': len(data['adaptations']),
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
    acc_ts, det_ts, adapt_ts = evaluate_with_adaptation(
        X, y, drift_points, mode=AdaptationMode.TYPE_SPECIFIC, cache_dir=cache_dir
    )
    results[AdaptationMode.TYPE_SPECIFIC] = {'accuracy': acc_ts, 'detections': det_ts, 'adaptations': adapt_ts}
    
    # Evaluate WITH simple adaptation
    print("\n[3] Evaluating WITH simple retrain adaptation...")
    acc_simple, det_simple, adapt_simple = evaluate_with_adaptation(
        X, y, drift_points, mode=AdaptationMode.SIMPLE
    )
    results[AdaptationMode.SIMPLE] = {'accuracy': acc_simple, 'detections': det_simple, 'adaptations': adapt_simple}
    
    # Evaluate WITHOUT adaptation
    print("\n[4] Evaluating WITHOUT adaptation (baseline)...")
    acc_none, det_none, adapt_none = evaluate_with_adaptation(
        X, y, drift_points, mode=AdaptationMode.NONE
    )
    results[AdaptationMode.NONE] = {'accuracy': acc_none, 'detections': det_none, 'adaptations': adapt_none}
    
    # Calculate metrics
    print("\n[5] Calculating metrics...")
    metrics = calculate_metrics(results, drift_points)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nDrift Type: {args.drift_type.upper()}")
    print("-"*40)
    for mode in [AdaptationMode.TYPE_SPECIFIC, AdaptationMode.SIMPLE, AdaptationMode.NONE]:
        m = metrics.get(mode, {})
        print(f"{mode:20s}: Accuracy = {m.get('overall_accuracy', 0):.4f}, "
              f"Post-drift = {m.get('post_drift_accuracy', 0):.4f}, "
              f"Adaptations = {m.get('n_adaptations', 0)}")
    
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
        f.write(f"Prequential Accuracy Evaluation Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Samples: {args.n_samples}, Drifts: {args.n_drifts}, Type: {args.drift_type}, Seed: {args.seed}\n\n")
        for mode, m in metrics.items():
            if isinstance(m, dict):
                f.write(f"\n{mode}:\n")
                for k, v in m.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{mode}: {m}\n")
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n✓ Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
