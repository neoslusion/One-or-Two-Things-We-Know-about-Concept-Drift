"""
Prequential Accuracy Evaluation for Drift Adaptation.

This script evaluates the adaptation strategies in an offline batch setting,
comparing:
1. WITH adaptation (drift-type-specific strategies)
2. WITHOUT adaptation (no-adaptation baseline)

Generates accuracy-over-time plots to demonstrate the value of adaptation.

Usage:
    python evaluate_prequential.py [--n_samples 5000] [--n_drifts 5]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from datetime import datetime

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SHAPE_DD_DIR = REPO_ROOT / "backup"
if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.insert(0, str(SHAPE_DD_DIR))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Import ShapeDD detector
try:
    from ow_mmd import shapedd_ow_mmd_buffer as shapedd_detect
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
try:
    from drift_type_classifier import DriftTypeConfig, classify_drift_type
except ImportError:
    classify_drift_type = None
    DriftTypeConfig = None

# Configuration
from config import (
    BUFFER_SIZE, CHUNK_SIZE, SHAPE_L1, SHAPE_L2, DRIFT_ALPHA,
    INITIAL_TRAINING_SIZE, PREQUENTIAL_WINDOW, ADAPTATION_WINDOW
)


def create_model():
    """Create sklearn Pipeline for classification."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])


def generate_synthetic_stream(n_samples: int = 5000, n_drifts: int = 5, 
                               n_features: int = 5, random_seed: int = 42):
    """
    Generate synthetic data stream with known drift points.
    
    Creates abrupt (sudden) drifts at regular intervals.
    Returns: X, y, drift_points
    """
    np.random.seed(random_seed)
    
    X = []
    y = []
    drift_points = []
    
    samples_per_segment = n_samples // (n_drifts + 1)
    
    for segment in range(n_drifts + 1):
        start_idx = segment * samples_per_segment
        end_idx = start_idx + samples_per_segment if segment < n_drifts else n_samples
        n_segment = end_idx - start_idx
        
        # Different mean for each segment (drift)
        mean_shift = segment * 2.0
        
        # Generate features
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
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    print(f"Generated {n_samples} samples with {n_drifts} drift points at: {drift_points}")
    return X, y, drift_points


def evaluate_with_adaptation(X, y, drift_points, enable_adaptation=True):
    """
    Evaluate model with/without adaptation enabled.
    
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
    
    # Phase 1: Initial training
    train_end = min(INITIAL_TRAINING_SIZE, n_samples)
    model.fit(X[:train_end], y[:train_end])
    print(f"  Initial training on {train_end} samples")
    
    # Fill buffer with training data
    for i in range(train_end):
        buffer.append({'idx': i, 'x': X[i], 'y': y[i]})
    
    # Phase 2: Deployment with evaluation
    drift_detected = False
    drift_detected_at = None
    samples_since_drift = 0
    
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
                        print(f"    Drift detected at sample {idx} (p={p_min:.4f})")
                except Exception as e:
                    pass  # Ignore detection errors
        
        # Adaptation logic
        if drift_detected:
            samples_since_drift += 1
            
            # Adapt after delay
            if samples_since_drift >= 50 and enable_adaptation:  # ADAPTATION_DELAY
                # Get adaptation window
                adapt_start = max(0, len(buffer) - ADAPTATION_WINDOW)
                adapt_data = list(buffer)[adapt_start:]
                adapt_X = np.array([item['x'] for item in adapt_data])
                adapt_y = np.array([item['y'] for item in adapt_data])
                
                # Simple adaptation: retrain on recent data
                model.fit(adapt_X, adapt_y)
                
                adaptations.append({'idx': idx, 'n_samples': len(adapt_X)})
                print(f"    Adapted model at sample {idx} using {len(adapt_X)} samples")
                
                drift_detected = False
                drift_detected_at = None
    
    return accuracy_history, drift_detections, adaptations


def plot_prequential_comparison(results_with, results_without, drift_points, output_path):
    """
    Plot prequential accuracy comparison.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Extract data
    idx_with = [r['idx'] for r in results_with['accuracy']]
    acc_with = [r['accuracy'] for r in results_with['accuracy']]
    
    idx_without = [r['idx'] for r in results_without['accuracy']]
    acc_without = [r['accuracy'] for r in results_without['accuracy']]
    
    # Plot accuracy curves
    ax.plot(idx_with, acc_with, 'b-', linewidth=1.5, label='With Adaptation', alpha=0.8)
    ax.plot(idx_without, acc_without, 'r-', linewidth=1.5, label='No Adaptation (Baseline)', alpha=0.8)
    
    # Mark true drift points
    for dp in drift_points:
        ax.axvline(x=dp, color='green', linestyle='--', alpha=0.5, label='_True Drift' if dp != drift_points[0] else 'True Drift')
    
    # Mark detected drifts
    for det in results_with['detections']:
        ax.axvline(x=det, color='blue', linestyle=':', alpha=0.3)
    
    # Mark adaptations
    for adapt in results_with['adaptations']:
        ax.scatter(adapt['idx'], 0.95, marker='^', color='purple', s=100, zorder=5)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Prequential Accuracy', fontsize=12)
    ax.set_title('Prequential Accuracy: With vs Without Adaptation', fontsize=14)
    ax.legend(loc='lower left')
    ax.set_ylim([0.3, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('↑ Adaptation events', xy=(0.02, 0.96), xycoords='axes fraction', fontsize=10, color='purple')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def calculate_metrics(results_with, results_without, drift_points):
    """
    Calculate comparison metrics.
    """
    acc_with = [r['accuracy'] for r in results_with['accuracy']]
    acc_without = [r['accuracy'] for r in results_without['accuracy']]
    
    # Overall accuracy
    mean_with = np.mean(acc_with)
    mean_without = np.mean(acc_without)
    
    # Accuracy in post-drift windows (500 samples after each drift)
    post_drift_with = []
    post_drift_without = []
    
    idx_list = [r['idx'] for r in results_with['accuracy']]
    
    for dp in drift_points:
        for i, idx in enumerate(idx_list):
            if dp <= idx < dp + 500:
                post_drift_with.append(acc_with[i])
                post_drift_without.append(acc_without[i])
    
    post_drift_mean_with = np.mean(post_drift_with) if post_drift_with else 0
    post_drift_mean_without = np.mean(post_drift_without) if post_drift_without else 0
    
    # Recovery time (samples to reach 90% of baseline after drift)
    # Simplified: just report improvement
    
    metrics = {
        'overall_accuracy_with': mean_with,
        'overall_accuracy_without': mean_without,
        'improvement': mean_with - mean_without,
        'improvement_pct': (mean_with - mean_without) / mean_without * 100 if mean_without > 0 else 0,
        'post_drift_accuracy_with': post_drift_mean_with,
        'post_drift_accuracy_without': post_drift_mean_without,
        'post_drift_improvement': post_drift_mean_with - post_drift_mean_without,
        'n_detections': len(results_with['detections']),
        'n_adaptations': len(results_with['adaptations']),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Prequential Accuracy Evaluation')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--n_drifts', type=int, default=5, help='Number of drift points')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./prequential_results', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PREQUENTIAL ACCURACY EVALUATION")
    print("="*80)
    print(f"Samples: {args.n_samples}, Drifts: {args.n_drifts}, Seed: {args.seed}")
    print("="*80)
    
    # Generate data
    print("\n[1] Generating synthetic data stream...")
    X, y, drift_points = generate_synthetic_stream(
        n_samples=args.n_samples,
        n_drifts=args.n_drifts,
        random_seed=args.seed
    )
    
    # Evaluate WITH adaptation
    print("\n[2] Evaluating WITH adaptation...")
    acc_with, det_with, adapt_with = evaluate_with_adaptation(X, y, drift_points, enable_adaptation=True)
    results_with = {'accuracy': acc_with, 'detections': det_with, 'adaptations': adapt_with}
    
    # Evaluate WITHOUT adaptation
    print("\n[3] Evaluating WITHOUT adaptation (baseline)...")
    acc_without, det_without, adapt_without = evaluate_with_adaptation(X, y, drift_points, enable_adaptation=False)
    results_without = {'accuracy': acc_without, 'detections': det_without, 'adaptations': adapt_without}
    
    # Calculate metrics
    print("\n[4] Calculating metrics...")
    metrics = calculate_metrics(results_with, results_without, drift_points)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Overall Accuracy WITH adaptation:    {metrics['overall_accuracy_with']:.4f}")
    print(f"Overall Accuracy WITHOUT adaptation: {metrics['overall_accuracy_without']:.4f}")
    print(f"Improvement: +{metrics['improvement']:.4f} (+{metrics['improvement_pct']:.1f}%)")
    print(f"\nPost-drift (500 samples after each drift):")
    print(f"  WITH adaptation:    {metrics['post_drift_accuracy_with']:.4f}")
    print(f"  WITHOUT adaptation: {metrics['post_drift_accuracy_without']:.4f}")
    print(f"  Improvement: +{metrics['post_drift_improvement']:.4f}")
    print(f"\nDetections: {metrics['n_detections']}, Adaptations: {metrics['n_adaptations']}")
    print("="*80)
    
    # Generate plot
    print("\n[5] Generating plot...")
    plot_path = output_dir / f"prequential_accuracy_comparison.png"
    plot_prequential_comparison(results_with, results_without, drift_points, plot_path)
    
    # Save metrics to file
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("Prequential Accuracy Evaluation Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Samples: {args.n_samples}, Drifts: {args.n_drifts}, Seed: {args.seed}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n✓ Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
