"""
OW-MMD vs MMD: Window Size Comparison Benchmark
================================================

Research Objective:
    Validate whether OW-MMD's sample efficiency (Bharti et al. ICML 2023)
    translates to better drift detection power, especially with small window sizes.

Hypothesis:
    OW-MMD's variance-optimal weighting should provide tighter null distributions,
    leading to better separation between drift and no-drift cases when sample
    sizes are small. This should manifest as higher F1 scores at smaller window sizes.

Experimental Design:
    - Window sizes: [50, 100, 150, 200, 300]
    - Methods: MMD, OW_MMD, ShapeDD, ShapeDD_OW_MMD (all with permutation test)
    - Datasets: Synthetic (gaussian_shift, gen_random, stagger) + Semi-synthetic Elec2
    - Metrics: Precision, Recall, F1, Detection Delay, Runtime
    - N_RUNS: 5 (quick validation) or 30 (full benchmark)

Fair Comparison:
    Both standard MMD and OW-MMD use permutation tests for p-value computation.
    The only difference is the weighting strategy in MMD computation.
"""

import sys
import time
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR.parent / 'backup'))
sys.path.insert(0, str(SCRIPT_DIR))

from ow_mmd import mmd_ow_permutation, shape_ow_mmd
from shape_dd import shape
from mmd import mmd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Quick validation vs full benchmark
QUICK_MODE = True  # Set to False for full 30-run benchmark

N_RUNS = 5 if QUICK_MODE else 30
N_PERM = 200 if QUICK_MODE else 500  # Permutations for p-value

STREAM_SIZE = 5000  # Samples per stream
N_DRIFT_EVENTS = 5  # Number of drifts per stream

# Window sizes to test (hypothesis: OW-MMD helps at small sizes)
WINDOW_SIZES = [50, 100, 150, 200, 300]

# Detection parameters
ACCEPTABLE_DELTA = 150  # Tolerance for matching detections to true drifts
COOLDOWN = 100  # Minimum samples between detections

# ShapeDD parameters (scale with window size)
def get_shape_params(window_size):
    """Get l1, l2 for ShapeDD based on window size."""
    l1 = max(15, window_size // 4)
    l2 = max(30, window_size // 2)
    return l1, l2

# Methods to compare
METHODS = ['MMD', 'OW_MMD', 'ShapeDD', 'ShapeDD_OW_MMD']

# Datasets
SYNTHETIC_DATASETS = [
    'gaussian_shift',
    'gen_random',
    'stagger',
]

RANDOM_SEEDS = [42 + i * 137 for i in range(N_RUNS)]

OUTPUT_DIR = SCRIPT_DIR / 'publication_figures'
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_gaussian_shift_stream(total_size, n_drift_events, seed=42):
    """
    Generate synthetic stream with Gaussian distribution shifts.
    
    This is the cleanest test case for MMD-based detectors:
    - Features are i.i.d. Gaussian
    - Drift = sudden mean shift of 2 standard deviations
    """
    np.random.seed(seed)
    n_features = 10
    shift_magnitude = 2.0
    
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    
    X = []
    current_mean = np.zeros(n_features)
    
    for i in range(n_drift_events + 1):
        segment = np.random.randn(segment_size, n_features) + current_mean
        X.append(segment)
        # Alternate shift direction
        shift = shift_magnitude * (1 if i % 2 == 0 else -1)
        current_mean = np.ones(n_features) * shift
    
    X = np.vstack(X)[:total_size]
    return X, drift_positions


def generate_gen_random_stream(total_size, n_drift_events, seed=42):
    """
    Generate synthetic stream with random feature changes.
    
    Similar to gaussian_shift but with different intensities per feature.
    """
    np.random.seed(seed)
    n_features = 5
    intensity = 1.5
    
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    
    X = []
    for i in range(n_drift_events + 1):
        mean = np.zeros(n_features)
        if i > 0:
            # Random shift direction and magnitude
            mean = np.random.randn(n_features) * intensity * (1 if i % 2 == 1 else -1)
        segment = np.random.randn(segment_size, n_features) + mean
        X.append(segment)
    
    X = np.vstack(X)[:total_size]
    return X, drift_positions


def generate_stagger_stream(total_size, n_drift_events, seed=42):
    """
    Generate STAGGER-like stream with concept changes.
    
    Features are categorical (encoded as binary), concepts change at drift points.
    """
    np.random.seed(seed)
    n_features = 5
    
    segment_size = total_size // (n_drift_events + 1)
    drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
    
    X = []
    for i in range(n_drift_events + 1):
        # Different feature distributions per segment
        if i % 3 == 0:
            segment = np.random.rand(segment_size, n_features) * 2
        elif i % 3 == 1:
            segment = np.random.rand(segment_size, n_features) * 2 + 1
        else:
            segment = np.random.rand(segment_size, n_features) * 2 - 0.5
        X.append(segment)
    
    X = np.vstack(X)[:total_size]
    return X, drift_positions


def load_elec2_semisynthetic(total_size, n_drift_events, seed=42):
    """
    Load Elec2 features with injected feature drift.
    
    This tests on real-world feature complexity with controlled drift.
    """
    try:
        from river.datasets import Elec2
        
        np.random.seed(seed)
        
        # Load Elec2 features
        X_list = []
        for i, (x, _) in enumerate(Elec2()):
            if i >= total_size:
                break
            X_list.append(list(x.values()))
        
        X = np.array(X_list)
        
        # Inject drift at specified positions
        segment_size = total_size // (n_drift_events + 1)
        drift_positions = [(i + 1) * segment_size for i in range(n_drift_events)]
        
        # Calculate feature stds for calibrated shifts
        feature_stds = np.std(X, axis=0)
        shift_magnitude = 3.0  # 3 sigma shift
        
        # Features to shift (first 5 have variance)
        features_to_shift = [0, 1, 2, 3, 4]
        
        cumulative_shift = np.zeros(X.shape[1])
        for i, drift_pos in enumerate(drift_positions):
            direction = 1 if i % 2 == 0 else -1
            for f in features_to_shift:
                cumulative_shift[f] += direction * shift_magnitude * feature_stds[f]
            X[drift_pos:] += cumulative_shift
        
        return X, drift_positions
        
    except ImportError:
        print("Warning: river not installed, using synthetic data instead")
        return generate_gaussian_shift_stream(total_size, n_drift_events, seed)


DATASET_GENERATORS = {
    'gaussian_shift': generate_gaussian_shift_stream,
    'gen_random': generate_gen_random_stream,
    'stagger': generate_stagger_stream,
    'elec2': load_elec2_semisynthetic,
}


# =============================================================================
# DETECTION METHODS
# =============================================================================

def detect_mmd(X, window_size, n_perm=500):
    """
    Standard MMD with permutation test.
    
    Sliding window approach: test each window for drift at its midpoint.
    """
    detections = []
    last_detection = -COOLDOWN
    
    for start in range(0, len(X) - window_size, window_size // 2):
        end = start + window_size
        window = X[start:end]
        center = (start + end) // 2
        
        # Skip if too close to last detection
        if center - last_detection < COOLDOWN:
            continue
        
        # Standard MMD with permutation test
        mmd_val, p_val = mmd(window, n_perm=n_perm)
        
        if p_val < 0.05:
            detections.append(center)
            last_detection = center
    
    return detections


def detect_ow_mmd(X, window_size, n_perm=500):
    """
    OW-MMD with permutation test.
    
    Same sliding window approach as standard MMD, but using optimal weights.
    """
    detections = []
    last_detection = -COOLDOWN
    
    for start in range(0, len(X) - window_size, window_size // 2):
        end = start + window_size
        window = X[start:end]
        center = (start + end) // 2
        
        if center - last_detection < COOLDOWN:
            continue
        
        # OW-MMD with permutation test
        mmd_val, p_val = mmd_ow_permutation(window, n_perm=n_perm)
        
        if p_val < 0.05:
            detections.append(center)
            last_detection = center
    
    return detections


def detect_shapedd(X, window_size, n_perm=500):
    """
    Original ShapeDD with standard MMD.
    """
    l1, l2 = get_shape_params(window_size)
    
    # Run ShapeDD on full stream
    result = shape(X, l1=l1, l2=l2, n_perm=n_perm)
    
    # Extract detections (p < 0.05) with cooldown
    detections = []
    last_detection = -COOLDOWN
    
    significant = np.where(result[:, 2] < 0.05)[0]
    for pos in significant:
        if pos - last_detection >= COOLDOWN:
            detections.append(pos)
            last_detection = pos
    
    return detections


def detect_shapedd_ow_mmd(X, window_size, n_perm=500):
    """
    ShapeDD with OW-MMD permutation test.
    """
    l1, l2 = get_shape_params(window_size)
    
    # Run ShapeDD with OW-MMD
    result = shape_ow_mmd(X, l1=l1, l2=l2, n_perm=n_perm)
    
    # Extract detections with cooldown
    detections = []
    last_detection = -COOLDOWN
    
    significant = np.where(result[:, 2] < 0.05)[0]
    for pos in significant:
        if pos - last_detection >= COOLDOWN:
            detections.append(pos)
            last_detection = pos
    
    return detections


DETECTORS = {
    'MMD': detect_mmd,
    'OW_MMD': detect_ow_mmd,
    'ShapeDD': detect_shapedd,
    'ShapeDD_OW_MMD': detect_shapedd_ow_mmd,
}


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_metrics(detections, true_drifts, stream_size, acceptable_delta=150):
    """
    Calculate precision, recall, F1, and detection delay.
    """
    if not detections:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'detection_delay': float('inf'),
            'tp': 0,
            'fp': 0,
            'fn': len(true_drifts)
        }
    
    # Match detections to true drifts
    matched_drifts = set()
    delays = []
    
    for det in detections:
        best_match = None
        best_dist = float('inf')
        
        for true_drift in true_drifts:
            dist = abs(det - true_drift)
            if dist <= acceptable_delta and dist < best_dist:
                if true_drift not in matched_drifts:
                    best_match = true_drift
                    best_dist = dist
        
        if best_match is not None:
            matched_drifts.add(best_match)
            delays.append(best_dist)
    
    tp = len(matched_drifts)
    fp = len(detections) - tp
    fn = len(true_drifts) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_delay = np.mean(delays) if delays else float('inf')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detection_delay': avg_delay,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark():
    """Run the full benchmark comparing OW-MMD vs MMD across window sizes."""
    
    print("=" * 70)
    print("OW-MMD vs MMD: Window Size Comparison Benchmark")
    print("=" * 70)
    print(f"Mode: {'QUICK VALIDATION' if QUICK_MODE else 'FULL BENCHMARK'}")
    print(f"N_RUNS: {N_RUNS}, N_PERM: {N_PERM}")
    print(f"Window sizes: {WINDOW_SIZES}")
    print(f"Methods: {METHODS}")
    print(f"Datasets: {SYNTHETIC_DATASETS + ['elec2']}")
    print("=" * 70)
    print()
    
    all_results = []
    
    datasets = SYNTHETIC_DATASETS + ['elec2']
    total_experiments = len(datasets) * len(WINDOW_SIZES) * len(METHODS) * N_RUNS
    current = 0
    
    for dataset_name in datasets:
        generator = DATASET_GENERATORS[dataset_name]
        
        for window_size in WINDOW_SIZES:
            for method in METHODS:
                detector = DETECTORS[method]
                
                run_results = []
                run_times = []
                
                for run_idx, seed in enumerate(RANDOM_SEEDS):
                    current += 1
                    
                    # Generate data
                    X, true_drifts = generator(STREAM_SIZE, N_DRIFT_EVENTS, seed=seed)
                    
                    # Run detector
                    start_time = time.time()
                    detections = detector(X, window_size, n_perm=N_PERM)
                    elapsed = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = calculate_metrics(detections, true_drifts, 
                                               STREAM_SIZE, ACCEPTABLE_DELTA)
                    metrics['runtime'] = elapsed
                    
                    run_results.append(metrics)
                    run_times.append(elapsed)
                    
                    # Progress update
                    if current % 10 == 0 or current == total_experiments:
                        print(f"\rProgress: {current}/{total_experiments} "
                              f"({100*current/total_experiments:.1f}%) - "
                              f"{dataset_name}/{method}/w={window_size}", end="")
                
                # Aggregate across runs
                avg_result = {
                    'dataset': dataset_name,
                    'window_size': window_size,
                    'method': method,
                    'precision_mean': np.mean([r['precision'] for r in run_results]),
                    'precision_std': np.std([r['precision'] for r in run_results]),
                    'recall_mean': np.mean([r['recall'] for r in run_results]),
                    'recall_std': np.std([r['recall'] for r in run_results]),
                    'f1_mean': np.mean([r['f1'] for r in run_results]),
                    'f1_std': np.std([r['f1'] for r in run_results]),
                    'delay_mean': np.mean([r['detection_delay'] for r in run_results 
                                          if r['detection_delay'] != float('inf')]) or 0,
                    'runtime_mean': np.mean(run_times),
                }
                
                all_results.append(avg_result)
    
    print("\n\nBenchmark complete!")
    return all_results


def analyze_results(results):
    """Analyze and display benchmark results."""
    
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    
    # Group by window size for comparison
    for window_size in WINDOW_SIZES:
        print(f"\n--- Window Size: {window_size} ---")
        subset = df[df['window_size'] == window_size]
        
        # Average across datasets
        summary = subset.groupby('method').agg({
            'precision_mean': 'mean',
            'recall_mean': 'mean',
            'f1_mean': 'mean',
            'delay_mean': 'mean',
            'runtime_mean': 'mean'
        }).round(3)
        
        print(summary.to_string())
    
    # Key comparison: OW-MMD vs MMD at each window size
    print("\n" + "=" * 90)
    print("KEY COMPARISON: OW-MMD vs MMD F1 Score by Window Size")
    print("=" * 90)
    
    comparison = []
    for window_size in WINDOW_SIZES:
        subset = df[df['window_size'] == window_size]
        
        mmd_f1 = subset[subset['method'] == 'MMD']['f1_mean'].mean()
        ow_mmd_f1 = subset[subset['method'] == 'OW_MMD']['f1_mean'].mean()
        shapedd_f1 = subset[subset['method'] == 'ShapeDD']['f1_mean'].mean()
        shapedd_ow_f1 = subset[subset['method'] == 'ShapeDD_OW_MMD']['f1_mean'].mean()
        
        comparison.append({
            'Window': window_size,
            'MMD': f"{mmd_f1:.3f}",
            'OW_MMD': f"{ow_mmd_f1:.3f}",
            'OW_MMD_Δ': f"{(ow_mmd_f1 - mmd_f1):+.3f}",
            'ShapeDD': f"{shapedd_f1:.3f}",
            'ShapeDD_OW': f"{shapedd_ow_f1:.3f}",
            'ShapeDD_Δ': f"{(shapedd_ow_f1 - shapedd_f1):+.3f}",
        })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    # Hypothesis test
    print("\n" + "=" * 90)
    print("HYPOTHESIS VALIDATION")
    print("=" * 90)
    
    # Check if OW-MMD improves more at smaller window sizes
    small_window = df[df['window_size'] == 50]
    large_window = df[df['window_size'] == 300]
    
    small_improvement = (
        small_window[small_window['method'] == 'OW_MMD']['f1_mean'].mean() -
        small_window[small_window['method'] == 'MMD']['f1_mean'].mean()
    )
    large_improvement = (
        large_window[large_window['method'] == 'OW_MMD']['f1_mean'].mean() -
        large_window[large_window['method'] == 'MMD']['f1_mean'].mean()
    )
    
    print(f"F1 improvement (OW_MMD over MMD):")
    print(f"  - At window=50:  {small_improvement:+.3f}")
    print(f"  - At window=300: {large_improvement:+.3f}")
    
    if small_improvement > large_improvement:
        print("\n✓ HYPOTHESIS SUPPORTED: OW-MMD provides larger improvement at smaller window sizes")
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED: OW-MMD improvement does not scale with window size reduction")
    
    return df


def create_visualizations(df):
    """Create and save visualization figures."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for methods
    colors = {
        'MMD': '#1f77b4',
        'OW_MMD': '#ff7f0e',
        'ShapeDD': '#2ca02c',
        'ShapeDD_OW_MMD': '#d62728'
    }
    
    # Plot 1: F1 Score vs Window Size (averaged across datasets)
    ax1 = axes[0, 0]
    for method in METHODS:
        method_data = df[df['method'] == method].groupby('window_size')['f1_mean'].mean()
        ax1.plot(method_data.index, method_data.values, 'o-', 
                label=method, color=colors[method], linewidth=2, markersize=8)
    ax1.set_xlabel('Window Size')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Window Size (Averaged Across Datasets)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detection Delay vs Window Size
    ax2 = axes[0, 1]
    for method in METHODS:
        method_data = df[df['method'] == method].groupby('window_size')['delay_mean'].mean()
        ax2.plot(method_data.index, method_data.values, 'o-', 
                label=method, color=colors[method], linewidth=2, markersize=8)
    ax2.set_xlabel('Window Size')
    ax2.set_ylabel('Detection Delay (samples)')
    ax2.set_title('Detection Delay vs Window Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: OW-MMD Improvement over MMD by Window Size
    ax3 = axes[1, 0]
    improvements = []
    for ws in WINDOW_SIZES:
        subset = df[df['window_size'] == ws]
        mmd_f1 = subset[subset['method'] == 'MMD']['f1_mean'].mean()
        ow_f1 = subset[subset['method'] == 'OW_MMD']['f1_mean'].mean()
        improvements.append(ow_f1 - mmd_f1)
    
    bars = ax3.bar(WINDOW_SIZES, improvements, color='#ff7f0e', alpha=0.7, width=30)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Window Size')
    ax3.set_ylabel('F1 Improvement (OW_MMD - MMD)')
    ax3.set_title('OW-MMD F1 Improvement Over Standard MMD')
    ax3.set_xticks(WINDOW_SIZES)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar, imp in zip(bars, improvements):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{imp:+.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Runtime comparison
    ax4 = axes[1, 1]
    x = np.arange(len(WINDOW_SIZES))
    width = 0.2
    
    for i, method in enumerate(METHODS):
        method_data = df[df['method'] == method].groupby('window_size')['runtime_mean'].mean()
        ax4.bar(x + i*width, method_data.values, width, label=method, color=colors[method], alpha=0.7)
    
    ax4.set_xlabel('Window Size')
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_title('Runtime by Method and Window Size')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(WINDOW_SIZES)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = OUTPUT_DIR / 'figure_window_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    
    plt.close()


def save_results(results, df):
    """Save results to files."""
    
    # Save raw results as JSON
    json_path = OUTPUT_DIR / 'window_comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Save summary table as CSV
    csv_path = OUTPUT_DIR / 'window_comparison_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")
    
    # Generate LaTeX table
    latex_path = OUTPUT_DIR / 'table_window_comparison.tex'
    
    with open(latex_path, 'w') as f:
        f.write("% OW-MMD vs MMD Window Size Comparison\n")
        f.write("% Auto-generated by benchmark_window_comparison.py\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{F1 Score Comparison: OW-MMD vs Standard MMD by Window Size}\n")
        f.write("\\label{tab:window-comparison}\n")
        f.write("\\begin{tabular}{ccccc}\n")
        f.write("\\toprule\n")
        f.write("Window & MMD & OW-MMD & ShapeDD & ShapeDD+OW-MMD \\\\\n")
        f.write("\\midrule\n")
        
        for ws in WINDOW_SIZES:
            subset = df[df['window_size'] == ws]
            mmd_f1 = subset[subset['method'] == 'MMD']['f1_mean'].mean()
            ow_f1 = subset[subset['method'] == 'OW_MMD']['f1_mean'].mean()
            shp_f1 = subset[subset['method'] == 'ShapeDD']['f1_mean'].mean()
            shp_ow_f1 = subset[subset['method'] == 'ShapeDD_OW_MMD']['f1_mean'].mean()
            
            # Bold the best in each row
            values = [mmd_f1, ow_f1, shp_f1, shp_ow_f1]
            best_idx = np.argmax(values)
            
            formatted = []
            for i, v in enumerate(values):
                if i == best_idx:
                    formatted.append(f"\\textbf{{{v:.3f}}}")
                else:
                    formatted.append(f"{v:.3f}")
            
            f.write(f"{ws} & {' & '.join(formatted)} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Starting benchmark...")
    start_time = time.time()
    
    # Run benchmark
    results = run_benchmark()
    
    # Analyze results
    df = analyze_results(results)
    
    # Create visualizations
    create_visualizations(df)
    
    # Save results
    save_results(results, df)
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
