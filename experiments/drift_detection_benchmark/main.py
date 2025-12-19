#!/usr/bin/env python3
"""
Drift Detection Benchmark - Main Entry Point

This script runs a comprehensive benchmark comparing multiple drift detection methods
across various synthetic and semi-real datasets.

Usage:
    cd experiments/drift_detection_benchmark
    python main.py
    
Or from project root:
    python -m experiments.drift_detection_benchmark.main

The benchmark includes:
1. Multiple independent runs for statistical validity
2. Window-based methods: D3, DAWIDD, MMD, KS, ShapeDD variants
3. Streaming methods: ADWIN, DDM, EDDM, HDDM (optional)
4. Statistical analysis with confidence intervals
5. Publication-quality visualizations
6. LaTeX table export for thesis

Results are saved to ./publication_figures/
"""

import gc
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Handle imports for both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Running as script: python main.py
    # Add parent directories to path for imports
    file_path = Path(__file__).resolve()
    package_dir = file_path.parent
    experiments_dir = package_dir.parent
    project_root = experiments_dir.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import using absolute paths
    from experiments.drift_detection_benchmark.config import (
        STREAM_SIZE,
        N_RUNS,
        RANDOM_SEEDS,
        CHUNK_SIZE,
        OVERLAP,
        WINDOW_METHODS,
        STREAMING_METHODS,
    )
    from experiments.drift_detection_benchmark.datasets import (
        DATASET_CATALOG,
        get_enabled_datasets,
        generate_drift_stream,
    )
    from experiments.drift_detection_benchmark.evaluation import (
        evaluate_drift_detector,
        evaluate_streaming_detector,
    )
    from experiments.drift_detection_benchmark.analysis import (
        run_statistical_analysis,
        generate_all_figures,
        export_all_tables,
    )
    from experiments.drift_detection_benchmark.analysis.statistics import print_results_summary
else:
    # Running as module: python -m experiments.drift_detection_benchmark.main
    from .config import (
        STREAM_SIZE,
        N_RUNS,
        RANDOM_SEEDS,
        CHUNK_SIZE,
        OVERLAP,
        WINDOW_METHODS,
        STREAMING_METHODS,
    )
    from .datasets import DATASET_CATALOG, get_enabled_datasets, generate_drift_stream
    from .evaluation import (
        evaluate_drift_detector,
        evaluate_streaming_detector,
    )
    from .analysis import (
        run_statistical_analysis,
        generate_all_figures,
        export_all_tables,
    )
    from .analysis.statistics import print_results_summary

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')


def run_benchmark():
    """
    Run the complete drift detection benchmark.

    Returns:
        tuple: (all_results, dataset_summaries) containing all experiment results
    """
    enabled_datasets = get_enabled_datasets()

    all_results = []
    dataset_summaries = []

    # Calculate expected totals for validation
    n_methods = len(WINDOW_METHODS) + len(STREAMING_METHODS)
    n_datasets = len(enabled_datasets)
    expected_experiments = N_RUNS * n_datasets * n_methods

    print(f"{'='*80}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'='*80}")
    print(f"  N_RUNS: {N_RUNS}")
    print(f"  Datasets: {n_datasets} ({', '.join([d[0] for d in enabled_datasets])})")
    print(f"  Window methods: {len(WINDOW_METHODS)} ({', '.join(WINDOW_METHODS)})")
    print(f"  Streaming methods: {len(STREAMING_METHODS)} ({', '.join(STREAMING_METHODS) if STREAMING_METHODS else 'None'})")
    print(f"  Expected total experiments: {expected_experiments}")
    print(f"{'='*80}\n")

    benchmark_start_time = time.time()

    # ========================================================================
    # OUTER LOOP: Multiple Independent Runs
    # ========================================================================
    for run_idx, seed in enumerate(RANDOM_SEEDS, 1):
        run_start_time = time.time()

        print(f"\n{'#'*80}")
        print(f"# RUN {run_idx}/{N_RUNS} (seed={seed})")
        print(f"# Progress: {run_idx/N_RUNS*100:.1f}% complete | Experiments so far: {len(all_results)}")
        print(f"{'#'*80}")

        for dataset_idx, (dataset_name, dataset_config) in enumerate(enabled_datasets, 1):
            print(f"\n{'='*80}")
            print(f"RUN {run_idx}/{N_RUNS} | DATASET {dataset_idx}/{len(enabled_datasets)}: {dataset_name.upper()}")
            print(f"{'='*80}")

            # Generate dataset with THIS RUN's seed
            X, y, true_drifts, info = generate_drift_stream(
                dataset_config,
                total_size=STREAM_SIZE,
                seed=seed  # Different seed per run
            )

            dataset_results = []

            # Evaluate window-based methods
            for method in WINDOW_METHODS:
                result = evaluate_drift_detector(
                    method, X, true_drifts,
                    chunk_size=CHUNK_SIZE,
                    overlap=OVERLAP
                )

                # Add metadata (including run information)
                result['paradigm'] = 'window'
                result['dataset'] = dataset_name
                result['n_features'] = info['n_features']
                result['n_drifts'] = info['n_drifts']
                result['drift_positions'] = true_drifts
                result['intens'] = info['intens']
                result['dims'] = info['dims']
                result['ground_truth_type'] = dataset_config.get('ground_truth_type', 'unknown')

                # Add run tracking for statistical analysis
                result['run_id'] = run_idx
                result['seed'] = seed

                dataset_results.append(result)
                all_results.append(result)

            # Evaluate streaming methods
            for method in STREAMING_METHODS:
                result = evaluate_streaming_detector(
                    method, X, y, true_drifts
                )

                result['paradigm'] = 'streaming'
                result['dataset'] = dataset_name
                result['n_features'] = info['n_features']
                result['n_drifts'] = info['n_drifts']
                result['drift_positions'] = true_drifts
                result['intens'] = info['intens']
                result['dims'] = info['dims']
                result['ground_truth_type'] = dataset_config.get('ground_truth_type', 'unknown')

                # Add run tracking
                result['run_id'] = run_idx
                result['seed'] = seed

                dataset_results.append(result)
                all_results.append(result)

            # Dataset summary (computed per run)
            if dataset_results:
                avg_f1 = np.mean([r['f1_score'] for r in dataset_results])
                detection_rate = np.mean([r['detection_rate'] for r in dataset_results])

                dataset_summaries.append({
                    'dataset': dataset_name,
                    'n_features': info['n_features'],
                    'n_drifts': info['n_drifts'],
                    'intens': info['intens'],
                    'avg_f1': avg_f1,
                    'detection_rate': detection_rate,
                    'run_id': run_idx,
                    'seed': seed
                })
            gc.collect()

        # Run summary
        run_elapsed = time.time() - run_start_time
        print(f"\n  [Run {run_idx} Summary] Completed in {run_elapsed:.1f}s | Total experiments: {len(all_results)}")

    # ========================================================================
    # FINAL VALIDATION AND SUMMARY
    # ========================================================================
    total_elapsed = time.time() - benchmark_start_time

    print(f"\n{'#'*80}")
    print(f"# ALL {N_RUNS} RUNS COMPLETED!")
    print(f"{'#'*80}")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Expected experiments: {expected_experiments}")
    print(f"  Total runtime: {total_elapsed/60:.1f} minutes")

    # Validation check
    if len(all_results) != expected_experiments:
        print(f"\n  WARNING: Experiment count mismatch!")
        print(f"      Expected {expected_experiments}, got {len(all_results)}")
        print(f"      This may indicate incomplete runs or configuration issues.")
    else:
        print(f"\n  Validation PASSED: All {expected_experiments} experiments completed successfully.")

    # Show experiment distribution
    print(f"\n  Experiments per run: {len(all_results) // N_RUNS if N_RUNS > 0 else 0}")
    print(f"  Experiments per dataset: {len(all_results) // n_datasets if n_datasets > 0 else 0}")
    print(f"{'#'*80}")

    return all_results, dataset_summaries


def main():
    """Main entry point for the benchmark."""
    print("\n" + "="*80)
    print("DRIFT DETECTION BENCHMARK")
    print("="*80)
    print("Comprehensive evaluation of concept drift detection methods")
    print("="*80 + "\n")

    # Run benchmark
    all_results, dataset_summaries = run_benchmark()

    if len(all_results) == 0:
        print("No results generated. Exiting.")
        return

    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    analysis_results = run_statistical_analysis(all_results, N_RUNS)

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print_results_summary(all_results, STREAM_SIZE)

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    generate_all_figures(all_results)

    # ========================================================================
    # EXPORT LATEX TABLES
    # ========================================================================
    print("\n" + "="*80)
    print("EXPORTING LATEX TABLES")
    print("="*80)
    export_all_tables(all_results, STREAM_SIZE)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Results saved to: ./publication_figures/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

