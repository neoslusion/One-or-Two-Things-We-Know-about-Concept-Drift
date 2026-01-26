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

Results are saved to:
- results/plots/ (Figures)
- results/tables/ (LaTeX Tables)
- results/raw/ (Raw Data)
"""

import gc
import sys
import time
import warnings
from pathlib import Path
import os

# OPTIMIZATION: Prevent thread oversubscription in parallel execution
# DAWIDD and MMD use matrix operations (BLAS/LAPACK). If each parallel job
# spawns multiple threads, the system thrashing destroys performance.
# We force single-threaded linear algebra for worker processes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
    from experiments.benchmark.config import (
        STREAM_SIZE,
        N_RUNS,
        RANDOM_SEEDS,
        CHUNK_SIZE,
        OVERLAP,
        WINDOW_METHODS,
        STREAMING_METHODS,
    )
    from data.catalog import DATASET_CATALOG, get_enabled_datasets
    from data.generators.benchmark_generators import generate_drift_stream
    from experiments.benchmark.evaluation import (
        evaluate_drift_detector,
        evaluate_streaming_detector,
    )
    from experiments.benchmark.analysis import (
        run_statistical_analysis,
        generate_all_figures,
        export_all_tables,
    )
    from experiments.benchmark.analysis.statistics import (
        print_results_summary,
        generate_statistical_report,
    )
    # Import unified output configuration
    from core.config import DETECTION_BENCHMARK_OUTPUTS
else:
    # Running as module: python -m experiments.benchmark.main
    from .config import (
        STREAM_SIZE,
        N_RUNS,
        RANDOM_SEEDS,
        CHUNK_SIZE,
        OVERLAP,
        WINDOW_METHODS,
        STREAMING_METHODS,
        N_JOBS,
    )
    from data.catalog import DATASET_CATALOG, get_enabled_datasets
    from data.generators.benchmark_generators import generate_drift_stream
    from .evaluation import (
        evaluate_drift_detector,
        evaluate_streaming_detector,
    )
    from .analysis import (
        run_statistical_analysis,
        generate_all_figures,
        export_all_tables,
    )
    from .analysis.statistics import print_results_summary, generate_statistical_report
    # Import unified output configuration
    from core.config import DETECTION_BENCHMARK_OUTPUTS

# Import joblib for parallel processing
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')


def _process_single_run(run_idx, seed, enabled_datasets, stream_size, chunk_size, overlap, window_methods, streaming_methods):
    """
    Process a single benchmark run (one seed).
    This function is executed in parallel by joblib.
    """
    # Note: We cannot use the global logger here as it writes to a shared file.
    # We will just return the results.
    print(f"Starting Run {run_idx} (seed={seed})...")
    
    run_results = []
    run_summaries = []
    
    start_time = time.time()

    for dataset_idx, (dataset_name, dataset_config) in enumerate(enabled_datasets, 1):
        # Generate dataset with THIS RUN's seed
        X, y, true_drifts, info = generate_drift_stream(
            dataset_config,
            total_size=stream_size,
            seed=seed
        )
        
        dataset_results = []

        # Evaluate window-based methods
        for method in window_methods:
            result = evaluate_drift_detector(
                method, X, true_drifts,
                chunk_size=chunk_size,
                overlap=overlap,
                verbose=False
            )

            # Add metadata
            result['paradigm'] = 'window'
            result['dataset'] = dataset_name
            result['n_features'] = info['n_features']
            result['n_drifts'] = info['n_drifts']
            result['drift_positions'] = true_drifts
            result['intens'] = info['intens']
            result['dims'] = info['dims']
            result['ground_truth_type'] = dataset_config.get('ground_truth_type', 'unknown')
            result['run_id'] = run_idx
            result['seed'] = seed

            dataset_results.append(result)
            run_results.append(result)

        # Evaluate streaming methods
        for method in streaming_methods:
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
            result['run_id'] = run_idx
            result['seed'] = seed

            dataset_results.append(result)
            run_results.append(result)

        # Dataset summary
        if dataset_results:
            avg_f1 = np.mean([r['f1_score'] for r in dataset_results])
            detection_rate = np.mean([r['detection_rate'] for r in dataset_results])
            
            run_summaries.append({
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

    print(f"Finished Run {run_idx} ({time.time() - start_time:.1f}s)")
    return run_results, run_summaries


def run_benchmark():
    """
    Run the complete drift detection benchmark.

    Returns:
        tuple: (all_results, dataset_summaries) containing all experiment results
    """
    from experiments.benchmark.utils.logging import get_logger, reset_logger
    
    # Reset and get fresh logger
    reset_logger()
    logger = get_logger(
        output_dir=str(DETECTION_BENCHMARK_OUTPUTS["log_file"].parent),
        log_to_file=True,
        verbose=True
    )
    
    enabled_datasets = get_enabled_datasets()

    all_results = []
    dataset_summaries = []

    # Calculate expected totals for validation
    all_methods = list(WINDOW_METHODS) + list(STREAMING_METHODS)
    n_methods = len(all_methods)
    n_datasets = len(enabled_datasets)
    expected_experiments = N_RUNS * n_datasets * n_methods

    # Log configuration
    logger.config(
        n_runs=N_RUNS,
        n_datasets=n_datasets,
        n_methods=n_methods,
        datasets=[d[0] for d in enabled_datasets],
        methods=all_methods
    )
    
    logger.start_benchmark()
    benchmark_start_time = time.time()

    print(f"\nStarting parallel execution with {N_JOBS} jobs...")

    # ========================================================================
    # PARALLEL EXECUTION: Multiple Independent Runs
    # ========================================================================
    parallel_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_process_single_run)(
            run_idx, seed, enabled_datasets, STREAM_SIZE, CHUNK_SIZE, OVERLAP, WINDOW_METHODS, STREAMING_METHODS
        )
        for run_idx, seed in enumerate(RANDOM_SEEDS, 1)
    )

    # Flatten results
    for run_res, run_sum in parallel_results:
        all_results.extend(run_res)
        dataset_summaries.extend(run_sum)

        # Log completion for each run (after the fact, for the record)
        # Note: run_res contains multiple dataset results. 
        # We can extract run_idx/seed from the first result.
        if run_res:
            rid = run_res[0]['run_id']
            # We don't have exact run time here unless we passed it back, 
            # but we can log that it's done.
            logger._log(f"Run {rid}/{N_RUNS} completed (parallel)")

    # ========================================================================
    # FINAL VALIDATION AND SUMMARY
    # ========================================================================
    total_elapsed = time.time() - benchmark_start_time
    logger.summary(len(all_results), expected_experiments, total_elapsed)
    logger.close()

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
    # NEMENYI POST-HOC TEST & CRITICAL DIFFERENCE DIAGRAM
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING STATISTICAL SIGNIFICANCE REPORT")
    print("="*80)
    import pandas as pd
    df_results = pd.DataFrame(all_results)
    statistical_report = generate_statistical_report(
        df_results,
        output_dir=str(DETECTION_BENCHMARK_OUTPUTS["statistical_tests"].parent)
    )

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    generate_all_figures(
        all_results,
        output_dir=str(DETECTION_BENCHMARK_OUTPUTS["f1_comparison"].parent)
    )

    # ========================================================================
    # EXPORT LATEX TABLES
    # ========================================================================
    print("\n" + "="*80)
    print("EXPORTING LATEX TABLES")
    print("="*80)
    export_all_tables(
        all_results,
        STREAM_SIZE,
        output_dir=str(DETECTION_BENCHMARK_OUTPUTS["methods_comparison"].parent)
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Results saved to: {DETECTION_BENCHMARK_OUTPUTS['results_pkl'].parent}")
    print(f"  LaTeX tables: {DETECTION_BENCHMARK_OUTPUTS['methods_comparison'].parent}")
    print(f"  Figures: {DETECTION_BENCHMARK_OUTPUTS['f1_comparison'].parent}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

