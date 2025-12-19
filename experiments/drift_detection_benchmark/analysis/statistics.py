"""
Statistical analysis and aggregation module.

Contains functions for:
- Confidence interval calculation
- Statistical significance testing (Wilcoxon, Friedman)
- Results aggregation and summary
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import warnings

warnings.filterwarnings('ignore')


def calculate_ci_95(data):
    """
    Calculate 95% confidence interval.

    Args:
        data: Array-like of values

    Returns:
        tuple: (lower_bound, upper_bound) of 95% CI
    """
    if len(data) < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
    return ci[0], ci[1]


def run_statistical_analysis(all_results, n_runs):
    """
    Run comprehensive statistical analysis on benchmark results.

    Args:
        all_results: List of result dictionaries from benchmark
        n_runs: Number of independent runs

    Returns:
        dict: Analysis results including aggregated stats and significance tests
    """
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS: {len(all_results)} total experiments from {n_runs} runs")
    print(f"{'='*80}\n")

    # Convert to DataFrame for easy analysis
    df_results = pd.DataFrame(all_results)

    # ========================================================================
    # 1. AGGREGATE STATISTICS WITH CONFIDENCE INTERVALS
    # ========================================================================
    print("\n" + "="*80)
    print("1. AGGREGATE RESULTS (Mean ± Std with 95% Confidence Intervals)")
    print("="*80)

    # Group by method and dataset
    aggregated = df_results.groupby(['method', 'dataset']).agg({
        'f1_score': ['mean', 'std', 'min', 'max', 'count'],
        'beta_score': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'mttd': ['mean', 'std'],
        'detection_rate': ['mean', 'std']
    }).round(3)

    # Calculate confidence intervals for key metrics
    ci_results = []
    for (method, dataset), group in df_results.groupby(['method', 'dataset']):
        f1_ci_low, f1_ci_high = calculate_ci_95(group['f1_score'].values)
        beta_ci_low, beta_ci_high = calculate_ci_95(group['beta_score'].values)

        ci_results.append({
            'method': method,
            'dataset': dataset,
            'f1_mean': group['f1_score'].mean(),
            'f1_std': group['f1_score'].std(),
            'f1_ci_low': f1_ci_low,
            'f1_ci_high': f1_ci_high,
            'beta_mean': group['beta_score'].mean(),
            'beta_std': group['beta_score'].std(),
            'beta_ci_low': beta_ci_low,
            'beta_ci_high': beta_ci_high,
            'n_runs': len(group)
        })

    df_ci = pd.DataFrame(ci_results)

    # Display aggregated results
    print("\nAggregated Results (sorted by F1 score):")
    print(aggregated.sort_values(('f1_score', 'mean'), ascending=False).head(20))

    print("\n\n95% Confidence Intervals (Top 10 by F1):")
    df_ci_sorted = df_ci.sort_values('f1_mean', ascending=False).head(10)
    for _, row in df_ci_sorted.iterrows():
        print(f"{row['method']:25s} | {row['dataset']:15s} | "
              f"F1: {row['f1_mean']:.3f} ± {row['f1_std']:.3f} "
              f"[{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}] | "
              f"β: {row['beta_mean']:.3f} ± {row['beta_std']:.3f} "
              f"[{row['beta_ci_low']:.3f}, {row['beta_ci_high']:.3f}]")

    # ========================================================================
    # 2. STATISTICAL SIGNIFICANCE TESTING
    # ========================================================================
    print("\n\n" + "="*80)
    print("2. STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)

    # Define methods to compare (focus on ShapeDD variants)
    shapedd_methods = ['ShapeDD', 'ShapeDD_SNR_Adaptive', 'ShapeDD_OW_MMD']
    baseline_methods = ['MMD', 'D3', 'DAWIDD', 'KS']

    # Filter for methods that actually have results
    available_methods = df_results['method'].unique()
    shapedd_methods = [m for m in shapedd_methods if m in available_methods]
    baseline_methods = [m for m in baseline_methods if m in available_methods]

    print(f"\nShapeDD variants: {shapedd_methods}")
    print(f"Baseline methods: {baseline_methods}")

    # ========================================================================
    # 2.1 Pairwise Wilcoxon Tests (ShapeDD variants vs Original)
    # ========================================================================
    wilcoxon_results = []
    if 'ShapeDD' in shapedd_methods and len(shapedd_methods) > 1:
        print("\n\n--- 2.1 Pairwise Wilcoxon Signed-Rank Tests ---")
        print("(Comparing improved ShapeDD variants against original ShapeDD)\n")

        baseline_shapedd = df_results[df_results['method'] == 'ShapeDD']['f1_score'].values

        for method in shapedd_methods:
            if method == 'ShapeDD':
                continue

            method_scores = df_results[df_results['method'] == method]['f1_score'].values

            # Ensure paired comparison (same datasets)
            if len(baseline_shapedd) == len(method_scores):
                statistic, p_value = wilcoxon(baseline_shapedd, method_scores)
                mean_diff = np.mean(method_scores) - np.mean(baseline_shapedd)

                # Cohen's d effect size
                pooled_std = np.sqrt((np.std(baseline_shapedd)**2 + np.std(method_scores)**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                effect_label = "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small" if abs(cohens_d) >= 0.2 else "negligible"

                print(f"{method:30s} vs ShapeDD:")
                print(f"  Mean F1 difference: {mean_diff:+.3f} ({significance})")
                print(f"  Wilcoxon p-value: {p_value:.5f}")
                print(f"  Cohen's d: {cohens_d:.3f} ({effect_label} effect)")
                print(f"  Interpretation: {'SIGNIFICANT improvement' if p_value < 0.05 and mean_diff > 0 else 'SIGNIFICANT decline' if p_value < 0.05 and mean_diff < 0 else 'No significant difference'}\n")

                wilcoxon_results.append({
                    'method': method,
                    'mean_diff': mean_diff,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                })

    # ========================================================================
    # 2.2 Friedman Test (Multiple Methods Comparison)
    # ========================================================================
    friedman_result = None
    all_methods = shapedd_methods + baseline_methods
    if len(all_methods) >= 3:
        print("\n--- 2.2 Friedman Test (Non-parametric ANOVA) ---")
        print(f"(Comparing {len(all_methods)} methods simultaneously)\n")

        # Prepare data for Friedman test
        scores_by_method = []
        for method in all_methods:
            method_scores = df_results[df_results['method'] == method]['f1_score'].values
            scores_by_method.append(method_scores)

        # Check if all methods have same number of observations
        lengths = [len(s) for s in scores_by_method]
        if len(set(lengths)) == 1:
            stat, p_value = friedmanchisquare(*scores_by_method)

            print(f"Friedman statistic: χ² = {stat:.3f}")
            print(f"p-value: {p_value:.5f}")

            if p_value < 0.05:
                print(f"\n✅ SIGNIFICANT difference detected among methods (p < 0.05)")
                print(f"   Interpretation: At least one method performs significantly different from others.")
            else:
                print(f"\n⚠️  No significant difference detected (p ≥ 0.05)")

            friedman_result = {'statistic': stat, 'p_value': p_value}
        else:
            print(f"⚠️  Cannot perform Friedman test: methods have different sample sizes")
            print(f"   Sample sizes: {dict(zip(all_methods, lengths))}")

    # ========================================================================
    # 3. SUMMARY TABLE FOR PUBLICATION
    # ========================================================================
    print("\n\n" + "="*80)
    print("3. PUBLICATION-READY SUMMARY TABLE")
    print("="*80)
    print("\nFormat: Method | Dataset | F1 ± std [95% CI] | β-score ± std\n")

    # Group by dataset and show top methods
    for dataset in df_results['dataset'].unique():
        print(f"\n--- {dataset.upper()} ---")
        dataset_ci = df_ci[df_ci['dataset'] == dataset].sort_values('f1_mean', ascending=False)

        for _, row in dataset_ci.head(5).iterrows():
            print(f"{row['method']:25s} | "
                  f"F1: {row['f1_mean']:.3f} ± {row['f1_std']:.3f} [{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}] | "
                  f"β: {row['beta_mean']:.3f} ± {row['beta_std']:.3f}")

    print("\n" + "="*80)
    print("Statistical analysis complete!")
    print("="*80)

    return {
        'df_results': df_results,
        'df_ci': df_ci,
        'aggregated': aggregated,
        'wilcoxon_results': wilcoxon_results,
        'friedman_result': friedman_result
    }


def print_results_summary(all_results, stream_size):
    """
    Print comprehensive results summary.

    Args:
        all_results: List of result dictionaries
        stream_size: Size of each data stream
    """
    if len(all_results) == 0:
        print("No results to analyze. Run the benchmark first.")
        return

    # Convert results to DataFrame
    results_df = pd.DataFrame([{
        'Dataset': r['dataset'],
        'Method': r['method'],
        'Paradigm': r.get('paradigm', 'unknown'),
        'N_Features': r.get('n_features', 0),
        'N_Drifts': len(r.get('drift_positions', [])),
        'Ground_Truth_Type': r.get('ground_truth_type', 'unknown'),
        'Intensity': r.get('intensity', 0),
        'TP': r.get('tp', 0),
        'FP': r.get('fp', 0),
        'FN': r.get('fn', 0),
        'Precision': r.get('precision', 0.0),
        'Recall': r.get('recall', 0.0),
        'F1': r.get('f1_score', 0.0),
        'MTTD': r.get('mttd', np.nan) if r.get('mttd') != float('inf') else np.nan,
        'Detection_Rate': r.get('detection_rate', 0.0),
        'N_Detections': r.get('n_detections', 0),
        'Runtime_s': r.get('runtime_s', 0.0)
    } for r in all_results])

    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(all_results)}")
    print(f"Methods evaluated: {results_df['Method'].nunique()}")
    print(f"Datasets used: {results_df['Dataset'].nunique()}")

    # Show ground truth breakdown
    gt_counts = results_df.groupby('Ground_Truth_Type')['Dataset'].nunique()
    print(f"\nGround Truth Breakdown:")
    for gt_type, count in gt_counts.items():
        print(f"  - {gt_type}: {count} dataset(s)")

    # Only use drift datasets for F1/Precision/Recall
    drift_results_df = results_df[results_df['N_Drifts'] > 0].copy()

    if len(drift_results_df) > 0:
        # Method rankings by F1
        print("\n" + "-" * 80)
        print("METHOD RANKINGS (by F1-Score)")
        print("-" * 80)

        method_f1 = drift_results_df.groupby('Method')['F1'].agg(['mean', 'std']).sort_values('mean', ascending=False)

        for rank, (method, row) in enumerate(method_f1.iterrows(), 1):
            marker = "★" if rank == 1 else " "
            print(f"{marker} {rank}. {method:<30} F1 = {row['mean']:.3f} ± {row['std']:.3f}")

        # Best method summary
        best_method = method_f1.index[0]
        best_stats = drift_results_df[drift_results_df['Method'] == best_method].agg({
            'F1': 'mean', 'Precision': 'mean', 'Recall': 'mean', 'MTTD': 'mean'
        })

        print(f"\nBest Method: {best_method}")
        print(f"  F1-Score:  {best_stats['F1']:.3f}")
        print(f"  Precision: {best_stats['Precision']:.3f}")
        print(f"  Recall:    {best_stats['Recall']:.3f}")
        print(f"  MTTD:      {best_stats['MTTD']:.1f} samples")

        # Runtime summary
        print("\n" + "-" * 80)
        print("RUNTIME ANALYSIS")
        print("-" * 80)

        runtime_summary = results_df.groupby('Method')['Runtime_s'].agg(['mean', 'std', 'sum'])
        runtime_summary = runtime_summary.sort_values('mean')

        for method, row in runtime_summary.iterrows():
            throughput = stream_size / row['mean'] if row['mean'] > 0 else 0
            print(f"  {method:<25} {row['mean']:>8.3f}s (±{row['std']:.3f}s) | {throughput:>8.0f} samples/s")

        print("\n" + "=" * 80)

    return results_df

