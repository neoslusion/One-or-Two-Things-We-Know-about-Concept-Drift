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
import matplotlib.pyplot as plt
import warnings

# Try to import scikit-posthocs for Nemenyi test
try:
    import scikit_posthocs as sp
    POSTHOCS_AVAILABLE = True
except ImportError:
    POSTHOCS_AVAILABLE = False
    print("Warning: scikit-posthocs not installed. Nemenyi test will be skipped.")

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
                # Convert to float to avoid type warnings
                p_value = float(p_value)
                mean_diff = float(np.mean(method_scores) - np.mean(baseline_shapedd))

                # Cohen's d effect size
                pooled_std = float(np.sqrt((np.std(baseline_shapedd)**2 + np.std(method_scores)**2) / 2))
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
                print(f"\nSIGNIFICANT difference detected among methods (p < 0.05)")
                print(f"   Interpretation: At least one method performs significantly different from others.")
            else:
                print(f"\nNo significant difference detected (p ≥ 0.05)")

            friedman_result = {'statistic': stat, 'p_value': p_value}
        else:
            print(f"Cannot perform Friedman test: methods have different sample sizes")
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


def run_nemenyi_posthoc(df_results, metric='f1_score', output_dir=None):
    """
    Run Nemenyi post-hoc test after Friedman test.

    The Nemenyi test performs pairwise comparisons between all methods
    to determine which specific pairs are significantly different.

    Args:
        df_results: DataFrame with columns ['method', 'dataset', metric]
        metric: Which metric to analyze (default: 'f1_score')
        output_dir: Directory to save results (optional)

    Returns:
        dict: Contains p-value matrix, average ranks, and CD value
    """
    if not POSTHOCS_AVAILABLE:
        print("scikit-posthocs not available. Cannot run Nemenyi test.")
        return None

    print("\n" + "="*80)
    print("NEMENYI POST-HOC TEST")
    print("="*80)

    # Pivot data: rows = datasets, columns = methods, values = metric scores
    pivot_df = df_results.pivot_table(
        index='dataset',
        columns='method',
        values=metric,
        aggfunc='mean'
    )

    print(f"\nAnalyzing {metric} across {len(pivot_df)} datasets and {len(pivot_df.columns)} methods")
    print(f"Methods: {list(pivot_df.columns)}")

    # Run Nemenyi test
    # scikit-posthocs expects data in long format or matrix format
    nemenyi_result = sp.posthoc_nemenyi_friedman(pivot_df.values)
    nemenyi_result.index = pivot_df.columns
    nemenyi_result.columns = pivot_df.columns

    print("\nNemenyi p-value matrix:")
    print(nemenyi_result.round(4))

    # Calculate average ranks
    ranks = pivot_df.rank(axis=1, ascending=False)  # Higher is better for F1
    avg_ranks = ranks.mean().sort_values()

    print("\nAverage Ranks (lower is better):")
    for method, rank in avg_ranks.items():
        print(f"  {method:30s}: {rank:.3f}")

    # Calculate Critical Difference (CD) for Nemenyi test
    k = len(pivot_df.columns)  # Number of methods
    n = len(pivot_df)  # Number of datasets

    # Critical value for Nemenyi test at alpha=0.05
    # q_alpha values for different k (from statistical tables)
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q_alpha = q_alpha_table.get(k, 3.0)  # Default to 3.0 if k not in table

    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    print(f"\nCritical Difference (CD) at α=0.05: {cd:.3f}")
    print(f"Methods with rank difference > {cd:.3f} are significantly different")

    # Identify significantly different pairs
    print("\nSignificantly different pairs (p < 0.05):")
    sig_pairs = []
    for i, m1 in enumerate(nemenyi_result.columns):
        for j, m2 in enumerate(nemenyi_result.columns):
            if i < j:
                p_val = float(nemenyi_result.iloc[i, j])
                if p_val < 0.05:
                    rank_diff = abs(avg_ranks[m1] - avg_ranks[m2])
                    print(f"  {m1} vs {m2}: p={p_val:.4f}, rank diff={rank_diff:.3f}")
                    sig_pairs.append((m1, m2, p_val))

    if len(sig_pairs) == 0:
        print("  No significantly different pairs found")

    # Save p-value matrix if output_dir specified
    if output_dir:
        import os
        nemenyi_result.to_csv(os.path.join(output_dir, 'nemenyi_pvalues.csv'))
        avg_ranks.to_csv(os.path.join(output_dir, 'average_ranks.csv'))
        print(f"\nResults saved to {output_dir}")

    return {
        'pvalue_matrix': nemenyi_result,
        'average_ranks': avg_ranks,
        'critical_difference': cd,
        'significant_pairs': sig_pairs,
        'pivot_data': pivot_df
    }


def plot_critical_difference_diagram(df_results, metric='f1_score', output_path=None):
    """
    Generate Critical Difference (CD) diagram for method comparison.

    The CD diagram shows average ranks of methods with horizontal bars
    connecting methods that are NOT significantly different (Nemenyi test).

    This is the standard visualization used in machine learning benchmark papers
    (Demsar 2006, "Statistical Comparisons of Classifiers over Multiple Data Sets").

    Args:
        df_results: DataFrame with columns ['method', 'dataset', metric]
        metric: Which metric to analyze (default: 'f1_score')
        output_path: Path to save the figure (optional)

    Returns:
        matplotlib figure object
    """
    if not POSTHOCS_AVAILABLE:
        print("scikit-posthocs not available. Cannot generate CD diagram.")
        return None

    # Pivot data
    pivot_df = df_results.pivot_table(
        index='dataset',
        columns='method',
        values=metric,
        aggfunc='mean'
    )

    # Calculate ranks (higher metric = rank 1)
    ranks = pivot_df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    # Calculate CD
    k = len(pivot_df.columns)
    n = len(pivot_df)
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    q_alpha = q_alpha_table.get(k, 3.0)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    # Run Nemenyi test
    nemenyi_result = sp.posthoc_nemenyi_friedman(pivot_df.values)
    nemenyi_result.index = pivot_df.columns
    nemenyi_result.columns = pivot_df.columns

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot settings
    methods = list(avg_ranks.index)
    ranks_values = list(avg_ranks.values)
    n_methods = len(methods)

    # Y positions for methods
    y_positions = list(range(n_methods))

    # Draw rank axis
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(-1, n_methods + 1)

    # Draw horizontal line for ranks
    ax.axhline(y=n_methods, color='black', linewidth=2)

    # Draw tick marks for ranks
    for i in range(1, n_methods + 1):
        ax.plot([i, i], [n_methods - 0.1, n_methods + 0.1], 'k-', linewidth=1)
        ax.text(i, n_methods + 0.3, str(i), ha='center', va='bottom', fontsize=10)

    # CD bar
    cd_x = 1
    ax.plot([cd_x, cd_x + cd], [n_methods + 0.8, n_methods + 0.8], 'k-', linewidth=2)
    ax.plot([cd_x, cd_x], [n_methods + 0.7, n_methods + 0.9], 'k-', linewidth=2)
    ax.plot([cd_x + cd, cd_x + cd], [n_methods + 0.7, n_methods + 0.9], 'k-', linewidth=2)
    ax.text(cd_x + cd/2, n_methods + 1.1, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=10)

    # Plot methods on left and right sides
    left_methods = []
    right_methods = []

    for i, (method, rank) in enumerate(avg_ranks.items()):
        if rank <= (n_methods + 1) / 2:
            left_methods.append((method, rank))
        else:
            right_methods.append((method, rank))

    # Draw left side methods (lower ranks = better)
    for i, (method, rank) in enumerate(left_methods):
        y = n_methods - 1 - i * 0.8
        ax.plot([rank, 0.3], [n_methods, y], 'k-', linewidth=1)
        ax.plot(rank, n_methods, 'ko', markersize=8)
        ax.text(0.2, y, method, ha='right', va='center', fontsize=11, fontweight='bold')

    # Draw right side methods (higher ranks = worse)
    for i, (method, rank) in enumerate(right_methods):
        y = n_methods - 1 - i * 0.8
        ax.plot([rank, n_methods + 0.7], [n_methods, y], 'k-', linewidth=1)
        ax.plot(rank, n_methods, 'ko', markersize=8)
        ax.text(n_methods + 0.8, y, method, ha='left', va='center', fontsize=11, fontweight='bold')

    # Draw bars connecting methods that are NOT significantly different
    # Group methods by non-significant differences
    groups = []
    used = set()

    for i, m1 in enumerate(methods):
        if m1 in used:
            continue
        group = [m1]
        for j, m2 in enumerate(methods):
            if i != j and m2 not in used:
                # Get p-value as float
                p_val = float(nemenyi_result.loc[m1, m2])
                if p_val >= 0.05:  # Not significantly different
                    # Check if m2 is within CD of all group members
                    if all(abs(avg_ranks[m2] - avg_ranks[g]) <= cd for g in group):
                        group.append(m2)
        if len(group) > 1:
            groups.append(group)
            used.update(group)

    # Draw connecting bars for non-significant groups
    bar_y = n_methods - 0.3
    for group in groups:
        group_ranks = [avg_ranks[m] for m in group]
        min_rank = min(group_ranks)
        max_rank = max(group_ranks)
        ax.plot([min_rank, max_rank], [bar_y, bar_y], 'b-', linewidth=4, alpha=0.6)
        bar_y -= 0.25

    ax.set_title(f'Critical Difference Diagram - {metric.replace("_", " ").title()}\n(Nemenyi test, α=0.05)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        # Save as PNG for LaTeX documents
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"CD diagram saved to {png_path}")

    return fig


def generate_statistical_report(df_results, output_dir):
    """
    Generate comprehensive statistical report with all tests and diagrams.

    Args:
        df_results: DataFrame with benchmark results
        output_dir: Directory to save all outputs

    Returns:
        dict: All statistical analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL REPORT")
    print("="*80)

    results = {}

    # 1. Nemenyi post-hoc test
    nemenyi_results = run_nemenyi_posthoc(df_results, metric='f1_score', output_dir=output_dir)
    results['nemenyi'] = nemenyi_results

    # 2. Critical Difference diagram
    cd_path = os.path.join(output_dir, 'critical_difference_f1.png')
    cd_fig = plot_critical_difference_diagram(df_results, metric='f1_score', output_path=cd_path)
    results['cd_figure'] = cd_fig

    # 3. Generate LaTeX table for statistical results
    if nemenyi_results:
        latex_stats = generate_statistical_latex_table(nemenyi_results, output_dir)
        results['latex_table'] = latex_stats

    print("\n" + "="*80)
    print("Statistical report complete!")
    print(f"Outputs saved to: {output_dir}")
    print("="*80)

    return results


def generate_statistical_latex_table(nemenyi_results, output_dir):
    """
    Generate LaTeX table summarizing statistical test results.

    Args:
        nemenyi_results: Results from run_nemenyi_posthoc
        output_dir: Directory to save the table

    Returns:
        str: LaTeX table string
    """
    import os

    avg_ranks = nemenyi_results['average_ranks']
    cd = nemenyi_results['critical_difference']

    # Create table
    latex = r"""\begin{table}[htbp]
\caption{Method rankings and statistical significance (Friedman-Nemenyi test, $\alpha$=0.05).
Methods connected by bars in the CD diagram are not significantly different.}
\label{tab:statistical_significance}
\begin{tabular}{lcc}
\toprule
Method & Average Rank & Rank \\
\midrule
"""

    for rank_pos, (method, avg_rank) in enumerate(avg_ranks.items(), 1):
        latex += f"{method} & {avg_rank:.3f} & {rank_pos} \\\\\n"

    latex += r"""\midrule
\multicolumn{3}{l}{Critical Difference (CD) = """ + f"{cd:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""

    # Save to file
    output_path = os.path.join(output_dir, 'table_IV_statistical_significance.tex')
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to {output_path}")

    return latex

