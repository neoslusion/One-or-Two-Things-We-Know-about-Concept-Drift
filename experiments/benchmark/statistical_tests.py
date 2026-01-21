"""
Statistical Significance Tests for Drift Detection Benchmark.

Implements standard statistical tests used in machine learning papers:
- Friedman test: Non-parametric test for comparing multiple methods
- Nemenyi post-hoc test: Pairwise comparison after Friedman
- Wilcoxon signed-rank test: Pairwise comparison for 2 methods

References:
    - Demšar (2006): "Statistical Comparisons of Classifiers over Multiple Data Sets"
    - García & Herrera (2008): "An Extension on Statistical Comparisons of Classifiers"

Usage:
    from experiments.benchmark.statistical_tests import (
        run_friedman_nemenyi_test,
        run_wilcoxon_test,
        generate_critical_difference_diagram
    )
    
    # Run Friedman + Nemenyi test
    results = run_friedman_nemenyi_test(scores_df)
    
    # Generate CD diagram
    generate_critical_difference_diagram(results, output_path)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings


def compute_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    Compute average ranks for each method across datasets.
    
    Args:
        scores: Matrix of shape (n_datasets, n_methods)
        higher_is_better: If True, higher scores get lower (better) ranks
        
    Returns:
        ranks: Matrix of shape (n_datasets, n_methods)
    """
    n_datasets, n_methods = scores.shape
    ranks = np.zeros_like(scores)
    
    for i in range(n_datasets):
        row = scores[i]
        if higher_is_better:
            # Higher is better: negate for ranking
            order = np.argsort(-row)
        else:
            order = np.argsort(row)
        
        # Handle ties with average rank
        sorted_indices = order
        current_rank = 1
        j = 0
        while j < n_methods:
            # Find all tied values
            tie_start = j
            while j < n_methods - 1 and row[sorted_indices[j]] == row[sorted_indices[j + 1]]:
                j += 1
            tie_end = j
            
            # Assign average rank to tied values
            avg_rank = (current_rank + current_rank + (tie_end - tie_start)) / 2
            for k in range(tie_start, tie_end + 1):
                ranks[i, sorted_indices[k]] = avg_rank
            
            current_rank += (tie_end - tie_start + 1)
            j += 1
    
    return ranks


def friedman_test(scores: np.ndarray) -> Tuple[float, float]:
    """
    Friedman test for comparing multiple methods across datasets.
    
    H₀: All methods perform equally (same median ranks)
    H₁: At least one method differs significantly
    
    Args:
        scores: Matrix of shape (n_datasets, n_methods)
        
    Returns:
        statistic: Friedman chi-squared statistic
        p_value: p-value of the test
    """
    n_datasets, n_methods = scores.shape
    
    # Compute ranks
    ranks = compute_ranks(scores, higher_is_better=True)
    
    # Average ranks per method
    avg_ranks = ranks.mean(axis=0)
    
    # Friedman statistic (chi-squared approximation)
    chi2 = (12 * n_datasets / (n_methods * (n_methods + 1))) * \
           (np.sum(avg_ranks ** 2) - (n_methods * (n_methods + 1) ** 2) / 4)
    
    # Corrected Friedman statistic (Iman-Davenport)
    # F = ((n-1) * chi2) / (n*(k-1) - chi2)
    # where n = n_datasets, k = n_methods
    if n_datasets * (n_methods - 1) - chi2 > 0:
        ff = ((n_datasets - 1) * chi2) / (n_datasets * (n_methods - 1) - chi2)
        # F-distribution with (k-1) and (k-1)(n-1) degrees of freedom
        p_value = 1 - stats.f.cdf(ff, n_methods - 1, (n_methods - 1) * (n_datasets - 1))
    else:
        # Fallback to chi-squared
        p_value = 1 - stats.chi2.cdf(chi2, n_methods - 1)
    
    return chi2, p_value


def nemenyi_critical_distance(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """
    Compute Nemenyi critical distance for post-hoc comparison.
    
    Two methods are significantly different if their average ranks
    differ by more than the critical distance.
    
    CD = q_α * sqrt(k(k+1) / (6n))
    
    where q_α is the critical value from studentized range distribution.
    
    Args:
        n_methods: Number of methods being compared
        n_datasets: Number of datasets
        alpha: Significance level (default 0.05)
        
    Returns:
        cd: Critical distance
    """
    # Critical values for Nemenyi test (from Demšar 2006)
    # q_α for α=0.05, different numbers of methods
    q_alpha_05 = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458,
        18: 3.489, 19: 3.517, 20: 3.544
    }
    
    q_alpha_10 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
        6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855,
        10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077,
        14: 3.120, 15: 3.159, 16: 3.196, 17: 3.230,
        18: 3.261, 19: 3.291, 20: 3.319
    }
    
    if alpha == 0.05:
        q_table = q_alpha_05
    elif alpha == 0.10:
        q_table = q_alpha_10
    else:
        warnings.warn(f"Alpha {alpha} not in table, using 0.05")
        q_table = q_alpha_05
    
    if n_methods not in q_table:
        # Approximate for large k
        q_alpha = 2.326 + 0.1 * np.log(n_methods)
    else:
        q_alpha = q_table[n_methods]
    
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    
    return cd


def run_friedman_nemenyi_test(
    results_df: pd.DataFrame,
    metric_column: str = "f1_score",
    method_column: str = "method",
    dataset_column: str = "dataset",
    alpha: float = 0.05
) -> Dict:
    """
    Run complete Friedman + Nemenyi analysis.
    
    Args:
        results_df: DataFrame with columns for method, dataset, and metric
        metric_column: Column name for the metric to compare
        method_column: Column name for method identifier
        dataset_column: Column name for dataset identifier
        alpha: Significance level
        
    Returns:
        Dictionary containing:
        - friedman_stat: Friedman test statistic
        - friedman_pvalue: p-value of Friedman test
        - reject_null: Whether to reject null hypothesis
        - avg_ranks: Average rank per method
        - critical_distance: Nemenyi CD
        - pairwise_significant: Matrix of significant differences
        - method_names: List of method names
    """
    # Pivot to get methods as columns, datasets as rows
    pivot = results_df.pivot_table(
        values=metric_column,
        index=dataset_column,
        columns=method_column,
        aggfunc='mean'
    )
    
    scores = pivot.values
    method_names = list(pivot.columns)
    dataset_names = list(pivot.index)
    
    n_datasets, n_methods = scores.shape
    
    # Friedman test
    friedman_stat, friedman_pvalue = friedman_test(scores)
    reject_null = friedman_pvalue < alpha
    
    # Compute ranks
    ranks = compute_ranks(scores, higher_is_better=True)
    avg_ranks = ranks.mean(axis=0)
    
    # Nemenyi critical distance
    cd = nemenyi_critical_distance(n_methods, n_datasets, alpha)
    
    # Pairwise significance matrix
    pairwise_significant = np.zeros((n_methods, n_methods), dtype=bool)
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if abs(avg_ranks[i] - avg_ranks[j]) > cd:
                pairwise_significant[i, j] = True
                pairwise_significant[j, i] = True
    
    return {
        "friedman_stat": friedman_stat,
        "friedman_pvalue": friedman_pvalue,
        "reject_null": reject_null,
        "avg_ranks": dict(zip(method_names, avg_ranks)),
        "critical_distance": cd,
        "pairwise_significant": pd.DataFrame(
            pairwise_significant,
            index=method_names,
            columns=method_names
        ),
        "method_names": method_names,
        "n_datasets": n_datasets,
        "n_methods": n_methods,
        "alpha": alpha,
        "ranks_matrix": pd.DataFrame(ranks, index=dataset_names, columns=method_names)
    }


def run_wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided"
) -> Dict:
    """
    Wilcoxon signed-rank test for pairwise method comparison.
    
    Non-parametric test for comparing two related samples.
    
    Args:
        scores_a: Scores for method A (one per dataset)
        scores_b: Scores for method B (one per dataset)
        alternative: "two-sided", "greater", or "less"
        
    Returns:
        Dictionary with test results
    """
    # Remove ties (datasets where both methods have same score)
    diff = scores_a - scores_b
    non_zero_mask = diff != 0
    
    if np.sum(non_zero_mask) < 5:
        warnings.warn("Fewer than 5 non-zero differences, test may be unreliable")
    
    statistic, p_value = stats.wilcoxon(
        scores_a[non_zero_mask],
        scores_b[non_zero_mask],
        alternative=alternative
    )
    
    return {
        "statistic": statistic,
        "p_value": p_value,
        "n_samples": len(scores_a),
        "n_non_zero": np.sum(non_zero_mask),
        "mean_diff": np.mean(diff),
        "median_diff": np.median(diff),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01
    }


def generate_significance_table(
    results: Dict,
    output_path: Optional[str] = None
) -> str:
    """
    Generate LaTeX table summarizing statistical test results.
    
    Args:
        results: Output from run_friedman_nemenyi_test
        output_path: Optional path to save LaTeX table
        
    Returns:
        LaTeX table string
    """
    method_names = results["method_names"]
    avg_ranks = results["avg_ranks"]
    cd = results["critical_distance"]
    friedman_p = results["friedman_pvalue"]
    
    # Sort by average rank
    sorted_methods = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    # Build LaTeX table
    lines = [
        r"\begin{tabular}{|l|c|c|}",
        r"\hline",
        r"\textbf{Method} & \textbf{Avg. Rank} & \textbf{Significantly Different From} \\",
        r"\hline"
    ]
    
    pairwise = results["pairwise_significant"]
    
    for method, rank in sorted_methods:
        # Find methods significantly different
        sig_diff = [m for m in method_names if pairwise.loc[method, m]]
        sig_str = ", ".join(sig_diff) if sig_diff else "-"
        
        # Bold if best rank
        if rank == sorted_methods[0][1]:
            lines.append(f"\\textbf{{{method}}} & \\textbf{{{rank:.2f}}} & {sig_str} \\\\")
        else:
            lines.append(f"{method} & {rank:.2f} & {sig_str} \\\\")
    
    lines.extend([
        r"\hline",
        r"\multicolumn{3}{|l|}{" + f"Friedman p-value: {friedman_p:.4f}, CD: {cd:.3f}" + r"} \\",
        r"\hline",
        r"\end{tabular}"
    ])
    
    latex_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)
    
    return latex_str


def generate_critical_difference_diagram(
    results: Dict,
    output_path: str,
    title: str = "Critical Difference Diagram"
):
    """
    Generate Critical Difference (CD) diagram.
    
    This is the standard visualization for Friedman-Nemenyi test results,
    showing average ranks and grouping methods that are not significantly different.
    
    Args:
        results: Output from run_friedman_nemenyi_test
        output_path: Path to save the figure
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    method_names = results["method_names"]
    avg_ranks = results["avg_ranks"]
    cd = results["critical_distance"]
    n_methods = results["n_methods"]
    
    # Sort methods by rank
    sorted_methods = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(3, n_methods * 0.5)))
    
    # Draw rank axis
    min_rank = 1
    max_rank = n_methods
    
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(0, 1)
    
    # Draw axis line
    ax.axhline(y=0.5, color='black', linewidth=1)
    
    # Draw tick marks and labels
    for r in range(1, n_methods + 1):
        ax.axvline(x=r, ymin=0.45, ymax=0.55, color='black', linewidth=1)
        ax.text(r, 0.4, str(r), ha='center', va='top', fontsize=10)
    
    # Draw CD bar
    ax.plot([1, 1 + cd], [0.9, 0.9], 'k-', linewidth=2)
    ax.text(1 + cd/2, 0.95, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot methods
    y_positions = np.linspace(0.7, 0.2, n_methods)
    
    for i, (method, rank) in enumerate(sorted_methods):
        y = y_positions[i]
        
        # Draw line from method name to rank
        if rank <= (min_rank + max_rank) / 2:
            # Left side
            ax.plot([rank, rank], [0.5, y], 'b-', linewidth=1)
            ax.plot([rank, min_rank - 0.3], [y, y], 'b-', linewidth=1)
            ax.text(min_rank - 0.35, y, method, ha='right', va='center', fontsize=10)
        else:
            # Right side
            ax.plot([rank, rank], [0.5, y], 'b-', linewidth=1)
            ax.plot([rank, max_rank + 0.3], [y, y], 'b-', linewidth=1)
            ax.text(max_rank + 0.35, y, method, ha='left', va='center', fontsize=10)
        
        # Mark the rank position
        ax.plot(rank, 0.5, 'bo', markersize=6)
    
    # Draw thick lines connecting methods not significantly different
    pairwise = results["pairwise_significant"]
    drawn_groups = []
    
    for i, (method_i, rank_i) in enumerate(sorted_methods):
        for j, (method_j, rank_j) in enumerate(sorted_methods[i+1:], i+1):
            if not pairwise.loc[method_i, method_j]:
                # Not significantly different - draw connector
                # Check if already part of a group
                if abs(rank_i - rank_j) <= cd:
                    ax.plot([rank_i, rank_j], [0.52, 0.52], 'k-', linewidth=3, alpha=0.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"CD diagram saved to: {output_path}")


def run_comprehensive_statistical_analysis(
    results_df: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = ["f1_score", "precision", "recall"]
) -> Dict:
    """
    Run complete statistical analysis and generate all outputs.
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save outputs
        metrics: List of metrics to analyze
        
    Returns:
        Dictionary with all analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"Statistical Analysis for: {metric}")
        print("="*60)
        
        # Run Friedman-Nemenyi
        results = run_friedman_nemenyi_test(results_df, metric_column=metric)
        all_results[metric] = results
        
        print(f"Friedman statistic: {results['friedman_stat']:.4f}")
        print(f"Friedman p-value: {results['friedman_pvalue']:.6f}")
        print(f"Reject null (α=0.05): {results['reject_null']}")
        print(f"Critical Distance: {results['critical_distance']:.3f}")
        
        print("\nAverage Ranks:")
        for method, rank in sorted(results['avg_ranks'].items(), key=lambda x: x[1]):
            print(f"  {method}: {rank:.3f}")
        
        # Generate outputs
        latex_path = os.path.join(output_dir, f"significance_{metric}.tex")
        generate_significance_table(results, latex_path)
        print(f"\nLaTeX table saved to: {latex_path}")
        
        cd_path = os.path.join(output_dir, f"cd_diagram_{metric}.png")
        generate_critical_difference_diagram(
            results, cd_path,
            title=f"Critical Difference Diagram ({metric.upper()})"
        )
    
    return all_results


# Example usage and test
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Simulate results: 5 methods, 7 datasets
    methods = ["SE-CDT", "CDT_MSW", "MMD", "HDDDM", "ADWIN"]
    datasets = ["SEA", "Hyperplane", "RBF", "Electricity", "Circles", "Sine1", "LED"]
    
    # Generate fake F1 scores (SE-CDT should be best)
    data = []
    for dataset in datasets:
        base_scores = [0.75, 0.60, 0.65, 0.55, 0.70]  # SE-CDT best
        for i, method in enumerate(methods):
            score = base_scores[i] + np.random.normal(0, 0.1)
            score = np.clip(score, 0, 1)
            data.append({
                "method": method,
                "dataset": dataset,
                "f1_score": score,
                "precision": score + np.random.normal(0, 0.05),
                "recall": score + np.random.normal(0, 0.05)
            })
    
    df = pd.DataFrame(data)
    
    # Run analysis
    results = run_comprehensive_statistical_analysis(
        df,
        output_dir="./test_statistical_output",
        metrics=["f1_score"]
    )
    
    print("\n✓ Statistical analysis complete!")
