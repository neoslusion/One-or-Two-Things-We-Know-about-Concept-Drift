"""
LaTeX table export module for thesis/publication.

Contains functions for generating publication-ready LaTeX tables:
- Comprehensive performance summary
- F1-Score by dataset
- Runtime statistics

Uses booktabs style and highlights best values with bold formatting.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def bold_best_in_column(series, higher_is_better=True):
    """Return series with best value wrapped in \\textbf{}."""
    if higher_is_better:
        best_val = series.max()
    else:
        best_val = series.min()
    return series.apply(lambda x: f"\\textbf{{{x}}}" if x == best_val else str(x))


def format_with_bold_best(df, metric_cols, higher_is_better_dict):
    """Format dataframe with bold best values per column."""
    df_formatted = df.copy()
    for col in metric_cols:
        if col in df_formatted.columns:
            higher = higher_is_better_dict.get(col, True)
            df_formatted[col] = bold_best_in_column(df_formatted[col], higher)
    return df_formatted


def export_all_tables(all_results, stream_size, output_dir="./publication_figures"):
    """
    Export all LaTeX tables for thesis.

    Args:
        all_results: List of result dictionaries from benchmark
        stream_size: Size of each data stream
        output_dir: Directory to save tables (default: ./publication_figures)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if len(all_results) == 0:
        print("ERROR: No results found. Please run the benchmark first.")
        return

    # Create results DataFrame
    results_df = pd.DataFrame(
        [
            {
                "Dataset": r["dataset"],
                "Method": r["method"],
                "N_Features": r.get("n_features", 0),
                "N_Drifts": len(r.get("drift_positions", [])),
                "Intensity": r.get("intensity", 0),
                "TP": r.get("tp", 0),
                "FP": r.get("fp", 0),
                "FN": r.get("fn", 0),
                "Precision": r.get("precision", 0.0),
                "Recall": r.get("recall", 0.0),  # This is EDR in thesis
                "F1": r.get("f1_score", 0.0),
                "MTTD": r.get("mttd", np.nan)
                if r.get("mttd") != float("inf")
                else np.nan,
                "Detection_Rate": r.get("detection_rate", 0.0),
                "N_Detections": r.get("n_detections", 0),
                "Runtime_s": r.get("runtime_s", 0.0),
            }
            for r in all_results
        ]
    )

    drift_results_df = results_df[results_df["N_Drifts"] > 0].copy()

    if len(drift_results_df) == 0:
        print("ERROR: No drift datasets found in results.")
        return

    print("=" * 80)
    print("LATEX TABLE EXPORT (Thesis Format)")
    print("=" * 80)

    # ========================================================================
    # TABLE I: Comprehensive Performance Summary (Aligned with Thesis)
    # ========================================================================
    # Thesis Format: Method | Precision | Recall (EDR) | F1 | Delay | FP
    print("\n[Table I] Comprehensive Performance Summary")

    method_stats = (
        drift_results_df.groupby("Method")
        .agg(
            {
                "Precision": "mean",
                "Recall": "mean",
                "F1": "mean",  # No std in main aggregate table in thesis usually
                "MTTD": "mean",  # Delay
                "FP": "mean",  # Thesis reports Avg FP per stream, not sum
            }
        )
        .round(3)
    )

    pub_table = pd.DataFrame(
        {
            "Method": method_stats.index,
            "Precision": method_stats["Precision"],
            "Recall": method_stats["Recall"],  # EDR
            "F1": method_stats["F1"],
            "Delay": method_stats["MTTD"].fillna(0).astype(int),
            "FP": method_stats["FP"].round(1),  # Avg FP
        }
    )

    pub_table = pub_table.sort_values("F1", ascending=False).reset_index(drop=True)

    # Find best values for bolding
    best_f1 = pub_table["F1"].max()
    best_precision = pub_table["Precision"].max()
    best_recall = pub_table["Recall"].max()
    best_delay = pub_table["Delay"].min()
    best_fp = pub_table["FP"].min()

    # Format metrics with bold for best
    def format_metric(val, best_val, higher_is_better=True, precision=3):
        fmt_str = f"{{:.{precision}f}}"
        formatted = fmt_str.format(val)
        if (higher_is_better and val == best_val) or (
            not higher_is_better and val == best_val
        ):
            return f"\\textbf{{{formatted}}}"
        return formatted

    pub_table["Precision"] = pub_table["Precision"].apply(
        lambda x: format_metric(x, best_precision)
    )
    pub_table["Recall"] = pub_table["Recall"].apply(
        lambda x: format_metric(x, best_recall)
    )
    pub_table["F1"] = pub_table["F1"].apply(lambda x: format_metric(x, best_f1))
    pub_table["Delay"] = pub_table["Delay"].apply(
        lambda x: format_metric(x, best_delay, higher_is_better=False, precision=0)
    )
    pub_table["FP"] = pub_table["FP"].apply(
        lambda x: format_metric(x, best_fp, higher_is_better=False, precision=1)
    )

    latex_table = pub_table[
        ["Method", "Precision", "Recall", "F1", "Delay", "FP"]
    ].copy()
    latex_table.columns = [
        "Method",
        "Precision",
        "Recall (EDR)",
        "F1-Score",
        "Delay",
        "False Pos.",
    ]

    latex_output = latex_table.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * 5,
        caption="Aggregate performance metrics averaged across all datasets. Best values are bolded.",
        label="tab:comparison_aggregate",
        position="htbp",
    )

    # Booktabs style
    latex_output = latex_output.replace("\\toprule", "\\toprule")
    latex_output = latex_output.replace("\\midrule", "\\midrule")
    latex_output = latex_output.replace("\\bottomrule", "\\bottomrule")

    # Output to correct thesis file location
    latex_file = Path("report/latex/tables/table_comparison_aggregate.tex")
    latex_file.parent.mkdir(exist_ok=True, parents=True)

    with open(latex_file, "w") as f:
        f.write(latex_output)
    print(f"  Saved: {latex_file}")

    print("\nTable I Preview:")
    print(latex_table.to_string(index=False))

    # ========================================================================
    # TABLE II: Performance by Dataset (Split to avoid overflow)
    # ========================================================================
    print("\n" + "-" * 80)
    print("[Table II] Performance by Dataset (Split)")

    f1_by_dataset = drift_results_df.pivot_table(
        values="F1", index="Method", columns="Dataset", aggfunc="mean"
    ).round(3)

    f1_by_dataset["Mean"] = f1_by_dataset.mean(axis=1).round(3)
    f1_by_dataset = f1_by_dataset.sort_values("Mean", ascending=False)

    # Format bolding
    f1_formatted = f1_by_dataset.copy()
    for col in f1_formatted.columns:
        best_val = f1_formatted[col].max()
        f1_formatted[col] = f1_formatted[col].apply(
            lambda x: f"\\textbf{{{x:.3f}}}" if x == best_val else f"{x:.3f}"
        )

    # Add rank
    f1_formatted.insert(0, "Rank", range(1, len(f1_formatted) + 1))

    # SPLIT LOGIC
    datasets = [c for c in f1_formatted.columns if c not in ["Rank", "Mean"]]
    mid = len(datasets) // 2

    # Table II-A: First half + Mean
    cols_a = ["Rank"] + datasets[:mid] + ["Mean"]
    table_a = f1_formatted[cols_a]

    # Table II-B: Second half + Mean
    cols_b = ["Rank"] + datasets[mid:] + ["Mean"]
    table_b = f1_formatted[cols_b]

    # Generate LaTeX for Part A
    latex_a = table_a.to_latex(
        escape=False,
        column_format="c" + "l" + "c" * (len(cols_a) - 1),
        caption="F1-Score by dataset (Part 1/2).",
        label="tab:f1_by_dataset_part1",
        position="htbp",
    ).replace("\\hline", "\\midrule")

    # Generate LaTeX for Part B
    latex_b = table_b.to_latex(
        escape=False,
        column_format="c" + "l" + "c" * (len(cols_b) - 1),
        caption="F1-Score by dataset (Part 2/2).",
        label="tab:f1_by_dataset_part2",
        position="htbp",
    ).replace("\\hline", "\\midrule")

    # Save to file (concatenated or separate files, usually separate is safer for thesis import)
    # But often thesis expects one file input. Let's write them sequentially.

    latex_file2 = output_dir / "table_II_f1_by_dataset.tex"
    with open(latex_file2, "w") as f:
        f.write(latex_a)
        f.write("\n\n")  # Separation
        f.write(latex_b)
    print(f"  Saved: {latex_file2} (Split into two tables)")

    print("\nTable II Preview:")
    print(f1_by_dataset.to_string())

    # ========================================================================
    # TABLE III: Runtime Statistics
    # ========================================================================
    print("\n" + "-" * 80)
    print("[Table III] Runtime Statistics")

    runtime_stats = (
        results_df.groupby("Method")
        .agg({"Runtime_s": ["mean", "std", "min", "max"]})
        .round(4)
    )
    runtime_stats.columns = ["Mean (s)", "Std (s)", "Min (s)", "Max (s)"]

    # Handle NaN/inf values in throughput calculation
    throughput = stream_size / runtime_stats["Mean (s)"]
    throughput = throughput.replace([np.inf, -np.inf], np.nan)
    runtime_stats["Throughput (samples/s)"] = throughput.fillna(0).round(0).astype(int)

    runtime_stats = runtime_stats.sort_values("Mean (s)")

    latex_runtime = runtime_stats.to_latex(
        escape=False,
        column_format="l" + "c" * len(runtime_stats.columns),
        caption="Runtime statistics by detection method. Throughput = samples processed per second.",
        label="tab:runtime_stats",
        position="htbp",
    )

    latex_file3 = output_dir / "table_III_runtime_stats.tex"
    with open(latex_file3, "w") as f:
        f.write(latex_runtime)
    print(f"  Saved: {latex_file3}")

    print("\nTable III Preview:")
    print(runtime_stats.to_string())

    print("\n" + "=" * 80)
    print(f"All LaTeX tables saved to: {output_dir.absolute()}")
    print("=" * 80)
