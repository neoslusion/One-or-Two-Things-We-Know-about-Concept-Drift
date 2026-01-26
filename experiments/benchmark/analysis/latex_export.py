"""
LaTeX table export module for thesis/publication.

Standardized to match the formatting in se_cdt_content.tex:
- Uses vertical bars |l|c|
- Uses \hline for all separators
- No booktabs dependency
"""

from pathlib import Path
import pandas as pd
import numpy as np
from core.config import generate_standard_table, format_metric, escape_latex, TABLES_DIR

def export_all_tables(all_results, stream_size, output_dir=TABLES_DIR):
    """
    Export all LaTeX tables for thesis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if len(all_results) == 0:
        print("ERROR: No results found. Please run the benchmark first.")
        return

    # Create results DataFrame
    results_df = pd.DataFrame([{
        "Dataset": r["dataset"],
        "Method": r["method"],
        "Precision": r.get("precision", 0.0),
        "Recall": r.get("recall", 0.0),
        "F1": r.get("f1_score", 0.0),
        "MTTD": r.get("mttd", np.nan) if r.get("mttd") != float("inf") else np.nan,
        "FP": r.get("fp", 0),
        "Runtime_s": r.get("runtime_s", 0.0),
        "N_Drifts": len(r.get("drift_positions", []))
    } for r in all_results])

    drift_results_df = results_df[results_df["N_Drifts"] > 0].copy()

    if len(drift_results_df) == 0:
        print("ERROR: No drift datasets found in results.")
        return

    # ========================================================================
    # TABLE I: Comprehensive Performance Summary
    # ========================================================================
    method_stats = drift_results_df.groupby("Method").agg({
        "Precision": "mean",
        "Recall": "mean",
        "F1": "mean",
        "MTTD": "mean",
        "FP": "mean"
    }).round(3)

    method_stats = method_stats.sort_values("F1", ascending=False)
    
    headers = ["Method", "Precision", "Recall (EDR)", "F1-Score", "Delay", "False Pos."]
    data = []
    
    # Find best values for bolding
    best_f1 = method_stats["F1"].max()
    
    for method, row in method_stats.iterrows():
        f1_str = f"{row['F1']:.3f}"
        if row['F1'] == best_f1:
            f1_str = f"\\textbf{{{f1_str}}}"
            
        data.append([
            escape_latex(method),
            f"{row['Precision']:.3f}",
            f"{row['Recall']:.3f}",
            f1_str,
            f"{row['MTTD']:.0f}" if not pd.isna(row['MTTD']) else "0",
            f"{row['FP']:.1f}"
        ])

    latex_output = generate_standard_table(headers, data, align="|l|c|c|c|c|c|")
    
    with open(output_dir / "table_I_comprehensive_performance.tex", "w") as f:
        f.write(latex_output)

    # ========================================================================
    # TABLE II: F1 by Dataset (Broken down into parts)
    # ========================================================================
    f1_by_dataset = drift_results_df.pivot_table(
        values="F1", index="Method", columns="Dataset", aggfunc="mean"
    ).round(3)
    
    # Calculate mean and sort
    f1_by_dataset["Mean"] = f1_by_dataset.mean(axis=1).round(3)
    f1_by_dataset = f1_by_dataset.sort_values("Mean", ascending=False)

    # Dataset name simplifier
    def simplify_name(name):
        name = name.replace("gen_random_", "Random ").replace("gaussian_shift_", "Gaussian ").replace("electricity_semisynthetic", "Electricity")
        name = name.replace("_", " ").title()
        name = name.replace("Mmd", "MMD").replace("Adw", "ADW").replace("Proper", "PROPER")
        return name

    # Process columns (datasets) excluding Mean first
    datasets = [c for c in f1_by_dataset.columns if c != "Mean"]
    
    # Chunk datasets into groups of 4
    chunk_size = 4
    for i, chunk_start in enumerate(range(0, len(datasets), chunk_size)):
        chunk_cols = datasets[chunk_start:chunk_start + chunk_size]
        
        # If this is the last chunk, append Mean
        is_last = (chunk_start + chunk_size >= len(datasets))
        if is_last:
            chunk_cols.append("Mean")
            
        # Prepare headers
        headers = ["Method"] + [simplify_name(c) for c in chunk_cols]
        data = []
        
        for method, row in f1_by_dataset.iterrows():
            row_data = [escape_latex(method)]
            for col in chunk_cols:
                val = row[col]
                if col == "Mean" and val == f1_by_dataset["Mean"].max():
                    row_data.append(f"\\textbf{{{val:.3f}}}")
                else:
                    row_data.append(f"{val:.3f}")
            data.append(row_data)

        # Generate table part
        part_num = i + 1
        latex_output = generate_standard_table(headers, data)
        
        filename = f"table_II_part{part_num}.tex"
        with open(output_dir / filename, "w") as f:
            f.write(latex_output)
            
        print(f"  Generated {filename}")

    # Keep original file as full backup (optional, or overwritten by full table if needed)
    # But user asked to break it down. I will write the full one too just in case 
    # but maybe with rotation or small font? No, just raw.
    headers = ["Method"] + list(f1_by_dataset.columns)
    data = []
    for method, row in f1_by_dataset.iterrows():
        data.append([escape_latex(method)] + [f"{val:.3f}" for val in row])
    
    full_output = generate_standard_table(headers, data)
    with open(output_dir / "table_II_f1_by_dataset.tex", "w") as f:
        f.write(full_output)

    print(f"  All tables exported to {output_dir}")
