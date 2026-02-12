#!/usr/bin/env python3
"""
Generate LaTeX Tables from Benchmark Results

Reads the benchmark results pickle file and generates LaTeX tables for the report.
Specifically handles:
1. Per-Drift-Type Performance Table (table_se_cdt_performance_by_type.tex)
2. SE-CDT Performance Summary (table_se_cdt_performance_summary.tex)
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    BENCHMARK_PROPER_OUTPUTS,
    TABLES_DIR,
    escape_latex,
    format_metric,
    generate_standard_table
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("TableGenerator")

def generate_drift_type_performance_table(results):
    """
    Generate Table: Classification Accuracy by Drift Type for SE-CDT.
    Focuses solely on SE-CDT performance.
    """
    # 1. Calculate SE-CDT Per-Type Accuracy
    confusion = {}
    total_counts = {}
    
    # Standard SE-CDT drift types mapping (GT -> Pred)
    drift_types = ["Sudden", "Blip", "Gradual", "Incremental", "Recurrent"]
    for dt in drift_types:
        confusion[dt] = 0
        total_counts[dt] = 0
        
    for res in results:
        if 'SE_Classifications' not in res:
            continue
            
        for item in res['SE_Classifications']:
            gt = item['gt_type']
            pred = item['pred']
            
            if gt in drift_types:
                total_counts[gt] += 1
                if gt == pred:
                    confusion[gt] += 1
                    
    per_class_acc = {}
    
    for dt in drift_types:
        if total_counts[dt] > 0:
            per_class_acc[dt] = (confusion[dt] / total_counts[dt]) * 100
        else:
            per_class_acc[dt] = 0.0
            
    # 2. Generate Table Data
    headers = ["Drift Type", "Accuracy [%]"]
    data = []
    
    # Rows for individual drift types
    for dt in drift_types:
        se_val = per_class_acc[dt]
        se_str = format_metric(se_val/100, "percentage")
            
        # Special note for Recurrent
        if dt == "Recurrent":
            se_str += "$^\\dagger$"
            
        data.append([escape_latex(dt), se_str])
        
    # Generate Latex
    # Standard table with 2 columns
    latex_output = generate_standard_table(headers, data, align="|l|c|")
    
    # Add footer manually
    footer = (
        "\\hline\n"
        "\\multicolumn{2}{|l|}{\\footnotesize $^\\dagger$Recurrent drift processed as separate Sudden events (no concept memory).}\\ \\n"
        "\\hline\n"
    )
    
    # Insert footer before \end{tabular}
    latex_output = latex_output.replace("\\end{tabular}", footer + "\\end{tabular}")

    output_path = TABLES_DIR / "table_se_cdt_performance_by_type.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_output)
        
    logger.info(f"Generated Drift Type Performance Table: {output_path}")
    print(f"   ✓ Generated: {output_path.name}")


def generate_performance_summary_table(results):
    """
    Generate Table: SE-CDT Performance Summary.
    Replaces the Fair Comparison table.
    """
    # 1. Compute SE-CDT classification accuracy (unsupervised)
    total_se = 0
    correct_cat_se = 0
    correct_sub_se = 0
    
    TCD_TYPES = {"Sudden", "Blip", "Recurrent"}
    
    for res in results:
        if 'SE_Classifications' not in res: continue
        total_se += len(res['SE_Classifications'])
        
        for item in res['SE_Classifications']:
            gt = item['gt_type']
            pred = item['pred']
            
            # Category accuracy (TCD vs PCD)
            is_gt_tcd = gt in TCD_TYPES
            is_pred_tcd = pred in TCD_TYPES
            if is_gt_tcd == is_pred_tcd:
                correct_cat_se += 1
            
            # Subcategory accuracy
            if gt == pred:
                correct_sub_se += 1
    
    cat_acc_se = (correct_cat_se / total_se * 100) if total_se > 0 else 0
    sub_acc_se = (correct_sub_se / total_se * 100) if total_se > 0 else 0
    
    # 2. Compute detection metrics (EDR)
    metrics_agg = {
        "SE": {"TP": 0, "FP": 0, "FN": 0, "Total": 0}
    }
    
    for r in results:
        if 'SE_STD_Metrics' in r:
            metrics_agg["SE"]["TP"] += r['SE_STD_Metrics']["TP"]
            metrics_agg["SE"]["FP"] += r['SE_STD_Metrics']["FP"]
            metrics_agg["SE"]["FN"] += r['SE_STD_Metrics']["FN"]
            metrics_agg["SE"]["Total"] += r['SE_STD_Metrics']["FN"] + r['SE_STD_Metrics']["TP"]
    
    se_edr = (metrics_agg["SE"]["TP"] / metrics_agg["SE"]["Total"] * 100) if metrics_agg["SE"]["Total"] > 0 else 0
    
    # 3. Generate Table Data
    headers = ["Metric", "Value", "Description"]
    data = []
    
    data.append(["Event Detection Rate (EDR)", format_metric(se_edr / 100, "percentage"), "Sensitivity to drift events"])
    data.append(["Category Accuracy (CAT)", format_metric(cat_acc_se / 100, "percentage"), "TCD vs PCD distinction"])
    data.append(["Subtype Accuracy (SUB)", format_metric(sub_acc_se / 100, "percentage"), "Exact drift type identification"])
    
    latex_output = generate_standard_table(headers, data, align="|l|c|l|")
    
    output_path = TABLES_DIR / "table_se_cdt_performance_summary.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_output)
        
    logger.info(f"Generated Performance Summary Table: {output_path}")
    print(f"   ✓ Generated: {output_path.name}")


def main():
    pkl_path = Path(BENCHMARK_PROPER_OUTPUTS["results_pkl"])
    if not pkl_path.exists():
        logger.error(f"Benchmark results not found at {pkl_path}")
        return
        
    try:
        results = pd.read_pickle(pkl_path)
        if isinstance(results, pd.DataFrame):
            results = results.to_dict('records')
            
        generate_drift_type_performance_table(results)
        generate_performance_summary_table(results)
        
    except Exception as e:
        logger.error(f"Failed to generate tables: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
