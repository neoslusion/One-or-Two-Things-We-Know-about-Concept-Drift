"""
aggregate_validation_results.py

Aggregates metrics from multiple independent runs of the monitoring system.
Generates a statistical report verifying system stability and performance.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy import stats


def parse_metrics_file(file_path):
    """Parse key metrics from a metrics_*.txt file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Extract Type-Specific Adaptation Metrics
    # We look for the "type_specific:" section
    data = {}

    # Regex patterns for key metrics under "type_specific:" block
    # Note: The file structure assumes "type_specific:" comes first or is distinct

    try:
        # Split by mode to ensure we get Type Specific
        sections = content.split("type_specific:")
        if len(sections) < 2:
            return None
        ts_section = sections[1].split("simple_retrain:")[0]  # Limit scope

        # Detection
        edr_match = re.search(r"EDR \(Recall\):\s+([\d\.]+)", ts_section)
        mdr_match = re.search(r"MDR \(Miss Rate\):\s+([\d\.]+)", ts_section)
        prec_match = re.search(r"Precision:\s+([\d\.]+)", ts_section)
        delay_match = re.search(r"Mean Delay:\s+([\d\.]+)", ts_section)

        # Adaptation
        acc_match = re.search(r"Overall Accuracy:\s+([\d\.]+)", ts_section)
        restore_match = re.search(r"Restoration Time:\s+([\d\.]+)", ts_section)

        if edr_match:
            data["EDR"] = float(edr_match.group(1))
        if mdr_match:
            data["MDR"] = float(mdr_match.group(1))
        if prec_match:
            data["Precision"] = float(prec_match.group(1))
        if delay_match:
            data["Mean_Delay"] = float(delay_match.group(1))
        if acc_match:
            data["Accuracy"] = float(acc_match.group(1))
        if restore_match:
            data["Restoration_Time"] = float(restore_match.group(1))

        # Get No Adaptation Accuracy for Improvement Calc
        none_section = content.split("no_adaptation:")[1]
        acc_none_match = re.search(r"Overall Accuracy:\s+([\d\.]+)", none_section)
        if acc_none_match:
            data["Baseline_Accuracy"] = float(acc_none_match.group(1))
            data["Improvement"] = data["Accuracy"] - data["Baseline_Accuracy"]

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    return data


def calculate_ci(data, confidence=0.95):
    """Calculate 95% Confidence Interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def main():
    parser = argparse.ArgumentParser(description="Aggregate Validation Results")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing run_* folders",
    )
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    all_metrics = []

    print(f"Scanning {base_dir} for results...")

    run_folders = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )

    for run_dir in run_folders:
        # Find metrics file (e.g., metrics_mixed.txt)
        metrics_files = list(run_dir.glob("metrics_*.txt"))
        if metrics_files:
            data = parse_metrics_file(metrics_files[0])
            if data:
                data["Run"] = run_dir.name
                all_metrics.append(data)

    if not all_metrics:
        print("No valid metrics found!")
        return

    df = pd.DataFrame(all_metrics)
    n_runs = len(df)

    print("\n" + "=" * 60)
    print(f"SYSTEM VALIDATION REPORT ({n_runs} Independent Runs)")
    print("=" * 60)

    metrics_to_report = [
        "Accuracy",
        "Improvement",
        "EDR",
        "Precision",
        "Restoration_Time",
        "Mean_Delay",
    ]

    print(f"{'Metric':<20} | {'Mean':<10} | {'Std Dev':<10} | {'95% CI':<20}")
    print("-" * 70)

    results_summary = {}

    for metric in metrics_to_report:
        if metric in df.columns:
            values = df[metric].dropna()
            mean, low, high = calculate_ci(values)
            std = np.std(values)

            # Formatting based on metric type
            if metric in ["Accuracy", "Improvement", "EDR", "Precision"]:
                fmt_mean = f"{mean:.2%}"
                fmt_std = f"{std:.2%}"
                fmt_ci = f"[{low:.2%}, {high:.2%}]"
            else:
                fmt_mean = f"{mean:.1f}"
                fmt_std = f"{std:.1f}"
                fmt_ci = f"[{low:.1f}, {high:.1f}]"

            print(f"{metric:<20} | {fmt_mean:<10} | {fmt_std:<10} | {fmt_ci:<20}")
            results_summary[metric] = mean

    print("=" * 60)

    # --- AUTOMATED JUDGMENT ---
    print("\n[AUTOMATED JUDGMENT]")

    # Criteria 1: Stability (Std Dev of Accuracy < 5%)
    acc_std = df["Accuracy"].std()
    if acc_std < 0.05:
        print("✅ STABILITY: PASS (Std Dev < 5%)")
    else:
        print(f"⚠️ STABILITY: WARNING (High Variance: {acc_std:.2%})")

    # Criteria 2: Effectiveness (Improvement > 0)
    imp_mean = df["Improvement"].mean()
    if imp_mean > 0.05:
        print(f"✅ EFFECTIVENESS: PASS (Avg Improvement +{imp_mean:.1%})")
    elif imp_mean > 0:
        print(f"⚠️ EFFECTIVENESS: MARGINAL (Avg Improvement +{imp_mean:.1%})")
    else:
        print(f"❌ EFFECTIVENESS: FAIL (No Improvement)")

    # Criteria 3: Detection Reliability (EDR > 0.8)
    edr_mean = df["EDR"].mean()
    if edr_mean > 0.8:
        print("✅ DETECTION: PASS (Recall > 80%)")
    else:
        print(f"⚠️ DETECTION: WARNING (Recall {edr_mean:.1%} is low)")

    # Save summary
    df.to_csv(base_dir / "aggregated_results.csv", index=False)
    print(f"\nDetailed CSV saved to {base_dir}/aggregated_results.csv")


if __name__ == "__main__":
    main()
