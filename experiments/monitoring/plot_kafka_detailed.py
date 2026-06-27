"""
Detailed Kafka System Report Generator
Reads: 
  - experiments/monitoring/shapedd_batches.csv (Detections)
  - experiments/monitoring/system_metrics.csv (Performance & Resources)
Output: 
  - experiments/monitoring/kafka_report_detailed.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import numpy as np

def generate_detailed_report():
    detect_log = "experiments/monitoring/shapedd_batches.csv"
    metrics_log = "experiments/monitoring/system_metrics.csv"
    output_path = "experiments/monitoring/kafka_report_detailed.png"

    # Check files
    if not os.path.exists(metrics_log):
        print(f"Error: Metrics log {metrics_log} not found. Run deployment first.")
        sys.exit(1)
        
    try:
        df_metrics = pd.read_csv(metrics_log)
        df_detect = pd.read_csv(detect_log) if os.path.exists(detect_log) else pd.DataFrame()
    except Exception as e:
        print(f"Error reading logs: {e}")
        sys.exit(1)
        
    if df_metrics.empty:
        print("Metrics log is empty.")
        sys.exit(0)

    # Setup Figure (4 Panels)
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1.5, 0.8, 1], hspace=0.3)
    
    # Shared X-axis (Index)
    ax_stream = plt.subplot(gs[0])
    ax_acc = plt.subplot(gs[1], sharex=ax_stream)
    ax_det = plt.subplot(gs[2], sharex=ax_stream)
    ax_res = plt.subplot(gs[3], sharex=ax_stream)

    # Ground Truth Drift (Known at 1500 for this scenario)
    GT_DRIFT = 1500
    
    # Color Scheme
    c_acc = "#2980b9" # Blue
    c_drift = "#e74c3c" # Red
    c_mem = "#8e44ad" # Purple
    c_cpu = "#27ae60" # Green

    # =========================================================================
    # Panel 1: Data Stream Stats
    # =========================================================================
    ax_stream.plot(df_metrics['idx'], df_metrics['feature_mean'], color='gray', alpha=0.6, label='Feature Mean')
    # Rolling mean for smoothness
    roll_mean = df_metrics['feature_mean'].rolling(window=20).mean()
    ax_stream.plot(df_metrics['idx'], roll_mean, color='black', linewidth=1.5, label='Rolling Mean (20)')
    
    # Mark GT Drift
    ax_stream.axvline(GT_DRIFT, color=c_drift, linestyle='--', linewidth=2, label='True Drift (1500)')
    ax_stream.fill_between(df_metrics['idx'], df_metrics['feature_mean'].min(), df_metrics['feature_mean'].max(), 
                           where=(df_metrics['idx'] >= GT_DRIFT), color=c_drift, alpha=0.05)

    ax_stream.set_ylabel("Feature Value (Mean)", fontsize=10, fontweight='bold')
    ax_stream.set_title("1. Data Stream Characteristic (Shift at 1500)", fontsize=12, fontweight='bold')
    ax_stream.legend(loc='upper left', fontsize=9)
    ax_stream.grid(True, alpha=0.3)
    plt.setp(ax_stream.get_xticklabels(), visible=False)

    # =========================================================================
    # Panel 2: Prequential Accuracy
    # =========================================================================
    ax_acc.plot(df_metrics['idx'], df_metrics['accuracy'], color=c_acc, linewidth=2, label='Prequential Accuracy')
    
    # Mark detected drifts
    if not df_detect.empty:
        for _, row in df_detect.iterrows():
            d_idx = row['detection_idx']
            # Vertical line for detection
            ax_acc.axvline(d_idx, color='orange', linestyle=':', label='Detection' if _ == 0 else "")
            
            # Annotation
            ax_acc.annotate(f"{row['drift_type']}\n({row['drift_category']})", 
                            xy=(d_idx, 0.5), xytext=(0, 20), textcoords="offset points",
                            ha='center', fontsize=8, color='black',
                            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8))

    ax_acc.axvline(GT_DRIFT, color=c_drift, linestyle='--', linewidth=2)
    ax_acc.set_ylabel("Accuracy", fontsize=10, fontweight='bold')
    ax_acc.set_ylim(0, 1.1)
    ax_acc.set_title("2. Model Accuracy & Drift Response", fontsize=12, fontweight='bold')
    ax_acc.legend(loc='lower left', fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    plt.setp(ax_acc.get_xticklabels(), visible=False)

    # =========================================================================
    # Panel 3: Drift Detection Score (P-Value)
    # =========================================================================
    if not df_detect.empty:
        # Plot p-values as stems or scatter
        markerline, stemlines, baseline = ax_det.stem(
            df_detect['detection_idx'], 
            df_detect['p_value'], 
            linefmt='r-', markerfmt='ro', basefmt='k-'
        )
        plt.setp(markerline, markersize=8)
        
        # Threshold line (0.05)
        ax_det.axhline(0.05, color='black', linestyle='--', label='Alpha (0.05)')
    else:
        ax_det.text(0.5, 0.5, "No Detections Logged", ha='center', transform=ax_det.transAxes)

    ax_det.set_yscale('log')
    ax_det.set_ylabel("P-Value (Log Scale)", fontsize=10, fontweight='bold')
    ax_det.set_title("3. SE-CDT Drift Confidence", fontsize=12, fontweight='bold')
    ax_det.grid(True, alpha=0.3)
    plt.setp(ax_det.get_xticklabels(), visible=False)

    # =========================================================================
    # Panel 4: System Resources
    # =========================================================================
    ax_res_cpu = ax_res.twinx()
    
    # Memory
    p1 = ax_res.fill_between(df_metrics['idx'], 0, df_metrics['memory_mb'], color=c_mem, alpha=0.3)
    ax_res.plot(df_metrics['idx'], df_metrics['memory_mb'], color=c_mem, linewidth=1.5, label='Memory (MB)')
    
    # CPU
    p2 = ax_res_cpu.plot(df_metrics['idx'], df_metrics['cpu_percent'], color=c_cpu, linewidth=1, alpha=0.6, label='CPU (%)')

    ax_res.set_ylabel("Memory (MB)", fontsize=10, fontweight='bold', color=c_mem)
    ax_res_cpu.set_ylabel("CPU (%)", fontsize=10, fontweight='bold', color=c_cpu)
    
    ax_res.set_xlabel("Sample Index", fontsize=12, fontweight='bold')
    ax_res.set_title("4. System Resource Consumption", fontsize=12, fontweight='bold')
    
    # Combined legend
    lines = [p1, p2[0]]
    labels = ['Memory (MB)', 'CPU (%)']
    ax_res.legend(lines, labels, loc='upper left', fontsize=9)
    
    ax_res.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Detailed Report Generated: {output_path}")

if __name__ == "__main__":
    generate_detailed_report()
