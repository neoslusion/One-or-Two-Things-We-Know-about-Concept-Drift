import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.benchmark.benchmark_proper import generate_mixed_stream, compute_mmd_sequence, WINDOW_SIZE
from core.config import BENCHMARK_PROPER_OUTPUTS, PLOTS_DIR

RESULTS_FILE = str(BENCHMARK_PROPER_OUTPUTS["results_pkl"])
OUTPUT_DIR = str(PLOTS_DIR)

def plot_scenario(scenario_name, seed=0):
    print(f"Visualizing {scenario_name} (Seed {seed})...")
    
    # 1. Re-generate Data (Match benchmark_proper.py 10-event logic)
    events = []
    length = 4000
    if scenario_name == "Mixed_A":
        pattern = [
            {"type": "Sudden", "width": 0},
            {"type": "Gradual", "width": 400},
            {"type": "Recurrent", "width": 400}
        ]
        for i in range(10):
            evt_template = pattern[i % 3]
            events.append({
                "type": evt_template["type"],
                "pos": 800 + i * 1000, 
                "width": evt_template["width"]
            })
        length = 800 + 10 * 1000 + 1000
        
    elif scenario_name == "Mixed_B":
        pattern = [
            {"type": "Blip", "width": 100},
            {"type": "Incremental", "width": 500},
            {"type": "Sudden", "width": 0}
        ]
        for i in range(10):
            evt_template = pattern[i % 3]
            events.append({
                "type": evt_template["type"],
                "pos": 800 + i * 1200, 
                "width": evt_template["width"]
            })
        length = 800 + 10 * 1200 + 1000

    elif scenario_name == "Repeated_Gradual":
        # Match benchmark_proper.py: pos = 800 + i*1000, width=300
        for i in range(10):
            events.append({
                "type": "Gradual",
                "pos": 800 + i * 1000,
                "width": 300
            })
        length = 800 + 10 * 1000 + 500

    elif scenario_name == "Repeated_Incremental":
        # Match benchmark_proper.py: pos = 800 + i*1200, width=600
        for i in range(10):
            events.append({
                "type": "Incremental",
                "pos": 800 + i * 1200,
                "width": 600
            })
        length = 800 + 10 * 1200 + 500
        
    X, y = generate_mixed_stream(events, length, seed)
    
    # Load Results to find detections
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        return
        
    df = pd.read_pickle(RESULTS_FILE)
    # Filter by scenario and seed
    row = df[(df["Scenario"] == scenario_name) & (df["Seed"] == seed)]
    if row.empty:
        print("No results found for this scenario/seed.")
        return
    
    res = row.iloc[0]
    cdt_dets = res["CDT_Detections"]
    se_class = res["SE_Classifications"]
    
    # 2. Compute MMD Signal (for SE Viz)
    mmd_sig = compute_mmd_sequence(X, WINDOW_SIZE, step=10)
    mmd_x_axis = np.arange(0, len(mmd_sig)) * 10  + WINDOW_SIZE # Align roughly with center of window
    
    # --- Figure 1: CDT_MSW (Stream Data + Detections) ---
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    
    # Plot feature mean for clarity
    feat_mean = np.mean(X, axis=1)
    ax1.plot(feat_mean, color='gray', alpha=0.5, label='Stream Mean')
    ax1.set_title(f"CDT_MSW: Stream Data & Detections ({scenario_name})")
    ax1.set_ylabel("Feature Value (Mean)")
    ax1.set_xlabel("Sample Index")
    
    # Plot GT Events
    for evt in events:
        pos = evt['pos']
        width = evt.get('width', 0)
        label = evt['type']
        ax1.axvline(pos, color='green', linestyle='-', linewidth=2, label='Ground Truth' if pos==800 else "")
        if width > 0:
            ax1.axvspan(pos, pos+width, color='green', alpha=0.1)
        ax1.text(pos, np.max(feat_mean)+0.5, label, color='green', fontweight='bold')
            
    # Plot CDT Detections
    for det in cdt_dets:
        pos = det['pos']
        dtype = det['type']
        ax1.axvline(pos, color='red', linestyle='--', linewidth=2, label='CDT Detect' if len(cdt_dets)>0 and pos==cdt_dets[0]['pos'] else "")
        ax1.text(pos, np.min(feat_mean)-0.5, dtype, color='red', rotation=90, verticalalignment='top')
        
    ax1.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/vis_{scenario_name.lower()}_CDT.png")
    print(f"Saved CDT plot to {OUTPUT_DIR}/vis_{scenario_name.lower()}_CDT.png")
    plt.close(fig1)

    # --- Figure 2: SE_CDT (MMD Signal + Classifications) ---
    # Create 2 subplots sharing x-axis
    fig2, (ax2_top, ax2_bottom) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Top Panel: Context (Stream Data) ---
    feat_mean = np.mean(X, axis=1)
    ax2_top.plot(feat_mean, color='gray', alpha=0.5, label='Stream Mean')
    ax2_top.set_title(f"SE_CDT: Stream Context & GT ({scenario_name})")
    ax2_top.set_ylabel("Feature Value")
    
    # Plot GT Events on Top Panel
    for evt in events:
        pos = evt['pos']
        width = evt.get('width', 0)
        label = evt['type']
        ax2_top.axvline(pos, color='green', linestyle='-', linewidth=2, label='Ground Truth' if pos==800 else "")
        if width > 0:
            ax2_top.axvspan(pos, pos+width, color='green', alpha=0.1)
        ax2_top.text(pos, np.max(feat_mean)+0.5, label, color='green', fontweight='bold')
    ax2_top.legend(loc='upper right')

    # --- Bottom Panel: MMD Signal ---
    ax2_bottom.plot(mmd_x_axis, mmd_sig, color='blue', label='ADW-MMD Signal')
    ax2_bottom.set_title(f"SE_CDT: MMD Signal & Classification")
    ax2_bottom.set_ylabel("MMD Value")
    ax2_bottom.set_xlabel("Sample Index")
    
    # Plot GT Regions on Bottom Panel (Faint)
    for evt in events:
        pos = evt['pos']
        width = evt.get('width', 0)
        ax2_bottom.axvline(pos, color='green', linestyle='--', alpha=0.3)
        if width > 0:
             ax2_bottom.axvspan(pos, pos+width, color='green', alpha=0.05)
         
    # Annotate SE Classifications
    for i, seq in enumerate(se_class):
        if i < len(events):
            evt_pos = events[i]['pos']
            pred = seq['pred']
            gt = seq['gt_type']
            color = 'green' if pred == gt else 'red'
            
            sig_idx = min(len(mmd_sig)-1, evt_pos // 10)
            sig_val = mmd_sig[sig_idx]
            
            ax2_bottom.annotate(f"Pred: {pred}\n(GT: {gt})", 
                         xy=(evt_pos, sig_val), 
                         xytext=(evt_pos, sig_val + 0.15), 
                         arrowprops=dict(facecolor=color, shrink=0.05),
                         color=color, fontweight='bold', ha='center')
            
    ax2_bottom.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/vis_{scenario_name.lower()}_SE.png")
    print(f"Saved SE plot to {OUTPUT_DIR}/vis_{scenario_name.lower()}_SE.png")
    plt.close(fig2)

if __name__ == "__main__":
    plot_scenario("Mixed_A", seed=0) # Sudden -> Gradual -> Recurrent
    plot_scenario("Mixed_B", seed=0) # Blip -> Incremental -> Sudden
    plot_scenario("Repeated_Gradual", seed=0)
    plot_scenario("Repeated_Incremental", seed=0)
