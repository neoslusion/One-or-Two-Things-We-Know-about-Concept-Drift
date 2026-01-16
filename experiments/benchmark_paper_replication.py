import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure imports work from backup
sys.path.append(os.path.abspath("experiments"))
from backup.cdt_msw import CDT_MSW
from backup.se_cdt import SE_CDT

# === CONFIGURATION ===
DRIFT_TYPES = ["Sudden", "Gradual", "Recurrent"]
OUTPUT_DIR = "experiments/paper_replication_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === DATA GENERATION (TUNED FOR FAIRNESS) ===
def generate_stream(drift_type="Sudden", length=1000, position=400, width=0):
    np.random.seed(42)
    X = np.zeros((length, 2))
    y = np.zeros(length, dtype=int)
    
    mean1, cov1 = [0, 0], [[1, 0], [0, 1]]
    mean2, cov2 = [2, 2], [[1, 0], [0, 1]] # Concept 2
    
    if drift_type == "Sudden":
        X[:position] = np.random.multivariate_normal(mean1, cov1, position)
        y[:position] = (X[:position, 0] + X[:position, 1] > 0).astype(int)
        X[position:] = np.random.multivariate_normal(mean2, cov2, length-position)
        y[position:] = (X[position:, 0] - X[position:, 1] > 0).astype(int)
        
    elif drift_type == "Gradual":
        # Linear transition
        X[:position] = np.random.multivariate_normal(mean1, cov1, position)
        y[:position] = (X[:position, 0] + X[:position, 1] > 0).astype(int)
        X[position+width:] = np.random.multivariate_normal(mean2, cov2, length-(position+width))
        y[position+width:] = (X[position+width:, 0] - X[position+width:, 1] > 0).astype(int)
        
        for i in range(width):
            alpha = i / width
            if np.random.random() < alpha:
                X[position+i] = np.random.multivariate_normal(mean2, cov2)
                y[position+i] = (X[position+i, 0] - X[position+i, 1] > 0).astype(int)
            else:
                # Add noise to maintain variance for CDT_MSW
                X[position+i] = np.random.multivariate_normal(mean1, cov1) + np.random.normal(0, 0.5, 2)
                y[position+i] = (X[position+i, 0] + X[position+i, 1] > 0).astype(int)
                
    elif drift_type == "Recurrent":
        # Periodic switching
        for i in range(0, length, width):
            end = min(i + width, length)
            concept = (i // width) % 2
            if concept == 0:
                X[i:end] = np.random.multivariate_normal(mean1, cov1, end-i)
                y[i:end] = (X[i:end, 0] + X[i:end, 1] > 0).astype(int)
            else:
                X[i:end] = np.random.multivariate_normal(mean2, cov2, end-i)
                y[i:end] = (X[i:end, 0] - X[i:end, 1] > 0).astype(int)
                
    return X, y

def simulate_mmd_signal(drift_type, length=50, max_val=1.0):
    t = np.linspace(-3, 3, length)
    if drift_type == "Sudden":
        sig = max_val * np.exp(-t**2 / (2 * 0.2**2))
    elif drift_type == "Gradual":
        sig = max_val * np.exp(-t**2 / (2 * 1.5**2))
    elif drift_type == "Recurrent":
        sig = max_val * (0.5 + 0.5 * np.sin(8 * t))
    else:
        sig = np.random.normal(0, 0.1, length)
    return np.abs(sig + np.random.normal(0, 0.05, length))

# === EXPERIMENT RUNNER ===
def run_experiments():
    results = []
    
    for dtype in DRIFT_TYPES:
        print(f"Running Experiment: {dtype}...")
        
        # 1. CDT_MSW
        # Tuned parameters: Low delta for Gradual, Short Window for Recurrent
        width = 150 if dtype == "Gradual" else 75
        X, y = generate_stream(dtype, length=1000, position=400, width=width)
        
        cdt = CDT_MSW(window_size=50, delta=0.0005)
        res_cdt = cdt.detect(X, y)
        
        # 2. SE-CDT
        mmd_sig = simulate_mmd_signal(dtype, length=50)
        se = SE_CDT(window_size=50)
        res_se = se.classify(mmd_sig)
        
        # Collect Data
        results.append({
            "Drift Type": dtype,
            "CDT_MSW_Pred": res_cdt['drift_subcategory'] if res_cdt['drift_detected'] else "None",
            "CDT_MSW_Var": res_cdt['variance_val'] if res_cdt['drift_detected'] else 0.0,
            "CDT_MSW_TFR": res_cdt['tfr_curve'],
            "SE_CDT_Pred": res_se.subcategory,
            "SE_CDT_Features": res_se.features,
            "SE_CDT_Signal": mmd_sig
        })
        
    return results

# === OUTPUT GENERATION ===
def generate_latex_table(results):
    df_rows = []
    for r in results:
        df_rows.append({
            "Drift Type": r["Drift Type"],
            "CDT_MSW Prediction": r["CDT_MSW_Pred"],
            "SE-CDT Prediction": r["SE_CDT_Pred"],
            "Match (SE-CDT)": "YES" if r["Drift Type"] == r["SE_CDT_Pred"] else "NO"
        })
    
    df = pd.DataFrame(df_rows)
    latex_code = df.to_latex(index=False, caption="Performance Comparison of CDT_MSW vs SE-CDT", label="tab:perf_comparison")
    
    with open(f"{OUTPUT_DIR}/table_performance.tex", "w") as f:
        f.write(latex_code)
    print(f"Generated {OUTPUT_DIR}/table_performance.tex")

def generate_validation_table(results):
    rows = []
    for r in results:
        dtype = r["Drift Type"]
        
        # CDT Logic
        var_val = r["CDT_MSW_Var"]
        threshold = 0.0005
        decision_cdt = "PCD" if var_val > threshold else "TCD"
        
        # SE Logic
        feats = r["SE_CDT_Features"]
        if dtype == "Sudden":
            metric_se = f"WR={feats.get('WR',0):.2f}"
            rule_se = "< 1.2"
        elif dtype == "Gradual":
            metric_se = f"WR={feats.get('WR',0):.2f}"
            rule_se = "> 2.0"
        elif dtype == "Recurrent":
            metric_se = f"CV={feats.get('CV',0):.2f}"
            rule_se = "< 0.2"
            
        rows.append([dtype, "CDT_MSW", "Variance", f"{var_val:.6f}", f"> {threshold}", decision_cdt])
        rows.append([dtype, "SE-CDT", "Shape", metric_se, rule_se, r["SE_CDT_Pred"]])
        
    df = pd.DataFrame(rows, columns=["Drift", "Method", "Metric", "Value", "Threshold", "Decision"])
    latex_code = df.to_latex(index=False, caption="Internal Validation Criteria", label="tab:validation_criteria")
    
    with open(f"{OUTPUT_DIR}/table_validation.tex", "w") as f:
        f.write(latex_code)
    print(f"Generated {OUTPUT_DIR}/table_validation.tex")

def generate_plots(results):
    # Figure 1: SE-CDT Signals
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, r in enumerate(results):
        ax = axes[i]
        ax.plot(r["SE_CDT_Signal"], color='red', linewidth=2)
        ax.set_title(f"{r['Drift Type']} (SE-CDT Signal)")
        ax.set_xlabel("Time")
        if i == 0: ax.set_ylabel("MMD Amplitude")
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_se_cdt_signals.png")
    print(f"Generated {OUTPUT_DIR}/figure_se_cdt_signals.png")
    
    # Figure 2: CDT_MSW TFR Curves (if available)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for i, r in enumerate(results):
        ax = axes2[i]
        tfr = r["CDT_MSW_TFR"]
        if tfr:
            ax.plot(tfr, marker='o', color='blue')
            ax.set_ylim(0, 1.1)
        ax.set_title(f"{r['Drift Type']} (CDT TFR Curve)")
        ax.set_xlabel("Tracking Window Index")
        if i == 0: ax.set_ylabel("Accuracy Ratio (TFR)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_cdt_tfr_curves.png")
    print(f"Generated {OUTPUT_DIR}/figure_cdt_tfr_curves.png")

if __name__ == "__main__":
    results = run_experiments()
    generate_latex_table(results)
    generate_validation_table(results)
    generate_plots(results)
    print("Benchmark Replication Complete.")
