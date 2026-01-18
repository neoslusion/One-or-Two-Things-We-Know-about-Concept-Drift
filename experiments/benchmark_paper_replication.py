import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os
from sklearn.metrics import f1_score
import sys
sys.path.append("../")
# Import CDT_MSW and SE_CDT (assume in same dir or backup; replace with your paths)
from backup.cdt_msw import CDT_MSW
from backup.se_cdt import SE_CDT

# Config from paper: window=50, tau=0.85, delta=0.005, n_adjoint=4, k=10
WINDOW_SIZE = 50
TAU = 0.85
DELTA = 0.005
N_ADJOINT = 4
K_TRACKING = 10

DRIFT_TYPES = ["Sudden", "Gradual", "Incremental", "Recurrent", "Blip"]
LENGTH = 3000  # Longer for multi-events
POSITIONS = [1000, 2000]  # Multiple drift positions (2 events; add more if needed)
WIDTH = 200  # For gradual/incremental
N_SEEDS = 10  # Avg over seeds for robust metrics
TOLERANCE = 50  # Position detect within ±50 samples is TP (paper-like tolerance)
OUTPUT_DIR = "benchmark_multi_drifts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Synthetic generation with multiple drifts (extend from your script)
def generate_stream(drift_type="Sudden", length=LENGTH, positions=POSITIONS, width=WIDTH):
    np.random.seed(42)  # Fixed for reproducible, but vary in loop
    X = np.random.randn(length, 5)  # 5 features multi-class
    y = (np.sum(X[:, :2], axis=1) > 0).astype(int)  # Base concept
    
    for pos in positions:
        end_pos = min(pos + width, length)
        if drift_type == "Sudden":
            X[pos:end_pos] += 2.0
            y[pos:end_pos] = (np.sum(X[pos:end_pos, 2:], axis=1) > 1).astype(int)
        
        elif drift_type == "Gradual":
            for i in range(width):
                alpha = i / width
                if np.random.random() < alpha:
                    X[pos + i] += 2.0 * alpha
                    y[pos + i] = (np.sum(X[pos + i, 2:], axis=0) > 1).astype(int)
                else:
                    X[pos + i] += np.random.normal(0, 0.5, 5)
        
        elif drift_type == "Incremental":
            step = 2.0 / (width // 10)
            for i in range(width):
                alpha = min(1.0, i / width)
                X[pos + i] += step * (i // (width // 10))
                y[pos + i] = (np.sum(X[pos + i, 2:], axis=0) > alpha).astype(int)
        
        elif drift_type == "Recurrent":
            period = width // 2
            for i in range(pos, end_pos, period):
                sub_end = min(i + period, end_pos)
                concept = ((i - pos) // period) % 2
                if concept == 1:
                    X[i:sub_end] += 2.0
                    y[i:sub_end] = (np.sum(X[i:sub_end, 2:], axis=1) > 1).astype(int)
        
        elif drift_type == "Blip":
            blip_width = width // 5
            X[pos:pos + blip_width] += 2.0
            y[pos:pos + blip_width] = (np.sum(X[pos:pos + blip_width, 2:], axis=1) > 1).astype(int)
    
    return X, y

# Real dataset with injected multi-drifts (paper uses Covertype/Electricity)
def get_real_with_drifts(name, positions=POSITIONS, shift=2.0):
    if name == "covertype":
        data = fetch_openml(name="covertype", version=1, as_frame=False)
    elif name == "electricity":
        # Assume CSV; replace with your URL/path
        df = pd.read_csv('https://raw.githubusercontent.com/scikit-multiflow/scikit-multiflow/master/src/skmultiflow/datasets/elec2.csv')
        data = df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(int)
    else:
        raise ValueError("Unknown")
    X, y = data[0][:LENGTH], data[1][:LENGTH]  # Truncate
    for pos in positions:
        X[pos:] += shift * np.random.randn(*X[pos:].shape)  # Inject shift
    return X, y

# Run for one type/seed
def run_for_type_seed(drift_type, seed, use_real=False):
    np.random.seed(seed)
    if use_real:
        X, y = get_real_with_drifts("covertype")
    else:
        X, y = generate_stream(drift_type)
    
    detector = CDT_MSW(WINDOW_SIZE, TAU, DELTA, N_ADJOINT, K_TRACKING)
    res_cdt = detector.detect(X, y)
    
    # Simulate MMD for SE_CDT per drift event (placeholder; integrate real ShapeDD if needed)
    mmd_sigs = [np.random.randn(50) + (1 if 'Sudden' in drift_type else 0.5) for _ in POSITIONS]
    se = SE_CDT(WINDOW_SIZE)
    res_ses = [se.classify(sig) for sig in mmd_sigs]
    
    return {
        "Drift Type": drift_type,
        "Seed": seed,
        "CDT_MSW": res_cdt,
        "SE_CDTs": res_ses,  # List for multi-events
        "GT_Positions": [p // WINDOW_SIZE for p in POSITIONS]  # Window positions
    }

# Metrics calc (detection rate, TCD/PCD acc, subcategory F1)
def calculate_metrics(results):
    metrics = []
    for res in results:
        gt_type = res["Drift Type"]
        gt_category = "TCD" if gt_type in ["Sudden", "Blip"] else "PCD"  # Map subcategory to category
        gt_sub = gt_type
        
        # CDT_MSW metrics
        detected_poss = [res["CDT_MSW"]["drift_position"]] if res["CDT_MSW"]["drift_detected"] else []  # Assume single detect; extend for multi
        tp = sum(1 for gt in res["GT_Positions"] if any(abs(det - gt) * WINDOW_SIZE <= TOLERANCE for det in detected_poss))
        detection_rate = tp / len(res["GT_Positions"])
        category_acc = 1 if res["CDT_MSW"]["drift_category"] == gt_category else 0
        sub_f1 = f1_score([gt_sub] * len(detected_poss), [res["CDT_MSW"]["drift_subcategory"]], average='macro', labels=DRIFT_TYPES) if detected_poss else 0
        
        # SE_CDT metrics (avg over events)
        se_category_acc = np.mean([1 if se.drift_type == gt_category else 0 for se in res["SE_CDTs"]])
        se_sub_f1 = f1_score([gt_sub] * len(res["SE_CDTs"]), [se.subcategory for se in res["SE_CDTs"]], average='macro', labels=DRIFT_TYPES)
        
        metrics.append({
            "Type": gt_type,
            "Seed": res["Seed"],
            "Detection Rate (CDT_MSW)": detection_rate,
            "TCD/PCD Acc (CDT_MSW)": category_acc,
            "Subcategory F1 (CDT_MSW)": sub_f1,
            "TCD/PCD Acc (SE_CDT)": se_category_acc,
            "Subcategory F1 (SE_CDT)": se_sub_f1
        })
    
    df = pd.DataFrame(metrics).groupby("Type").mean().reset_index()  # Avg over seeds
    latex = df.to_latex(index=False, caption="Benchmark Metrics", label="tab:metrics")
    with open(os.path.join(OUTPUT_DIR, "metrics.tex"), "w") as f:
        f.write(latex)
    df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))

# Full benchmark
def run_benchmark():
    results = []
    for dtype in DRIFT_TYPES:
        for seed in range(N_SEEDS):
            res = run_for_type_seed(dtype, seed)
            results.append(res)
    
    calculate_metrics(results)
    
    # Plots example for first seed per type
    for dtype in DRIFT_TYPES:
        res = next(r for r in results if r["Drift Type"] == dtype and r["Seed"] == 0)
        plt.figure()
        plt.plot(res["CDT_MSW"]["tfr_curve"] if "tfr_curve" in res["CDT_MSW"] else [])
        plt.title(f"TFR Multi-Drifts {dtype}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"tfr_multi_{dtype}.png"))
    
    print("Benchmark done! Check", OUTPUT_DIR)

if __name__ == "__main__":
    run_benchmark()
