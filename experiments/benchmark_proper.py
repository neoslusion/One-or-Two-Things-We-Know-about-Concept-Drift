import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.pool import Pool
import os
import time
import sys
import logging
import argparse
from datetime import datetime

# Ensure imports work
sys.path.append(os.path.abspath("experiments"))
from backup.cdt_msw import CDT_MSW
from backup.se_cdt import SE_CDT
from backup.mmd_variants import mmd_adw

# === CONFIGURATION ===
WINDOW_SIZE = 50
TAU = 0.85
DELTA = 0.005
N_ADJOINT = 4
K_TRACKING = 10
OUTPUT_DIR = "experiments/drift_detection_benchmark/publication_figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "benchmark_proper_detailed.pkl")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SE_CDT_Benchmark")

# === DATA GENERATION ===
def generate_mixed_stream(events, length=None, seed=42):
    """
    Generate a stream with multiple drift events.
    """
    rng = np.random.RandomState(seed) # Use controlled local RNG
    
    if length is None:
        last_event = max([e['pos'] + e.get('width', 0) for e in events])
        length = last_event + 1000
        
    X = rng.randn(length, 5)
    # Base concept: sum(feat 0,1) > 0
    y = (np.sum(X[:, :2], axis=1) > 0).astype(int)
    
    events = sorted(events, key=lambda x: x['pos'])
    
    for evt in events:
        dtype = evt['type']
        pos = evt['pos']
        width = evt.get('width', 200)
        end_pos = min(pos + width, length)
        
        if dtype == "Sudden":
            X[pos:] += 2.0
            y[pos:] = (np.sum(X[pos:, 2:], axis=1) > 1).astype(int)
            
        elif dtype == "Gradual":
            for i in range(width):
                if pos + i >= length: break
                alpha = i / width
                if rng.random() < alpha:
                    X[pos + i] += 2.0 
                    y[pos + i] = (np.sum(X[pos + i, 2:], axis=0) > 1).astype(int)
                
            if end_pos < length:
                X[end_pos:] += 2.0
                y[end_pos:] = (np.sum(X[end_pos:, 2:], axis=1) > 1).astype(int)
                
        elif dtype == "Incremental":
            step = 2.0 / (width // 10) if width > 0 else 0
            for i in range(width):
                if pos + i >= length: break
                alpha = min(1.0, i / width)
                X[pos + i] += step * (i // (width // 10))
                y[pos + i] = (np.sum(X[pos + i, 2:], axis=0) > 1).astype(int) if alpha > 0.5 else (np.sum(X[pos + i, 2:], axis=0) > 0).astype(int)
                
            if end_pos < length:
                X[end_pos:] += 2.0
                y[end_pos:] = (np.sum(X[end_pos:, 2:], axis=1) > 1).astype(int)

        elif dtype == "Recurrent":
            period = width // 2
            for i in range(pos, length, period): 
                sub_end = min(i + period, length)
                concept = ((i - pos) // period) % 2
                if concept == 1:
                    X[i:sub_end] += 2.0
                    y[i:sub_end] = (np.sum(X[i:sub_end, 2:], axis=1) > 1).astype(int)
                    
        elif dtype == "Blip":
            blip_width = width // 5
            end_blip = min(pos + blip_width, length)
            X[pos:end_blip] += 2.0
            y[pos:end_blip] = (np.sum(X[pos:end_blip, 2:], axis=1) > 1).astype(int)
            
    return X, y

# === REAL SIGNAL EXTRACTION ===
def compute_mmd_sequence(X, window_size=WINDOW_SIZE, step=10):
    n = len(X)
    mmd_curve = []
    
    for i in range(0, n - 2*window_size, step):
        window = X[i : i + 2*window_size]
        if len(window) < 2*window_size: break
        val, _ = mmd_adw(window, s=window_size)
        mmd_curve.append(val)
        
    return np.array(mmd_curve)

# === CONTINUOUS DETECTOR WRAPPER ===
def run_continuous_cdt(X, y):
    detections = []
    current_ptr = 0
    n = len(X)
    
    while current_ptr < n - 2*WINDOW_SIZE:
        stream_X = X[current_ptr:]
        stream_y = y[current_ptr:]
        
        if len(stream_X) < 2*WINDOW_SIZE: break
        
        detector = CDT_MSW(WINDOW_SIZE, TAU, DELTA, N_ADJOINT, K_TRACKING)
        res = detector.detect(stream_X, stream_y)
        
        if res['drift_detected']:
            abs_pos = current_ptr + res['drift_position'] * WINDOW_SIZE
            detections.append({
                "pos": abs_pos,
                "type": res['drift_subcategory'],
                "var": res.get('variance_val', 0)
            })
            drift_len_samples = res['drift_length'] * WINDOW_SIZE
            current_ptr = max(abs_pos + drift_len_samples, abs_pos + WINDOW_SIZE)
        else:
            break
            
    return detections

def run_mixed_experiment(params):
    scenario = params["scenario"]
    seed = params["seed"]
    
    events = []
    
    if scenario == "Mixed_A":
        # Cycle: Sudden -> Gradual -> Recurrent
        pattern = [
            {"type": "Sudden", "width": 0},
            {"type": "Gradual", "width": 400},
            {"type": "Recurrent", "width": 400}
        ]
        for i in range(10):
            evt = pattern[i % 3]
            events.append({"type": evt["type"], "pos": 800 + i * 1000, "width": evt["width"]})

    elif scenario == "Mixed_B":
         # Cycle: Blip -> Incremental -> Sudden
        pattern = [
            {"type": "Blip", "width": 100},
            {"type": "Incremental", "width": 500},
            {"type": "Sudden", "width": 0}
        ]
        for i in range(10):
            evt = pattern[i % 3]
            events.append({"type": evt["type"], "pos": 800 + i * 1200, "width": evt["width"]})

    elif scenario == "Repeated_Sudden":
        events = [{"type": "Sudden", "pos": 800 + i*800, "width": 0} for i in range(10)]

    elif scenario == "Repeated_Gradual":
        events = [{"type": "Gradual", "pos": 800 + i*1000, "width": 1000} for i in range(10)]

    elif scenario == "Repeated_Recurrent":
        events = [{"type": "Recurrent", "pos": 800 + i*1000, "width": 400} for i in range(10)]

    elif scenario == "Repeated_Incremental":
        events = [{"type": "Incremental", "pos": 800 + i*1200, "width": 600} for i in range(10)]
    
    else: # Control
        events = [{"type": "Sudden", "pos": 1000, "width": 0}]
        
    # Generate Stream
    length = events[-1]['pos'] + 2000
    X, y = generate_mixed_stream(events, length, seed)
    
    # 1. CDT Continuous
    t0 = time.time()
    cdt_detections = run_continuous_cdt(X, y)
    dt_cdt = time.time() - t0
    

    # 2. SE Classification
    t0 = time.time()
    mmd_sig = compute_mmd_sequence(X, WINDOW_SIZE, step=10)
    se = SE_CDT(WINDOW_SIZE)
    
    se_classifications = []
    for evt in events:
        target_idx = evt['pos'] // 10
        # Check if we have enough signal
        if target_idx + 50 < len(mmd_sig):
            slice_start = max(0, target_idx - 50)
            slice_end = target_idx + 50
            sig_slice = mmd_sig[slice_start:slice_end]
            
            res_se = se.classify(sig_slice)
            se_classifications.append({
                "gt_type": evt['type'],
                "pred": res_se.subcategory,
                "features": res_se.features
            })
        
    return {
        "Scenario": scenario,
        "Seed": seed,
        "CDT_Detections": cdt_detections,
        "SE_Classifications": se_classifications,
        "Events": events
    }

def print_summary(results):
    total_se = 0
    correct_se = 0
    
    print("\n" + "="*60)
    print("SE-CDT CLASSIFICATION SUMMARY")
    print("="*60)
    
    confusion = {}
    
    for res in results:
        for item in res['SE_Classifications']:
            gt = item['gt_type']
            pred = item['pred']
            
            if gt not in confusion: confusion[gt] = {}
            if pred not in confusion[gt]: confusion[gt][pred] = 0
            
            confusion[gt][pred] += 1
            total_se += 1
            if gt == pred:
                correct_se += 1
            elif gt == "Blip" and pred == "Sudden":
                pass
                
    print(f"Total Events Classified: {total_se}")
    print(f"Overall Accuracy: {correct_se/total_se:.2%}" if total_se > 0 else "N/A")
    
    print("-" * 60)
    print(f"{'True Base':<15} | {'Predicted Distribution'}")
    print("-" * 60)
    
    for gt, preds in confusion.items():
        total_gt = sum(preds.values())
        pred_str = ", ".join([f"{k}: {v} ({v/total_gt:.1%})" for k, v in preds.items()])
        print(f"{gt:<15} | {pred_str}")
    print("-" * 60)



def run_benchmark_proper():
    tasks = []
    scenarios = ["Mixed_A", "Mixed_B", "Repeated_Gradual", "Repeated_Incremental", "Repeated_Sudden"]
    n_seeds = 10 # Increased to 10 for better coverage
    
    for sc in scenarios:
        for seed in range(n_seeds): 
            tasks.append({"scenario": sc, "seed": seed})
            
    logger.info(f"Starting Comprehensive Benchmark with {len(tasks)} tasks...")
    
    cpu_count = max(1, multiprocessing.cpu_count() - 2)
    pool = Pool(cpu_count)
    
    results = []
    for i, res in enumerate(pool.imap(run_mixed_experiment, tasks)):
        results.append(res)
        if (i+1) % 5 == 0:
            logger.info(f"Processed {i+1}/{len(tasks)} tasks...")
    
    pool.close()
    pool.join()
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_pickle(OUTPUT_FILE)
    logger.info(f"Benchmark Complete. Results saved to {OUTPUT_FILE}")
    
    # Print Summary
    print_summary(results)

if __name__ == "__main__":
    run_benchmark_proper()
