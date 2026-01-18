import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.pool import Pool
import os
import time
import sys

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
OUTPUT_FILE = "experiments/benchmark_proper_results.pkl"
DRIFT_TYPES = ["Sudden", "Gradual", "Recurrent", "Blip", "Incremental"]
SEEDS = range(10)

# === DATA GENERATION ===
def generate_stream(drift_type="Sudden", length=2000, position=1000, width=200, seed=42):
    # Legacy wrapper for single type
    return generate_mixed_stream([{"type": drift_type, "pos": position, "width": width}], length, seed)

def generate_mixed_stream(events, length=None, seed=42):
    """
    Generate a stream with multiple drift events.
    events: list of dicts {'type': 'Sudden', 'pos': 1000, 'width': 200}
    """
    np.random.seed(seed)
    
    # Estimate required length if not provided
    if length is None:
        last_event = max([e['pos'] + e.get('width', 0) for e in events])
        length = last_event + 1000
        
    X = np.random.randn(length, 5)
    # Base concept: sum(feat 0,1) > 0
    y = (np.sum(X[:, :2], axis=1) > 0).astype(int)
    
    current_concept_idx = 0 
    # concept 0: > 0
    # concept 1: > 1 (shift +2)
    # concept 2: > 2 (shift +4) ... or just toggle 0/1
    
    # Sort events by position
    events = sorted(events, key=lambda x: x['pos'])
    
    for evt in events:
        dtype = evt['type']
        pos = evt['pos']
        width = evt.get('width', 200)
        end_pos = min(pos + width, length)
        
        # Apply drift logic modification on top of base
        # Simple strategy: Concept B is always "X+2, sum>1" relative to Concept A
        # For mixed, we toggle between state A and B, or additive shift?
        # Let's use Additive Shift (Global Drift) to avoid confusion.
        
        if dtype == "Sudden":
            X[pos:] += 2.0
            # Update labels for new concept (threshold increases)
            y[pos:] = (np.sum(X[pos:, 2:], axis=1) > 1).astype(int)
            
        elif dtype == "Gradual":
            for i in range(width):
                if pos + i >= length: break
                alpha = i / width
                if np.random.random() < alpha:
                    X[pos + i] += 2.0 
                    y[pos + i] = (np.sum(X[pos + i, 2:], axis=0) > 1).astype(int)
                else:
                    # Keep old Concept A logic (0 mean)
                    pass
            # Finalize
            if end_pos < length:
                X[end_pos:] += 2.0
                y[end_pos:] = (np.sum(X[end_pos:, 2:], axis=1) > 1).astype(int)
                
        elif dtype == "Incremental":
            step = 2.0 / (width // 10)
            for i in range(width):
                if pos + i >= length: break
                alpha = min(1.0, i / width)
                # Incremental shift of mean
                X[pos + i] += step * (i // (width // 10))
                # Label threshold shifts? Or just feature shift implies label change?
                # Simple: Feature Shift, Fixed Function? No, Concept Drift = P(y|X) changes.
                # Here we shift X, so P(y|X) changes if decision boundary stays same.
                # Let's keep decision boundary formula consistent but features move.
                y[pos + i] = (np.sum(X[pos + i, 2:], axis=0) > 1).astype(int) if alpha > 0.5 else (np.sum(X[pos + i, 2:], axis=0) > 0).astype(int)
                
            if end_pos < length:
                X[end_pos:] += 2.0
                y[end_pos:] = (np.sum(X[end_pos:, 2:], axis=1) > 1).astype(int)

        elif dtype == "Recurrent":
            period = width // 2
            for i in range(pos, length, period): # Reoccurs until end? Or just for width?
                # Usually Recurrent means flipping back and forth
                # Let's say it lasts until next event or end
                sub_end = min(i + period, length)
                concept = ((i - pos) // period) % 2
                if concept == 1:
                    X[i:sub_end] += 2.0
                    y[i:sub_end] = (np.sum(X[i:sub_end, 2:], axis=1) > 1).astype(int)
                else:
                    # Revert to base (0)
                    # X is already base 0 from init
                    # Need to un-shift if we accumulated? 
                    # Our logic is additive X[pos:] so previous drifts stack.
                    # This implies "Recurrent" only works well as isolated event in this simpler gen.
                    pass
                    
        elif dtype == "Blip":
            blip_width = width // 5
            end_blip = min(pos + blip_width, length)
            X[pos:end_blip] += 2.0
            y[pos:end_blip] = (np.sum(X[pos:end_blip, 2:], axis=1) > 1).astype(int)
            # After blip, it reverts (since we only added to slice, remaining X is 0)
            
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
    """
    Run CDT_MSW in a loop to detect multiple drifts.
    Resets reference window after detection/adaptation.
    """
    detections = []
    current_ptr = 0
    n = len(X)
    
    # We simulate an online process
    # But CDT_MSW design is: detect -> growth -> tracking
    # Ideally: Detect drift -> Collect new ref -> Continue
    
    while current_ptr < n - 2*WINDOW_SIZE:
        # Slice stream from current pointer
        stream_X = X[current_ptr:]
        stream_y = y[current_ptr:]
        
        if len(stream_X) < 2*WINDOW_SIZE: break
        
        detector = CDT_MSW(WINDOW_SIZE, TAU, DELTA, N_ADJOINT, K_TRACKING)
        res = detector.detect(stream_X, stream_y)
        
        if res['drift_detected']:
            # Absolute position
            abs_pos = current_ptr + res['drift_position'] * WINDOW_SIZE
            
            det_info = {
                "pos": abs_pos,
                "type": res['drift_subcategory'],
                "var": res.get('variance_val', 0)
            }
            detections.append(det_info)
            
            # ADAPTATION SIMULATION:
            # Skip past the drift + length to collect new reference
            # Assume we need 1 window length to stabilize?
            # Paper says: Full Reset -> Collect New.
            drift_len_samples = res['drift_length'] * WINDOW_SIZE
            current_ptr = abs_pos + drift_len_samples
            
            # Ensure we advance at least window size to avoid stuck
            current_ptr = max(current_ptr, abs_pos + WINDOW_SIZE)
        else:
            break # No more drifts detected
            
    return detections

def run_continuous_se(X):
    """
    Run SE_CDT on full signal.
    SE_CDT usually takes a signal curve and classifies it.
    For continuous, we need to Peak Pick the signal and classify each peak.
    """
    mmd_signal = compute_mmd_sequence(X, WINDOW_SIZE, step=10)
    
    # Simple Peak Detection on MMD Signal
    # Threshold = mean + 3*std? Or dynamic?
    threshold = np.mean(mmd_signal) + 2.0 * np.std(mmd_signal)
    
    # Find peaks (simple)
    peaks = []
    # This is a naive peak detector for benchmark
    # In real app, SE_CDT has 'monitor' mode? 
    # backup/se_cdt.py seems to be classifier-only.
    # We will simulate: If MMD > Threshold, extract window, classify.
    
    se_results = []
    # Skip logic ... 
    
    # Let's just return the signal stats for the KNOWN drift regions 
    # to be fair comparison with Ground Truth events?
    # Or implement a peak finder.
    
    return mmd_signal # Return raw for analysis or simple peak find?

# === WORKER FUNCTION ===
# === WORKER FUNCTION ===
def run_single_experiment(params):
    dtype = params["drift_type"]
    seed = params["seed"]
    
    # 1. Generate Data
    X, y = generate_stream(dtype, seed=seed)
    
    # 2. CDT_MSW (Supervised)
    start_cdt = time.time()
    detector = CDT_MSW(WINDOW_SIZE, TAU, DELTA, N_ADJOINT, K_TRACKING)
    res_cdt = detector.detect(X, y)
    time_cdt = time.time() - start_cdt
    
    # 3. SE_CDT (Unsupervised Hybrid)
    start_se = time.time()
    
    # Extract REAL signal from data
    mmd_signal = compute_mmd_sequence(X, WINDOW_SIZE, step=10)
    
    # Run SE Classification on real signal
    se = SE_CDT(WINDOW_SIZE)
    # Note: SE_CDT expects a signal length approx commensurate with window analysis
    res_se = se.classify(mmd_signal)
    
    time_se = time.time() - start_se
    
    return {
        "Drift Type": dtype,
        "Seed": seed,
        "CDT_Prediction": res_cdt["drift_subcategory"] if res_cdt["drift_detected"] else "None",
        "CDT_Position": res_cdt["drift_position"] if res_cdt["drift_detected"] else -1,
        "CDT_Variance": res_cdt.get("variance_val", 0.0),
        "CDT_Time": time_cdt,
        "SE_Prediction": res_se.subcategory,
        "SE_Features": res_se.features,
        "SE_Time": time_se
    }

def run_mixed_experiment(params):
    scenario = params["scenario"] # "Mixed_A", "Mixed_B"
    seed = params["seed"]
    
    events = []
    length = 4000
    
    if scenario == "Mixed_A":
        # Cycle: Sudden -> Gradual -> Recurrent -> ...
        # 10 Events total
        pattern = [
            {"type": "Sudden", "width": 0},
            {"type": "Gradual", "width": 400},
            {"type": "Recurrent", "width": 400}
        ]
        events = []
        for i in range(10):
            evt_template = pattern[i % 3]
            events.append({
                "type": evt_template["type"],
                "pos": 800 + i * 1000, # 1000 spacing
                "width": evt_template["width"]
            })
        length = 800 + 10 * 1000 + 1000 # ~12000

    elif scenario == "Mixed_B":
         # Cycle: Blip -> Incremental -> Sudden -> ...
        pattern = [
            {"type": "Blip", "width": 100},
            {"type": "Incremental", "width": 500},
            {"type": "Sudden", "width": 0}
        ]
        events = []
        for i in range(10):
            evt_template = pattern[i % 3]
            events.append({
                "type": evt_template["type"],
                "pos": 800 + i * 1200, # Wider spacing for Incremental
                "width": evt_template["width"]
            })
        length = 800 + 10 * 1200 + 1000 # ~14000

    elif scenario == "Repeated_Sudden":
        # Consistency Check: 10 Sudden Drifts
        events = [{"type": "Sudden", "pos": 800 + i*800, "width": 0} for i in range(10)]
        length = 800 + 10*800 + 500

    elif scenario == "Repeated_Gradual":
        # Consistency Check: 10 Gradual Drifts 
        events = [{"type": "Gradual", "pos": 800 + i*1000, "width": 300} for i in range(10)]
        length = 800 + 10*1000 + 500

    elif scenario == "Repeated_Recurrent":
        # Consistency Check: 10 Recurrent Drifts
        events = [{"type": "Recurrent", "pos": 800 + i*1000, "width": 400} for i in range(10)]
        length = 800 + 10*1000 + 500

    elif scenario == "Repeated_Incremental":
        # Consistency Check: 10 Incremental Drifts
        # Incremental needs more space usually? 
        events = [{"type": "Incremental", "pos": 800 + i*1200, "width": 600} for i in range(10)]
        length = 800 + 10*1200 + 500
    else:
        # Control: Single Sudden
        events = [{"type": "Sudden", "pos": 1000, "width": 0}]
        length = 2000
        
    X, y = generate_mixed_stream(events, length, seed)
    
    # 1. CDT Continuous
    t0 = time.time()
    cdt_detections = run_continuous_cdt(X, y)
    dt_cdt = time.time() - t0
    
    # 2. SE Continuous
    t0 = time.time()
    mmd_sig = compute_mmd_sequence(X, WINDOW_SIZE, step=10)
    se = SE_CDT(WINDOW_SIZE)
    # Classify the WHOLE signal? No, SE_CDT is window-based classifier.
    # We cheat slightly for benchmark: Construct "Ground Truth Windows"
    # and ask SE_CDT to classify the signal at those positions.
    # This evaluates SE_CDT's classification accuracy given detection.
    
    se_classifications = []
    for evt in events:
        # Find signal peak near event pos
        # MMD Step=10. Pos=800 -> Index=80
        target_idx = evt['pos'] // 10
        # Extract slice around target
        slice_start = max(0, target_idx - 50)
        slice_end = min(len(mmd_sig), target_idx + 50)
        sig_slice = mmd_sig[slice_start:slice_end]
        
        if len(sig_slice) > 10:
            res_se = se.classify(sig_slice)
            se_classifications.append({
                "gt_type": evt['type'],
                "pred": res_se.subcategory,
                "features": res_se.features
            })
        
    dt_se = time.time() - t0
    
    return {
        "Scenario": scenario,
        "Seed": seed,
        "CDT_Detections": cdt_detections,
        "SE_Classifications": se_classifications,
        "Events": events
    }

# === MAIN RUNNER ===
def run_benchmark_proper():
    tasks = []
    
    # 1. Standard Benchmark (Single Event)
    # for dtype in DRIFT_TYPES:
    #     for seed in SEEDS:
    #         tasks.append({"drift_type": dtype, "seed": seed})
            
    # 2. Comprehensive Multi-Drift (Mixed & Consistency)
    scenarios = ["Mixed_A", "Mixed_B", "Repeated_Gradual", "Repeated_Incremental", "Repeated_Sudden"]
    for sc in scenarios:
        for seed in range(5): 
            tasks.append({"scenario": sc, "seed": seed})
            
    print(f"Starting Comprehensive Benchmark with {len(tasks)} tasks...")
    
    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    pool = Pool(cpu_count)
    
    results = []
    results = []
    for res in pool.imap(run_mixed_experiment, tasks):
        results.append(res)
        if len(results) % 5 == 0:
            print(f"Processed {len(results)}/{len(tasks)} tasks...")
    
    pool.close()
    pool.join()
    
    # Save Standard Results
    df = pd.DataFrame(results)
    df.to_pickle(OUTPUT_FILE)
    print(f"Standard Benchmark Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark_proper()
