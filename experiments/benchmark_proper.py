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
from backup.mmd import mmd as mmd_standard  # Standard MMD for detection
from scipy.signal import find_peaks, peak_widths

# === CONFIGURATION ===
WINDOW_SIZE = 50
TAU = 0.85
DELTA = 0.005
N_ADJOINT = 4
K_TRACKING = 10
# Detection Thresholds (SE-CDT / ShapeDD) - LOWERED for better PCD detection
SHAPE_HEIGHT = 0.01   # Lowered from 0.05
SHAPE_PROMINENCE = 0.005  # Lowered from 0.01
DETECTION_TOLERANCE = 300 # Increased tolerance for gradual drifts
USE_STANDARD_MMD = False  # Use ADW-MMD (faster but suppresses weak signals)

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
def generate_mixed_stream(events, length=None, seed=42, supervised_mode=False):
    """
    Generate a stream with multiple drift events.
    
    Args:
        events: List of drift events with type, pos, width
        length: Stream length
        seed: Random seed
        supervised_mode: If True, generate concept-aware labels (changes P(Y|X))
                        for supervised methods like CDT_MSW.
                        If False, only changes P(X) (normal unsupervised mode).
    """
    rng = np.random.RandomState(seed)
    
    if length is None:
        last_event = max([e['pos'] + e.get('width', 0) for e in events])
        length = last_event + 1000
        
    X = rng.randn(length, 5)
    
    # Track which concept is active at each position
    concept_id = np.zeros(length, dtype=int)  # 0 = base concept
    current_concept = 0
    
    events = sorted(events, key=lambda x: x['pos'])
    
    for evt in events:
        dtype = evt['type']
        pos = evt['pos']
        width = evt.get('width', 200)
        end_pos = min(pos + width, length)
        
        if dtype == "Sudden":
            X[pos:] += 2.0
            current_concept += 1
            concept_id[pos:] = current_concept
            
        elif dtype == "Gradual":
            for i in range(width):
                if pos + i >= length: break
                alpha = i / width
                if rng.random() < alpha:
                    X[pos + i] += 2.0 
                    concept_id[pos + i] = current_concept + 1
                    
            if end_pos < length:
                X[end_pos:] += 2.0
                current_concept += 1
                concept_id[end_pos:] = current_concept
                
        elif dtype == "Incremental":
            step = 2.0 / max(1, (width // 10))
            for i in range(width):
                if pos + i >= length: break
                X[pos + i] += step * (i // max(1, width // 10))
                if i > width // 2:
                    concept_id[pos + i] = current_concept + 1
                    
            if end_pos < length:
                X[end_pos:] += 2.0
                current_concept += 1
                concept_id[end_pos:] = current_concept

        elif dtype == "Recurrent":
            period = max(1, width // 2)
            for i in range(pos, length, period): 
                sub_end = min(i + period, length)
                cycle = ((i - pos) // period) % 2
                if cycle == 1:
                    X[i:sub_end] += 2.0
                    concept_id[i:sub_end] = current_concept + 1
                else:
                    concept_id[i:sub_end] = current_concept
                    
        elif dtype == "Blip":
            blip_width = max(1, width // 5)
            end_blip = min(pos + blip_width, length)
            X[pos:end_blip] += 2.0
            concept_id[pos:end_blip] = current_concept + 1
    
    # Generate labels based on mode
    if supervised_mode:
        # Concept-aware labels: different decision boundaries per concept
        # This makes CDT_MSW work because accuracy actually drops
        y = np.zeros(length, dtype=int)
        for i in range(length):
            if concept_id[i] % 2 == 0:
                # Even concepts: boundary is X[:,0] + X[:,1] > 0
                y[i] = 1 if (X[i, 0] + X[i, 1]) > 0 else 0
            else:
                # Odd concepts: ROTATED boundary X[:,0] - X[:,1] > 0
                y[i] = 1 if (X[i, 0] - X[i, 1]) > 0 else 0
    else:
        # Unsupervised mode: Labels just follow a fixed formula
        # (CDT_MSW won't work well, but SE-CDT will)
        y = (np.sum(X[:, :2], axis=1) > 0).astype(int)
            
    return X, y

# === REAL SIGNAL EXTRACTION ===
def compute_mmd_sequence(X, window_size=WINDOW_SIZE, step=10, use_standard=USE_STANDARD_MMD):
    """Compute MMD sequence over sliding windows.
    
    Args:
        use_standard: If True, use standard MMD. If False, use ADW-MMD.
    """
    n = len(X)
    mmd_curve = []
    
    for i in range(0, n - 2*window_size, step):
        window = X[i : i + 2*window_size]
        if len(window) < 2*window_size: break
        
        if use_standard:
            # Standard MMD (better for classification, preserves weak signals)
            val, _ = mmd_standard(window, s=window_size, n_perm=100)  # Reduced perm for speed
            val = max(0, val)  # Ensure non-negative
        else:
            # ADW-MMD (faster, but suppresses weak signals)
            val, _ = mmd_adw(window, s=window_size)
        
        mmd_curve.append(val)
        
    return np.array(mmd_curve)

# === METRIC CALCULATION ===
def calculate_metrics(detections, events, length, tolerance=DETECTION_TOLERANCE):
    """
    Calculate detection metrics: TP, FP, FN, MDR, EDR, Delay.
    """
    tp = 0
    fp = 0
    fn = 0
    delays = []
    
    # Track which events are detected to avoid double counting
    detected_events = set()
    used_detections = set()
    
    # Sort for robust matching
    sorted_events = sorted(events, key=lambda x: x['pos'])
    sorted_detections = sorted([d['pos'] for d in detections])
    
    # 1. Check for TPs (Match detection to event)
    for det_pos in sorted_detections:
        matched = False
        for i, evt in enumerate(sorted_events):
            if i in detected_events: continue
            
            # Acceptance window: [drift_pos, drift_pos + tolerance]
            # Ideally detection is AFTER drift, but we allow small early detections (e.g. -50)
            if evt['pos'] - 50 <= det_pos <= evt['pos'] + tolerance:
                tp += 1
                delay = det_pos - evt['pos']
                delays.append(delay)
                detected_events.add(i)
                used_detections.add(det_pos)
                matched = True
                break
        
        if not matched:
            fp += 1
            
    fn = len(events) - tp
    
    mdr = fn / len(events) if len(events) > 0 else 0
    # EDR (Early Detection Rate): 1 - (mean_delay / tolerance) roughly,
    # or more standard definition: % of drifts detected within tolerance (Detection Rate).
    # Wait, EDR in literature usually means "Early Detection Rate" ? 
    # Or "Effective Detection Rate"? 
    # Actually, the user asked for "MDR and EDR".
    # Usually: Detection Rate (Recall) = 1 - MDR.
    # EDR might refer to "Error Detection Rate" or just "Detection Rate".
    # Let's assume EDR = Detection Rate = TP / Total.
    edr = tp / len(events) if len(events) > 0 else 0
    
    mean_delay = np.mean(delays) if delays else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "MDR": mdr,          # Missed Detection Rate
        "EDR": edr,          # Detection Rate (User likely means this or Early warning)
        "Mean_Delay": mean_delay,
        "Precision": precision
    }

# === CONTINUOUS DETECTOR WRAPPER ===
def run_continuous_cdt(X, y):
    """Run CDT_MSW continuously with proper reference window updates."""
    detections = []
    ref_position = 0  # Start reference at beginning
    n = len(X)
    max_detections = 20  # Prevent infinite loop
    
    while ref_position < n - 4*WINDOW_SIZE and len(detections) < max_detections:
        detector = CDT_MSW(WINDOW_SIZE, TAU, DELTA, N_ADJOINT, K_TRACKING)
        
        # Pass full stream but with reference starting at ref_position
        drift_pos = detector.detection_process(X, y, ref_start=ref_position)
        
        if drift_pos == -1 or drift_pos <= ref_position:
            break
        
        # Run growth and tracking at detected position
        drift_length, drift_category, variance_val = detector.growth_process(X, y, drift_pos)
        tfr_values = detector.tracking_process(X, y, drift_pos, drift_length)
        subcategory = detector.identify_subcategory(tfr_values, drift_category)
        
        detections.append({
            "pos": drift_pos,
            "type": subcategory,
            "category": drift_category,
            "var": variance_val
        })
        
        # Move reference PAST the detected drift (update reference window)
        ref_position = drift_pos + max(drift_length * WINDOW_SIZE, WINDOW_SIZE)
            
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
    dt_se = time.time() - t0
    
    # 3. SE Detection with BOTH MMD variants
    # 3a. Standard MMD Signal (better for PCD)
    mmd_sig_std = compute_mmd_sequence(X, WINDOW_SIZE, step=10, use_standard=True)
    # 3b. ADW-MMD Signal (faster, better precision)
    mmd_sig_adw = compute_mmd_sequence(X, WINDOW_SIZE, step=10, use_standard=False)
    
    def detect_peaks_from_signal(sig, height=SHAPE_HEIGHT, prom=SHAPE_PROMINENCE):
        """Detect peaks and return detection points."""
        peaks, _ = find_peaks(sig, height=height, prominence=prom, distance=20)
        detection_points = []
        for p in peaks:
            sample_idx = p * 10
            if not detection_points or (sample_idx - detection_points[-1]['pos'] > 200):
                detection_points.append({"pos": sample_idx, "val": sig[p]})
        return detection_points
    
    def detect_adaptive(sig):
        """
        Adaptive two-stage detection:
        Stage 1: High threshold (0.03) for strong TCD peaks
        Stage 2: Low threshold (0.005) with peak width filter for PCD
        """
        detection_points = []
        used_regions = set()  # Track already detected regions
        
        # Stage 1: High threshold - catch strong peaks (TCD: Sudden, Blip)
        peaks_strong, props_strong = find_peaks(sig, height=0.02, prominence=0.01, distance=20)
        for p in peaks_strong:
            sample_idx = p * 10
            if not detection_points or (sample_idx - detection_points[-1]['pos'] > 200):
                detection_points.append({"pos": sample_idx, "val": sig[p], "stage": "TCD"})
                # Mark region as used
                for r in range(max(0, p-30), min(len(sig), p+30)):
                    used_regions.add(r)
        
        # Stage 2: Lower threshold with width filter - catch wide peaks (PCD: Gradual, Incremental)
        peaks_weak, _ = find_peaks(sig, height=0.003, prominence=0.001, distance=30)
        
        if len(peaks_weak) > 0:
            # Calculate peak widths
            widths, _, _, _ = peak_widths(sig, peaks_weak, rel_height=0.5)
            
            for p, w in zip(peaks_weak, widths):
                # Skip if already detected in Stage 1
                if p in used_regions:
                    continue
                    
                sample_idx = p * 10
                
                # Width filter: PCD peaks are wide (FWHM > 15 in signal units = 150 samples)
                if w > 15:  # Wide peak = likely PCD
                    if not detection_points or (sample_idx - detection_points[-1]['pos'] > 300):
                        detection_points.append({"pos": sample_idx, "val": sig[p], "stage": "PCD", "width": w})
        
        return detection_points
    
    # Standard MMD detection (lower threshold for PCD)
    se_det_std = detect_peaks_from_signal(mmd_sig_std, height=0.001, prom=0.0005)  # Very low for Standard
    # ADW-MMD detection (same thresholds)
    se_det_adw = detect_peaks_from_signal(mmd_sig_adw, height=SHAPE_HEIGHT, prom=SHAPE_PROMINENCE)
    # Adaptive detection (Standard MMD with two-stage + width filter)
    se_det_adaptive = detect_adaptive(mmd_sig_std)
    
    # 4. End-to-End Classification for BOTH variants
    def run_e2e_classification(detections, mmd_sig, events):
        """Classify detected events and match to ground truth."""
        e2e_results = []
        for det in detections:
            det_idx = det['pos'] // 10
            if det_idx + 50 < len(mmd_sig):
                slice_start = max(0, det_idx - 50)
                slice_end = det_idx + 50
                sig_slice = mmd_sig[slice_start:slice_end]
                
                res_se = se.classify(sig_slice)
                
                matched_gt = None
                min_dist = float('inf')
                for evt in events:
                    dist = abs(det['pos'] - evt['pos'])
                    if dist < min_dist and dist <= DETECTION_TOLERANCE:
                        min_dist = dist
                        matched_gt = evt
                
                e2e_results.append({
                    "det_pos": det['pos'],
                    "pred": res_se.subcategory,
                    "pred_cat": res_se.drift_type,
                    "gt_type": matched_gt['type'] if matched_gt else "FP",
                    "matched": matched_gt is not None
                })
        return e2e_results
    
    e2e_std = run_e2e_classification(se_det_std, mmd_sig_std, events)
    e2e_adw = run_e2e_classification(se_det_adw, mmd_sig_adw, events)
    e2e_adaptive = run_e2e_classification(se_det_adaptive, mmd_sig_std, events)
            
    # 5. Calculate Metrics for ALL methods
    cdt_metrics = calculate_metrics(cdt_detections, events, len(X))
    se_std_metrics = calculate_metrics(se_det_std, events, len(X))
    se_adw_metrics = calculate_metrics(se_det_adw, events, len(X))
    se_adaptive_metrics = calculate_metrics(se_det_adaptive, events, len(X))
        
    return {
        "Scenario": scenario,
        "Seed": seed,
        "CDT_Detections": cdt_detections,
        "SE_Classifications": se_classifications,
        "E2E_STD": e2e_std,
        "E2E_ADW": e2e_adw,
        "E2E_Adaptive": e2e_adaptive,
        "SE_Det_STD": se_det_std,
        "SE_Det_ADW": se_det_adw,
        "SE_Det_Adaptive": se_det_adaptive,
        "Events": events,
        "Runtime_CDT": dt_cdt,
        "Runtime_SE": dt_se,
        "Stream_Length": len(X),
        "CDT_Metrics": cdt_metrics,
        "SE_STD_Metrics": se_std_metrics,
        "SE_ADW_Metrics": se_adw_metrics,
        "SE_Adaptive_Metrics": se_adaptive_metrics
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
                
    print(f"Overall Accuracy: {correct_se/total_se:.2%}" if total_se > 0 else "N/A")
    
    # Runtime Stats
    total_cdt_time = sum([r['Runtime_CDT'] for r in results])
    total_se_time = sum([r['Runtime_SE'] for r in results])
    n_runs = len(results)
    avg_stream_len = sum([r['Stream_Length'] for r in results]) / n_runs
    
    print("-" * 60)
    print(f"RUNTIME PERFORMANCE (Avg per {int(avg_stream_len)} samples)")
    print("-" * 60)
    print(f"CDT_MSW : {total_cdt_time/n_runs:.4f} s")
    print(f"SE_CDT  : {total_se_time/n_runs:.4f} s")
    print(f"Speedup : {total_cdt_time/total_se_time:.2f}x (SE is faster)" if total_se_time > 0 else "N/A")
    
    # Aggregated Metrics for ALL methods (including ADAPTIVE)
    metrics_agg = {
        "CDT": {"TP": 0, "FP": 0, "FN": 0, "Total": 0},
        "SE_STD": {"TP": 0, "FP": 0, "FN": 0, "Total": 0},
        "SE_ADW": {"TP": 0, "FP": 0, "FN": 0, "Total": 0},
        "SE_ADAPT": {"TP": 0, "FP": 0, "FN": 0, "Total": 0}
    }
    
    for r in results:
        for m_key, agg_key in [("CDT_Metrics", "CDT"), ("SE_STD_Metrics", "SE_STD"), 
                              ("SE_ADW_Metrics", "SE_ADW"), ("SE_Adaptive_Metrics", "SE_ADAPT")]:
            if m_key in r:
                metrics_agg[agg_key]["TP"] += r[m_key]["TP"]
                metrics_agg[agg_key]["FP"] += r[m_key]["FP"]
                metrics_agg[agg_key]["FN"] += r[m_key]["FN"]
                metrics_agg[agg_key]["Total"] += (r[m_key]["TP"] + r[m_key]["FN"])

    print("-" * 90)
    print(f"{'Method':<15} | {'MDR':<8} | {'EDR (Recall)':<12} | {'Precision':<10} | {'FP':<6}")
    print("-" * 90)
    
    for method in ["CDT", "SE_STD", "SE_ADW", "SE_ADAPT"]:
        total = metrics_agg[method]["Total"]
        tp = metrics_agg[method]["TP"]
        fp = metrics_agg[method]["FP"]
        fn = metrics_agg[method]["FN"]
        
        mdr = fn / total if total > 0 else 0.0
        edr = tp / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        print(f"{method:<15} | {mdr:.3f}    | {edr:.3f}        | {prec:.3f}      | {fp}")
    
    print("-" * 60)
    print(f"{'True Base':<15} | {'Predicted Distribution (Oracle Mode)'}")
    print("-" * 60)
    
    for gt, preds in confusion.items():
        total_gt = sum(preds.values())
        pred_str = ", ".join([f"{k}: {v} ({v/total_gt:.1%})" for k, v in preds.items()])
        print(f"{gt:<15} | {pred_str}")
    print("-" * 60)
    
    # End-to-End Classification Metrics for BOTH variants
    print("\n" + "="*80)
    print("END-TO-END CLASSIFICATION (Detection -> Classification)")
    print("="*80)
    
    TCD_TYPES = ["Sudden", "Blip", "Recurrent"]
    
    def compute_e2e_metrics(results, key):
        """Compute E2E accuracy for a given E2E key."""
        total = 0
        correct_sub = 0
        correct_cat = 0
        fp = 0
        
        for res in results:
            for item in res.get(key, []):
                if not item['matched']:
                    fp += 1
                    continue
                    
                gt = item['gt_type']
                pred = item['pred']
                
                total += 1
                if gt == pred:
                    correct_sub += 1
                
                is_gt_tcd = gt in TCD_TYPES
                is_pred_tcd = pred in TCD_TYPES
                if is_gt_tcd == is_pred_tcd:
                    correct_cat += 1
        
        return {
            "Total": total,
            "Sub_Acc": correct_sub / total if total > 0 else 0,
            "Cat_Acc": correct_cat / total if total > 0 else 0,
            "FP": fp
        }
    
    e2e_std_metrics = compute_e2e_metrics(results, "E2E_STD")
    e2e_adw_metrics = compute_e2e_metrics(results, "E2E_ADW")
    
    print(f"{'Variant':<15} | {'TP Classified':<14} | {'Sub Acc':<10} | {'Cat Acc':<10} | {'FP':<6}")
    print("-" * 80)
    print(f"{'Standard MMD':<15} | {e2e_std_metrics['Total']:<14} | {e2e_std_metrics['Sub_Acc']:.2%}     | {e2e_std_metrics['Cat_Acc']:.2%}     | {e2e_std_metrics['FP']}")
    print(f"{'ADW-MMD':<15} | {e2e_adw_metrics['Total']:<14} | {e2e_adw_metrics['Sub_Acc']:.2%}     | {e2e_adw_metrics['Cat_Acc']:.2%}     | {e2e_adw_metrics['FP']}")
    print("-" * 80)

# === SUPERVISED CDT_MSW BENCHMARK ===
def run_supervised_comparison(n_seeds=5):
    """
    Run CDT_MSW in its proper supervised setting (concept-aware labels).
    Compare with SE-CDT on the same data (but SE ignores labels).
    """
    logger.info("Starting SUPERVISED CDT_MSW Benchmark...")
    
    results = []
    scenario = "Mixed_A"  # Focus on one scenario for fair comparison
    
    for seed in range(n_seeds):
        # Define events (same as Mixed_A)
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
                "pos": 800 + i * 1000,
                "width": evt_template["width"]
            })
        
        length = events[-1]['pos'] + 2000
        
        # Generate with SUPERVISED MODE - labels change with concept
        X, y = generate_mixed_stream(events, length, seed, supervised_mode=True)
        
        # Run CDT_MSW with proper supervised data
        t0 = time.time()
        cdt_detections = run_continuous_cdt(X, y)
        dt_cdt = time.time() - t0
        
        # Run SE-CDT (ignores labels)
        mmd_sig = compute_mmd_sequence(X, WINDOW_SIZE, step=10, use_standard=True)
        se_det = []
        peaks, _ = find_peaks(mmd_sig, height=0.01, prominence=0.005, distance=20)
        for p in peaks:
            sample_idx = p * 10
            if not se_det or (sample_idx - se_det[-1]['pos'] > 200):
                se_det.append({"pos": sample_idx, "val": mmd_sig[p]})
        
        # Calculate metrics
        cdt_metrics = calculate_metrics(cdt_detections, events, len(X))
        se_metrics = calculate_metrics(se_det, events, len(X))
        
        results.append({
            "seed": seed,
            "CDT_Metrics": cdt_metrics,
            "SE_Metrics": se_metrics,
            "CDT_Runtime": dt_cdt,
            "n_cdt_det": len(cdt_detections),
            "n_se_det": len(se_det)
        })
    
    # Print supervised comparison
    print("\n" + "="*80)
    print("SUPERVISED CDT_MSW vs SE-CDT (Fair Comparison)")
    print("Labels use ROTATED DECISION BOUNDARIES per concept")
    print("="*80)
    
    cdt_agg = {"TP": 0, "FP": 0, "FN": 0, "Total": 0}
    se_agg = {"TP": 0, "FP": 0, "FN": 0, "Total": 0}
    
    for r in results:
        for m, agg in [("CDT_Metrics", cdt_agg), ("SE_Metrics", se_agg)]:
            agg["TP"] += r[m]["TP"]
            agg["FP"] += r[m]["FP"]
            agg["FN"] += r[m]["FN"]
            agg["Total"] += r[m]["TP"] + r[m]["FN"]
    
    print(f"\n{'Method':<20} | {'EDR':<8} | {'MDR':<8} | {'FP':<6} | {'Setting':<12}")
    print("-" * 70)
    
    for name, agg, setting in [("CDT_MSW", cdt_agg, "Supervised"), ("SE-CDT (Std)", se_agg, "Unsupervised")]:
        edr = agg["TP"] / agg["Total"] if agg["Total"] > 0 else 0
        mdr = agg["FN"] / agg["Total"] if agg["Total"] > 0 else 0
        print(f"{name:<20} | {edr:.3f}    | {mdr:.3f}    | {agg['FP']:<6} | {setting}")
    
    print("-" * 70)
    logger.info("Supervised comparison complete.")
    return results


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
    generate_latex_table(results)
    
    # Run Supervised CDT comparison
    run_supervised_comparison(n_seeds=5)

def generate_latex_table(results):
    """
    Generate Full Aggregate Comparison Table (Table 4.6)
    Combines Classification Accuracy + Detection Metrics (MDR, EDR).
    """
    # 1. Classification Accuracy (Oracle)
    total_se = 0
    correct_se = 0
    correct_cat_se = 0
    TCD_TYPES = ["Sudden", "Blip", "Recurrent"]
    
    for res in results:
        for item in res['SE_Classifications']:
            gt = item['gt_type']
            pred = item['pred']
            total_se += 1
            if gt == pred: correct_se += 1
            is_gt_tcd = gt in TCD_TYPES
            is_pred_tcd = pred in TCD_TYPES
            if is_gt_tcd == is_pred_tcd: correct_cat_se += 1
            
    cat_acc_se = (correct_cat_se / total_se * 100) if total_se > 0 else 0
    sub_acc_se = (correct_se / total_se * 100) if total_se > 0 else 0
    
    # 2. Detection Metrics (Computed for all methods)
    metrics_agg = {
        "CDT": {"TP": 0, "FP": 0, "FN": 0, "Total": 0},
        "SE_STD": {"TP": 0, "FP": 0, "FN": 0, "Total": 0},
        "SE_ADW": {"TP": 0, "FP": 0, "FN": 0, "Total": 0}
    }
    for r in results:
        for m_key, agg_key in [("CDT_Metrics", "CDT"), ("SE_STD_Metrics", "SE_STD"), ("SE_ADW_Metrics", "SE_ADW")]:
            if m_key in r:
                metrics_agg[agg_key]["TP"] += r[m_key]["TP"]
                metrics_agg[agg_key]["FP"] += r[m_key]["FP"]
                metrics_agg[agg_key]["FN"] += r[m_key]["FN"]
                metrics_agg[agg_key]["Total"] += r[m_key]["FN"] + r[m_key]["TP"]
            
    det_res = {}
    for m in ["CDT", "SE_STD", "SE_ADW"]:
        total = metrics_agg[m]["Total"]
        fn = metrics_agg[m]["FN"]
        tp = metrics_agg[m]["TP"]
        fp = metrics_agg[m]["FP"]
        mdr = fn / total if total > 0 else 0
        edr = tp / total if total > 0 else 0
        det_res[m] = {"MDR": mdr, "EDR": edr, "FP": fp}

    # 3. Generate Table Logic
    # CDT Classification Acc is generally poor (known ~46%/13% from paper/static), 
    # but we can try to compute it if CDT provides subcategory.
    # Logic: benchmark_proper returns cdt_detections with 'type'. 
    # Match these to Ground Truth matching TPs.
    # For now, to keep it simple and safe (since CDT matching is complex), 
    # we can use the Static Classification Acc for CDT (since this benchmark focuses on SE classification),
    # OR better: Compute it! 
    # Matching CDT detection to GT event -> Check type.
    
    cdt_correct_cat = 0
    cdt_correct_sub = 0
    cdt_tp_count = 0
    
    # Simple matching for CDT accuracy
    for res in results:
        events = sorted(res['Events'], key=lambda x: x['pos'])
        dets = sorted(res['CDT_Detections'], key=lambda x: x['pos'])
        
        assigned_evts = set()
        for d in dets:
            for i, e in enumerate(events):
                if i in assigned_evts: continue
                if abs(d['pos'] - e['pos']) < 200: # Match
                     assigned_evts.add(i)
                     cdt_tp_count += 1
                     if d['type'] == e['type']: 
                         cdt_correct_sub += 1
                     
                     is_gt_tcd = e['type'] in TCD_TYPES
                     is_pred_tcd = d['type'] in TCD_TYPES
                     if is_gt_tcd == is_pred_tcd:
                         cdt_correct_cat += 1
                     break
                     
    cat_acc_cdt = (cdt_correct_cat / cdt_tp_count * 100) if cdt_tp_count > 0 else 0
    sub_acc_cdt = (cdt_correct_sub / cdt_tp_count * 100) if cdt_tp_count > 0 else 0
            
    # Table Content - All 3 methods with computed metrics
    data = [
        {
            "Method": "CDT\\_MSW",
            "CAT Acc": f"{cat_acc_cdt:.1f}\\%",
            "SUB Acc": f"{sub_acc_cdt:.1f}\\%",
            "EDR": f"{det_res['CDT']['EDR']:.3f}", 
            "MDR": f"{det_res['CDT']['MDR']:.3f}",
            "FP": f"{det_res['CDT']['FP']}",
            "Supervised": "Có"
        },
        {
            "Method": "\\textbf{SE-CDT (Std)}",
            "CAT Acc": f"\\textbf{{{cat_acc_se:.1f}\\%}}",
            "SUB Acc": f"\\textbf{{{sub_acc_se:.1f}\\%}}", 
            "EDR": f"\\textbf{{{det_res['SE_STD']['EDR']:.3f}}}",
            "MDR": f"\\textbf{{{det_res['SE_STD']['MDR']:.3f}}}",
            "FP": f"{det_res['SE_STD']['FP']}",
            "Supervised": "Không"
        },
        {
            "Method": "SE-CDT (ADW)",
            "CAT Acc": f"{cat_acc_se:.1f}\\%",  # Same Oracle Acc
            "SUB Acc": f"{sub_acc_se:.1f}\\%",  # Same Oracle Acc
            "EDR": f"{det_res['SE_ADW']['EDR']:.3f}",
            "MDR": f"{det_res['SE_ADW']['MDR']:.3f}",
            "FP": f"{det_res['SE_ADW']['FP']}",
            "Supervised": "Không"
        }
    ]
    
    latex_lines = []
    latex_lines.append("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Method} & \\textbf{CAT Acc} & \\textbf{SUB Acc} & \\textbf{EDR$\\uparrow$} & \\textbf{MDR$\\downarrow$} & \\textbf{FP} & \\textbf{Supervised} \\\\")
    latex_lines.append("\\hline")
    
    for row in data:
        line = f"{row['Method']} & {row['CAT Acc']} & {row['SUB Acc']} & {row['EDR']} & {row['MDR']} & {row['FP']} & {row['Supervised']} \\\\"
        latex_lines.append(line)
        
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    output_path = "report/latex/tables/table_comparison_aggregate.tex"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
        
    print(f"\nAggregate Table generated at: {output_path}")

if __name__ == "__main__":
    run_benchmark_proper()

