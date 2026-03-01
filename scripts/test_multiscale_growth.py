#!/usr/bin/env python3
"""
Multi-scale MMD Growth Process Experiment.

Compares single-scale vs multi-scale Growth Process for TCD/PCD classification.
Generates 5 drift types, runs SE-CDT on each, and measures classification accuracy.

Usage:
    python scripts/test_multiscale_growth.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from data.generators.drift_generators import ConceptDriftStreamGenerator
from core.detectors.se_cdt import SE_CDT
from core.detectors.mmd_variants import shapedd_adw_mmd_proper


# ─── Ground Truth Mapping ─────────────────────────────────────────────────────
DRIFT_GROUND_TRUTH = {
    "Sudden":      {"category": "TCD", "subcategory": "Sudden"},
    "Blip":        {"category": "TCD", "subcategory": "Blip"},
    "Recurrent":   {"category": "TCD", "subcategory": "Recurrent"},
    "Gradual":     {"category": "PCD", "subcategory": "Gradual"},
    "Incremental": {"category": "PCD", "subcategory": "Incremental"},
}


@dataclass
class ExperimentResult:
    drift_type: str
    seed: int
    detected: bool
    predicted_category: str   # TCD or PCD
    predicted_subcategory: str
    true_category: str
    true_subcategory: str
    drift_length: int
    time_ms: float
    method: str  # "single" or "multi"
    wr_values: Dict[int, float] = None  # scale -> WR
    divergence: float = 0.0


def generate_drift_data(drift_type: str, seed: int, 
                        n_samples: int = 1500, n_features: int = 5,
                        magnitude: float = 2.5) -> Tuple[np.ndarray, int]:
    """Generate a window of data containing one drift event."""
    gen = ConceptDriftStreamGenerator(n_features=n_features, seed=seed)
    
    drift_pos = n_samples // 2  # Drift at midpoint
    
    if drift_type == "Sudden":
        X, _, pos = gen.generate_sudden_drift(
            length=n_samples, drift_position=drift_pos, magnitude=magnitude
        )
    elif drift_type == "Blip":
        X, _, pos = gen.generate_blip_drift(
            length=n_samples, drift_position=drift_pos, 
            blip_width=80, magnitude=magnitude
        )
    elif drift_type == "Recurrent":
        X, _, positions = gen.generate_recurrent_drift(
            length=n_samples, drift_position=drift_pos, 
            period=300, magnitude=magnitude
        )
        pos = positions[0] if isinstance(positions, list) else positions
    elif drift_type == "Gradual":
        X, _, pos = gen.generate_gradual_drift(
            length=n_samples, drift_position=drift_pos, 
            transition_width=400, magnitude=magnitude
        )
    elif drift_type == "Incremental":
        X, _, pos = gen.generate_incremental_drift(
            length=n_samples, drift_position=drift_pos, 
            transition_width=400, magnitude=magnitude
        )
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")
    
    return X, pos


def run_single_scale_growth(window: np.ndarray, detector: SE_CDT, 
                            mmd_trace: np.ndarray) -> Tuple[int, float]:
    """Run the single-scale (original) growth process."""
    t0 = time.time()
    drift_length = detector._growth_process_single(mmd_trace)
    t1 = time.time()
    return drift_length, (t1 - t0) * 1000


def run_multiscale_growth(window: np.ndarray, detector: SE_CDT,
                          mmd_trace: np.ndarray) -> Tuple[int, Dict[int, float], float, float]:
    """Run the multi-scale growth process, returning diagnostics."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, peak_widths
    
    scales = [30, 50, 100]
    n = len(window)
    l2 = detector.l2
    
    t0 = time.time()
    
    # Compute WR at each scale
    wr_by_scale = {}
    for l1_scale in scales:
        if l1_scale == detector.l1 and mmd_trace is not None:
            trace = mmd_trace
        else:
            min_needed = 2 * l1_scale + l2
            if n < min_needed:
                wr_by_scale[l1_scale] = 0.0
                continue
            _, _, trace, _ = shapedd_adw_mmd_proper(
                window, l1=l1_scale, l2=l2, alpha=1.0
            )
        
        wr = detector._compute_wr_from_trace(trace)
        wr_by_scale[l1_scale] = wr
    
    wr_small = wr_by_scale.get(scales[0], 0.0)
    wr_large = wr_by_scale.get(scales[-1], 0.0)
    
    if wr_large < 0.01:
        divergence = 0.0
        drift_length = 1
    else:
        divergence = wr_small / wr_large
        # Composite PCD score (matching se_cdt.py)
        is_pcd = (divergence > 0.50 and wr_small > 0.15) or wr_small > 0.20
        if is_pcd:
            drift_length = max(2, int(wr_small * n / 3))
        else:
            drift_length = 1
    
    t1 = time.time()
    return drift_length, wr_by_scale, divergence, (t1 - t0) * 1000


def run_experiment(n_runs: int = 10, n_samples: int = 1500):
    """Run the full comparison experiment."""
    drift_types = ["Sudden", "Blip", "Recurrent", "Gradual", "Incremental"]
    
    detector = SE_CDT(window_size=50, l2=150)
    
    results: List[ExperimentResult] = []
    
    print("=" * 80)
    print("Multi-scale MMD Growth Process Experiment")
    print(f"N_RUNS={n_runs}, N_SAMPLES={n_samples}")
    print("=" * 80)
    
    for drift_type in drift_types:
        gt = DRIFT_GROUND_TRUTH[drift_type]
        print(f"\n--- {drift_type} (GT: {gt['category']}) ---")
        
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 7
            
            try:
                # Generate data
                X, drift_pos = generate_drift_data(drift_type, seed, n_samples)
                
                # Run detection first (get MMD trace)
                _, _, mmd_trace, _ = shapedd_adw_mmd_proper(
                    X, l1=50, l2=150, alpha=0.05
                )
                
                if len(mmd_trace) < 5:
                    print(f"  Run {run_idx}: No MMD trace (too short)")
                    continue
                
                # --- Single-scale ---
                dl_single, time_single = run_single_scale_growth(X, detector, mmd_trace)
                cat_single = "PCD" if dl_single > 1 else "TCD"
                
                # Classify with single-scale drift_length
                res_single = detector.classify(mmd_trace, drift_length=dl_single)
                
                results.append(ExperimentResult(
                    drift_type=drift_type, seed=seed, detected=True,
                    predicted_category=cat_single,
                    predicted_subcategory=res_single.subcategory,
                    true_category=gt["category"],
                    true_subcategory=gt["subcategory"],
                    drift_length=dl_single,
                    time_ms=time_single,
                    method="single"
                ))
                
                # --- Multi-scale ---
                dl_multi, wr_vals, div, time_multi = run_multiscale_growth(X, detector, mmd_trace)
                cat_multi = "PCD" if dl_multi > 1 else "TCD"
                
                res_multi = detector.classify(mmd_trace, drift_length=dl_multi)
                
                results.append(ExperimentResult(
                    drift_type=drift_type, seed=seed, detected=True,
                    predicted_category=cat_multi,
                    predicted_subcategory=res_multi.subcategory,
                    true_category=gt["category"],
                    true_subcategory=gt["subcategory"],
                    drift_length=dl_multi,
                    time_ms=time_multi,
                    method="multi",
                    wr_values=wr_vals,
                    divergence=div
                ))
                
                # Print per-run diagnostics
                wr_str = ", ".join(f"l1={k}:{v:.3f}" for k, v in sorted(wr_vals.items()))
                print(f"  Run {run_idx}: Single→{cat_single}(dl={dl_single}) | "
                      f"Multi→{cat_multi}(dl={dl_multi}, div={div:.2f}) | WR: {wr_str}")
                
            except Exception as e:
                print(f"  Run {run_idx}: ERROR - {e}")
    
    return results


def print_summary(results: List[ExperimentResult]):
    """Print comparison summary table."""
    drift_types = ["Sudden", "Blip", "Recurrent", "Gradual", "Incremental"]
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY: Category Accuracy (TCD vs PCD)")
    print("=" * 80)
    
    header = f"{'Drift Type':<15} {'GT':>5} | {'Single-Scale':>15} | {'Multi-Scale':>15} | {'Improvement':>12}"
    print(header)
    print("-" * len(header))
    
    total_single_correct = 0
    total_multi_correct = 0
    total_count = 0
    
    for dt in drift_types:
        gt_cat = DRIFT_GROUND_TRUTH[dt]["category"]
        
        single_results = [r for r in results if r.drift_type == dt and r.method == "single"]
        multi_results = [r for r in results if r.drift_type == dt and r.method == "multi"]
        
        n = len(single_results)
        if n == 0:
            continue
        
        single_correct = sum(1 for r in single_results if r.predicted_category == r.true_category)
        multi_correct = sum(1 for r in multi_results if r.predicted_category == r.true_category)
        
        single_acc = single_correct / n * 100
        multi_acc = multi_correct / n * 100
        
        improvement = multi_acc - single_acc
        imp_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        
        print(f"{dt:<15} {gt_cat:>5} | {single_acc:>12.1f}% | {multi_acc:>12.1f}% | {imp_str:>12}")
        
        total_single_correct += single_correct
        total_multi_correct += multi_correct
        total_count += n
    
    print("-" * len(header))
    
    overall_single = total_single_correct / total_count * 100 if total_count > 0 else 0
    overall_multi = total_multi_correct / total_count * 100 if total_count > 0 else 0
    overall_imp = overall_multi - overall_single
    imp_str = f"+{overall_imp:.1f}%" if overall_imp > 0 else f"{overall_imp:.1f}%"
    
    print(f"{'OVERALL':<15} {'':>5} | {overall_single:>12.1f}% | {overall_multi:>12.1f}% | {imp_str:>12}")
    
    # Timing comparison
    print("\n" + "=" * 80)
    print("TIMING COMPARISON")
    print("=" * 80)
    
    single_times = [r.time_ms for r in results if r.method == "single"]
    multi_times = [r.time_ms for r in results if r.method == "multi"]
    
    if single_times and multi_times:
        print(f"Single-scale: {np.mean(single_times):.1f}ms ± {np.std(single_times):.1f}ms")
        print(f"Multi-scale:  {np.mean(multi_times):.1f}ms ± {np.std(multi_times):.1f}ms")
        print(f"Overhead:     {np.mean(multi_times)/max(np.mean(single_times), 0.001):.1f}x")
    
    # Divergence analysis
    print("\n" + "=" * 80)
    print("CROSS-SCALE DIVERGENCE ANALYSIS")
    print("=" * 80)
    
    for dt in drift_types:
        multi_results = [r for r in results if r.drift_type == dt and r.method == "multi"]
        if multi_results:
            divs = [r.divergence for r in multi_results]
            gt_cat = DRIFT_GROUND_TRUTH[dt]["category"]
            print(f"{dt:<15} ({gt_cat}): divergence = {np.mean(divs):.3f} ± {np.std(divs):.3f} "
                  f"(range: {min(divs):.3f} - {max(divs):.3f})")
    
    # Subcategory accuracy
    print("\n" + "=" * 80)
    print("SUBCATEGORY ACCURACY (5-class)")
    print("=" * 80)
    
    for method in ["single", "multi"]:
        method_results = [r for r in results if r.method == method]
        correct = sum(1 for r in method_results if r.predicted_subcategory == r.true_subcategory)
        total = len(method_results)
        acc = correct / total * 100 if total > 0 else 0
        label = "Single-scale" if method == "single" else "Multi-scale "
        print(f"{label}: {acc:.1f}% ({correct}/{total})")


if __name__ == "__main__":
    results = run_experiment(n_runs=10, n_samples=1500)
    print_summary(results)
