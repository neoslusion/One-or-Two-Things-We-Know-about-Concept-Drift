#!/usr/bin/env python
"""
Quick Prequential Test for CAT Accuracy Validation
Tests SE-CDT in end-to-end prequential setting with single scenario.

Usage:
    python scripts/test_prequential_cat.py
    python scripts/test_prequential_cat.py --drift_type incremental --n_samples 3000
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.monitoring.evaluate_prequential import (
    generate_synthetic_stream,
    evaluate_with_adaptation,
    AdaptationMode,
    calculate_metrics
)


def main():
    parser = argparse.ArgumentParser(
        description="Quick prequential test for CAT accuracy validation"
    )
    parser.add_argument(
        "--drift_type",
        type=str,
        default="mixed",
        choices=["sudden", "gradual", "incremental", "recurrent", "mixed"],
        help="Drift type to test (default: mixed)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3000,
        help="Number of samples (default: 3000 for quick test)"
    )
    parser.add_argument(
        "--n_drifts",
        type=int,
        default=3,
        help="Number of drift events (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--w_ref",
        type=int,
        default=50,
        help="Reference window size (default: 50)"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.15,
        help="Detection threshold (default: 0.15)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PREQUENTIAL VALIDATION - CAT ACCURACY IMPROVEMENT")
    print("="*80)
    print(f"Drift Type: {args.drift_type}")
    print(f"Stream Length: {args.n_samples} samples")
    print(f"Drift Events: {args.n_drifts}")
    print(f"SE-CDT Config: w_ref={args.w_ref}, thresh={args.thresh}")
    print("="*80 + "\n")
    
    # Generate data
    print("[1/3] Generating synthetic data stream...")
    X, y, drift_points, drift_types = generate_synthetic_stream(
        n_samples=args.n_samples,
        n_drifts=args.n_drifts,
        drift_type=args.drift_type,
        random_seed=args.seed
    )
    print(f"  Generated: {len(X)} samples, {len(drift_points)} drift points")
    print(f"  Drift positions: {drift_points}")
    print(f"  Drift types: {drift_types}")
    
    # Evaluate with type-specific adaptation
    print("\n[2/3] Evaluating with type-specific adaptation...")
    acc_ts, det_ts, adapt_ts, times_ts = evaluate_with_adaptation(
        X, y, drift_points,
        mode=AdaptationMode.TYPE_SPECIFIC,
        w_ref=args.w_ref,
        sudden_thresh=args.thresh
    )
    
    results = {
        AdaptationMode.TYPE_SPECIFIC: {
            "accuracy": acc_ts,
            "detections": det_ts,
            "adaptations": adapt_ts,
            "classification_times": times_ts
        }
    }
    
    # Calculate metrics
    print("\n[3/3] Calculating metrics...")
    metrics = calculate_metrics(results, drift_points)
    
    # Analyze classifications
    TCD_TYPES = {"Sudden", "Blip", "Recurrent"}
    cat_correct = 0
    cat_total = 0
    
    type_counts = {}
    for adapt in adapt_ts:
        if 'drift_type' in adapt:
            dtype = adapt['drift_type']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            # Compare with ground truth
            adapt_idx = adapt.get('idx', -1)
            for i, dp in enumerate(drift_points):
                if abs(adapt_idx - dp) < 250:  # Match within tolerance
                    gt_type = drift_types[i] if i < len(drift_types) else "Unknown"
                    is_gt_tcd = gt_type in TCD_TYPES
                    is_pred_tcd = dtype in TCD_TYPES
                    cat_total += 1
                    if is_gt_tcd == is_pred_tcd:
                        cat_correct += 1
                    break
    
    cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("PREQUENTIAL TEST RESULTS")
    print("="*80)
    print(f"Detection Metrics:")
    ts_metrics = metrics.get(AdaptationMode.TYPE_SPECIFIC, {})
    print(f"  EDR (Detection Rate): {ts_metrics.get('edr', 0):.3f}")
    print(f"  MDR (Miss Rate):      {ts_metrics.get('mdr', 0):.3f}")
    print(f"  Mean Delay:           {ts_metrics.get('mean_delay', 0):.1f} samples")
    print(f"  False Positives:      {ts_metrics.get('fp', 0)}")
    
    print(f"\nClassification Analysis:")
    print(f"  Matched Classifications: {cat_total}")
    print(f"  CAT Accuracy (TCD/PCD):  {cat_accuracy:.1f}%")
    print(f"\nDetected Drift Types:")
    for dtype, count in sorted(type_counts.items()):
        is_tcd = " (TCD)" if dtype in TCD_TYPES else " (PCD)"
        print(f"    {dtype:12s}{is_tcd}: {count} detections")
    
    print(f"\nPrequential Accuracy:")
    print(f"  Final Accuracy: {acc_ts[-1] if acc_ts else 0:.3f}")
    print(f"  Mean Accuracy:  {sum(acc_ts)/len(acc_ts) if acc_ts else 0:.3f}")
    
    print("\n" + "-"*80)
    print("THRESHOLD VALIDATION:")
    print("  Changes: Sudden (wr<0.15, snr>2.0), Incremental (lts>0.5)")
    print(f"  CAT Accuracy: {cat_accuracy:.1f}%")
    if cat_accuracy >= 75.0:
        print("  ✓ SUCCESS: Meets 75% target")
    else:
        print(f"  ✗ BELOW TARGET: {cat_accuracy:.1f}% < 75%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
