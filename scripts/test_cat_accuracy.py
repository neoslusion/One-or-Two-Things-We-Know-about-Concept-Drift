#!/usr/bin/env python
"""
Quick Test Script for CAT Accuracy Improvement
Validates classification threshold changes in SE-CDT without full benchmark.

Usage:
    python scripts/test_cat_accuracy.py
    python scripts/test_cat_accuracy.py --scenarios Mixed_A Repeated_Incremental --seeds 3
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmark.benchmark_proper import run_quick_validation


def main():
    parser = argparse.ArgumentParser(
        description="Quick validation test for CAT accuracy improvements"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["Repeated_Incremental", "Mixed_A"],
        choices=["Mixed_A", "Mixed_B", "Repeated_Sudden", "Repeated_Gradual", 
                 "Repeated_Incremental", "Repeated_Recurrent"],
        help="Scenarios to test (default: Repeated_Incremental Mixed_A)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=2,
        help="Number of random seeds per scenario (default: 2)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SE-CDT CLASSIFICATION THRESHOLD VALIDATION")
    print("="*80)
    print(f"Testing scenarios: {', '.join(args.scenarios)}")
    print(f"Seeds per scenario: {args.seeds}")
    print(f"Total experiments: {len(args.scenarios) * args.seeds}")
    print("="*80 + "\n")
    
    # Run validation
    metrics = run_quick_validation(scenarios=args.scenarios, n_seeds=args.seeds)
    
    # Print summary
    print("\n" + "="*80)
    print("THRESHOLD TUNING SUMMARY")
    print("="*80)
    print("\nChanges Applied:")
    print("  1. Sudden detection: wr < 0.15 (was 0.12), snr > 2.0 (was 2.5)")
    print("  2. Incremental detection: lts > 0.5 (was 0.3)")
    print("  3. Compound conditions: ms > 0.6 (was 0.5), lts > 0.3 (was 0.1)")
    print("\nExpected Improvements:")
    print("  - CAT Accuracy: 60% → 75-80%")
    print("  - TCD Accuracy: 25% → 60-70%")
    print("  - Incremental Accuracy: 40% → 25-30% (acceptable trade-off)")
    print("\nActual Results:")
    print(f"  - CAT Accuracy: {metrics['cat_accuracy']:.1f}%")
    print(f"  - SUB Accuracy: {metrics['sub_accuracy']:.1f}%")
    print(f"  - Incremental Accuracy: {metrics['per_class_accuracy'].get('Incremental', 0):.1f}%")
    print(f"  - Sudden Accuracy: {metrics['per_class_accuracy'].get('Sudden', 0):.1f}%")
    
    # Evaluation
    cat_target = 75.0
    cat_achieved = metrics['cat_accuracy'] >= cat_target
    
    print("\n" + "-"*80)
    if cat_achieved:
        print(f"✓ SUCCESS: CAT accuracy {metrics['cat_accuracy']:.1f}% meets target ({cat_target}%)")
    else:
        print(f"✗ BELOW TARGET: CAT accuracy {metrics['cat_accuracy']:.1f}% < {cat_target}%")
        print("  Consider further threshold adjustments or test with more seeds.")
    print("="*80 + "\n")
    
    return metrics


if __name__ == "__main__":
    main()
