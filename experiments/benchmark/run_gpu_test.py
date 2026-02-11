"""
run_gpu_test.py

Targeted benchmark for Electricity dataset to verify CPU usage/performance.
"""

import sys
import os
import time
import numpy as np

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data.generators.benchmark_generators import generate_drift_stream
from experiments.benchmark.evaluation import evaluate_drift_detector
from experiments.benchmark.config import CHUNK_SIZE, OVERLAP

# Check Torch status (should be disabled/unavailable for MMD)
try:
    import torch
    from core.detectors.mmd_variants import HAS_TORCH

    print(f"Torch Installed: {True}")
    print(f"MMD GPU Enabled (HAS_TORCH): {HAS_TORCH}")
except ImportError:
    print("Torch not found")


def run_test():
    print("\nRunning Targeted CPU Benchmark: Electricity Semi-Synthetic")
    print("=" * 60)

    # Config for Elec2
    dataset_config = {
        "type": "electricity_semisynthetic",
        "n_drift_events": 5,
        "params": {"data_path": "data/real_world/electricity-normalized.csv"},
    }

    # Generate Data
    print("Generating stream...")
    try:
        X, y, true_drifts, info = generate_drift_stream(
            dataset_config, total_size=5000, seed=42
        )
        print(f"Stream generated: {len(X)} samples, {len(true_drifts)} drifts")
    except Exception as e:
        print(f"FAILED to generate stream: {e}")
        return

    # Methods to test
    methods = ["MMD", "ShapeDD", "ShapeDD_MMDAgg", "MMD_ADW", "ShapeDD_ADW_MMD"]

    for method in methods:
        print(f"\nTesting {method}...")
        try:
            start = time.time()
            result = evaluate_drift_detector(
                method,
                X,
                true_drifts,
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP,
                verbose=False,
            )
            duration = time.time() - start
            f1 = result["f1_score"]

            print(f"  -> Finished in {duration:.4f}s")
            print(f"  -> F1 Score: {f1:.4f}")

        except Exception as e:
            print(f"  -> FAILED: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    run_test()
