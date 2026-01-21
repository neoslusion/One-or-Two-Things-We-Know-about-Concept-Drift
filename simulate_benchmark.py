import numpy as np
import time
from experiments.backup.shape_dd import shape
from sklearn.metrics.pairwise import pairwise_kernels


def simulate_full_benchmark():
    print("Simulating Full Benchmark Workload (CPU)")
    print("-" * 60)

    # Setup mirroring benchmark config
    STREAM_SIZE = 10000
    CHUNK_SIZE = 150
    OVERLAP = 100
    L1 = 50
    L2 = 150
    N_PERM = 100  # Reduced for simulation speed (benchmark uses 500-2500)

    # Generate stream
    X = np.random.randn(STREAM_SIZE, 5)

    # Create sliding windows
    stride = CHUNK_SIZE - OVERLAP  # 50
    n_windows = (STREAM_SIZE - CHUNK_SIZE) // stride + 1

    print(f"Stream: {STREAM_SIZE} samples")
    print(f"Windows: {n_windows} (Size {CHUNK_SIZE}, Stride {stride})")
    print(f"Algorithm: ShapeDD (L1={L1}, L2={L2}, Perm={N_PERM})")
    print("-" * 60)

    # Run Simulation
    start_total = time.time()
    times = []

    # Run first 50 windows to estimate
    n_test = 50
    print(f"Running first {n_test} windows...")

    for i in range(n_test):
        start = X[i * stride : i * stride + CHUNK_SIZE]
        if len(start) < CHUNK_SIZE:
            break

        t0 = time.time()
        # ShapeDD execution
        _ = shape(start, l1=L1, l2=L2, n_perm=N_PERM)
        t1 = time.time()
        times.append(t1 - t0)

    avg_time = np.mean(times)
    total_est = avg_time * n_windows

    print("-" * 60)
    print(f"Avg Time per Window: {avg_time:.4f}s")
    print(f"Estimated Total Time: {total_est:.2f}s ({total_est / 60:.2f} min)")
    print(f"Throughput: {1 / avg_time:.1f} windows/sec")


if __name__ == "__main__":
    simulate_full_benchmark()
