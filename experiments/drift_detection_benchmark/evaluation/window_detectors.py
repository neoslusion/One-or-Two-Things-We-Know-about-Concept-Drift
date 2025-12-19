"""
Window-based drift detector evaluation.

Contains the evaluate_drift_detector function for evaluating window-based
drift detection methods including:
- D3, DAWIDD, MMD, KS
- ShapeDD variants (buffer-based approach)
- MMD_OW, ShapeDD_OW_MMD
"""

import time
import numpy as np
from collections import deque

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backup')))

from shape_dd import shape, shape_snr_adaptive
from d3 import d3
from dawidd import dawidd
from mmd import mmd
from ow_mmd import mmd_ow, shapedd_ow_mmd_buffer
from ks import ks

from ..config import (
    SHAPE_L1, SHAPE_L2, SHAPE_N_PERM, COOLDOWN, CHUNK_SIZE, OVERLAP
)
from ..utils import create_sliding_windows
from .metrics import calculate_detection_metrics_enhanced


def evaluate_drift_detector(method_name, X, true_drifts, chunk_size=None, overlap=None):
    """
    Evaluate drift detector on a stream (NO MODEL ADAPTATION).

    Two approaches:
    1. ShapeDD methods: Use BUFFER-BASED approach
       - Maintain rolling buffer of samples (750 samples)
       - Run ShapeDD on buffer periodically
       - Check recent chunks within buffer for drift
       - Scalable for large streams (only processes 750 samples at a time)

    2. Other methods: Use SLIDING WINDOW approach
       - Process stream in overlapping windows
       - Run detector on each window

    IMPORTANT: All ShapeDD methods now use CONSISTENT window sizes:
    - L1 = 50 (reference window)
    - L2 = 150 (test window)

    This ensures fair comparison and isolates algorithmic improvements
    from window size effects.

    Args:
        method_name: Name of the drift detection method
        X: Feature matrix of shape (n_samples, n_features)
        true_drifts: List of true drift positions
        chunk_size: Detection window size (default: from config)
        overlap: Window overlap (default: from config)

    Returns:
        dict: Results including detections, metrics, and runtime
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = OVERLAP

    print(f"Running: {method_name}")

    start_time = time.time()
    detections = []
    last_detection = -10**9

    # METHOD 1: Buffer-based approach for ShapeDD methods
    if 'ShapeDD' in method_name:

        # Configuration
        BUFFER_SIZE = 750           # Large rolling buffer
        CHECK_FREQUENCY = 150       # How often to check for drift

        # Rolling buffer (stores recent samples)
        buffer = deque(maxlen=BUFFER_SIZE)

        print(f"  Buffer size: {BUFFER_SIZE} samples")
        print(f"  Check frequency: every {CHECK_FREQUENCY} samples")
        print(f"  Window sizes: L1={SHAPE_L1}, L2={SHAPE_L2} (CONSISTENT for fair comparison)")
        print(f"  Processing stream...")

        # Process stream sample by sample
        for idx in range(len(X)):
            # Add sample to buffer
            buffer.append({'x': X[idx], 'idx': idx})

            # Check for drift periodically (every CHECK_FREQUENCY samples)
            if len(buffer) >= BUFFER_SIZE and idx % CHECK_FREQUENCY == 0:

                # Step 1: Extract buffer data
                buffer_list = list(buffer)
                buffer_X = np.array([item['x'] for item in buffer_list])  # Shape: (BUFFER_SIZE, n_features)
                buffer_indices = np.array([item['idx'] for item in buffer_list])

                try:
                    if method_name == 'ShapeDD':
                        # Original ShapeDD (no adaptive features)
                        shp_results = shape(buffer_X, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM)

                    elif method_name == 'ShapeDD_SNR_Adaptive':
                        shp_results = shape_snr_adaptive(buffer_X, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM)

                    elif method_name == 'ShapeDD_OW_MMD':
                        shp_results = shapedd_ow_mmd_buffer(buffer_X, SHAPE_L1, SHAPE_L2, gamma='auto')

                    # Step 3: Check recent chunk within buffer for drift
                    # Look at last CHECK_FREQUENCY samples in buffer
                    chunk_start = max(0, len(buffer_X) - CHECK_FREQUENCY)
                    recent_pvalues = shp_results[chunk_start:, 2]  # p-values for recent chunk

                    # Step 4: Check if drift detected
                    min_pvalue = recent_pvalues.min()
                    trigger = min_pvalue < 0.05  # Significance threshold

                    if trigger:
                        # Find exact position of drift in buffer
                        drift_pos_in_chunk = int(np.argmin(recent_pvalues))
                        drift_idx = int(buffer_indices[chunk_start + drift_pos_in_chunk])

                        # Record detection (with cooldown to avoid duplicates)
                        if drift_idx - last_detection >= COOLDOWN:
                            detections.append(drift_idx)
                            last_detection = drift_idx
                            print(f"    [Sample {idx}] DRIFT DETECTED at position {drift_idx} (p-value: {min_pvalue:.6f})")

                except Exception as e:
                    pass  # Skip failed detections

    # METHOD 2: Sliding window approach for other methods
    else:
        # Create sliding windows
        windows, window_centers = create_sliding_windows(X, chunk_size, overlap)
        print(f"  Processing {len(windows)} windows...")

        for window_idx, (window, center_idx) in enumerate(zip(windows, window_centers)):
            try:
                # Method-specific detection
                if method_name == 'D3':
                    score = d3(window)
                    trigger = score > 0.7

                elif method_name == 'DAWIDD':
                    _, p_value = dawidd(window, 'rbf')
                    trigger = p_value < 0.05

                elif method_name == 'MMD':
                    stat, p_value = mmd(window)
                    trigger = p_value < 0.05

                elif method_name == 'KS':
                    p_value = ks(window)
                    trigger = p_value < 0.05

                elif method_name == 'MMD_OW':
                    # Optimally-Weighted MMD
                    stat, threshold = mmd_ow(window, gamma='auto')
                    trigger = stat > threshold

                else:
                    trigger = False

                # Record detection if triggered and outside cooldown
                if trigger and (center_idx - last_detection >= COOLDOWN):
                    detections.append(center_idx)
                    last_detection = center_idx

            except Exception as e:
                pass  # Skip failed detections

    # Calculate metrics
    end_time = time.time()
    metrics = calculate_detection_metrics_enhanced(
        detections, true_drifts, len(X)
    )

    return {
        'method': method_name,
        'detections': detections,
        'stream_size': len(X),
        'runtime_s': end_time - start_time,
        **metrics
    }

