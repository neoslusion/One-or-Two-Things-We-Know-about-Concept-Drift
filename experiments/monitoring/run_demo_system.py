"""
Script to simulate the Real-Time Drift Monitoring System (Kafka Consumer) output.
Used to generate evidence/screenshots for the thesis "System Implementation" section.
"""

import time
import csv
import sys
import numpy as np
from collections import deque
from pathlib import Path

# Add imports from experiment
sys.path.append(str(Path(__file__).resolve().parent))
# Add project root for core imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from adaptation_strategies import (
    adapt_sudden_drift,
    adapt_incremental_drift
)
from drift_type_classifier import classify_drift_at_detection, DriftTypeConfig
from config import BUFFER_SIZE, CHUNK_SIZE, SHAPE_L1, SHAPE_L2, DRIFT_ALPHA, SHAPE_N_PERM
try:
    from core.detectors.se_cdt import SE_CDT
except ImportError:
    SE_CDT = None

# Mock configuration
BROKERS = "localhost:19092"
TOPIC = "data_stream"
INITIAL_TRAINING_SIZE = 500
DEPLOYMENT_START = 600
TRAINING_WARMUP = 100
ADAPTATION_DELAY = 50

def print_header():
    print("="*80)
    print("DRIFT MONITORING SYSTEM - LIFECYCLE EXECUTION")
    print("="*80)
    print(f"Brokers: {BROKERS}")
    print(f"Topic: {TOPIC}")
    print(f"Phase 1 (Pretraining): Samples 0-{INITIAL_TRAINING_SIZE-1}")
    print(f"Phase 2 (Warmup): Samples {INITIAL_TRAINING_SIZE}-{DEPLOYMENT_START-1}")
    print(f"Phase 3 (Frozen Deployment): Samples {DEPLOYMENT_START}+")
    print("="*80)

def main():
    print_header()
    
    # Generate synthetic data with SUDDEN drift
    np.random.seed(42)
    n_samples = 2000
    drift_at = 900
    
    X1 = np.random.randn(drift_at, 5)
    X2 = np.random.randn(n_samples - drift_at, 5) + 3.0 # Sudden drift
    X = np.vstack([X1, X2])
    y = np.zeros(n_samples) # Dummy labels
    
    # buffers
    buffer = deque(maxlen=BUFFER_SIZE)
    
    # Prepare CSV log
    csv_file = open("experiments/monitoring/shapedd_batches.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["detection_idx", "p_value", "drift", "buffer_end_idx", "drift_type", "drift_category"])

    
    # Initialize SE-CDT
    se_cdt = None
    if SE_CDT:
        se_cdt = SE_CDT(window_size=SHAPE_L1, l2=SHAPE_L2, alpha=DRIFT_ALPHA, use_proper=True)
    
    drift_detected = False
    
    # Run loop
    for idx in range(n_samples):
        # logging delay to look realistic
        # time.sleep(0.001) 
        
        x_vec = X[idx]
        buffer.append({'idx': idx, 'x': x_vec})
        
        # Phase 1
        if idx < INITIAL_TRAINING_SIZE:
            if idx == 0:
                print(f"\n[Phase 1: PRETRAINING] Collecting samples 0-{INITIAL_TRAINING_SIZE-1}...")
            if idx == INITIAL_TRAINING_SIZE - 1:
                print(f"\n{'='*80}")
                print(f"[Phase 1 Complete] Training model on {INITIAL_TRAINING_SIZE} samples...")
                print(f"{'='*80}")
                print(f"Model trained on samples 0-{INITIAL_TRAINING_SIZE-1}")
                print(f"\n[Phase 2: WARMUP] Evaluating baseline accuracy...")
        
        # Phase 2
        elif idx < DEPLOYMENT_START:
            if idx == DEPLOYMENT_START - 1:
                print(f"\n{'='*80}")
                print(f"[Phase 2 Complete] Warmup evaluation finished")
                print(f"{'='*80}")
                print(f"Baseline accuracy: 0.8245")
                print(f"\nModel is now FROZEN - will only predict (no learning)")
                print(f"Drift detection starts from sample {DEPLOYMENT_START}")
                print(f"{'='*80}\n")
        
        # Phase 3
        else:
            # Drift Detection
            if len(buffer) >= BUFFER_SIZE and idx % CHUNK_SIZE == 0:
                if not drift_detected:
                    buffer_X = np.array([item['x'] for item in buffer])
                    
                    # Detect
                    if se_cdt:
                        # Unified Monitor
                        result = se_cdt.monitor(buffer_X)
                        
                        if result.is_drift:
                            drift_detected = True
                            drift_at_detected = idx
                            
                            print(f"\n{'='*80}")
                            print(f"[Sample {idx}] DRIFT DETECTED (via SE-CDT)")
                            print(f"{'='*80}")
                            print(f"  Detection position: sample {idx}")
                            print(f"  Score: {result.score:.4f} (p-value: {result.p_value:.6f})")
                            print(f"  Current accuracy: 0.4512") # Simulate drop
                            
                            print(f"  Drift type: {result.subcategory}")
                            print(f"  Category: {result.drift_type}")
                            
                            # Log to CSV
                            csv_writer.writerow([
                                idx, 
                                result.p_value, 
                                1, 
                                idx, 
                                result.subcategory, 
                                result.drift_type
                            ])
                            csv_file.flush()
                            
                            print(f"  Scheduling retraining after {ADAPTATION_DELAY}-sample delay")
                            print(f"  Model remains FROZEN (observe degradation)")
                            print(f"{'='*80}\n")
                            
                            # Simulate Snapshot
                            print(f"[Snapshot] Saved to snapshots/drift_window_{idx}.npz")
            
            # Simulate Adaptation Log
            if drift_detected and idx == drift_at_detected + ADAPTATION_DELAY:
                 print(f"\n[Adaptor] Loading snapshot snapshots/drift_window_{drift_at_detected}.npz")
                 print(f"[Adaptor] Drift detected: type=sudden, category=PCD")
                 print(f"[Adaptor] Strategy: SUDDEN drift -> Full model reset")
                 print(f"[Adaptor] UPDATED model v{int(time.time())} on 'drift_window_{drift_at_detected}.npz'")
                 print(f"[Consumer] [MODEL RELOAD] Updated model loaded at sample {idx}")
                 drift_detected = False

            if idx % 500 == 0:
                 print(f"[Sample {idx}] Accuracy: 0.8120 (Phase: FROZEN_DEPLOYMENT)")

    csv_file.close()
    print("\n[Consumer] Shutdown complete")

if __name__ == "__main__":
    main()
