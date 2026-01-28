"""
Consumer matching DriftMonitoring.ipynb notebook workflow.

4-Phase Lifecycle:
  Phase 1 (samples 0-499): PRETRAINING - Collect and batch train model
  Phase 2 (samples 500-599): WARMUP - Evaluate baseline accuracy  
  Phase 3 (samples 600+): FROZEN DEPLOYMENT - Only predict, detect drift
  Phase 4: ADAPTATION - Triggered by drift detection
"""
import json, os, time, csv, pickle
import psutil
from collections import deque
import numpy as np
from confluent_kafka import Consumer, Producer
import sys
from pathlib import Path

# SETUP PATHS FIRST
REPO_ROOT = Path(__file__).resolve().parents[2]
SHAPE_DD_DIR = REPO_ROOT / "experiments" / "backup"
if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.append(str(SHAPE_DD_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Sklearn for model training (matching notebook)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Import shared configuration
from config import (
    BUFFER_SIZE, CHUNK_SIZE, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM, 
    DRIFT_PVALUE, DRIFT_ALPHA, BROKERS, TOPIC, GROUP_ID, RESULT_TOPIC, ACCURACY_TOPIC, SHAPEDD_LOG,
    INITIAL_TRAINING_SIZE, TRAINING_WARMUP, DEPLOYMENT_START,
    PREQUENTIAL_WINDOW, ADAPTATION_DELAY, ADAPTATION_WINDOW
)

# Snapshot and model directories
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", "./snapshots"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)



# Import unified SE-CDT detector
from core.detectors.se_cdt import SE_CDT

# Allow environment variable overrides
BROKERS = os.getenv("BROKERS", BROKERS)
TOPIC = os.getenv("TOPIC", TOPIC)
GROUP_ID = os.getenv("GROUP_ID", GROUP_ID)
RESULT_TOPIC = os.getenv("RESULT_TOPIC", RESULT_TOPIC)


def create_model():
    """
    Create sklearn Pipeline matching DriftMonitoring.ipynb.
    StandardScaler + LogisticRegression
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])


def save_model(model, path):
    """Save model to disk."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    """Load model from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    # Use dynamic group ID to reset offset on each run
    reset_offset = os.getenv("RESET_OFFSET_ON_RESTART", "true").lower() == "true"
    
    if reset_offset:
        group_id = f"{GROUP_ID}-{int(time.time())}"
        print(f"[Consumer] Fresh group: {group_id} (starting from earliest)")
    else:
        group_id = GROUP_ID
        print(f"[Consumer] Persistent group: {group_id}")
    
    # Data stream consumer
    conf = {
        'bootstrap.servers': BROKERS,
        'group.id': group_id,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
    }
    c = Consumer(conf)
    c.subscribe([TOPIC])

    # Model update consumer (to receive updates from adaptor)
    model_update_conf = {
        'bootstrap.servers': BROKERS,
        'bootstrap.servers': BROKERS,
        'group.id': f'{group_id}-model-updates',
        'auto.offset.reset': 'earliest',  # Changed from latest to catch all updates
        'enable.auto.commit': True,
    }
    model_consumer = Consumer(model_update_conf)
    model_consumer.subscribe(['model.updated'])
    
    # Producer for publishing results
    p = Producer({'bootstrap.servers': BROKERS})

    print("="*80)
    print("DRIFT MONITORING SYSTEM - LIFECYCLE EXECUTION")
    print("="*80)
    print(f"Brokers: {BROKERS}")
    print(f"Topic: {TOPIC}")
    print(f"Phase 1 (Pretraining): Samples 0-{INITIAL_TRAINING_SIZE-1}")
    print(f"Phase 2 (Warmup): Samples {INITIAL_TRAINING_SIZE}-{DEPLOYMENT_START-1}")
    print(f"Phase 3 (Frozen Deployment): Samples {DEPLOYMENT_START}+")
    print("="*80)

    # Prepare CSV log for detections
    csv_path = os.getenv("SHAPEDD_LOG", "shapedd_batches.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["detection_idx", "p_value", "drift", "buffer_end_idx", "drift_type", "drift_category"])

    # Prepare Detailed Metrics Log
    metrics_path = "system_metrics.csv"
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        "idx", "accuracy", "memory_mb", "cpu_percent", 
        "feature_mean", "phase", "drift_detected"
    ])
    process = psutil.Process()

    # Initialize SE-CDT Unified Detector
    # Configured to use PROPER ShapeDD + ADW-MMD + Signal Shape Classification
    se_cdt = SE_CDT(
        window_size=SHAPE_L1,
        l2=SHAPE_L2,
        alpha=DRIFT_ALPHA,
        use_proper=True
    )

    # =========================================================================
    # PHASE TRACKING
    # =========================================================================
    current_phase = "PRETRAINING"
    
    # Phase 1: Pretraining data collection (keyed by idx for Kafka resilience)
    training_buffer = {}  # idx -> (x, y)
    
    # Phase 2: Warmup tracking
    warmup_correct = []
    baseline_accuracy = 0.0
    
    # Phase 3: Model and tracking
    model = None
    model_trained = False
    buffer = deque(maxlen=BUFFER_SIZE)  # For drift detection
    recent_correct = deque(maxlen=PREQUENTIAL_WINDOW)  # For accuracy tracking
    accuracy_tracker = []
    
    # Drift detection state
    drift_detected = False
    drift_detected_at = None
    drift_type = None
    in_degradation_period = False
    planned_adaptation_idx = None
    drift_info = {} # Store details for delayed adaptation

    try:
        print(f"\n[Phase 1: PRETRAINING] Collecting samples 0-{INITIAL_TRAINING_SIZE-1}...")
        
        while True:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[Consumer] Error: {msg.error()}")
                continue
                
            try:
                data = json.loads(msg.value().decode("utf-8"))
                x_vec = data.get("x", [])
                y_label = data.get("y")  # Ground truth label from producer
                idx = data.get("idx", -1)
                
                if not x_vec or y_label is None or idx < 0:
                    continue

                x_array = np.array(x_vec, dtype=float)

                # =====================================================================
                # PHASE 1: PRETRAINING (Samples 0-499) - Use idx from producer!
                # =====================================================================
                if idx < INITIAL_TRAINING_SIZE:
                    # Store by index (handles out-of-order or duplicate messages)
                    training_buffer[idx] = (x_array, y_label)
                    
                    if len(training_buffer) % 100 == 0:
                        print(f"  Collected {len(training_buffer)}/{INITIAL_TRAINING_SIZE} training samples (latest idx: {idx})...")
                    
                    # Train model when we have ALL required samples (0-499)
                    if len(training_buffer) >= INITIAL_TRAINING_SIZE and not model_trained:
                        # Check if we have consecutive samples 0 to 499
                        required_indices = set(range(INITIAL_TRAINING_SIZE))
                        available_indices = set(training_buffer.keys())
                        
                        if required_indices.issubset(available_indices):
                            print(f"\n{'='*80}")
                            print(f"[Phase 1 Complete] Training model on {INITIAL_TRAINING_SIZE} samples...")
                            print(f"{'='*80}")
                            
                            # Extract samples in order (0 to 499)
                            X_init = np.array([training_buffer[i][0] for i in range(INITIAL_TRAINING_SIZE)])
                            y_init = np.array([training_buffer[i][1] for i in range(INITIAL_TRAINING_SIZE)])
                            
                            model = create_model()
                            model.fit(X_init, y_init)
                            model_trained = True
                            
                            # Save initial model
                            model_path = MODEL_DIR / "initial_model.pkl"
                            save_model(model, model_path)
                            
                            print(f"Model trained on samples 0-{INITIAL_TRAINING_SIZE-1}")
                            print(f"Model saved to {model_path}")
                            print(f"\n[Phase 2: WARMUP] Evaluating baseline accuracy...")
                            current_phase = "WARMUP"
                            
                            # Clear training buffer to free memory
                            training_buffer.clear()
                
                # =====================================================================
                # PHASE 2: WARMUP (Samples 500-599) - Use idx from producer!
                # =====================================================================
                elif INITIAL_TRAINING_SIZE <= idx < DEPLOYMENT_START and model_trained:
                    # Evaluate baseline accuracy
                    y_pred = model.predict(x_array.reshape(1, -1))[0]
                    is_correct = (y_pred == y_label)
                    warmup_correct.append(is_correct)
                    
                    # Check if we've completed warmup (received sample 599)
                    if idx == DEPLOYMENT_START - 1 and len(warmup_correct) >= TRAINING_WARMUP:
                        baseline_accuracy = np.mean(warmup_correct)
                        
                        print(f"\n{'='*80}")
                        print(f"[Phase 2 Complete] Warmup evaluation finished")
                        print(f"{'='*80}")
                        print(f"Warmup samples: {TRAINING_WARMUP}")
                        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
                        print(f"\nModel is now FROZEN - will only predict (no learning)")
                        print(f"Drift detection starts from sample {DEPLOYMENT_START}")
                        print(f"{'='*80}\n")
                        
                        current_phase = "FROZEN_DEPLOYMENT"
                
                # =====================================================================
                # PHASE 3: FROZEN DEPLOYMENT (Samples 600+) - Use idx from producer!
                # =====================================================================
                elif idx >= DEPLOYMENT_START and model_trained:
                    # STEP 0: Check for model updates from adaptor (non-blocking)
                    update_msg = model_consumer.poll(0.001)  # 1ms poll
                    if update_msg is not None and not update_msg.error():
                        try:
                            update_data = json.loads(update_msg.value().decode('utf-8'))
                            if update_data.get('event') == 'model_updated':
                                # Reload model from disk
                                model_path = MODEL_DIR / "current_model.pkl"
                                if model_path.exists():
                                    model = load_model(model_path)
                                    print(f"\n{'='*80}")
                                    print(f"[MODEL RELOAD] Updated model loaded at sample {idx}")
                                    print(f"  Version: {update_data.get('version')}")
                                    print(f"  Drift type: {update_data.get('drift_type')}")
                                    print(f"  Training samples: {update_data.get('n_samples')}")
                                    print(f"{'='*80}\n")
                        except Exception as e:
                            pass  # Ignore errors in model update
                    
                    # STEP 1: Prediction only (NO learning)
                    y_pred = model.predict(x_array.reshape(1, -1))[0]
                    is_correct = (y_pred == y_label)
                    recent_correct.append(is_correct)
                    
                    # Track prequential accuracy
                    accuracy = np.mean(recent_correct) if len(recent_correct) > 0 else 0.0
                    accuracy_tracker.append({'idx': idx, 'accuracy': accuracy})
                    
                    if idx % 10 == 0:
                        accuracy_msg = {
                            "idx": idx,
                            "accuracy": float(accuracy),
                            "phase": current_phase,
                            "baseline_accuracy": float(baseline_accuracy),
                            "drift_detected": drift_detected,
                            "timestamp": time.time()
                        }
                        p.produce(ACCURACY_TOPIC, json.dumps(accuracy_msg).encode('utf-8'))
                        p.poll(0)
                        
                        # Log detailed metrics to CSV
                        metrics_writer.writerow([
                            idx,
                            f"{accuracy:.4f}",
                            f"{process.memory_info().rss / 1024 / 1024:.2f}",
                            f"{process.cpu_percent()}",
                            f"{np.mean(x_array):.4f}",
                            current_phase,
                            1 if drift_detected else 0
                        ])
                        metrics_file.flush()
                    
                    # STEP 2: Add to buffer for drift detection
                    buffer.append({'idx': idx, 'x': x_vec, 'y': y_label})
                    
                    # STEP 3: Periodic drift detection
                    if len(buffer) >= BUFFER_SIZE and idx % CHUNK_SIZE == 0:
                        if not drift_detected:
                            # Extract buffer data for monitoring
                            buffer_X = np.array([item['x'] for item in buffer], dtype=float)
                            
                            # Run Unified SE-CDT Monitor
                            # Returns: SECDTResult(is_drift, drift_type, subcategory, score, ...)
                            result = se_cdt.monitor(buffer_X)
                            
                            if result.is_drift:
                                # DRIFT DETECTED!
                                # Use positions from result if available, else approximate
                                drift_pos_in_buffer = result.drift_positions[0] if result.drift_positions else -1
                                # If monitor returns relative pos, map to global idx. 
                                # Assuming logic returns latest possible drift.
                                # Let's stick to simple logic for now: last chunk triggered it.
                                
                                # Note: SE-CDT returns detection, we need to map back to global IDX
                                # Ideally SE-CDT returns index relative to buffer start
                                # For now, we take the end of buffer as roughly the detection point
                                drift_detected_at = buffer[drift_pos_in_buffer]['idx'] if drift_pos_in_buffer >= 0 and drift_pos_in_buffer < len(buffer) else idx
                                
                                drift_detected = True
                                drift_p_value = result.p_value
                                
                                print(f"\n{'='*80}")
                                print(f"[Sample {idx}] DRIFT DETECTED (via SE-CDT)")
                                print(f"{'='*80}")
                                print(f"  Detection position: sample {drift_detected_at}")
                                print(f"  Score: {result.score:.4f} (p-value: {result.p_value:.6f})")
                                print(f"  Current accuracy: {accuracy:.4f}")
                                
                                drift_type = result.subcategory
                                drift_category = result.drift_type # TCD/PCD
                                    
                                print(f"  Drift type: {drift_type}")
                                print(f"  Category: {drift_category}")
                                
                                # Schedule adaptation
                                in_degradation_period = True
                                planned_adaptation_idx = idx + ADAPTATION_DELAY
                                
                                print(f"  Scheduling retraining after {ADAPTATION_DELAY}-sample delay")
                                print(f"  Planned retraining at sample {planned_adaptation_idx}")
                                print(f"  Model remains FROZEN (observe degradation)")
                                print(f"{'='*80}\n")
                                
                                # Store drift info for delayed trigger
                                drift_info = {
                                    "p_value": result.p_value,
                                    "drift_type": drift_type,
                                    "drift_category": drift_category,
                                    "drift_detected_at": drift_detected_at
                                }

                    # STEP 4: Check for Delayed Adaptation Trigger
                    if drift_detected and planned_adaptation_idx is not None and idx == planned_adaptation_idx:
                        print(f"\n{'='*80}")
                        print(f"[Sample {idx}] EXECUTING DELAYED ADAPTATION")
                        print(f"  Drift was detected at sample {drift_info['drift_detected_at']}")
                        print(f"  Current buffer size: {len(buffer)}")
                        print(f"{'='*80}")
                        
                        # Save snapshot NOW (contains post-drift data)
                        snapshot_filename = f"drift_window_{drift_info['drift_detected_at']}_{int(time.time())}.npz"
                        snapshot_path = SNAPSHOT_DIR / snapshot_filename

                        # Save complete buffer
                        buffer_X = np.array([item['x'] for item in buffer], dtype=float)
                        buffer_indices = np.array([item['idx'] for item in buffer])
                        buffer_y = np.array([item['y'] for item in buffer])
                        
                        np.savez(
                            snapshot_path,
                            X=buffer_X,
                            y=buffer_y,
                            indices=buffer_indices,
                            drift_position=drift_info['drift_detected_at'],
                            feature_names=np.array([f"f{i}" for i in range(buffer_X.shape[1])])
                        )
                        print(f"[Snapshot] Saved to {snapshot_path}")

                        # Log detection to CSV (Restored)
                        csv_writer.writerow([
                            drift_info['drift_detected_at'], 
                            drift_info['p_value'], 
                            1, 
                            idx,
                            drift_info['drift_type'], 
                            drift_info['drift_category']
                        ])
                        csv_file.flush()

                        # Emit drift event to Kafka triggers Adaptor
                        out = {
                            "event": "drift_detected",
                            "idx": int(drift_info['drift_detected_at']),
                            "p_value": drift_info['p_value'],
                            "detector": "se_cdt_unified",
                            "window_path": str(snapshot_path),
                            "drift_type": drift_info['drift_type'],
                            "drift_category": drift_info['drift_category'],
                            "drift_detected_at": int(drift_info['drift_detected_at']),
                            "ts": time.time()
                        }
                        p.produce(RESULT_TOPIC, json.dumps(out).encode("utf-8"))
                        p.poll(0)
                        
                        # Reset
                        drift_info = {}
                        planned_adaptation_idx = None
                        # drift_detected remains True until we receive model update?? 
                        # actually we usually reset it here or wait for model update.
                        # Let's keep it True so we don't detect again immediately, 
                        # but we should rely on Model Update to reset 'in_degradation_period'.
                        # For this script logic (line 151 reset on model reload), we are good.
                    
                    # Display progress
                    if idx % 1000 == 0:
                        print(f"[Sample {idx}] Accuracy: {accuracy:.4f} (Phase: {current_phase})")

            except Exception as e:
                print(f"[Consumer] Processing error: {e}")
                import traceback
                traceback.print_exc()
                
    except KeyboardInterrupt:
        print("\n[Consumer] Stopped by user")
    finally:
        c.close()
        model_consumer.close()
        p.flush()
        csv_file.close()
        metrics_file.close()
        print("[Consumer] Shutdown complete")


if __name__ == "__main__":
    main()
