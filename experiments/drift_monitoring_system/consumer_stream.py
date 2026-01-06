"""
Consumer matching DriftMonitoring.ipynb notebook workflow.

4-Phase Lifecycle:
  Phase 1 (samples 0-499): PRETRAINING - Collect and batch train model
  Phase 2 (samples 500-599): WARMUP - Evaluate baseline accuracy  
  Phase 3 (samples 600+): FROZEN DEPLOYMENT - Only predict, detect drift
  Phase 4: ADAPTATION - Triggered by drift detection
"""
import json, os, time, csv, pickle
from collections import deque
import numpy as np
from confluent_kafka import Consumer, Producer
import sys
from pathlib import Path

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

# Make experiments/backup importable for ow_mmd
REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_DD_DIR = REPO_ROOT / "experiments" / "backup"
if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.append(str(SHAPE_DD_DIR))

# Use ShapeDD_OW_MMD (best performer in benchmark - F1=0.623)
from ow_mmd import shapedd_ow_mmd_buffer as shapedd_detect

# Import drift type classifier
from drift_type_classifier import classify_drift_at_detection, DriftTypeConfig

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
        'group.id': f'{group_id}-model-updates',
        'auto.offset.reset': 'latest',  # Only new updates
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

    # Prepare CSV log
    csv_path = os.getenv("SHAPEDD_LOG", SHAPEDD_LOG)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["detection_idx", "p_value", "drift", "buffer_end_idx", "drift_type", "drift_category"])

    # Drift type classifier config
    drift_type_cfg = DriftTypeConfig(
        w_ref=250,
        w_basic=100,
        sudden_len_thresh=250
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
                    
                    # Publish accuracy metrics to Kafka (every 10 samples for real-time viz)
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
                    
                    # STEP 2: Add to buffer for drift detection
                    buffer.append({'idx': idx, 'x': x_vec, 'y': y_label})
                    
                    # STEP 3: Periodic drift detection
                    if len(buffer) >= BUFFER_SIZE and idx % CHUNK_SIZE == 0:
                        if not drift_detected:
                            # Extract buffer data
                            buffer_list = list(buffer)
                            buffer_X = np.array([item['x'] for item in buffer_list], dtype=float)
                            buffer_indices = np.array([item['idx'] for item in buffer_list])
                            buffer_y = np.array([item['y'] for item in buffer_list])
                            
                            # Run ShapeDD_OW_MMD (best performer: F1=0.623, high precision)
                            shp_results = shapedd_detect(buffer_X, l1=SHAPE_L1, l2=SHAPE_L2)
                            
                            # Check recent chunk for drift
                            chunk_start = max(0, len(buffer_X) - CHUNK_SIZE)
                            chunk_pvals = shp_results[chunk_start:, 2]
                            p_min = float(chunk_pvals.min())
                            
                            if p_min < DRIFT_ALPHA:
                                # DRIFT DETECTED!
                                det_pos_in_chunk = int(np.argmin(chunk_pvals))
                                drift_detected_at = int(buffer_indices[chunk_start + det_pos_in_chunk])
                                drift_detected = True
                                drift_p_value = p_min
                                
                                print(f"\n{'='*80}")
                                print(f"[Sample {idx}] DRIFT DETECTED")
                                print(f"{'='*80}")
                                print(f"  Detection position: sample {drift_detected_at}")
                                print(f"  p-value: {p_min:.6f}")
                                print(f"  Current accuracy: {accuracy:.4f}")
                                
                                # Classify drift type
                                drift_pos_in_buffer = np.where(buffer_indices == drift_detected_at)[0]
                                if len(drift_pos_in_buffer) > 0:
                                    drift_idx_in_buffer = int(drift_pos_in_buffer[0])
                                    drift_type_result = classify_drift_at_detection(
                                        X=buffer_X,
                                        drift_idx=drift_idx_in_buffer,
                                            cfg=drift_type_cfg
                                        )
                                    drift_type = drift_type_result['subcategory']
                                    
                                    print(f"  Drift type: {drift_type}")
                                    print(f"  Category: {drift_type_result['category']}")
                                
                                # Schedule adaptation
                                in_degradation_period = True
                                planned_adaptation_idx = idx + ADAPTATION_DELAY
                                
                                print(f"  Scheduling retraining after {ADAPTATION_DELAY}-sample delay")
                                print(f"  Planned retraining at sample {planned_adaptation_idx}")
                                print(f"  Model remains FROZEN (observe degradation)")
                                print(f"{'='*80}\n")
                                
                                # Save snapshot with INDICES and DRIFT POSITION
                                snapshot_filename = f"drift_window_{drift_detected_at}_{int(time.time())}.npz"
                                snapshot_path = SNAPSHOT_DIR / snapshot_filename

                                # Save complete buffer with metadata
                                np.savez(
                                    snapshot_path,
                                    X=buffer_X,
                                    y=buffer_y,
                                    indices=buffer_indices,
                                    drift_position=drift_detected_at,  # CRITICAL: Mark drift position
                                    feature_names=np.array([f"f{i}" for i in range(buffer_X.shape[1])])
                                )
                                print(f"[Snapshot] Saved to {snapshot_path}")
                                print(f"  Shape: {buffer_X.shape}")
                                print(f"  Drift position: {drift_detected_at}")
                                print(f"  Index range: {buffer_indices[0]} to {buffer_indices[-1]}\n")

                                # Log detection
                                csv_writer.writerow([
                                    drift_detected_at, p_min, 1, buffer_indices[-1],
                                    drift_type, drift_type_result.get('category', 'undetermined')
                                ])
                                csv_file.flush()

                                # Emit drift event to Kafka
                                out = {
                                    "event": "drift_detected",
                                    "idx": int(drift_detected_at),
                                    "p_value": p_min,
                                    "detector": "shapedd",
                                    "window_path": str(snapshot_path),
                                    "drift_type": drift_type,
                                    "drift_category": drift_type_result.get('category', 'undetermined'),
                                    "drift_detected_at": int(drift_detected_at),  # Explicit position
                                    "ts": time.time()
                                }
                                p.produce(RESULT_TOPIC, json.dumps(out).encode("utf-8"))
                                p.poll(0)
                    
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
        print("[Consumer] Shutdown complete")


if __name__ == "__main__":
    main()
