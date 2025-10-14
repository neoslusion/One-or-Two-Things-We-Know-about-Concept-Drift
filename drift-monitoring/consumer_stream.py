import json, os, time, csv
from collections import deque
import numpy as np
from confluent_kafka import Consumer, Producer
import sys
from pathlib import Path

# Import shared configuration
from config import (BUFFER_SIZE, CHUNK_SIZE, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM, 
                   DRIFT_PVALUE, BROKERS, TOPIC, GROUP_ID, RESULT_TOPIC, SHAPEDD_LOG)

# Snapshot directory for saving drift windows
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", "./snapshots"))
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Make experiments/backup importable for shape_dd
REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_DD_DIR = REPO_ROOT / "experiments" / "backup"
if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.append(str(SHAPE_DD_DIR))

from shape_dd import shape_adaptive as shape

# Import drift type classifier
from drift_type_classifier import classify_drift_at_detection, DriftTypeConfig

# Allow environment variable overrides
BROKERS = os.getenv("BROKERS", BROKERS)
TOPIC = os.getenv("TOPIC", TOPIC)
GROUP_ID = os.getenv("GROUP_ID", GROUP_ID)
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", str(BUFFER_SIZE)))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", str(CHUNK_SIZE)))
DRIFT_PVALUE = float(os.getenv("DRIFT_PVALUE", str(DRIFT_PVALUE)))
RESULT_TOPIC = os.getenv("RESULT_TOPIC", RESULT_TOPIC)

# Drift type classification configuration
ENABLE_DRIFT_TYPE_CLASSIFICATION = os.getenv("ENABLE_DRIFT_TYPE_CLASSIFICATION", "true").lower() == "true"
HISTORY_BUFFER_SIZE = int(os.getenv("HISTORY_BUFFER_SIZE", "500"))  # For drift type classification

def main():
    conf = {
        'bootstrap.servers': BROKERS,
        'group.id': GROUP_ID,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'auto.commit.interval.ms': 1000,
    }
    c = Consumer(conf)
    p = Producer({'bootstrap.servers': BROKERS})
    c.subscribe([TOPIC])

    print(f"[shapedd] Brokers={BROKERS} Topic={TOPIC} Group={GROUP_ID} Buffer={BUFFER_SIZE} Chunk={CHUNK_SIZE} alpha={DRIFT_PVALUE}")
    print(f"[shapedd] Drift Type Classification: {'ENABLED' if ENABLE_DRIFT_TYPE_CLASSIFICATION else 'DISABLED'}")

    # Prepare CSV log for visualization
    csv_path = os.getenv("SHAPEDD_LOG", SHAPEDD_LOG)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["detection_idx", "p_value", "drift", "buffer_end_idx", "drift_type", "drift_category"])

    # Initialize drift type classifier config
    drift_type_cfg = DriftTypeConfig(
        w_ref=200, 
        w_basic=50,
        sudden_len_thresh=150 # Inherited from notebook
    )

    # Maintain history buffer for drift type classification (circular buffer)
    history_buffer = deque(maxlen=HISTORY_BUFFER_SIZE)

    buffer = []  # Accumulate samples for processing
    sample_count = 0
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[shapedd] consumer error: {msg.error()}")
                continue
            try:
                data = json.loads(msg.value().decode("utf-8"))
                x_vec = data.get("x", [])
                idx = data.get("idx", -1)
                if not x_vec:
                    continue
                    
                buffer.append({"idx": idx, "x": x_vec})
                history_buffer.append({"idx": idx, "x": x_vec})  # Also add to history for classification
                sample_count += 1

                # Process every BUFFER_SIZE samples (like experiment processes full dataset)
                if len(buffer) >= BUFFER_SIZE:
                    print(f"[shapedd] Processing buffer of {len(buffer)} samples...")
                    
                    # Extract data matrix
                    X = np.array([item["x"] for item in buffer], dtype=float)
                    indices = np.array([item["idx"] for item in buffer])
                    
                    # Run shape on entire buffer (like experiment)
                    shp_full = shape(X, SHAPE_L1, SHAPE_L2, SHAPE_N_PERM)
                    
                    # Create batches like experiment
                    n_samples = len(X)
                    batches = []
                    for i in range(0, n_samples - CHUNK_SIZE + 1, CHUNK_SIZE):
                        batch_indices = np.arange(i, min(i + CHUNK_SIZE, n_samples))
                        batches.append(batch_indices)
                    
                    # Process each batch and check for drift
                    detection_count = 0
                    for b in batches:
                        # Extract p-values for this batch (column 2 in ConceptDrift_Pipeline!)
                        batch_pvals = shp_full[b, 2]  # Column 2 contains p-values
                        p_min = float(batch_pvals.min())
                        drift = p_min < DRIFT_PVALUE
                        
                        if drift:
                            # Find detection position (argmin of p-values within batch)
                            det_pos_in_batch = int(np.argmin(batch_pvals))
                            det_idx_in_buffer = b[det_pos_in_batch]  # Index in current buffer
                            det_idx = indices[det_idx_in_buffer]  # Map to original stream index

                            print(f"[shapedd] DRIFT detected at idx={det_idx} p_min={p_min:.6f}")

                            # === DRIFT TYPE CLASSIFICATION ===
                            drift_type_info = {"subcategory": "undetermined", "category": "undetermined", "note": ""}

                            if ENABLE_DRIFT_TYPE_CLASSIFICATION:
                                try:
                                    # Build classification history from history_buffer
                                    history_data = np.array([item["x"] for item in history_buffer], dtype=float)
                                    history_indices = np.array([item["idx"] for item in history_buffer])

                                    # Find drift position in history buffer
                                    drift_pos_in_history = np.where(history_indices == det_idx)[0]

                                    if len(drift_pos_in_history) > 0:
                                        drift_idx_in_history = int(drift_pos_in_history[0])

                                        # Classify drift type using history buffer
                                        drift_type_result = classify_drift_at_detection(
                                            X=history_data,
                                            drift_idx=drift_idx_in_history,
                                            cfg=drift_type_cfg
                                        )

                                        drift_type_info = drift_type_result
                                        print(f"[shapedd] Drift Type: {drift_type_result['subcategory']} "
                                              f"(category: {drift_type_result['category']}, "
                                              f"length: {drift_type_result['drift_length']})")
                                    else:
                                        drift_type_info["note"] = "Drift index not found in history buffer"
                                        print(f"[shapedd] Warning: Could not classify drift type - index not in history")

                                except Exception as e:
                                    drift_type_info["note"] = f"Classification error: {str(e)}"
                                    print(f"[shapedd] Error classifying drift type: {e}")

                            # Log detection with drift type
                            csv_writer.writerow([
                                det_idx,
                                p_min,
                                1,
                                indices[-1],
                                drift_type_info.get('subcategory', 'undetermined'),
                                drift_type_info.get('category', 'undetermined')
                            ])
                            csv_file.flush()

                            # Save drift window snapshot for adaptor
                            snapshot_filename = f"drift_window_{det_idx}_{int(time.time())}.npz"
                            snapshot_path = SNAPSHOT_DIR / snapshot_filename

                            # Extract window around drift point (use batch data)
                            window_X = X[b]
                            window_indices = indices[b]

                            # Save snapshot with feature data (no labels for unsupervised drift detection)
                            np.savez(
                                snapshot_path,
                                X=window_X,
                                indices=window_indices,
                                feature_names=np.array([f"f{i}" for i in range(window_X.shape[1])])
                            )
                            print(f"[shapedd] Saved snapshot to {snapshot_path}")

                            # Emit to Kafka with proper format for adaptor (including drift type)
                            out = {
                                "event": "drift_detected",
                                "idx": int(det_idx),
                                "p_value": p_min,
                                "alpha": DRIFT_PVALUE,
                                "drift": True,
                                "detector": "shapedd",
                                "window_path": str(snapshot_path),
                                "buffer_end_idx": int(indices[-1]),
                                "drift_type": drift_type_info.get('subcategory', 'undetermined'),
                                "drift_category": drift_type_info.get('category', 'undetermined'),
                                "drift_length": drift_type_info.get('drift_length', 0),
                                "classification_note": drift_type_info.get('note', ''),
                                "ts": time.time()
                            }
                            p.produce(RESULT_TOPIC, json.dumps(out).encode("utf-8"))
                            p.poll(0)
                            detection_count += 1
                    
                    print(f"[shapedd] Processed {len(batches)} batches from buffer, {detection_count} drifts detected")
                    buffer.clear()
            except Exception as e:
                print(f"[shapedd-batch] processing error: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        c.close()
        try:
            csv_file.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()


