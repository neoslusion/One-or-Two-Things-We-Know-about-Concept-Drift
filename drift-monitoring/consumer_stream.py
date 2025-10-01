import json, os, time, csv
from collections import deque
import numpy as np
from confluent_kafka import Consumer, Producer
import sys
from pathlib import Path

# Make experiments/backup importable for shape_dd
REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_DD_DIR = REPO_ROOT / "experiments" / "backup"
if str(SHAPE_DD_DIR) not in sys.path:
    sys.path.append(str(SHAPE_DD_DIR))

from shape_dd import shape

BROKERS = os.getenv("BROKERS", "localhost:19092")
TOPIC = os.getenv("TOPIC", "sensor.stream")
GROUP_ID = os.getenv("GROUP_ID", "shapedd-detector")
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "10000"))  # Process every 10k samples
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))     # Batch size for shape algorithm
DRIFT_PVALUE = float(os.getenv("DRIFT_PVALUE", "0.05"))
RESULT_TOPIC = os.getenv("RESULT_TOPIC", "drift.results")

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
    # Prepare CSV log for visualization
    csv_path = os.getenv("SHAPEDD_LOG", "shapedd_batches.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["detection_idx", "p_value", "drift", "buffer_end_idx"])
    
    buffer = []  # Accumulate 10k samples
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
                sample_count += 1
                
                # Process every BUFFER_SIZE samples (like experiment processes full dataset)
                if len(buffer) >= BUFFER_SIZE:
                    print(f"[shapedd] Processing buffer of {len(buffer)} samples...")
                    
                    # Extract data matrix
                    X = np.array([item["x"] for item in buffer], dtype=float)
                    indices = np.array([item["idx"] for item in buffer])
                    
                    # Run shape on entire buffer (like experiment)
                    shp_full = shape(X, 50, CHUNK_SIZE, 2500)
                    
                    # Create batches like experiment
                    n_samples = len(X)
                    batches = []
                    for i in range(0, n_samples - CHUNK_SIZE + 1, CHUNK_SIZE):
                        batch_indices = np.arange(i, min(i + CHUNK_SIZE, n_samples))
                        batches.append(batch_indices)
                    
                    # Process each batch and check for drift
                    for b in batches:
                        # Extract p-values for this batch (column 2 in ConceptDrift_Pipeline!)
                        batch_pvals = shp_full[b, 2]  # Column 2 contains p-values
                        p_min = float(batch_pvals.min())
                        drift = p_min < DRIFT_PVALUE
                        
                        if drift:
                            # Find detection position (argmin of p-values within batch)
                            det_pos_in_batch = int(np.argmin(batch_pvals))
                            det_idx = indices[b[det_pos_in_batch]]  # Map to original stream index
                            
                            status = "DRIFT"
                            print(f"[shapedd] DRIFT detected at idx={det_idx} p_min={p_min:.6f}")
                            
                            # Log detection
                            csv_writer.writerow([det_idx, p_min, 1, indices[-1]])
                            csv_file.flush()
                            
                            # Emit to Kafka
                            out = {
                                "detection_idx": int(det_idx),
                                "p_value": p_min,
                                "alpha": DRIFT_PVALUE,
                                "drift": True,
                                "buffer_end_idx": int(indices[-1]),
                                "ts": time.time()
                            }
                            p.produce(RESULT_TOPIC, json.dumps(out).encode("utf-8"))
                            p.poll(0)
                    
                    print(f"[shapedd] Processed {len(batches)} batches from buffer")
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


