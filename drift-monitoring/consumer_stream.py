import json, os, time, csv
from collections import deque
import numpy as np
from confluent_kafka import Consumer, Producer
from consumer_shapedd import shape_adaptive

BROKERS = os.getenv("BROKERS", "localhost:19092")
TOPIC = os.getenv("TOPIC", "sensor.stream")
GROUP_ID = os.getenv("GROUP_ID", "shapedd-detector")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "250"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
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

    print(f"[shapedd-batch] Brokers={BROKERS} Topic={TOPIC} Group={GROUP_ID} Batch={BATCH_SIZE} Th={DRIFT_THRESHOLD} alpha={DRIFT_PVALUE}")
    # Prepare CSV log for visualization
    csv_path = os.getenv("SHAPEDD_LOG", "shapedd_batches.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["batch_end_idx", "score", "p_min", "drift", "batch_size", "est_change_idx"])
    buf = []
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
                buf.append(x_vec)
                if len(buf) >= BATCH_SIZE:
                    X = np.asarray(buf, dtype=float)
                    # Align with experiment: l1=50, l2=BATCH_SIZE, n_perm=2500
                    res = shape_adaptive(X, 50, BATCH_SIZE, 2500)
                    # Use p-value rule (col 1): drift if min p-value < alpha
                    peak_vals = res[:, 0]
                    p_vals = res[:, 1]
                    peak_pos = int(np.argmax(peak_vals))
                    p_min_pos = int(np.argmin(p_vals))
                    score = float(peak_vals[peak_pos])
                    p_min = float(p_vals[p_min_pos])
                    drift = p_min < DRIFT_PVALUE
                    status = "DRIFT" if drift else "ok"
                    # Map estimated change point to stream index using p_min location
                    est_change_idx = int(idx - (BATCH_SIZE - 1) + p_min_pos)
                    print(f"[shapedd-batch] idx={idx} n={len(buf)} score={score:.4f} p_min={p_min:.4g} status={status} est_cp={est_change_idx}")
                    # Write batch summary for plotting
                    csv_writer.writerow([idx, score, p_min, int(drift), len(buf), est_change_idx])
                    csv_file.flush()
                    # Emit to Kafka result topic for Console visualization
                    out = {"batch_end_idx": idx, "score": score, "p_min": p_min, "alpha": DRIFT_PVALUE, "drift": bool(drift), "batch_size": len(X), "est_change_idx": est_change_idx, "ts": time.time()}
                    p.produce(RESULT_TOPIC, json.dumps(out).encode("utf-8"))
                    p.poll(0)
                    buf.clear()
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


