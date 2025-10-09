import os
import json
import time
import signal
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

# Kafka client: kafka-python
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# River (online ML)
from river import preprocessing, linear_model, optim

# ------------------ ENV & defaults ------------------

RESULT_TOPIC = os.getenv("RESULT_TOPIC", "drift.results")
MODEL_UPDATED_TOPIC = os.getenv("MODEL_UPDATED_TOPIC", "model.updated")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")
GROUP_ID = os.getenv("GROUP_ID", "adaptor-group")

SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", "./snapshots"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "./models/current_model.pkl"))
TASK_TYPE = os.getenv("TASK_TYPE", "classification")  # "classification" | "regression"

# Nếu muốn bỏ qua update khi snapshot không có nhãn:
ALLOW_UNLABELED_UPDATE = os.getenv("ALLOW_UNLABELED_UPDATE", "false").lower() == "true"

# ------------------ utils ------------------

def log(msg: str):
    print(f"[Adaptor] {msg}", flush=True)

def err(msg: str):
    print(f"[Adaptor][ERROR] {msg}", flush=True)

def make_model():
    """Tạo River pipeline cơ bản theo loại bài toán."""
    if TASK_TYPE == "classification":
        return preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(0.01), l2=1e-4
        )
    # regression
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.01), l2=1e-4
    )

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def _to_records(X: np.ndarray, feature_names: Optional[List[str]] = None):
    """Chuyển numpy -> list[dict] (kiểu River)."""
    n, d = X.shape
    if feature_names is None:
        feature_names = [str(j) for j in range(d)]
    records = []
    for i in range(n):
        rec = {feature_names[j]: float(X[i, j]) for j in range(d)}
        records.append(rec)
    return records

def learn_many(model, X: np.ndarray, y: Optional[np.ndarray], feature_names: Optional[List[str]] = None):
    """Online update từng mẫu cho River model."""
    if y is None:
        log("Snapshot không có nhãn (y). Performing unsupervised adaptation using data statistics.")
        # For unsupervised drift, update the scaler with new data distribution
        # This allows the model to adapt to new data characteristics
        if ALLOW_UNLABELED_UPDATE:
            Xrecs = _to_records(X, feature_names)
            # Update just the scaler (preprocessing) part without labels
            for xi in Xrecs:
                # Transform to update internal statistics
                _ = model.transform_one(xi)
        return model

    Xrecs = _to_records(X, feature_names)
    for xi, yi in zip(Xrecs, y):
        model = model.learn_one(xi, yi)
    return model

def load_snapshot(path: Path):
    """Đọc file .npz snapshot."""
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"] if "y" in data.files else None
    feature_names = data["feature_names"].tolist() if "feature_names" in data.files else None
    return X, y, feature_names

def publish_model_updated(prod: KafkaProducer, model_version: int, extra: dict):
    payload = {
        "event": "model_updated",
        "version": model_version,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    try:
        prod.send(MODEL_UPDATED_TOPIC, value=payload)
        prod.flush(5)
        log(f"Published model_updated v{model_version} -> {MODEL_UPDATED_TOPIC}")
    except KafkaError as e:
        err(f"Gửi model_updated thất bại: {e}")

# ------------------ main loop ------------------

def main():
    # Kafka IO
    consumer = KafkaConsumer(
        RESULT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=0,
        heartbeat_interval_ms=3000,
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda d: json.dumps(d).encode("utf-8"),
        linger_ms=50,
        acks="all",
        retries=5,
    )

    # Load or init model
    if MODEL_PATH.exists():
        model = load_model()
        log(f"Nạp model từ {MODEL_PATH}")
    else:
        model = make_model()
        log("Chưa có model sẵn — khởi tạo model mới.")

    # versioning đơn giản: dựa trên mtime file
    model_version = int(MODEL_PATH.stat().st_mtime) if MODEL_PATH.exists() else 0

    running = True
    def _stop(*_):
        nonlocal running
        running = False
        log("Stopping...")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log(f"Listening events from topic '{RESULT_TOPIC}' (bootstrap: {KAFKA_BOOTSTRAP})")

    while running:
        try:
            records_map = consumer.poll(timeout_ms=1000, max_records=100)
            if not records_map:
                continue

            for tp, records in records_map.items():
                for rec in records:
                    evt = rec.value
                    if not isinstance(evt, dict):
                        continue
                    if evt.get("event") != "drift_detected":
                        # có thể hỗ trợ "bootstrap_ready" v.v. trong tương lai
                        continue

                    idx = evt.get("idx")
                    pval = evt.get("p_value")
                    wpath = evt.get("window_path")
                    detector = evt.get("detector", "unknown")

                    if not wpath:
                        err(f"Drift@{idx} (p={pval}) nhưng thiếu window_path.")
                        continue

                    # Hỗ trợ path tương đối (dưới SNAPSHOT_DIR) hoặc tuyệt đối
                    snap_path = Path(wpath)
                    if not snap_path.is_absolute():
                        snap_path = SNAPSHOT_DIR / snap_path.name

                    if not snap_path.exists():
                        err(f"Snapshot không tồn tại: {snap_path}")
                        continue

                    t0 = time.time()
                    try:
                        X, y, feature_names = load_snapshot(snap_path)
                    except Exception as e:
                        err(f"Lỗi đọc snapshot {snap_path}: {e}")
                        continue

                    n = int(X.shape[0]) if isinstance(X, np.ndarray) else 0
                    if n == 0:
                        err(f"Snapshot rỗng: {snap_path}")
                        continue

                    # cập nhật model
                    model = learn_many(model, X, y, feature_names)
                    save_model(model)
                    dt = time.time() - t0

                    # update version
                    model_version = int(MODEL_PATH.stat().st_mtime)

                    log(f"UPDATED model v{model_version} on window '{snap_path.name}' "
                        f"(n={n}, p={pval}, detector={detector}) in {dt:.2f}s -> {MODEL_PATH}")

                    # publish thông báo model_updated (optional)
                    publish_model_updated(
                        producer,
                        model_version,
                        extra={
                            "source_snapshot": str(snap_path),
                            "idx": idx,
                            "p_value": pval,
                            "detector": detector,
                            "n_samples": n,
                            "task": TASK_TYPE,
                        },
                    )

        except Exception as e:
            err(f"Loop error: {e}")
            time.sleep(1)

    consumer.close()
    try:
        producer.flush(2)
        producer.close()
    except Exception:
        pass
    log("Bye.")


if __name__ == "__main__":
    main()
