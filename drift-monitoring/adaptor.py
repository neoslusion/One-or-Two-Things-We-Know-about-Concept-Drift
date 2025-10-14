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

# Drift-type-specific adaptation
ENABLE_DRIFT_TYPE_ADAPTATION = os.getenv("ENABLE_DRIFT_TYPE_ADAPTATION", "true").lower() == "true"

# Model cache directory for recurring drift
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "./models/cache"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


def adapt_sudden_drift(model, X: np.ndarray, y: Optional[np.ndarray], feature_names: Optional[List[str]] = None):
    """
    Strategy for SUDDEN drift: Full model reset and retrain.

    Sudden drift indicates abrupt change - best to start fresh.
    """
    log("Strategy: SUDDEN drift → Full model reset and retrain")

    # Create fresh model
    new_model = make_model()

    # Warm start with drift window data
    if y is not None:
        new_model = learn_many(new_model, X, y, feature_names)
        log(f"  Retrained on {len(X)} samples from drift window")
    elif ALLOW_UNLABELED_UPDATE:
        # Unsupervised: at least update scaler
        new_model = learn_many(new_model, X, y, feature_names)
        log(f"  Updated scaler with {len(X)} unlabeled samples")

    return new_model


def adapt_incremental_drift(model, X: np.ndarray, y: Optional[np.ndarray], feature_names: Optional[List[str]] = None):
    """
    Strategy for INCREMENTAL drift: Continue gradual online updates.

    Incremental drift is monotonic and progressive - keep existing model and adapt gradually.
    """
    log("Strategy: INCREMENTAL drift → Gradual online updates (maintain model)")

    # Continue incremental learning with existing model
    model = learn_many(model, X, y, feature_names)
    log(f"  Applied incremental updates with {len(X)} samples")

    return model


def adapt_gradual_drift(model, X: np.ndarray, y: Optional[np.ndarray], feature_names: Optional[List[str]] = None):
    """
    Strategy for GRADUAL drift: Weighted updates with sample decay.

    Gradual drift is non-monotonic with oscillations - use weighted learning.
    """
    log("Strategy: GRADUAL drift → Weighted updates (recent samples prioritized)")

    if y is None and not ALLOW_UNLABELED_UPDATE:
        log("  No labels and unlabeled update disabled - skipping")
        return model

    # Apply weights: more recent samples get higher weight
    n = len(X)
    Xrecs = _to_records(X, feature_names)

    if y is not None:
        # Supervised: use weighted learning
        # Simple linear weighting: earlier samples get lower weight
        for i, (xi, yi) in enumerate(zip(Xrecs, y)):
            weight = (i + 1) / n  # 0.0 to 1.0
            # River doesn't directly support sample weights in learn_one
            # Approximation: repeat recent samples more
            if weight > 0.5:  # Focus on recent half
                model = model.learn_one(xi, yi)
        log(f"  Applied weighted updates focusing on recent {n//2} samples")
    else:
        # Unsupervised: update scaler with all samples
        model = learn_many(model, X, y, feature_names)
        log(f"  Updated scaler with {n} samples")

    return model


def adapt_recurrent_drift(model, X: np.ndarray, y: Optional[np.ndarray],
                          feature_names: Optional[List[str]] = None,
                          drift_idx: Optional[int] = None):
    """
    Strategy for RECURRENT drift: Use cached model if pattern repeats.

    Recurrent drift means return to previous distribution - try to reuse old models.
    """
    log("Strategy: RECURRENT drift → Check for cached model")

    # For now, simple implementation: check if we have any cached models
    # In production, you'd implement pattern matching (e.g., using distribution similarity)

    cached_models = list(MODEL_CACHE_DIR.glob("model_*.pkl"))

    if cached_models and y is not None:
        # TODO: Implement pattern matching to find best cached model
        # For now, use most recent cache
        latest_cache = max(cached_models, key=lambda p: p.stat().st_mtime)
        log(f"  Found cached model: {latest_cache.name}")

        try:
            with open(latest_cache, "rb") as f:
                cached_model = pickle.load(f)

            # Fine-tune cached model with current drift window
            cached_model = learn_many(cached_model, X, y, feature_names)
            log(f"  Loaded cached model and fine-tuned with {len(X)} samples")
            return cached_model

        except Exception as e:
            err(f"  Failed to load cached model: {e}")

    # Fallback: treat as incremental if no suitable cache
    log("  No suitable cached model found → fallback to incremental update")
    return adapt_incremental_drift(model, X, y, feature_names)


def adapt_blip_drift(model, X: np.ndarray, y: Optional[np.ndarray], feature_names: Optional[List[str]] = None):
    """
    Strategy for BLIP: Ignore or minimal update.

    Blip is a very short temporary anomaly - likely noise, don't overreact.
    """
    log("Strategy: BLIP drift → Minimal update (likely temporary noise)")

    # Option 1: Completely ignore
    # return model

    # Option 2: Very conservative update (only if labeled)
    if y is not None:
        # Only update with small subset to avoid overreaction
        n_samples = min(5, len(X))
        model = learn_many(model, X[:n_samples], y[:n_samples], feature_names)
        log(f"  Conservative update with {n_samples} samples only")
    else:
        log("  Ignoring unlabeled blip")

    return model


def cache_model(model, drift_idx: int):
    """Cache current model for potential reuse with recurring drift."""
    cache_path = MODEL_CACHE_DIR / f"model_{drift_idx}_{int(time.time())}.pkl"
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(model, f)
        log(f"  Cached model to {cache_path.name}")
    except Exception as e:
        err(f"  Failed to cache model: {e}")

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
    log(f"Drift-Type Adaptation: {'ENABLED' if ENABLE_DRIFT_TYPE_ADAPTATION else 'DISABLED'}")

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
                    drift_type = evt.get("drift_type", "undetermined")
                    drift_category = evt.get("drift_category", "undetermined")

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

                    # === DRIFT-TYPE-SPECIFIC ADAPTATION ===
                    log(f"Drift detected: type={drift_type}, category={drift_category}")

                    if ENABLE_DRIFT_TYPE_ADAPTATION and drift_type != "undetermined":
                        # Select strategy based on drift type
                        if drift_type == "sudden":
                            model = adapt_sudden_drift(model, X, y, feature_names)
                        elif drift_type == "incremental":
                            model = adapt_incremental_drift(model, X, y, feature_names)
                        elif drift_type == "gradual":
                            model = adapt_gradual_drift(model, X, y, feature_names)
                        elif drift_type == "recurrent":
                            model = adapt_recurrent_drift(model, X, y, feature_names, idx)
                            # Cache model for future recurrences
                            cache_model(model, idx)
                        elif drift_type == "blip":
                            model = adapt_blip_drift(model, X, y, feature_names)
                        else:
                            # Fallback to default
                            log(f"Unknown drift type '{drift_type}' → using default strategy")
                            model = learn_many(model, X, y, feature_names)
                    else:
                        # Fallback: default strategy (original behavior)
                        if not ENABLE_DRIFT_TYPE_ADAPTATION:
                            log("Drift-type adaptation disabled → using default strategy")
                        else:
                            log(f"Drift type undetermined → using default strategy")
                        model = learn_many(model, X, y, feature_names)

                    save_model(model)
                    dt = time.time() - t0

                    # update version
                    model_version = int(MODEL_PATH.stat().st_mtime)

                    log(f"UPDATED model v{model_version} on window '{snap_path.name}' "
                        f"(n={n}, p={pval}, type={drift_type}, detector={detector}) in {dt:.2f}s -> {MODEL_PATH}")

                    # publish thông báo model_updated (optional)
                    publish_model_updated(
                        producer,
                        model_version,
                        extra={
                            "source_snapshot": str(snap_path),
                            "idx": idx,
                            "p_value": pval,
                            "detector": detector,
                            "drift_type": drift_type,
                            "drift_category": drift_category,
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
