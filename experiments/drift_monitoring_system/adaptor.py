import os
import json
import time
import signal
import pickle
from pathlib import Path

import numpy as np

# Kafka client: confluent_kafka (consistent with consumer)
from confluent_kafka import Consumer, Producer, KafkaError

# Sklearn for batch learning (matches MultiDetectorEvaluation.ipynb)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Import adaptation strategies
from adaptation_strategies import (
    adapt_sudden_drift,
    adapt_incremental_drift,
    adapt_gradual_drift,
    adapt_recurrent_drift,
    adapt_blip_drift,
    cache_model_with_distribution
)

# ------------------ ENV & defaults ------------------

# Import ADAPTATION_WINDOW from config
from config import ADAPTATION_WINDOW

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
    """
    Create sklearn Pipeline matching MultiDetectorEvaluation.ipynb.
    
    Uses batch learning with StandardScaler + LogisticRegression.
    Model is FROZEN after training (no online learning during deployment).
    """
    if TASK_TYPE == "classification":
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
    else:
        # For regression (not used in current experiments)
        from sklearn.linear_model import LinearRegression
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_snapshot(path: Path):
    """
    Load .npz snapshot file with drift metadata.
    
    Returns:
        tuple: (X, y, feature_names, indices, drift_position)
    """
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"] if "y" in data.files else None
    feature_names = data["feature_names"].tolist() if "feature_names" in data.files else None
    indices = data["indices"] if "indices" in data.files else None
    drift_position = int(data["drift_position"]) if "drift_position" in data.files else None
    return X, y, feature_names, indices, drift_position

def publish_model_updated(prod: Producer, model_version: int, extra: dict):
    payload = {
        "event": "model_updated",
        "version": model_version,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    try:
        prod.produce(MODEL_UPDATED_TOPIC, json.dumps(payload).encode('utf-8'))
        prod.flush()
        log(f"Published model_updated v{model_version} -> {MODEL_UPDATED_TOPIC}")
    except Exception as e:
        err(f"Failed to publish model_updated: {e}")

# ------------------ main loop ------------------

def main():
    # Kafka IO using confluent_kafka
    consumer_conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP,
        'group.id': GROUP_ID,
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True,
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([RESULT_TOPIC])

    producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP})

    # Load or init model
    if MODEL_PATH.exists():
        model = load_model()
        log(f"Loaded model from {MODEL_PATH}")
    else:
        model = make_model()
        log("No existing model — initialized new model")

    # Simple versioning based on file mtime
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
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                err(f"Consumer error: {msg.error()}")
                continue

            try:
                evt = json.loads(msg.value().decode('utf-8'))
                if not isinstance(evt, dict):
                    continue
                if evt.get("event") != "drift_detected":
                    continue

                idx = evt.get("idx")
                pval = evt.get("p_value")
                wpath = evt.get("window_path")
                detector = evt.get("detector", "unknown")
                drift_type = evt.get("drift_type", "undetermined")
                drift_category = evt.get("drift_category", "undetermined")

                if not wpath:
                    err(f"Drift@{idx} (p={pval}) missing window_path")
                    continue

                # Support relative or absolute paths
                snap_path = Path(wpath)
                if not snap_path.is_absolute():
                    snap_path = SNAPSHOT_DIR / snap_path.name

                if not snap_path.exists():
                    err(f"Snapshot not found: {snap_path}")
                    continue

                t0 = time.time()
                try:
                    X, y, feature_names, indices, drift_position = load_snapshot(snap_path)
                except Exception as e:
                    err(f"Error loading snapshot {snap_path}: {e}")
                    continue

                # === CRITICAL: FILTER TO POST-DRIFT DATA ONLY ===
                # Matching DriftMonitoring.ipynb to avoid pre-drift contamination
                if drift_position is not None and indices is not None:
                    # Filter to samples >= drift_position
                    post_drift_mask = indices >= drift_position
                    X_post = X[post_drift_mask]
                    y_post = y[post_drift_mask] if y is not None else None
                    
                    # Limit to adaptation window
                    if len(X_post) > ADAPTATION_WINDOW:
                        X_adapt = X_post[-ADAPTATION_WINDOW:]
                        y_adapt = y_post[-ADAPTATION_WINDOW:] if y_post is not None else None
                    else:
                        X_adapt = X_post
                        y_adapt = y_post
                    
                    # Verify no contamination
                    pre_drift_count = int(np.sum(~post_drift_mask))
                    post_drift_count = int(np.sum(post_drift_mask))
                    train_indices = indices[post_drift_mask]
                    if len(train_indices) > 0:
                        train_range = f"{train_indices[0]} to {train_indices[-1]}"
                    else:
                        train_range = "N/A"
                    
                    log(f"Snapshot loaded: {snap_path.name}")
                    log(f"  Total samples: {len(X)}")
                    log(f"  Drift position: {drift_position}")
                    log(f"  Pre-drift samples (filtered out): {pre_drift_count}")
                    log(f"  Post-drift samples: {post_drift_count}")
                    log(f"  Training data range: samples {train_range}")
                    log(f"  Adaptation samples: {len(X_adapt)}")
                    
                    if pre_drift_count > 0:
                        log(f"  Filtered {pre_drift_count} pre-drift samples to avoid contamination")
                else:
                    # Fallback: no drift position metadata, use all data
                    log(f"WARNING: No drift position metadata - using all snapshot data")
                    X_adapt = X
                    y_adapt = y

                n = int(X_adapt.shape[0]) if isinstance(X_adapt, np.ndarray) else 0
                if n == 0:
                    err(f"No post-drift samples available for adaptation")
                    continue

                # === DRIFT-TYPE-SPECIFIC ADAPTATION ===
                log(f"Drift detected: type={drift_type}, category={drift_category}")

                if ENABLE_DRIFT_TYPE_ADAPTATION and drift_type != "undetermined":
                    # Select strategy based on drift type (using FILTERED post-drift data)
                    if drift_type == "sudden":
                        model = adapt_sudden_drift(make_model, X_adapt, y_adapt, feature_names, ALLOW_UNLABELED_UPDATE)
                    elif drift_type == "incremental":
                        model = adapt_incremental_drift(model, X_adapt, y_adapt, feature_names)
                    elif drift_type == "gradual":
                        model = adapt_gradual_drift(model, X_adapt, y_adapt, feature_names)
                    elif drift_type == "recurrent":
                        model = adapt_recurrent_drift(model, X_adapt, y_adapt, feature_names, MODEL_CACHE_DIR)
                        # Cache model with distribution for future recurrences
                        cache_model_with_distribution(model, X_adapt, idx, MODEL_CACHE_DIR)
                    elif drift_type == "blip":
                        model = adapt_blip_drift(model, X_adapt, y_adapt, feature_names)
                    else:
                        log(f"Unknown drift type '{drift_type}' → incremental update")
                        model = adapt_incremental_drift(model, X_adapt, y_adapt, feature_names)
                else:
                    # Fallback: default incremental strategy
                    log("Drift-type undetermined → incremental update")
                    model = adapt_incremental_drift(model, X_adapt, y_adapt, feature_names)

                save_model(model)
                dt = time.time() - t0

                # Update version
                model_version = int(MODEL_PATH.stat().st_mtime)

                log(f"UPDATED model v{model_version} on '{snap_path.name}' "
                    f"(n={n}, p={pval}, type={drift_type}) in {dt:.2f}s")

                # Publish model_updated notification
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

            except json.JSONDecodeError as e:
                err(f"JSON decode error: {e}")

        except Exception as e:
            err(f"Loop error: {e}")
            time.sleep(1)

    consumer.close()
    producer.flush()
    log("Bye.")


if __name__ == "__main__":
    main()
