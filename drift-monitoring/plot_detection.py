import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from confluent_kafka import Consumer

def load_batch_results(log_path):
    """Load batch detection results from CSV."""
    idx, score, p_min, drift = [], [], [], []
    with open(log_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            idx.append(int(row["batch_end_idx"]))
            score.append(float(row["score"]))
            p_min.append(float(row.get("p_min", "nan")))
            drift.append(int(row["drift"]))
    return np.array(idx), np.array(score), np.array(p_min), np.array(drift)

def load_streaming_data(brokers, topic, max_samples=5000):
    """Load recent streaming data from Kafka topic."""
    consumer = Consumer({
        'bootstrap.servers': brokers,
        'group.id': 'plot-reader',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False
    })
    consumer.subscribe([topic])
    
    data_points = []
    try:
        while len(data_points) < max_samples:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                break
            if msg.error():
                continue
            try:
                data = json.loads(msg.value().decode('utf-8'))
                x_vec = data.get('x', [])
                idx = data.get('idx', -1)
                if x_vec and idx >= 0:
                    data_points.append({'idx': idx, 'x': x_vec})
            except:
                continue
    finally:
        consumer.close()
    
    if not data_points:
        return np.array([]), np.array([])
    
    # Sort by index and extract features
    data_points.sort(key=lambda d: d['idx'])
    indices = np.array([d['idx'] for d in data_points])
    features = np.array([d['x'] for d in data_points])
    return indices, features

def main():
    # Load batch detection results
    log_path = os.getenv("SHAPEDD_LOG", "shapedd_batches.csv")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    batch_idx, score, p_min, drift = load_batch_results(log_path)
    
    # Load streaming data for visualization
    brokers = os.getenv("BROKERS", "localhost:19092")
    topic = os.getenv("TOPIC", "sensor.stream")
    print(f"Loading streaming data from {brokers}/{topic}...")
    data_idx, features = load_streaming_data(brokers, topic, max_samples=5000)
    
    # Create visualization similar to shape_dd notebook
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original data stream (first dimension) with drift detection
    if len(features) > 0:
        # Color points based on detected drift batches
        colors = np.zeros(len(data_idx))
        for i, b_idx in enumerate(batch_idx[drift == 1]):
            # Mark points in drift batches with different color
            mask = (data_idx >= b_idx - 250) & (data_idx <= b_idx)
            colors[mask] = 1
        
        scatter = axes[0].scatter(data_idx, features[:, 0], c=colors, 
                                cmap='coolwarm', alpha=0.6, s=2)
        axes[0].set_title('First Dimension with Detected Drift')
        axes[0].set_ylabel('Feature 1 Value')
        axes[0].grid(True, alpha=0.3)
        
        # Mark drift detection points
        drift_batches = batch_idx[drift == 1]
        if len(drift_batches) > 0:
            axes[0].vlines(drift_batches, ymin=features[:, 0].min(), ymax=features[:, 0].max(), 
                         colors="red", linestyles="--", alpha=0.7, label="Drift detected")
            axes[0].legend()
    
    # Plot 2: Shape-adaptive scores
    axes[1].plot(batch_idx, score, 'o-', color="#1f77b4", label="Max peak score")
    axes[1].scatter(batch_idx[drift==1], score[drift==1], color="red", s=50, 
                   label="DRIFT detected", zorder=5)
    axes[1].set_title('ShapeDD Scores by Batch')
    axes[1].set_ylabel('Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: P-values
    if np.isfinite(p_min).any():
        axes[2].plot(batch_idx, p_min, 'o-', color="#2ca02c", label="Min p-value")
        axes[2].axhline(0.05, color="#d62728", linestyle="--", alpha=0.7, label="α=0.05")
        axes[2].scatter(batch_idx[drift==1], p_min[drift==1], color="red", s=50, 
                       label="DRIFT detected", zorder=5)
        axes[2].set_title('ShapeDD P-values by Batch')
        axes[2].set_xlabel('Sample Index (Batch End)')
        axes[2].set_ylabel('P-value')
        axes[2].set_yscale('log')
        axes[2].grid(True, which="both", alpha=0.3)
        axes[2].legend()
    else:
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


