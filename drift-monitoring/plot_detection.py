import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from confluent_kafka import Consumer

def load_detection_results(log_path):
    det_idx, p_vals, drift_flags, buffer_ends = [], [], [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                # sanitize fields
                d = row.get("detection_idx", "").replace("\x00", "").strip()
                pv = row.get("p_value", "").replace("\x00", "").strip()
                df = row.get("drift", "").replace("\x00", "").strip()
                be = row.get("buffer_end_idx", "").replace("\x00", "").strip()
                if not d or not pv or not df or not be:
                    continue
                det_idx.append(int(float(d)))
                p_vals.append(float(pv))
                drift_flags.append(int(float(df)))
                buffer_ends.append(int(float(be)))
            except Exception:
                # skip malformed lines
                continue
    return np.array(det_idx), np.array(p_vals), np.array(drift_flags), np.array(buffer_ends)
    
def load_streaming_data(brokers, topic, max_samples=20000):
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
    # Load detection results
    log_path = os.getenv("SHAPEDD_LOG", "shapedd_batches.csv")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    det_idx, p_vals, drift_flags, buffer_ends = load_detection_results(log_path)
    
    # Load streaming data for visualization
    brokers = os.getenv("BROKERS", "localhost:19092")
    topic = os.getenv("TOPIC", "sensor.stream")
    print(f"Loading streaming data from {brokers}/{topic}...")
    data_idx, features = load_streaming_data(brokers, topic, max_samples=20000)
    
    # Create visualization similar to shape_dd notebook
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original data stream (first dimension) - match notebook exactly
    if len(features) > 0:
        # Create drift indicator: mark detected drift points
        drift_indicator = np.zeros(len(data_idx))
        
        # Mark detected drift points
        for d_idx in det_idx:
            # Find closest data point to detection
            closest_pos = np.argmin(np.abs(data_idx - d_idx))
            drift_indicator[closest_pos] = 1
        
        # Plot exactly like notebook: range(len(X)), X[:, 0], c=y
        axes[0].scatter(range(len(features)), features[:, 0], c=drift_indicator, 
                       cmap='coolwarm', alpha=0.6)
        axes[0].set_title('First Dimension with Detected Drift')
        axes[0].set_ylabel('Value')
        axes[0].grid(True)
        
        # Add vertical lines for drift detection points
        if len(det_idx) > 0:
            for d_idx in det_idx:
                # Find position in data array corresponding to detection
                pos_in_array = np.where(data_idx <= d_idx)[0]
                if len(pos_in_array) > 0:
                    axes[0].axvline(x=pos_in_array[-1], color='red', linestyle='--', alpha=0.7)
    
    # Plot 2: P-values for detections
    if len(det_idx) > 0 and np.isfinite(p_vals).any():
        axes[1].scatter(det_idx, p_vals, c='red', s=50, label='Detected drift p-values')
        axes[1].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='Threshold (0.05)')
        axes[1].set_title('ShapeDD Detection P-values')
        axes[1].set_xlabel('Sample Position')
        axes[1].set_ylabel('P-value')
        axes[1].set_yscale('log')
        axes[1].grid(True)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No drift detections yet', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('ShapeDD Detection P-values')
        axes[1].set_xlabel('Sample Position')
        axes[1].set_ylabel('P-value')
    
    # Hide the third subplot since we only need 2 plots like the notebook
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


