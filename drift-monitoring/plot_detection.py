import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from confluent_kafka import Consumer

# Import shared configuration
from config import BROKERS, TOPIC, SHAPEDD_LOG

def load_detection_results(log_path):
    """Load drift detection results from CSV log."""
    det_idx = []
    with open(log_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                d = row.get("detection_idx", "").replace("\x00", "").strip()
                if d:
                    det_idx.append(int(float(d)))
            except Exception:
                continue
    return np.array(det_idx)

def load_streaming_data(brokers, topic, max_samples=20000):
    """Load streaming data with drift indicators from Kafka topic."""
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
                drift = data.get('drift', 0)  # True drift indicator from gen_random
                if x_vec and idx >= 0:
                    data_points.append({'idx': idx, 'x': x_vec, 'drift': drift})
            except:
                continue
    finally:
        consumer.close()
    
    if not data_points:
        return np.array([]), np.array([]), np.array([])
    
    # Sort by index and extract data
    data_points.sort(key=lambda d: d['idx'])
    indices = np.array([d['idx'] for d in data_points])
    features = np.array([d['x'] for d in data_points])
    drift_indicators = np.array([d['drift'] for d in data_points])
    return indices, features, drift_indicators

def main():
    # Load detection results
    log_path = os.getenv("SHAPEDD_LOG", SHAPEDD_LOG)
    det_idx = []
    if os.path.exists(log_path):
        det_idx = load_detection_results(log_path)
        print(f"Loaded {len(det_idx)} detections from {log_path}")
    else:
        print(f"No log file found at {log_path}, showing data only")
    
    # Load streaming data
    brokers = os.getenv("BROKERS", BROKERS)
    topic = os.getenv("TOPIC", TOPIC)
    print(f"Loading streaming data from {brokers}/{topic}...")
    data_idx, features, drift_indicators = load_streaming_data(brokers, topic, max_samples=20000)
    
    if len(features) == 0:
        print("No streaming data available")
        return
    
    # Create simple 2-plot visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Data with true drift points (from gen_random)
    ax1.scatter(range(len(features)), features[:, 0], c=drift_indicators, 
               cmap='coolwarm', alpha=0.7, s=15)
    ax1.set_title('Data Stream with True Drift Points (from gen_random)')
    ax1.set_ylabel('Feature Value (First Dimension)')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for drift indicator
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('True Drift Level', rotation=270, labelpad=15)
    
    # Plot 2: Detection points (aligned with data)
    detection_signal = np.zeros(len(features))
    if len(det_idx) > 0:
        # Mark detection positions
        for d_idx in det_idx:
            # Find closest data point to detection
            closest_pos = np.argmin(np.abs(data_idx - d_idx))
            if closest_pos < len(detection_signal):
                detection_signal[closest_pos] = 1
        
        # Plot detection signal
        ax2.scatter(range(len(features)), detection_signal, c=detection_signal, 
                   cmap='Reds', alpha=0.8, s=50)
        ax2.set_title(f'ShapeDD Detection Points ({len(det_idx)} detections)')
        
        # Add vertical lines for detections
        for d_idx in det_idx:
            closest_pos = np.argmin(np.abs(data_idx - d_idx))
            if closest_pos < len(features):
                ax2.axvline(x=closest_pos, color='red', linestyle='--', alpha=0.7)
    else:
        ax2.scatter(range(len(features)), detection_signal, c='gray', alpha=0.3, s=15)
        ax2.set_title('ShapeDD Detection Points (No detections yet)')
    
    ax2.set_ylabel('Detection (0=No, 1=Yes)')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Print summary
    true_drift_points = np.sum(drift_indicators > 0)
    print(f"\nSummary:")
    print(f"  Total samples: {len(features):,}")
    print(f"  True drift points: {true_drift_points}")
    print(f"  Detected drift points: {len(det_idx)}")
    if len(det_idx) > 0:
        print(f"  Detection positions: {list(det_idx)}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


