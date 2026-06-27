import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from confluent_kafka import Consumer
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import shared configuration
from experiments.monitoring.config import BROKERS, TOPIC, SHAPEDD_LOG

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
    
    # Create simple 2-plot visualization with proper alignment
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Use stream indices for consistent x-axis across both plots
    x_axis = data_idx  # Use actual stream indices
    
    # Plot 1: Data with true drift points (from gen_random)
    scatter1 = ax1.scatter(x_axis, features[:, 0], c=drift_indicators, 
                          cmap='coolwarm', alpha=0.7, s=15)
    ax1.set_title('Data Stream with True Drift Points (from gen_random)')
    ax1.set_ylabel('Feature Value (First Dimension)')
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for true drift transitions
    drift_transitions = np.where(np.diff(drift_indicators) != 0)[0] + 1
    for transition_idx in drift_transitions:
        if transition_idx < len(x_axis):
            stream_pos = x_axis[transition_idx]
            ax1.axvline(x=stream_pos, color='blue', linestyle='-', alpha=0.6, linewidth=2)
            ax1.text(stream_pos, ax1.get_ylim()[1]*0.9, f'True\n{stream_pos}', 
                    rotation=90, ha='center', va='top', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7))
    
    # Plot 2: Detection points (perfectly aligned using same x-axis)
    detection_signal = np.zeros(len(features))
    detection_x_positions = []
    
    if len(det_idx) > 0:
        # Mark detection positions in the data array
        for d_idx in det_idx:
            # Find closest data point to detection
            closest_pos = np.argmin(np.abs(data_idx - d_idx))
            if closest_pos < len(detection_signal):
                detection_signal[closest_pos] = 1
                detection_x_positions.append(x_axis[closest_pos])
        
        # Plot detection signal using same x-axis as data
        ax2.scatter(x_axis, detection_signal, c=detection_signal, 
                   cmap='Reds', alpha=0.8, s=50)
        ax2.set_title(f'ShapeDD Detection Points ({len(det_idx)} detections)')
        
        # Add vertical lines for detections at exact positions
        for det_x_pos in detection_x_positions:
            ax2.axvline(x=det_x_pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax2.text(det_x_pos, 0.8, f'Det\n{det_x_pos}', 
                    rotation=90, ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
    else:
        ax2.scatter(x_axis, detection_signal, c='gray', alpha=0.3, s=15)
        ax2.set_title('ShapeDD Detection Points (No detections yet)')
    
    ax2.set_ylabel('Detection')
    ax2.set_xlabel('Stream Index')
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
    
    # Ensure both plots have identical x-axis ranges for perfect alignment
    x_min, x_max = x_axis.min(), x_axis.max()
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    
    # Add matching grid lines for visual alignment verification
    ax1.grid(True, alpha=0.3, which='both')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add text annotation about alignment
    fig.suptitle('Drift Detection Analysis - Aligned Plots for Comparison', fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    # Save to plots directory
    from core.config import PLOTS_DIR
    output_path = PLOTS_DIR / "realtime_detection_snapshot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Snapshot saved to: {output_path}")
    # plt.show()

if __name__ == "__main__":
    main()


