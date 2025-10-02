import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from confluent_kafka import Consumer

# Import shared configuration
from config import BROKERS, TOPIC, SHAPEDD_LOG, BUFFER_SIZE, DRIFT_PVALUE

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
    log_path = os.getenv("SHAPEDD_LOG", SHAPEDD_LOG)
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    det_idx, p_vals, drift_flags, buffer_ends = load_detection_results(log_path)
    
    # Load streaming data for visualization
    brokers = os.getenv("BROKERS", BROKERS)
    topic = os.getenv("TOPIC", TOPIC)
    print(f"Loading streaming data from {brokers}/{topic}...")
    data_idx, features = load_streaming_data(brokers, topic, max_samples=20000)
    
    # Create visualization similar to shape_dd notebook
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original data stream (first dimension) - match notebook exactly
    if len(features) > 0:
        # Create drift indicator: mark detected drift points
        drift_indicator = np.zeros(len(data_idx))
        
        # Mark detected drift points
        detection_positions = []
        for d_idx in det_idx:
            # Find closest data point to detection
            closest_pos = np.argmin(np.abs(data_idx - d_idx))
            drift_indicator[closest_pos] = 1
            detection_positions.append(closest_pos)
        
        # Plot exactly like notebook: range(len(X)), X[:, 0], c=y
        scatter = axes[0].scatter(range(len(features)), features[:, 0], c=drift_indicator, 
                       cmap='coolwarm', alpha=0.6, s=20)
        axes[0].set_title('Data Stream with Detected Drift Points')
        axes[0].set_ylabel('Feature Value (First Dimension)')
        axes[0].set_xlabel('Sample Index')
        axes[0].grid(True, alpha=0.3)
        
        # Add vertical lines for drift detection points with annotations
        if len(det_idx) > 0:
            for i, d_idx in enumerate(det_idx):
                # Find position in data array corresponding to detection
                pos_in_array = np.where(data_idx <= d_idx)[0]
                if len(pos_in_array) > 0:
                    pos = pos_in_array[-1]
                    axes[0].axvline(x=pos, color='red', linestyle='--', alpha=0.8, linewidth=2)
                    # Add detection annotation
                    axes[0].annotate(f'Drift {i+1}\n(idx={d_idx})', 
                                   xy=(pos, features[pos, 0]), 
                                   xytext=(10, 10), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                                   fontsize=8, color='white')
        
        # Add colorbar to explain drift indicator
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Drift Detected', rotation=270, labelpad=15)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['No Drift', 'Drift Detected'])
    else:
        axes[0].text(0.5, 0.5, 'No streaming data available', ha='center', va='center', 
                    transform=axes[0].transAxes, fontsize=14)
        axes[0].set_title('Data Stream with Detected Drift Points')
        axes[0].set_ylabel('Feature Value')
    
    # Plot 2: P-values for detections
    if len(det_idx) > 0 and np.isfinite(p_vals).any():
        # Filter out invalid p-values for better visualization
        valid_mask = np.isfinite(p_vals) & (p_vals > 0) & (p_vals <= 1)
        valid_det_idx = det_idx[valid_mask]
        valid_p_vals = p_vals[valid_mask]
        
        if len(valid_p_vals) > 0:
            # Color points based on significance level
            colors = ['red' if p < 0.05 else 'orange' for p in valid_p_vals]
            axes[1].scatter(valid_det_idx, valid_p_vals, c=colors, s=80, 
                           alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Add threshold line
            axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                           label='Significance Threshold (α=0.05)')
            
            # Add annotations for significant detections
            for i, (idx, p_val) in enumerate(zip(valid_det_idx, valid_p_vals)):
                if p_val < 0.05:
                    axes[1].annotate(f'p={p_val:.3f}', xy=(idx, p_val), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
        
        axes[1].set_title('ShapeDD Detection P-values (Statistical Significance)')
        axes[1].set_xlabel('Sample Position (Stream Index)')
        axes[1].set_ylabel('P-value (log scale)')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Set y-axis limits for better visualization
        if len(valid_p_vals) > 0:
            min_p = max(min(valid_p_vals) * 0.1, 1e-6)
            axes[1].set_ylim(min_p, 1.0)
    else:
        axes[1].text(0.5, 0.5, 'No drift detections with valid p-values yet', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('ShapeDD Detection P-values')
        axes[1].set_xlabel('Sample Position')
        axes[1].set_ylabel('P-value')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Summary information (instead of hiding it)
    axes[2].axis('off')
    
    # Create summary text
    summary_text = f"""System Configuration:
    • Buffer Size: {BUFFER_SIZE:,} samples
    • Detection Method: ShapeDD
    • P-value Threshold: {DRIFT_PVALUE}
    
Detection Summary:
    • Total Data Points: {len(features):,}
    • Drift Detections: {len(det_idx)}
    • Valid P-values: {len(p_vals[np.isfinite(p_vals)]) if len(p_vals) > 0 else 0}
    • Significant Detections: {len(p_vals[(p_vals < 0.05) & np.isfinite(p_vals)]) if len(p_vals) > 0 else 0}
    """
    
    if len(det_idx) > 0:
        summary_text += f"\nDetection Positions: {list(det_idx)}"
        if len(p_vals) > 0 and np.isfinite(p_vals).any():
            valid_p = p_vals[np.isfinite(p_vals)]
            summary_text += f"\nMin P-value: {np.min(valid_p):.6f}"
            summary_text += f"\nMean P-value: {np.mean(valid_p):.6f}"
    
    axes[2].text(0.05, 0.95, summary_text, transform=axes[2].transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


