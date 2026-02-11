"""
Real-time Drift Detection Visualization

Displays live data stream with aligned plots (following plot_detection.py style):
- Plot 1: Data stream with true drift indicators from generator
- Plot 2: Detection points with drift type classification
- Real-time updates with proper alignment

Usage:
    python plot_detection_realtime.py
"""

import os
import csv
import json
import time
from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from confluent_kafka import Consumer
import sys

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import shared configuration
from experiments.monitoring.config import BROKERS, TOPIC, SHAPEDD_LOG, ACCURACY_TOPIC

# Configuration
MAX_DISPLAY_POINTS = 3000  # Maximum points to display in plot
UPDATE_INTERVAL_MS = 500   # Update plot every 500ms
POLL_TIMEOUT_SEC = 0.05    # Kafka poll timeout

# Drift type colors (matching plot_detection.py style)
DRIFT_TYPE_COLORS = {
    'sudden': '#DC143C',        # Crimson red
    'incremental': '#FF8C00',   # Dark orange
    'gradual': '#FFD700',       # Gold
    'recurrent': '#32CD32',     # Lime green
    'blip': '#1E90FF',          # Dodger blue
    'undetermined': '#808080'   # Gray
}


class RealtimeDriftVisualizer:
    """Real-time visualization with accuracy tracking (matching DriftMonitoring.ipynb)."""

    def __init__(self, brokers, topic, accuracy_topic, log_path):
        self.brokers = brokers
        self.topic = topic
        self.accuracy_topic = accuracy_topic
        self.log_path = log_path

        # Data buffers (ordered by stream index)
        self.stream_indices = deque(maxlen=MAX_DISPLAY_POINTS)
        self.stream_features = deque(maxlen=MAX_DISPLAY_POINTS)
        self.stream_drift_indicators = deque(maxlen=MAX_DISPLAY_POINTS)

        # Accuracy tracking buffers
        self.accuracy_indices = deque(maxlen=MAX_DISPLAY_POINTS)
        self.accuracy_values = deque(maxlen=MAX_DISPLAY_POINTS)
        self.baseline_accuracy = None

        # Detection tracking
        self.detections = []  # List of {idx, p_value, type, category}
        self.last_processed_line = 0  # Track number of lines processed (not file position)

        # Kafka consumer for live data stream
        # Use dynamic group ID + 'earliest' to always start from beginning
        import time as time_module
        plot_group_id = f'realtime-plotter-{int(time_module.time())}'
        
        self.consumer = Consumer({
            'bootstrap.servers': brokers,
            'group.id': plot_group_id,  # Dynamic group ID for fresh start
            'auto.offset.reset': 'earliest',  # Start from beginning
            'enable.auto.commit': True
        })
        self.consumer.subscribe([topic])
        print(f"[RT-Plot] Data consumer group: {plot_group_id} (starting from earliest)")

        # Kafka consumer for accuracy metrics
        accuracy_group_id = f'realtime-accuracy-{int(time_module.time())}'
        self.accuracy_consumer = Consumer({
            'bootstrap.servers': brokers,
            'group.id': accuracy_group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        })
        self.accuracy_consumer.subscribe([accuracy_topic])
        print(f"[RT-Plot] Accuracy consumer group: {accuracy_group_id} (starting from earliest)")

        # Setup matplotlib (3-plot layout: data, detections, accuracy)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        self.setup_plots()

        # Statistics
        self.total_samples = 0
        self.total_detections = 0
        self.total_accuracy_points = 0
        self.last_status_time = time.time()

    def setup_plots(self):
        """Setup 3-plot layout matching DriftMonitoring.ipynb."""
        # Plot 1: Data stream with true drift
        self.ax1.set_title('Data Stream with True Drift Points (SEA Dataset)', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Feature Value (First Dimension)', fontsize=11)
        self.ax1.grid(True, alpha=0.3)

        # Plot 2: Detection signal
        self.ax2.set_title('ShapeDD Detection Points (Real-time)', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('Detection', fontsize=11)
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.grid(True, alpha=0.3)

        # Plot 3: Model Accuracy (NEW - matching notebook)
        self.ax3.set_title('Model Accuracy Over Time', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('Accuracy', fontsize=11)
        self.ax3.set_xlabel('Stream Index', fontsize=11)
        self.ax3.set_ylim(-0.05, 1.05)
        self.ax3.grid(True, alpha=0.3)

        # Main title
        self.fig.suptitle('Real-time Drift Detection and Model Performance', fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.96)

    def update_stream_data(self):
        """Poll Kafka for new streaming data."""
        new_samples = 0
        start_time = time.time()

        # Poll for up to 100ms or 100 messages
        while (time.time() - start_time) < 0.1 and new_samples < 100:
            msg = self.consumer.poll(timeout=POLL_TIMEOUT_SEC)
            if msg is None:
                break
            if msg.error():
                continue

            try:
                data = json.loads(msg.value().decode('utf-8'))
                x_vec = data.get('x', [])
                idx = data.get('idx', -1)
                drift_indicator = data.get('drift_indicator', 0)

                if x_vec and idx >= 0:
                    self.stream_indices.append(idx)
                    self.stream_features.append(x_vec[0])  # First feature
                    self.stream_drift_indicators.append(drift_indicator)
                    self.total_samples += 1
                    new_samples += 1
            except Exception:
                continue

        return new_samples

    def update_accuracy_data(self):
        """Poll Kafka for new accuracy metrics."""
        new_points = 0
        start_time = time.time()

        # Poll for up to 100ms or 50 messages
        while (time.time() - start_time) < 0.1 and new_points < 50:
            msg = self.accuracy_consumer.poll(timeout=POLL_TIMEOUT_SEC)
            if msg is None:
                break
            if msg.error():
                continue

            try:
                data = json.loads(msg.value().decode('utf-8'))
                idx = data.get('idx', -1)
                accuracy = data.get('accuracy', 0.0)
                baseline = data.get('baseline_accuracy', 0.0)

                if idx >= 0:
                    self.accuracy_indices.append(idx)
                    self.accuracy_values.append(accuracy)
                    if baseline > 0 and self.baseline_accuracy is None:
                        self.baseline_accuracy = baseline
                    self.total_accuracy_points += 1
                    new_points += 1
            except Exception:
                continue

        return new_points

    def update_detections(self):
        """Load new detections from CSV log, filtering to match visible data range."""
        if not os.path.exists(self.log_path):
            return 0

        new_detections = 0
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                reader = csv.DictReader(f)
                
                # Skip lines we've already processed
                current_line = 0
                for row in reader:
                    current_line += 1
                    
                    # Skip already processed lines
                    if current_line <= self.last_processed_line:
                        continue
                    
                    try:
                        det_idx = int(float(row.get('detection_idx', '0')))
                        p_value = float(row.get('p_value', '1.0'))
                        drift_type = row.get('drift_type', 'undetermined').strip()
                        drift_category = row.get('drift_category', 'undetermined').strip()

                        # Note: No filtering needed - CSV is cleared on restart
                        # All detections in CSV are from current run

                        # Avoid duplicates
                        if not any(d['idx'] == det_idx for d in self.detections):
                            self.detections.append({
                                'idx': det_idx,
                                'p_value': p_value,
                                'type': drift_type,
                                'category': drift_category
                            })
                            self.total_detections += 1
                            new_detections += 1
                            print(f"[RT-Plot] New detection: idx={det_idx}, type={drift_type}, p={p_value:.4f}")
                    except Exception as e:
                        continue

                # Update line counter to number of data lines processed (not including header)
                self.last_processed_line = current_line
        except Exception:
            pass

        return new_detections

    def plot_frame(self, frame):
        """Update plot for animation frame (3-plot layout with accuracy)."""
        # Update data from sources
        new_samples = self.update_stream_data()
        new_accuracy = self.update_accuracy_data()
        new_detections = self.update_detections()

        # Print status periodically
        if time.time() - self.last_status_time > 5:
            print(f"[RT-Plot] Samples: {self.total_samples}, Accuracy points: {self.total_accuracy_points}, "
                  f"Detections: {self.total_detections}, Buffer: {len(self.stream_indices)}/{MAX_DISPLAY_POINTS}")
            self.last_status_time = time.time()

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # === PLOT 1: Data Stream with True Drift ===
        if len(self.stream_indices) > 0:
            indices = np.array(self.stream_indices)
            features = np.array(self.stream_features)
            drift_indicators = np.array(self.stream_drift_indicators)

            # Scatter plot colored by drift indicator
            self.ax1.scatter(indices, features, c=drift_indicators, 
                           cmap='coolwarm', alpha=0.7, s=15)

            # Mark true drift transitions with vertical lines
            drift_transitions = np.where(np.diff(drift_indicators) != 0)[0] + 1
            for trans_idx in drift_transitions:
                if trans_idx < len(indices):
                    stream_pos = indices[trans_idx]
                    self.ax1.axvline(x=stream_pos, color='blue', linestyle='-', 
                                   alpha=0.6, linewidth=2)
                    self.ax1.text(stream_pos, self.ax1.get_ylim()[1]*0.9, 
                                f'True\n{stream_pos}', rotation=90, ha='center', 
                                va='top', fontsize=8, 
                                bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='blue', alpha=0.7))

            self.ax1.set_title('Data Stream with True Drift Points (SEA Dataset)', 
                             fontsize=12, fontweight='bold')
            self.ax1.set_ylabel('Feature Value (First Dimension)', fontsize=11)
            self.ax1.grid(True, alpha=0.3)

            # === PLOT 2: Detection Signal (Aligned) ===
            # Create detection signal array (same length as data)
            detection_signal = np.zeros(len(features))
            detection_positions = []

            if self.detections:
                for det in self.detections:
                    det_idx = det['idx']
                    # Find closest data point to detection
                    closest_pos = np.argmin(np.abs(indices - det_idx))
                    if closest_pos < len(detection_signal):
                        detection_signal[closest_pos] = 1
                        detection_positions.append((indices[closest_pos], det))

                # Plot detection signal with drift type colors
                for pos_idx, (x_pos, det) in enumerate(detection_positions):
                    drift_type = det['type']
                    color = DRIFT_TYPE_COLORS.get(drift_type, DRIFT_TYPE_COLORS['undetermined'])
                    
                    # Scatter point
                    self.ax2.scatter([x_pos], [1], c=[color], s=100, 
                                   edgecolors='black', linewidths=1.5, zorder=10)
                    
                    # Vertical line
                    self.ax2.axvline(x=x_pos, color=color, linestyle='--', 
                                   alpha=0.7, linewidth=2)
                    
                    # Label
                    self.ax2.text(x_pos, 0.8, f'{drift_type}\n{x_pos}', 
                                rotation=90, ha='center', va='center', fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor=color, alpha=0.7))

                # Fill remaining points with gray
                non_detection_mask = detection_signal == 0
                if np.any(non_detection_mask):
                    self.ax2.scatter(indices[non_detection_mask], 
                                   detection_signal[non_detection_mask], 
                                   c='gray', alpha=0.3, s=15)

                title_suffix = f'({len(self.detections)} detections)'
            else:
                # No detections yet - show all zeros
                self.ax2.scatter(indices, detection_signal, c='gray', alpha=0.3, s=15)
                title_suffix = '(No detections yet)'

            self.ax2.set_title(f'ShapeDD Detection Points {title_suffix}', 
                             fontsize=12, fontweight='bold')
            self.ax2.set_ylabel('Detection', fontsize=11)
            self.ax2.set_ylim(-0.1, 1.1)
            self.ax2.grid(True, alpha=0.3)

            # === PLOT 3: Model Accuracy Over Time (NEW) ===
            if len(self.accuracy_indices) > 0:
                acc_indices = np.array(self.accuracy_indices)
                acc_values = np.array(self.accuracy_values)
                
                # Plot accuracy line
                self.ax3.plot(acc_indices, acc_values, 'b-', linewidth=2, label='Prequential Accuracy')
                
                # Mark baseline accuracy (if available)
                if self.baseline_accuracy is not None:
                    self.ax3.axhline(y=self.baseline_accuracy, color='green', linestyle='--', 
                                   linewidth=2, alpha=0.7, label=f'Baseline ({self.baseline_accuracy:.3f})')
                
                # Mark drift detection points with vertical lines
                for det in self.detections:
                    det_idx = det['idx']
                    drift_type = det['type']
                    color = DRIFT_TYPE_COLORS.get(drift_type, DRIFT_TYPE_COLORS['undetermined'])
                    self.ax3.axvline(x=det_idx, color=color, linestyle='--', 
                                   alpha=0.6, linewidth=1.5)
                
                # Add current accuracy annotation
                if len(acc_values) > 0:
                    current_acc = acc_values[-1]
                    self.ax3.text(0.02, 0.95, f'Current: {current_acc:.4f}', 
                                transform=self.ax3.transAxes, 
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                self.ax3.legend(loc='upper right', fontsize=9)
                accuracy_title = f'Model Accuracy Over Time ({len(self.accuracy_indices)} points)'
            else:
                # No accuracy data yet
                self.ax3.text(0.5, 0.5, 'Waiting for accuracy data...', 
                            transform=self.ax3.transAxes, ha='center', va='center',
                            fontsize=14, color='gray')
                accuracy_title = 'Model Accuracy Over Time (No data yet)'
            
            self.ax3.set_title(accuracy_title, fontsize=12, fontweight='bold')
            self.ax3.set_ylabel('Accuracy', fontsize=11)
            self.ax3.set_xlabel('Stream Index', fontsize=11)
            self.ax3.set_ylim(-0.05, 1.05)
            self.ax3.grid(True, alpha=0.3)

            # Ensure all plots have same x-axis range
            x_min, x_max = indices.min(), indices.max()
            self.ax1.set_xlim(x_min, x_max)
            self.ax2.set_xlim(x_min, x_max)
            self.ax3.set_xlim(x_min, x_max)

        else:
            # No data yet - show waiting message
            self.ax1.text(0.5, 0.5, 'Waiting for data...', 
                        transform=self.ax1.transAxes, ha='center', va='center',
                        fontsize=14, color='gray')
            self.ax2.text(0.5, 0.5, 'Waiting for data...', 
                        transform=self.ax2.transAxes, ha='center', va='center',
                        fontsize=14, color='gray')
            self.ax3.text(0.5, 0.5, 'Waiting for data...', 
                        transform=self.ax3.transAxes, ha='center', va='center',
                        fontsize=14, color='gray')
            
            self.ax1.set_title('Data Stream with True Drift Points (SEA Dataset)', 
                             fontsize=12, fontweight='bold')
            self.ax2.set_title('ShapeDD Detection Points (Real-time)', 
                             fontsize=12, fontweight='bold')
            self.ax3.set_title('Model Accuracy Over Time', 
                             fontsize=12, fontweight='bold')
            self.ax3.set_xlabel('Stream Index', fontsize=11)

        # Update window title with stats
        self.fig.canvas.manager.set_window_title(
            f'Real-time Drift Monitor - {self.total_samples} samples, {self.total_detections} detections, Acc: {self.total_accuracy_points} pts'
        )

    def run(self):
        """Start the real-time visualization."""
        print("=" * 70)
        print("Real-time Drift Detection with Accuracy Tracking")
        print("=" * 70)
        print(f"Kafka Broker: {self.brokers}")
        print(f"Data Topic: {self.topic}")
        print(f"Accuracy Topic: {self.accuracy_topic}")
        print(f"Log File: {self.log_path}")
        print(f"Update Interval: {UPDATE_INTERVAL_MS}ms")
        print(f"Max Display Points: {MAX_DISPLAY_POINTS}")
        print("=" * 70)
        print("\nStarting visualization... Press Ctrl+C to stop.")
        print()

        # Create animation
        ani = FuncAnimation(
            self.fig,
            self.plot_frame,
            interval=UPDATE_INTERVAL_MS,
            blit=False,
            cache_frame_data=False
        )

        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n\nStopping visualization...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("Closing Kafka consumers...")
        self.consumer.close()
        self.accuracy_consumer.close()
        
        # Print final summary
        print("\n" + "=" * 70)
        print("Final Summary:")
        print(f"  Total samples processed: {self.total_samples:,}")
        print(f"  Total accuracy points: {self.total_accuracy_points:,}")
        print(f"  Total detections: {self.total_detections}")
        if self.baseline_accuracy is not None:
            print(f"  Baseline accuracy: {self.baseline_accuracy:.4f}")
        if len(self.accuracy_values) > 0:
            print(f"  Final accuracy: {self.accuracy_values[-1]:.4f}")
        if self.detections:
            print(f"  Detection indices: {[d['idx'] for d in self.detections]}")
            # Count by type
            type_counts = {}
            for det in self.detections:
                dtype = det['type']
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            print(f"  By type: {dict(type_counts)}")
        print("=" * 70)
        print("Done!")


def main():
    """Main entry point."""
    # Configuration from environment
    brokers = os.getenv("BROKERS", BROKERS)
    topic = os.getenv("TOPIC", TOPIC)
    accuracy_topic = os.getenv("ACCURACY_TOPIC", ACCURACY_TOPIC)
    log_path = os.getenv("SHAPEDD_LOG", SHAPEDD_LOG)

    # Validate log path
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Warning: Log file not found at {log_path}")
        print("Detections will appear once consumer starts writing to log.")
        print()

    # Create and run visualizer
    visualizer = RealtimeDriftVisualizer(brokers, topic, accuracy_topic, log_path)
    visualizer.run()


if __name__ == "__main__":
    main()
