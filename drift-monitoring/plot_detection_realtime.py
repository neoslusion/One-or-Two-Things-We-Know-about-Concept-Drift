"""
Real-time Drift Detection and Classification Visualization

Displays live data stream with:
- Feature values over time
- Drift detection points with type classification
- Color-coded drift types (sudden, incremental, gradual, recurrent, blip)
- True drift indicators from generator

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
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from confluent_kafka import Consumer

# Import shared configuration
from config import BROKERS, TOPIC, SHAPEDD_LOG

# Configuration
MAX_DISPLAY_POINTS = 2000  # Maximum points to display in plot
UPDATE_INTERVAL_MS = 1000  # Update plot every 1 second
POLL_TIMEOUT_SEC = 0.1     # Kafka poll timeout

# Drift type colors (matching professional visualization standards)
DRIFT_TYPE_COLORS = {
    'sudden': '#FF4444',        # Bright red - immediate action
    'incremental': '#FF9944',   # Orange - gradual change
    'gradual': '#FFDD44',       # Yellow - slow oscillation
    'recurrent': '#44FF44',     # Green - pattern repeat
    'blip': '#4444FF',          # Blue - temporary noise
    'undetermined': '#888888'   # Gray - unknown
}

DRIFT_TYPE_MARKERS = {
    'sudden': 'v',         # Triangle down - sudden drop
    'incremental': '^',    # Triangle up - gradual rise
    'gradual': 's',        # Square - stable oscillation
    'recurrent': 'o',      # Circle - recurring
    'blip': 'x',           # X - noise
    'undetermined': 'd'    # Diamond - unknown
}


class RealtimeDriftVisualizer:
    """Real-time visualization of drift detection with type classification."""

    def __init__(self, brokers, topic, log_path):
        self.brokers = brokers
        self.topic = topic
        self.log_path = log_path

        # Data buffers
        self.stream_indices = deque(maxlen=MAX_DISPLAY_POINTS)
        self.stream_features = deque(maxlen=MAX_DISPLAY_POINTS)
        self.stream_drift_indicators = deque(maxlen=MAX_DISPLAY_POINTS)

        # Detection tracking
        self.detections = []  # List of {idx, p_value, type, category}
        self.last_log_position = 0

        # Kafka consumer for live data
        self.consumer = Consumer({
            'bootstrap.servers': brokers,
            'group.id': 'realtime-plotter',
            'auto.offset.reset': 'latest',  # Only new data
            'enable.auto.commit': True
        })
        self.consumer.subscribe([topic])

        # Setup matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=(16, 10))
        self.setup_plots()

        # Statistics
        self.total_samples = 0
        self.total_detections = 0

    def setup_plots(self):
        """Setup the subplot layout."""
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Main plot: Data stream with detections
        self.ax_main = self.fig.add_subplot(gs[0:2, :])
        self.ax_main.set_title('Real-time Data Stream with Drift Detection', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('Stream Index', fontsize=11)
        self.ax_main.set_ylabel('Feature Value (First Dimension)', fontsize=11)
        self.ax_main.grid(True, alpha=0.3)

        # Bottom left: Drift type distribution
        self.ax_types = self.fig.add_subplot(gs[2, 0])
        self.ax_types.set_title('Drift Type Distribution', fontsize=11, fontweight='bold')
        self.ax_types.set_ylabel('Count', fontsize=10)

        # Bottom right: Statistics
        self.ax_stats = self.fig.add_subplot(gs[2, 1])
        self.ax_stats.set_title('Real-time Statistics', fontsize=11, fontweight='bold')
        self.ax_stats.axis('off')

        # Add legend for drift types
        legend_elements = [
            mpatches.Patch(color=DRIFT_TYPE_COLORS['sudden'], label='Sudden (TCD)'),
            mpatches.Patch(color=DRIFT_TYPE_COLORS['incremental'], label='Incremental (PCD)'),
            mpatches.Patch(color=DRIFT_TYPE_COLORS['gradual'], label='Gradual (PCD)'),
            mpatches.Patch(color=DRIFT_TYPE_COLORS['recurrent'], label='Recurrent (PCD)'),
            mpatches.Patch(color=DRIFT_TYPE_COLORS['blip'], label='Blip (noise)'),
            mpatches.Patch(color='blue', label='True Drift (gen_random)', alpha=0.3)
        ]
        self.fig.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

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
                drift_indicator = data.get('drift', 0)

                if x_vec and idx >= 0:
                    self.stream_indices.append(idx)
                    self.stream_features.append(x_vec[0])  # First feature for visualization
                    self.stream_drift_indicators.append(drift_indicator)
                    self.total_samples += 1
                    new_samples += 1
            except Exception as e:
                continue

        return new_samples

    def update_detections(self):
        """Load new detections from CSV log."""
        if not os.path.exists(self.log_path):
            return 0

        new_detections = 0
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                # Skip to last known position
                f.seek(self.last_log_position)

                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        det_idx = int(float(row.get('detection_idx', '0')))
                        p_value = float(row.get('p_value', '1.0'))
                        drift_type = row.get('drift_type', 'undetermined').strip()
                        drift_category = row.get('drift_category', 'undetermined').strip()

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
                    except Exception:
                        continue

                # Update position
                self.last_log_position = f.tell()
        except Exception as e:
            pass

        return new_detections

    def plot_frame(self, frame):
        """Update plot for animation frame."""
        # Update data from sources
        new_samples = self.update_stream_data()
        new_detections = self.update_detections()

        # Clear axes
        self.ax_main.clear()
        self.ax_types.clear()
        self.ax_stats.clear()

        # === Main Plot: Data Stream ===
        if len(self.stream_indices) > 0:
            indices = np.array(self.stream_indices)
            features = np.array(self.stream_features)
            drift_indicators = np.array(self.stream_drift_indicators)

            # Plot data points colored by true drift indicator
            scatter = self.ax_main.scatter(
                indices, features,
                c=drift_indicators,
                cmap='coolwarm',
                alpha=0.5,
                s=20,
                label='Data points'
            )

            # Mark true drift transitions with vertical lines
            drift_changes = np.where(np.diff(drift_indicators) != 0)[0]
            for change_idx in drift_changes:
                if change_idx < len(indices) - 1:
                    x_pos = indices[change_idx + 1]
                    self.ax_main.axvline(
                        x=x_pos,
                        color='blue',
                        linestyle='-',
                        alpha=0.3,
                        linewidth=2,
                        label='True drift' if change_idx == drift_changes[0] else None
                    )

            # Plot detections with drift type markers
            for detection in self.detections:
                det_idx = detection['idx']
                drift_type = detection['type']

                # Find closest point in stream for y-position
                if len(indices) > 0:
                    closest_pos = np.argmin(np.abs(indices - det_idx))
                    y_pos = features[closest_pos] if closest_pos < len(features) else 0
                else:
                    y_pos = 0

                # Plot detection marker
                color = DRIFT_TYPE_COLORS.get(drift_type, DRIFT_TYPE_COLORS['undetermined'])
                marker = DRIFT_TYPE_MARKERS.get(drift_type, DRIFT_TYPE_MARKERS['undetermined'])

                self.ax_main.scatter(
                    [det_idx], [y_pos],
                    c=[color],
                    marker=marker,
                    s=200,
                    edgecolors='black',
                    linewidths=2,
                    zorder=10,
                    alpha=0.9
                )

                # Add vertical line for detection
                self.ax_main.axvline(
                    x=det_idx,
                    color=color,
                    linestyle='--',
                    alpha=0.6,
                    linewidth=1.5
                )

                # Add label
                self.ax_main.text(
                    det_idx,
                    self.ax_main.get_ylim()[1] * 0.95,
                    f'{drift_type}\n{det_idx}',
                    rotation=90,
                    ha='right',
                    va='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='black')
                )

        self.ax_main.set_title(
            f'Real-time Data Stream with Drift Detection (Live)',
            fontsize=14,
            fontweight='bold'
        )
        self.ax_main.set_xlabel('Stream Index', fontsize=11)
        self.ax_main.set_ylabel('Feature Value (First Dimension)', fontsize=11)
        self.ax_main.grid(True, alpha=0.3)

        # === Drift Type Distribution ===
        if self.detections:
            type_counts = {}
            for det in self.detections:
                dtype = det['type']
                type_counts[dtype] = type_counts.get(dtype, 0) + 1

            # Sort by predefined order
            drift_types = ['sudden', 'incremental', 'gradual', 'recurrent', 'blip', 'undetermined']
            types = [t for t in drift_types if t in type_counts]
            counts = [type_counts[t] for t in types]
            colors = [DRIFT_TYPE_COLORS[t] for t in types]

            bars = self.ax_types.bar(types, counts, color=colors, alpha=0.8, edgecolor='black')

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                self.ax_types.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{int(count)}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

            self.ax_types.set_ylabel('Count', fontsize=10)
            self.ax_types.set_xticklabels(types, rotation=45, ha='right', fontsize=9)
        else:
            self.ax_types.text(
                0.5, 0.5,
                'No detections yet',
                ha='center',
                va='center',
                transform=self.ax_types.transAxes,
                fontsize=12,
                color='gray'
            )

        self.ax_types.set_title('Drift Type Distribution', fontsize=11, fontweight='bold')

        # === Statistics Panel ===
        self.ax_stats.axis('off')

        stats_text = f"""
        Total Samples: {self.total_samples:,}
        Total Detections: {self.total_detections}

        Buffer Size: {len(self.stream_indices)} / {MAX_DISPLAY_POINTS}

        Latest Detection:
        """

        if self.detections:
            latest = self.detections[-1]
            stats_text += f"""  Index: {latest['idx']}
          Type: {latest['type']}
          Category: {latest['category']}
          P-value: {latest['p_value']:.6f}
        """
        else:
            stats_text += "  None yet"

        stats_text += f"""

        Status: {'🟢 LIVE' if new_samples > 0 else '🔴 WAITING'}
        Update Rate: ~{UPDATE_INTERVAL_MS}ms
        """

        self.ax_stats.text(
            0.05, 0.95,
            stats_text,
            transform=self.ax_stats.transAxes,
            fontsize=10,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

        self.ax_stats.set_title('Real-time Statistics', fontsize=11, fontweight='bold')

        # Update window title with status
        self.fig.canvas.manager.set_window_title(
            f'Drift Detection Monitor - {self.total_samples} samples, {self.total_detections} detections'
        )

    def run(self):
        """Start the real-time visualization."""
        print("=" * 60)
        print("Real-time Drift Detection Visualization")
        print("=" * 60)
        print(f"Kafka Broker: {self.brokers}")
        print(f"Data Topic: {self.topic}")
        print(f"Log File: {self.log_path}")
        print(f"Update Interval: {UPDATE_INTERVAL_MS}ms")
        print(f"Max Display Points: {MAX_DISPLAY_POINTS}")
        print("=" * 60)
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
        print("Closing Kafka consumer...")
        self.consumer.close()
        print("Done!")


def main():
    """Main entry point."""
    # Configuration from environment
    brokers = os.getenv("BROKERS", BROKERS)
    topic = os.getenv("TOPIC", TOPIC)
    log_path = os.getenv("SHAPEDD_LOG", SHAPEDD_LOG)

    # Validate log path
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Warning: Log file not found at {log_path}")
        print("Detections will appear once consumer starts writing to log.")
        print()

    # Create and run visualizer
    visualizer = RealtimeDriftVisualizer(brokers, topic, log_path)
    visualizer.run()


if __name__ == "__main__":
    main()
