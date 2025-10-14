"""
Advanced Real-time Drift Visualization with Multi-Feature Support

Features:
- Multi-dimensional feature visualization
- Drift type timeline
- Detection latency analysis
- P-value tracking
- Performance metrics

Usage:
    python plot_detection_advanced.py [--features 2]
"""

import os
import csv
import json
import time
import argparse
from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from confluent_kafka import Consumer

# Import shared configuration
from config import BROKERS, TOPIC, SHAPEDD_LOG

# Configuration
MAX_DISPLAY_POINTS = 2000
UPDATE_INTERVAL_MS = 1000
POLL_TIMEOUT_SEC = 0.1

# Drift type visualization
DRIFT_TYPE_COLORS = {
    'sudden': '#FF4444', 'incremental': '#FF9944', 'gradual': '#FFDD44',
    'recurrent': '#44FF44', 'blip': '#4444FF', 'undetermined': '#888888'
}

DRIFT_CATEGORY_COLORS = {
    'TCD': '#FF6666',  # Transient - Red
    'PCD': '#6666FF',  # Progressive - Blue
    'undetermined': '#888888'
}


class AdvancedDriftVisualizer:
    """Advanced real-time drift visualization with detailed analytics."""

    def __init__(self, brokers, topic, log_path, n_features=2):
        self.brokers = brokers
        self.topic = topic
        self.log_path = log_path
        self.n_features = n_features

        # Data buffers
        self.stream_indices = deque(maxlen=MAX_DISPLAY_POINTS)
        self.stream_features = deque(maxlen=MAX_DISPLAY_POINTS)  # Will store all features
        self.stream_drift_indicators = deque(maxlen=MAX_DISPLAY_POINTS)

        # Detection tracking
        self.detections = []
        self.detection_timeline = deque(maxlen=100)  # Track last 100 detections
        self.last_log_position = 0

        # Performance metrics
        self.true_drift_positions = []
        self.detected_drift_positions = []
        self.p_values = deque(maxlen=100)

        # Kafka consumer
        self.consumer = Consumer({
            'bootstrap.servers': brokers,
            'group.id': 'advanced-plotter',
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True
        })
        self.consumer.subscribe([topic])

        # Statistics
        self.total_samples = 0
        self.total_detections = 0
        self.last_update_time = time.time()

        # Setup matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=(18, 12))
        self.setup_plots()

    def setup_plots(self):
        """Setup advanced subplot layout."""
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.4, wspace=0.4)

        # Row 1: Feature plots
        self.ax_features = []
        for i in range(min(self.n_features, 3)):
            ax = self.fig.add_subplot(gs[0, i])
            ax.set_title(f'Feature {i}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            self.ax_features.append(ax)

        # Row 2: Main combined view
        self.ax_main = self.fig.add_subplot(gs[1, :])
        self.ax_main.set_title('Combined Feature View with Drift Detection', fontsize=12, fontweight='bold')
        self.ax_main.set_ylabel('Normalized Value', fontsize=10)
        self.ax_main.grid(True, alpha=0.3)

        # Row 3: Analytics
        self.ax_timeline = self.fig.add_subplot(gs[2, :2])
        self.ax_timeline.set_title('Drift Type Timeline', fontsize=11, fontweight='bold')
        self.ax_timeline.set_xlabel('Detection Number', fontsize=9)
        self.ax_timeline.set_ylabel('Drift Type', fontsize=9)

        self.ax_pvalue = self.fig.add_subplot(gs[2, 2])
        self.ax_pvalue.set_title('P-value History', fontsize=11, fontweight='bold')
        self.ax_pvalue.set_xlabel('Detection #', fontsize=9)
        self.ax_pvalue.set_ylabel('P-value', fontsize=9)
        self.ax_pvalue.set_yscale('log')

        # Row 4: Statistics and metrics
        self.ax_distribution = self.fig.add_subplot(gs[3, 0])
        self.ax_distribution.set_title('Drift Type Count', fontsize=10, fontweight='bold')

        self.ax_category = self.fig.add_subplot(gs[3, 1])
        self.ax_category.set_title('TCD vs PCD', fontsize=10, fontweight='bold')

        self.ax_stats = self.fig.add_subplot(gs[3, 2])
        self.ax_stats.set_title('Metrics', fontsize=10, fontweight='bold')
        self.ax_stats.axis('off')

        # Overall title
        self.fig.suptitle('Advanced Drift Detection Monitor', fontsize=16, fontweight='bold')

        # Legend
        legend_elements = [
            mpatches.Patch(color=DRIFT_TYPE_COLORS['sudden'], label='Sudden'),
            mpatches.Patch(color=DRIFT_TYPE_COLORS['incremental'], label='Incremental'),
            mpatches.Patch(color=DRIFT_TYPE_COLORS['gradual'], label='Gradual'),
            mpatches.Patch(color=DRIFT_TYPE_COLORS['recurrent'], label='Recurrent'),
            mpatches.Patch(color=DRIFT_TYPE_COLORS['blip'], label='Blip'),
        ]
        self.fig.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=5, framealpha=0.9)

    def update_stream_data(self):
        """Poll Kafka for new data."""
        new_samples = 0
        start_time = time.time()

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
                    self.stream_features.append(x_vec[:self.n_features])  # Store multiple features
                    self.stream_drift_indicators.append(drift_indicator)

                    # Track true drift positions
                    if drift_indicator == 1 and (not self.true_drift_positions or idx > self.true_drift_positions[-1]):
                        self.true_drift_positions.append(idx)

                    self.total_samples += 1
                    new_samples += 1
            except Exception:
                continue

        return new_samples

    def update_detections(self):
        """Load new detections from CSV."""
        if not os.path.exists(self.log_path):
            return 0

        new_detections = 0
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                f.seek(self.last_log_position)
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        det_idx = int(float(row.get('detection_idx', '0')))
                        p_value = float(row.get('p_value', '1.0'))
                        drift_type = row.get('drift_type', 'undetermined').strip()
                        drift_category = row.get('drift_category', 'undetermined').strip()

                        if not any(d['idx'] == det_idx for d in self.detections):
                            detection = {
                                'idx': det_idx,
                                'p_value': p_value,
                                'type': drift_type,
                                'category': drift_category,
                                'timestamp': time.time()
                            }
                            self.detections.append(detection)
                            self.detection_timeline.append(detection)
                            self.p_values.append(p_value)
                            self.detected_drift_positions.append(det_idx)
                            self.total_detections += 1
                            new_detections += 1
                    except Exception:
                        continue

                self.last_log_position = f.tell()
        except Exception:
            pass

        return new_detections

    def plot_frame(self, frame):
        """Update all plots."""
        # Update data
        new_samples = self.update_stream_data()
        new_detections = self.update_detections()

        # Clear all axes
        for ax in self.ax_features:
            ax.clear()
        self.ax_main.clear()
        self.ax_timeline.clear()
        self.ax_pvalue.clear()
        self.ax_distribution.clear()
        self.ax_category.clear()
        self.ax_stats.clear()

        if len(self.stream_indices) == 0:
            return

        indices = np.array(self.stream_indices)
        features = np.array(self.stream_features)
        drift_indicators = np.array(self.stream_drift_indicators)

        # === Individual Feature Plots ===
        for i, ax in enumerate(self.ax_features):
            if i < features.shape[1]:
                # Plot feature with true drift background
                ax.scatter(indices, features[:, i], c=drift_indicators,
                          cmap='coolwarm', alpha=0.6, s=15)

                # Mark detections
                for det in self.detections:
                    if det['idx'] in indices:
                        pos = np.where(indices == det['idx'])[0][0]
                        ax.axvline(det['idx'], color=DRIFT_TYPE_COLORS[det['type']],
                                  linestyle='--', alpha=0.7, linewidth=2)

                ax.set_title(f'Feature {i}', fontsize=10, fontweight='bold')
                ax.set_ylabel('Value', fontsize=9)
                ax.grid(True, alpha=0.3)

        # === Main Combined Plot ===
        # Normalize features for combined view
        if features.shape[1] > 0:
            features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

            for i in range(min(self.n_features, 3)):
                if i < features_norm.shape[1]:
                    self.ax_main.plot(indices, features_norm[:, i],
                                     alpha=0.6, linewidth=1, label=f'Feature {i}')

            # Mark true drift
            drift_changes = np.where(np.diff(drift_indicators) != 0)[0]
            for change_idx in drift_changes:
                if change_idx < len(indices) - 1:
                    self.ax_main.axvline(indices[change_idx + 1], color='blue',
                                        linestyle='-', alpha=0.2, linewidth=3)

            # Mark detections with type
            for det in self.detections:
                color = DRIFT_TYPE_COLORS[det['type']]
                self.ax_main.axvline(det['idx'], color=color, linestyle='--',
                                    alpha=0.7, linewidth=2, label=f"{det['type']} @{det['idx']}")

                # Add annotation
                y_pos = self.ax_main.get_ylim()[1] * 0.9
                self.ax_main.text(det['idx'], y_pos, det['type'],
                                 rotation=90, ha='right', va='top', fontsize=7,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

        self.ax_main.set_title('Combined Feature View with Drift Detection', fontsize=12, fontweight='bold')
        self.ax_main.set_xlabel('Stream Index', fontsize=10)
        self.ax_main.set_ylabel('Normalized Value', fontsize=10)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend(loc='upper left', fontsize=7, ncol=4)

        # === Drift Type Timeline ===
        if self.detection_timeline:
            timeline = list(self.detection_timeline)
            det_numbers = list(range(len(timeline)))
            drift_types = [d['type'] for d in timeline]

            # Create color-coded scatter
            for i, (det_num, dtype) in enumerate(zip(det_numbers, drift_types)):
                self.ax_timeline.scatter(det_num, i, c=DRIFT_TYPE_COLORS[dtype],
                                        s=100, edgecolors='black', linewidths=1)
                self.ax_timeline.text(det_num, i, dtype[:3], ha='center', va='center',
                                     fontsize=6, fontweight='bold')

            self.ax_timeline.set_yticks(range(len(timeline)))
            self.ax_timeline.set_yticklabels([f"#{i+1}" for i in range(len(timeline))], fontsize=7)
            self.ax_timeline.set_xlabel('Detection Sequence', fontsize=9)

        self.ax_timeline.set_title('Drift Type Timeline', fontsize=11, fontweight='bold')
        self.ax_timeline.grid(True, alpha=0.3, axis='x')

        # === P-value History ===
        if self.p_values:
            pvals = list(self.p_values)
            self.ax_pvalue.plot(range(len(pvals)), pvals, 'o-', color='purple', alpha=0.7)
            self.ax_pvalue.axhline(y=0.05, color='red', linestyle='--',
                                   alpha=0.5, label='α=0.05')
            self.ax_pvalue.set_ylabel('P-value (log scale)', fontsize=9)
            self.ax_pvalue.legend(fontsize=7)

        self.ax_pvalue.set_title('P-value History', fontsize=11, fontweight='bold')
        self.ax_pvalue.grid(True, alpha=0.3)

        # === Drift Type Distribution ===
        if self.detections:
            type_counts = {}
            for det in self.detections:
                dtype = det['type']
                type_counts[dtype] = type_counts.get(dtype, 0) + 1

            types = sorted(type_counts.keys())
            counts = [type_counts[t] for t in types]
            colors = [DRIFT_TYPE_COLORS[t] for t in types]

            self.ax_distribution.barh(types, counts, color=colors, alpha=0.8, edgecolor='black')
            for i, (t, c) in enumerate(zip(types, counts)):
                self.ax_distribution.text(c, i, f' {c}', va='center', fontsize=9, fontweight='bold')

        self.ax_distribution.set_title('Drift Type Count', fontsize=10, fontweight='bold')
        self.ax_distribution.set_xlabel('Count', fontsize=9)

        # === TCD vs PCD ===
        if self.detections:
            category_counts = {}
            for det in self.detections:
                cat = det['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1

            categories = list(category_counts.keys())
            counts = [category_counts[c] for c in categories]
            colors = [DRIFT_CATEGORY_COLORS.get(c, '#888888') for c in categories]

            self.ax_category.pie(counts, labels=categories, colors=colors,
                                autopct='%1.0f%%', startangle=90,
                                textprops={'fontsize': 9, 'fontweight': 'bold'})

        self.ax_category.set_title('TCD vs PCD', fontsize=10, fontweight='bold')

        # === Statistics ===
        self.ax_stats.axis('off')

        # Calculate metrics
        update_rate = self.total_samples / max(1, time.time() - self.last_update_time) if hasattr(self, 'last_update_time') else 0
        detection_rate = len(self.detections) / max(1, self.total_samples) * 1000 if self.total_samples > 0 else 0

        stats_text = f"""
Samples: {self.total_samples:,}
Detections: {self.total_detections}
True Drifts: {len(self.true_drift_positions)}

Sampling Rate:
  {update_rate:.1f} samples/sec

Detection Rate:
  {detection_rate:.2f} per 1K samples

Buffer: {len(self.stream_indices)}/{MAX_DISPLAY_POINTS}

Status: {'🟢 LIVE' if new_samples > 0 else '⏸ IDLE'}
        """

        self.ax_stats.text(0.1, 0.9, stats_text.strip(), transform=self.ax_stats.transAxes,
                          fontsize=9, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        self.ax_stats.set_title('Metrics', fontsize=10, fontweight='bold')

    def run(self):
        """Start visualization."""
        print("=" * 70)
        print("Advanced Drift Detection Visualization")
        print("=" * 70)
        print(f"Kafka: {self.brokers}")
        print(f"Topic: {self.topic}")
        print(f"Log: {self.log_path}")
        print(f"Features: {self.n_features}")
        print("=" * 70)
        print("\nPress Ctrl+C to stop.\n")

        ani = FuncAnimation(self.fig, self.plot_frame, interval=UPDATE_INTERVAL_MS,
                           blit=False, cache_frame_data=False)

        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.consumer.close()
            print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Advanced drift detection visualization')
    parser.add_argument('--features', type=int, default=2, help='Number of features to display')
    args = parser.parse_args()

    brokers = os.getenv("BROKERS", BROKERS)
    topic = os.getenv("TOPIC", TOPIC)
    log_path = os.getenv("SHAPEDD_LOG", SHAPEDD_LOG)

    visualizer = AdvancedDriftVisualizer(brokers, topic, log_path, n_features=args.features)
    visualizer.run()


if __name__ == "__main__":
    main()
